import signal
import threading
import functools
import numpy as np

from collections import namedtuple
from enum import Enum
from multiprocessing import Process, Queue, Value
from abc import ABC, abstractmethod
from opendbc.car.honda.values import CruiseButtons
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.tools.sim.lib.common import SimulatorState, World
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors

QueueMessage = namedtuple("QueueMessage", ["type", "info"], defaults=[None])

class QueueMessageType(Enum):
  START_STATUS = 0
  CONTROL_COMMAND = 1
  TERMINATION_INFO = 2
  CLOSE_STATUS = 3

def control_cmd_gen(cmd: str):
  return QueueMessage(QueueMessageType.CONTROL_COMMAND, cmd)

def rk_loop(function, hz, exit_event: threading.Event):
  rk = Ratekeeper(hz, None)
  while not exit_event.is_set():
    function()
    rk.keep_time()


class SimulatorBridge(ABC):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality):
    set_params_enabled()
    self.params = Params()
    self.params.put_bool("AlphaLongitudinalEnabled", True, block=True)

    self.rk = Ratekeeper(100, None)

    self.dual_camera = dual_camera
    self.high_quality = high_quality

    self._exit_event: threading.Event | None = None
    self._threads = []
    self._keep_alive = True
    self.started = Value('i', False)
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self.simulator_state = SimulatorState()

    self.world: World | None = None

    self.past_startup_engaged = False
    self.startup_button_prev = True

    self.test_run = False

  def _on_shutdown(self, signal, frame):
    self.shutdown()

  def shutdown(self):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      self._run(q)
    finally:
      self.close("bridge terminated")

  def close(self, reason):
    self.started.value = False

    if self._exit_event is not None:
      self._exit_event.set()

    if self.world is not None:
      self.world.close(reason)

  def run(self, queue, retries=-1):
    bridge_p = Process(name="bridge", target=self.bridge_keep_alive, args=(queue, retries))
    bridge_p.start()
    return bridge_p

  def print_status(self):
    print(
    f"""
State:
Ignition: {self.simulator_state.ignition} Engaged: {self.simulator_state.is_engaged}
    """)

  @abstractmethod
  def spawn_world(self, q: Queue, /) -> World:
    pass

  def _run(self, q: Queue):
    self.world = self.spawn_world(q)

    self.simulated_car = SimulatedCar()
    self.simulated_sensors = SimulatedSensors(self.dual_camera)

    # diagnostic-only: watch what the planner/model actually want (#30693 vehicle_not_moving)
    try:
      import openpilot.cereal.messaging as _messaging
      self._dbg_sm = _messaging.SubMaster(['longitudinalPlan', 'modelV2', 'carState', 'selfdriveState'])
    except Exception:
      self._dbg_sm = None

    self._exit_event = threading.Event()

    self.simulated_car_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_car.update, self.simulator_state),
                                                                        100, self._exit_event))
    self.simulated_car_thread.start()

    self.simulated_camera_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_sensors.send_camera_images, self.world),
                                                                        20, self._exit_event))
    self.simulated_camera_thread.start()

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      self.world.tick()

    while self._keep_alive:
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0

      self.simulator_state.cruise_button = 0
      self.simulator_state.left_blinker = False
      self.simulator_state.right_blinker = False

      throttle_manual = steer_manual = brake_manual = 0.

      # Read manual controls
      if not q.empty():
        message = q.get()
        if message.type == QueueMessageType.CONTROL_COMMAND:
          m = message.info.split('_')
          if m[0] == "steer":
            steer_manual = float(m[1])
          elif m[0] == "throttle":
            throttle_manual = float(m[1])
          elif m[0] == "brake":
            brake_manual = float(m[1])
          elif m[0] == "cruise":
            if m[1] == "down":
              self.simulator_state.cruise_button = CruiseButtons.DECEL_SET
            elif m[1] == "up":
              self.simulator_state.cruise_button = CruiseButtons.RES_ACCEL
            elif m[1] == "cancel":
              self.simulator_state.cruise_button = CruiseButtons.CANCEL
            elif m[1] == "main":
              self.simulator_state.cruise_button = CruiseButtons.MAIN
          elif m[0] == "blinker":
            if m[1] == "left":
              self.simulator_state.left_blinker = True
            elif m[1] == "right":
              self.simulator_state.right_blinker = True
          elif m[0] == "ignition":
            self.simulator_state.ignition = not self.simulator_state.ignition
          elif m[0] == "reset":
            self.world.reset()
          elif m[0] == "quit":
            break

      self.simulator_state.user_brake = brake_manual
      self.simulator_state.user_gas = throttle_manual
      self.simulator_state.user_torque = steer_manual * -10000

      steer_manual = steer_manual * -40

      # Update openpilot on current sensor state
      self.simulated_sensors.update(self.simulator_state, self.world)

      self.simulated_car.sm.update(0)
      self.simulator_state.is_engaged = self.simulated_car.sm['selfdriveState'].active

      # always-on model diagnostic (does modeld hallucinate a lead?), not gated by engagement #30693
      try:
        self._dbg2 = getattr(self, "_dbg2", 0) + 1
        if self._dbg2 % 50 == 0 and self._dbg_sm is not None:
          self._dbg_sm.update(0)
          mv = self._dbg_sm['modelV2']
          leads = mv.leadsV3
          lp = max((ld.prob for ld in leads), default=-1.0)
          lx = leads[0].x[0] if len(leads) and len(leads[0].x) else -1.0
          print(f"[mdl] engaged={self.simulator_state.is_engaged} leadProb={lp:.2f} leadX={lx:.1f} "
                f"mdlA={mv.action.desiredAcceleration:+.3f} mdlCurv={mv.action.desiredCurvature:+.4f}", flush=True)
      except Exception:
        pass

      if self.simulator_state.is_engaged:
        accel_cmd = self.simulated_car.sm['carControl'].actuators.accel
        throttle_op = np.clip(accel_cmd / 1.6, 0.0, 1.0)
        brake_op = np.clip(-accel_cmd / 4.0, 0.0, 1.0)
        steer_op = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

        # defensive diagnostic (must never crash the bridge): is openpilot commanding
        # forward accel, or holding/braking? distinguishes perception vs actuation. #30693
        try:
          self._dbgn = getattr(self, "_dbgn", 0) + 1
          if self._dbgn % 40 == 0:
            plan_v = plan_a = mdl_a = mdl_c = -99.0
            has_lead = "?"
            v_set = long_src = exp_mode = "?"
            if self._dbg_sm is not None:
              self._dbg_sm.update(0)
              lp = self._dbg_sm['longitudinalPlan']
              if len(lp.speeds): plan_v = lp.speeds[-1]
              if len(lp.accels): plan_a = lp.accels[-1]
              has_lead = lp.hasLead
              long_src = lp.longitudinalPlanSource
              act = self._dbg_sm['modelV2'].action
              mdl_a, mdl_c = act.desiredAcceleration, act.desiredCurvature
              v_set = f"{self._dbg_sm['carState'].cruiseState.speed:.1f}"
              exp_mode = self._dbg_sm['selfdriveState'].experimentalMode
            print(f"[lng] accel={accel_cmd:+.3f} vEgo={self.simulator_state.speed:.2f} "
                  f"planV={plan_v:.2f} planA={plan_a:+.3f} lead={has_lead} src={long_src} "
                  f"vSet={v_set} exp={exp_mode} mdlA={mdl_a:+.3f} mdlCurv={mdl_c:+.4f}", flush=True)
        except Exception:
          pass

        self.past_startup_engaged = True
      elif not self.past_startup_engaged and self.simulated_car.sm['selfdriveState'].engageable:
        self.simulator_state.cruise_button = CruiseButtons.DECEL_SET if self.startup_button_prev else CruiseButtons.MAIN # force engagement on startup
        self.startup_button_prev = not self.startup_button_prev

      throttle_out = throttle_op if self.simulator_state.is_engaged else throttle_manual
      brake_out = brake_op if self.simulator_state.is_engaged else brake_manual
      steer_out = steer_op if self.simulator_state.is_engaged else steer_manual

      self.world.apply_controls(steer_out, throttle_out, brake_out)
      self.world.read_state()
      self.world.read_sensors(self.simulator_state)

      if self.world.exit_event.is_set():
        self.shutdown()

      if self.rk.frame % self.TICKS_PER_FRAME == 0:
        self.world.tick()
        self.world.read_cameras()

      # don't print during test, so no print/IO Block between OP and metadrive processes
      if not self.test_run and self.rk.frame % 25 == 0:
        self.print_status()

      self.started.value = True

      self.rk.keep_time()
