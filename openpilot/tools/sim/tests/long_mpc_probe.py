#!/usr/bin/env python3
"""Standalone probe: does the acados longitudinal MPC solve a simple launch on this host?

Solves "accelerate from standstill to v_cruise, no lead" 100 times -- the exact
problem the sim's planner fails on the macOS CI runner (QP error status 3,
issue #30693). Independent of the rest of the openpilot stack.
"""
from collections import Counter

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc

def main():
  mpc = LongitudinalMpc(dt=0.05)
  radar_state = log.RadarState.new_message().as_reader()
  personality = log.LongitudinalPersonality.standard

  # mpc.run() calls reset() on failure, which ZEROES solution_status before we can
  # read it -- count resets to see the true failure count
  resets = {"n": 0}
  orig_reset = mpc.reset
  def counting_reset():
    resets["n"] += 1
    orig_reset()
  mpc.reset = counting_reset

  statuses: Counter = Counter()
  v_ego, a_ego = 0.0, 0.0
  for i in range(100):
    mpc.set_weights(prev_accel_constraint=(i > 0), personality=personality)
    mpc.set_cur_state(v_ego, a_ego)
    mpc.update(radar_state, v_cruise=11.1, personality=personality)
    statuses[mpc.solution_status] += 1
    # roll the state forward along the solution like the planner does
    v_ego = float(np.interp(0.05, np.linspace(0, 10, len(mpc.v_solution)), mpc.v_solution))
    a_ego = float(np.interp(0.05, np.linspace(0, 10, len(mpc.a_solution)), mpc.a_solution))

  print(f"[probe] statuses over 100 launch solves: {dict(statuses)} resets={resets['n']}")
  print(f"[probe] final v={v_ego:.2f} a={a_ego:.2f}")
  print(f"[probe] {'PASS' if resets['n'] == 0 and v_ego > 0.05 else 'FAIL'}")

if __name__ == "__main__":
  main()
