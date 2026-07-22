import time

import numpy as np

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.cereal import messaging

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.lib.common import W, H

# modeld's tinygrad warp kernels bake the device (VENUS-aligned) NV12 layout at
# compile time: stride=align(W,128), uv_offset=stride*align(H,32). A packed
# stride=W buffer therefore reads as row-shifted garbage (and out-of-bounds) to
# the model -- the sim must allocate and fill device-identical buffers. #30693
STRIDE, Y_HEIGHT, UV_HEIGHT, YUV_SIZE = get_nv12_info(W, H)
UV_OFFSET = STRIDE * Y_HEIGHT


def rgb_to_nv12(rgb):
  """Convert RGB image to NV12 (YUV420) format using BT.601 coefficients."""
  h, w = rgb.shape[:2]
  r = rgb[:, :, 0].astype(np.int32)
  g = rgb[:, :, 1].astype(np.int32)
  b = rgb[:, :, 2].astype(np.int32)

  # Y plane - BT.601 coefficients (matches original OpenCL kernel)
  y = (((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16
  y = np.clip(y, 0, 255).astype(np.uint8)

  # Subsample RGB for UV (2x2 box filter)
  r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2

  # U and V planes
  u = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)

  # Interleave UV for NV12 format
  uv = np.empty((h // 2, w), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v

  # pack into the VENUS-aligned layout the model warp expects (row-padded planes)
  buf = np.zeros(YUV_SIZE, dtype=np.uint8)
  buf[:h * STRIDE].reshape(h, STRIDE)[:, :w] = y
  buf[UV_OFFSET:UV_OFFSET + (h // 2) * STRIDE].reshape(h // 2, STRIDE)[:, :w] = uv
  return buf.tobytes()


class Camerad:
  """Simulates the camerad daemon"""
  def __init__(self, dual_camera):
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    # device-identical NV12 buffers (same call as system/camerad/cameras/camera_common.cc)
    self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_ROAD, 5, W, H, YUV_SIZE, STRIDE, UV_OFFSET)
    if dual_camera:
      self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, W, H, YUV_SIZE, STRIDE, UV_OFFSET)

    self.vipc_server.start_listener()

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def rgb_to_yuv(self, rgb):
    """Convert RGB to NV12 YUV format."""
    assert rgb.shape == (H, W, 3), f"{rgb.shape}"
    assert rgb.dtype == np.uint8
    return rgb_to_nv12(rgb)

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    # Use wall-clock monotonic time (same clock as the simulated IMU's logMonoTime).
    # frame_id*0.05 is a sim-time clock starting at 0, which is thousands of seconds
    # behind the IMU-driven filter clock in locationd, so cameraOdometry was always
    # rejected as "older than the max rewind threshold" and the pose filter never
    # validated (no engagement). See issue #30693.
    eof = int(time.monotonic() * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)
