#!/usr/bin/env python3
"""Probe which Panda3D render backends work on this host and whether any is
hardware-accelerated (Metal via ANGLE / MoltenVK+Zink), and at what fps.

Goal (#30693): the free macOS runner's OpenGL falls back to Apple's software
renderer; if we can route Panda3D through Metal we can render full quality fast
and drop the SIMPLE_RENDER/low-res/flat-card workarounds.
"""
import os
import sys
import time


def probe_backend(backend, extra_prc=""):
  # each backend in a subprocess: a failed GSG init can hard-crash
  from panda3d.core import loadPrcFileData, GraphicsPipeSelection, GraphicsEngine
  from panda3d.core import FrameBufferProperties, WindowProperties, GraphicsPipe, Texture, GraphicsOutput
  loadPrcFileData("", "window-type none")
  loadPrcFileData("", "notify-level-display warning")
  loadPrcFileData("", f"load-display {backend}")
  if extra_prc:
    loadPrcFileData("", extra_prc)

  sel = GraphicsPipeSelection.get_global_ptr()
  sel.load_aux_modules()
  pipe = sel.make_pipe(backend) if hasattr(sel, "make_pipe") else sel.make_module_pipe(backend)
  if pipe is None:
    print(f"  pipe=None (backend {backend} not loadable)")
    return
  engine = GraphicsEngine.get_global_ptr()
  fb = FrameBufferProperties()
  fb.set_rgba_bits(8, 8, 8, 8)
  fb.set_depth_bits(24)
  win = engine.make_output(pipe, "probe", 0, fb, WindowProperties.size(512, 256),
                           GraphicsPipe.BF_refuse_window, None)
  if win is None:
    print(f"  make_output=None (no buffer for {backend})")
    return
  engine.open_windows()
  gsg = win.get_gsg()
  if gsg is None:
    print(f"  gsg=None for {backend}")
    return
  print(f"  VENDOR   = {gsg.get_driver_vendor()}")
  print(f"  RENDERER = {gsg.get_driver_renderer()}")
  print(f"  VERSION  = {gsg.get_driver_version()}")


def main():
  backend = sys.argv[1] if len(sys.argv) > 1 else "pandagl"
  extra = sys.argv[2] if len(sys.argv) > 2 else ""
  print(f"[probe] backend={backend} extra_prc={extra!r}")
  try:
    probe_backend(backend, extra)
  except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[probe] {backend} FAILED: {e}")


if __name__ == "__main__":
  main()
