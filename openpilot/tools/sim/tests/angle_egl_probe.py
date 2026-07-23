#!/usr/bin/env python3
"""Minimal EGL+ANGLE probe (no panda3d): does ANGLE give HARDWARE Metal rendering
on this host? Loads Chrome's libEGL/libGLESv2 (ANGLE), makes a pbuffer context,
and prints GL_RENDERER. If it says 'ANGLE (Apple, ... Metal ...)' the paravirtual
GPU renders on hardware and a panda3d-GLES2 build is worth it. #30693
"""
import ctypes
import ctypes.util
import sys

EGL_DEFAULT_DISPLAY = 0
EGL_NO_CONTEXT = 0
EGL_NO_SURFACE = 0
EGL_PBUFFER_BIT = 0x0001
EGL_OPENGL_ES2_BIT = 0x0004
EGL_SURFACE_TYPE = 0x3033
EGL_RENDERABLE_TYPE = 0x3040
EGL_RED_SIZE = 0x3024
EGL_GREEN_SIZE = 0x3023
EGL_BLUE_SIZE = 0x3022
EGL_NONE = 0x3038
EGL_WIDTH = 0x3057
EGL_HEIGHT = 0x3056
EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_OPENGL_ES_API = 0x30A0
GL_RENDERER = 0x1F01
GL_VENDOR = 0x1F00
GL_VERSION = 0x1F02


def main(egl_path, gles_path):
  gles = ctypes.CDLL(gles_path, mode=ctypes.RTLD_GLOBAL)
  egl = ctypes.CDLL(egl_path, mode=ctypes.RTLD_GLOBAL)

  egl.eglGetDisplay.restype = ctypes.c_void_p
  egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
  dpy = egl.eglGetDisplay(ctypes.c_void_p(EGL_DEFAULT_DISPLAY))
  print(f"display={dpy}")

  major = ctypes.c_int(); minor = ctypes.c_int()
  if not egl.eglInitialize(ctypes.c_void_p(dpy), ctypes.byref(major), ctypes.byref(minor)):
    print(f"eglInitialize FAILED err=0x{egl.eglGetError():x}"); return
  print(f"EGL {major.value}.{minor.value}")

  egl.eglQueryString.restype = ctypes.c_char_p
  egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
  print("EGL_VENDOR:", egl.eglQueryString(ctypes.c_void_p(dpy), 0x3053))
  print("EGL_VERSION:", egl.eglQueryString(ctypes.c_void_p(dpy), 0x3054))

  cfg_attrs = (ctypes.c_int * 11)(
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
    EGL_NONE)
  config = ctypes.c_void_p(); num = ctypes.c_int()
  egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
  if not egl.eglChooseConfig(ctypes.c_void_p(dpy), cfg_attrs, ctypes.byref(config), 1, ctypes.byref(num)) or num.value == 0:
    print(f"eglChooseConfig FAILED err=0x{egl.eglGetError():x}"); return

  egl.eglBindAPI(ctypes.c_int(EGL_OPENGL_ES_API))
  pb_attrs = (ctypes.c_int * 5)(EGL_WIDTH, 512, EGL_HEIGHT, 256, EGL_NONE)
  egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
  egl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  surf = egl.eglCreatePbufferSurface(ctypes.c_void_p(dpy), config, pb_attrs)

  ctx_attrs = (ctypes.c_int * 3)(EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE)
  egl.eglCreateContext.restype = ctypes.c_void_p
  egl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  ctx = egl.eglCreateContext(ctypes.c_void_p(dpy), config, ctypes.c_void_p(EGL_NO_CONTEXT), ctx_attrs)
  if not ctx:
    print(f"eglCreateContext FAILED err=0x{egl.eglGetError():x}"); return

  egl.eglMakeCurrent.argtypes = [ctypes.c_void_p]*4
  if not egl.eglMakeCurrent(ctypes.c_void_p(dpy), ctypes.c_void_p(surf), ctypes.c_void_p(surf), ctypes.c_void_p(ctx)):
    print(f"eglMakeCurrent FAILED err=0x{egl.eglGetError():x}"); return

  gles.glGetString.restype = ctypes.c_char_p
  gles.glGetString.argtypes = [ctypes.c_int]
  print("GL_VENDOR  :", gles.glGetString(GL_VENDOR))
  print("GL_RENDERER:", gles.glGetString(GL_RENDERER))
  print("GL_VERSION :", gles.glGetString(GL_VERSION))


if __name__ == "__main__":
  try:
    main(sys.argv[1], sys.argv[2])
  except Exception as e:
    import traceback; traceback.print_exc(); print("PROBE FAILED:", e)
