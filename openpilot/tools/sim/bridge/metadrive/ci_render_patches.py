"""Env-gated render simplifications for running the MetaDrive bridge on CI.

Free GitHub Actions runners have no GPU; Mesa LLVMpipe software rendering
cannot sustain the default MetaDrive pipeline in real time (issue #30693).
Each flag removes render work without changing what openpilot's camera sees
structurally (road layout, lane lines, horizon).

Must be called before the MetaDrive engine is created — Prc settings load at
engine init and the terrain patch replaces a method used during reset.
"""
import os
import platform


def _apply_macos_headless_patches():
  """macOS GitHub runners have no display and no hardware OpenGL (GL falls back to
  Apple's software renderer). Panda3D's Cocoa offscreen path needs several fixes to
  open a headless software-GL buffer and render the scene. Metal compute (modeld) is
  hardware-accelerated separately. See issue #30693."""
  import metadrive.engine.core.engine_core as ec
  # metadrive forces offscreen->onscreen on mac (needs a real window, impossible
  # headless); keep it offscreen. Side effect: it then picks simplepbr use_330=False
  # (GLSL 120) which won't compile on Apple's GL 4.1 core profile -> force use_330.
  ec.is_mac = lambda: False
  import metadrive.third_party.simplepbr as spbr
  # keep the scene PBR shader (_recompile_pbr), skip the main-window tonemap
  # post-process (FilterManager.render_scene_into returns None headless -> crash)
  spbr.Pipeline._setup_tonemapping = lambda self: setattr(self, "tonemap_quad", None)
  _orig_spbr_init = spbr.init
  ec.init = lambda **kw: _orig_spbr_init(**{**kw, "use_330": True})
  # ShowBase's default framebuffer props (multisample/stencil/srgb) make the first
  # Cocoa offscreen buffer fail with "no usable pixel format"; inject minimal props
  from direct.showbase import ShowBase as SB
  from panda3d.core import FrameBufferProperties
  _orig_doopen = SB.ShowBase._doOpenWindow
  def _doopen(self, *a, **kw):
    if kw.get("fbprops") is None:
      fb = FrameBufferProperties()
      fb.setRgbColor(True)
      fb.setRgbaBits(8, 8, 8, 8)
      fb.setDepthBits(24)
      kw["fbprops"] = fb
    return _orig_doopen(self, *a, **kw)
  SB.ShowBase._doOpenWindow = _doopen
  # PSSM shadow buffer can't be created headless; skip it entirely (the CI terrain
  # shader is flat-lit and samples no shadow map)
  from metadrive.engine.core.pssm import PSSM
  def _pssm_skip(self):
    self.use_pssm = False
    try:
      self.engine.render.set_shader_inputs(use_pssm=False)
    except Exception:
      pass
  PSSM.init = _pssm_skip


def apply_ci_render_patches():
  if platform.system() == "Darwin":
    _apply_macos_headless_patches()


  if os.environ.get("METADRIVE_NO_MSAA"):
    # metadrive's EngineCore forces 8x MSAA at import time; Prc settings are
    # last-write-wins, so this must load before the engine is created
    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "framebuffer-multisample 0")
    loadPrcFileData("", "multisamples 0")

  if os.environ.get("METADRIVE_NO_SHADOWS"):
    # PSSM renders the scene into a 2-split shadow atlas every frame;
    # deactivate the shadow buffer but keep its shader inputs bound so the
    # stock terrain shader stays valid
    from metadrive.engine.core.pssm import PSSM
    pssm_init_orig = PSSM.init
    def pssm_init_no_render(self):
      pssm_init_orig(self)
      self.buffer.set_active(False)
      self.use_pssm = False
      self.engine.render.set_shader_inputs(use_pssm=False)
    PSSM.init = pssm_init_no_render

  if os.environ.get("METADRIVE_FLAT_TERRAIN_CARD"):
    # the PG-map terrain is flat (physics already uses a plane); replace the
    # chunked ShaderTerrainMesh — whose draw path dominates LLVMpipe frame
    # time — with a single flat quad carrying the same shader inputs
    from metadrive.constants import CameraTagStateKey, CamMask
    from metadrive.engine.core.terrain import Terrain
    from panda3d.core import Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter, Shader

    def _gen_card(self, size, heightfield, attribute_tex, target_triangle_width=10, engine=None):
      engine = engine or self.engine
      vdata = GeomVertexData("terrain_card", GeomVertexFormat.getV3t2(), Geom.UHStatic)
      vdata.setNumRows(4)
      vw = GeomVertexWriter(vdata, "vertex")
      uw = GeomVertexWriter(vdata, "texcoord")
      for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
        vw.addData3(x, y, 0)
        uw.addData2(x, y)
      tris = GeomTriangles(Geom.UHStatic)
      tris.addVertices(0, 1, 2)
      tris.addVertices(0, 2, 3)
      geom = Geom(vdata)
      geom.addPrimitive(tris)
      node = GeomNode("terrain_card")
      node.addGeom(geom)
      self._mesh_terrain = self.origin.attach_new_node(node)
      self._mesh_terrain.setTwoSided(True)
      # the main window camera's RGB tag state applies the stock terrain
      # shader, which asserts on ShaderTerrainMesh.* inputs only the real
      # terrain node provides — hide the card from it (the openpilot sensor
      # cameras carry their own tag state with the card shaders)
      self._mesh_terrain.hide(CamMask.MainCam)
      here = os.path.dirname(os.path.abspath(__file__))
      self._mesh_terrain.set_shader(Shader.load(Shader.SL_GLSL,
                                                os.path.join(here, "terrain_card.vert.glsl"),
                                                os.path.join(here, "terrain_ci.frag.glsl")))
      self._mesh_terrain.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.RGB, self.SEMANTIC_LABEL)
      self._mesh_terrain.setTag(CameraTagStateKey.Depth, self.SEMANTIC_LABEL)
      self._terrain_shader_set = False  # rebind inputs to the new node
      self._set_terrain_shader(engine, attribute_tex)
      self._mesh_terrain.set_scale(size, size, 1)
      self._mesh_terrain.set_pos(-size / 2, -size / 2, 0)

    Terrain._generate_mesh_vis_terrain = _gen_card
