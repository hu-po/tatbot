"""MJCF assembly: the blue arm, its pen and wrist camera, the scene camera, and whatever the episode adds.

A scene is text plus an in-memory asset dict (meshes and textures as bytes),
compiled once per episode. Everything that moves during an episode is either
an arm hinge or a mocap body, so a step is ``qpos``/``mocap`` writes and
kinematics -- there is no physics to integrate.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from tatbot_travel import assets
from tatbot_travel.camera import RenderPlan
from tatbot_travel.tool import PenGeometry, load_pen, pen_meshes
from tatbot_travel.urdf_chain import Chain, Visual, load_chain, to_mjcf

ARM = "left"  # the blue arm (config/arm-labels.json)
CAMERA = "wrist"
SCENE_CAMERA = "scene"
# Geoms only the scene camera draws: the wrist view shows the pen cradle from real pixels (``selfview``).
# MuJoCo hides geom group 3 unless a render's options enable it.
SCENE_ONLY_GROUP = 3
GAP_SITE = "gap_point"
LENS_SITE = "lens_face"
OPTICAL_FRAME = f"{ARM}/realsense_color_optical_frame"
PEN_BODY = f"{ARM}/tattoo_pen"
ARM_LINKS = {"base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6",
             "carriage_left", "carriage_right", "realsense_mount_d405"}
# Appearance groups that belong to the robot; everything else is the world.
ROBOT_MATERIALS = ("arm", "d405", "cradle", "cube", "shell", "chrome", "lens")


def material_class(visual: Visual, *, render_cradle: bool,
                   scene_view: bool = False) -> str | tuple[str, int] | None:
    """Which appearance group a URDF visual belongs to (``None`` hides it).

    The pen's URDF cylinders are replaced by a textured lathe of the same
    datasheet profile, and the pen cradle (with its clamp cap) is normally
    drawn from real pixels (``selfview``), so both are hidden from the wrist
    camera unless asked for; with a scene camera they are drawn in
    ``SCENE_ONLY_GROUP`` for it alone. The tag cube on the same mount is out
    of the wrist camera's view.
    """
    link = visual.link.split("/", 1)[-1]
    if link in ARM_LINKS:
        return "arm"
    if link == "realsense_link":
        return "d405"
    if link == "ee_mount":
        material = "cube" if "fiducial_cube" in str(visual.params.get("file", "")) else "cradle"
        if render_cradle:
            return material
        return (material, SCENE_ONLY_GROUP) if scene_view else None
    return None


def scene_camera_xml(pose: np.ndarray, base_pos, fovy_deg: float) -> str:
    """The scene camera, fixed on the rig: ``pose`` is its optical frame (x right, y down, z forward) in the
    arm base frame, which MuJoCo's camera frame (y up, looking down -z) flips about x."""
    matrix = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    pos = np.asarray(base_pos, dtype=float) + pose[:3, 3]
    return (f'<camera name="{SCENE_CAMERA}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
            f'quat="{w:.7f} {x:.7f} {y:.7f} {z:.7f}" fovy="{fovy_deg:.6f}"/>')


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).save(buffer, format="PNG")
    return buffer.getvalue()


def _rgba(values) -> str:
    return " ".join(f"{float(c):.4g}" for c in values)


@dataclass
class Material:
    rgba: tuple[float, float, float, float]
    specular: float = 0.3
    shininess: float = 0.3
    reflectance: float = 0.0
    emission: float = 0.0
    texture: str | None = None
    texrepeat: tuple[float, float] | None = None

    def xml(self, name: str) -> str:
        attrs = [f'name="{name}"', f'rgba="{_rgba(self.rgba)}"', f'specular="{self.specular:.4g}"',
                 f'shininess="{self.shininess:.4g}"', f'reflectance="{self.reflectance:.4g}"',
                 f'emission="{self.emission:.4g}"']
        if self.texture:
            attrs.append(f'texture="{self.texture}"')
        if self.texrepeat:
            attrs.append(f'texrepeat="{self.texrepeat[0]:.4g} {self.texrepeat[1]:.4g}" texuniform="true"')
        return f"<material {' '.join(attrs)}/>"


def default_robot_materials() -> dict[str, Material]:
    return {
        "arm": Material((0.10, 0.10, 0.10, 1), 0.35, 0.4),
        "d405": Material((0.12, 0.12, 0.13, 1), 0.3, 0.3),
        "cradle": Material((0.12, 0.12, 0.13, 1), 0.2, 0.2),
        "cube": Material((0.10, 0.10, 0.11, 1), 0.15, 0.1),
        "shell": Material((0.94, 0.94, 0.93, 1), 0.4, 0.5),
        "chrome": Material((0.75, 0.77, 0.80, 1), 1.0, 1.0),
        "lens": Material((0.08, 0.10, 0.14, 1), 1.0, 1.0),
    }


@dataclass
class SceneBuilder:
    """Accumulates MJCF fragments and asset bytes for one episode's scene."""

    plan: RenderPlan
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_plan: RenderPlan | None = None  # the scene camera's render, when the episode has one
    scene_pose: np.ndarray | None = None  # its optical frame in the arm base frame (4x4)
    gap_m: float = 0.020
    carriage: float = 0.0
    render_cradle: bool = False
    materials: dict[str, Material] = field(default_factory=default_robot_materials)
    assets: dict[str, bytes] = field(default_factory=dict)
    asset_xml: list[str] = field(default_factory=list)
    world_xml: list[str] = field(default_factory=list)
    lights_xml: list[str] = field(default_factory=list)
    skybox: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.6, 0.6, 0.62), (0.2, 0.2, 0.22))
    headlight: tuple[float, float] = (0.35, 0.25)  # diffuse, ambient
    chain: Chain | None = None
    pen: PenGeometry | None = None

    def __post_init__(self):
        self.chain = self.chain or load_chain(assets.urdf_path(), ARM)
        self.pen = self.pen or load_pen()

    def add_texture(self, name: str, rgb: np.ndarray) -> None:
        self.assets[f"{name}.png"] = png_bytes(rgb)
        self.asset_xml.append(f'<texture name="{name}" type="2d" file="{name}.png"/>')

    def add_mesh_obj(self, name: str, obj_text: str) -> None:
        self.assets[f"{name}.obj"] = obj_text.encode()
        self.asset_xml.append(f'<mesh name="{name}" file="{name}.obj" inertia="shell"/>')

    def _pen_geoms(self) -> str:
        geoms = []
        for name, text in pen_meshes(self.pen).items():
            if f"{name}.obj" not in self.assets:
                self.add_mesh_obj(name, text)
            material = name.split("_")[1]
            geoms.append(f'<geom type="mesh" mesh="{name}" material="{material}" contype="0" '
                         f'conaffinity="0" group="1" density="0"/>')
        return "".join(geoms)

    def arm_extra(self) -> dict[str, str]:
        lens = self.pen.lens_z
        return {
            OPTICAL_FRAME: f'<camera name="{CAMERA}" quat="0 1 0 0" fovy="{self.plan.fovy_deg:.6f}"/>',
            PEN_BODY: (self._pen_geoms()
                       + f'<site name="{LENS_SITE}" pos="0 0 {lens:.6f}" size="0.001" group="5"/>'
                       + f'<site name="{GAP_SITE}" pos="0 0 {lens + self.gap_m:.6f}" size="0.001" group="5"/>'),
        }

    def xml(self) -> str:
        scene_view = self.scene_plan is not None
        arm = to_mjcf(self.chain, base_pos=self.base_pos, carriage=self.carriage,
                      material_for=lambda v: material_class(v, render_cradle=self.render_cradle,
                                                            scene_view=scene_view),
                      extra=self.arm_extra())
        for name, path in arm.mesh_files.items():
            self.assets[f"{name}.stl"] = path.read_bytes()
        top, bottom = self.skybox
        materials = "\n    ".join(m.xml(n) for n, m in self.materials.items())
        hd, ha = self.headlight
        newline = "\n    "
        plans = [self.plan] + ([self.scene_plan] if scene_view else [])
        offwidth, offheight = max(p.width for p in plans), max(p.height for p in plans)
        scene_camera = (scene_camera_xml(self.scene_pose, self.base_pos, self.scene_plan.fovy_deg)
                        if scene_view else "")
        return f"""<mujoco model="travel">
  <compiler angle="radian" autolimits="true" boundmass="0.001" boundinertia="1e-7"/>
  <statistic extent="1" center="0.3 0 0.1"/>
  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.004" zfar="30" haze="0"/>
    <headlight diffuse="{hd:.4g} {hd:.4g} {hd:.4g}" ambient="{ha:.4g} {ha:.4g} {ha:.4g}" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{_rgba(top)}" rgb2="{_rgba(bottom)}"
             width="256" height="1536"/>
    {arm.meshes}
    {newline.join(self.asset_xml)}
    {materials}
  </asset>
  <worldbody>
    {newline.join(self.lights_xml)}
{arm.body}
    {scene_camera}
    {newline.join(self.world_xml)}
  </worldbody>
</mujoco>"""

    def compile(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(self.xml(), self.assets)
