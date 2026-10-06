"""One episode's world: every appearance and geometry choice, sampled from the episode seed.

The robot is fixed (the blue arm, its pen, its wrist camera, and the scene
camera on the rig); everything else is randomised: the table, the room behind
it and the people in it, the rig's clutter (racks, gear, paper, cables, tag
boards), the lights, the phantom's size and skin, the ink on it (Sharpie
lines, tatbot flash tattoos, or both), the hands that move the phantom, the
pen's chrome reflections, both cameras' mounting and calibration spread, and
how the pen cradle's real-pixel layer is lit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tatbot_travel import ink, realbank, textures
from tatbot_travel.appearance import (
    LightingConfig,
    SkinConfig,
    SurfaceConfig,
    sample_lighting,
    sample_skin,
    sample_surface,
)
from tatbot_travel.camera import SCENE_CAMERA_POSE, Intrinsics, RenderPlan, render_plan
from tatbot_travel.lines import sample_chart_curve
from tatbot_travel.people import build_hand, standing_body
from tatbot_travel.phantom import Phantom, build_phantom, obj_text
from tatbot_travel.scene import ARM, CAMERA, SCENE_ONLY_GROUP, Material, SceneBuilder
from tatbot_travel.selfview import SelfViewLook
from tatbot_travel.shell import SkinShell, chart_skin
from tatbot_travel.tabletop import TabletopConfig, TabletopItem, add_tabletop

HIDDEN = np.array([0.0, 0.0, -10.0])  # where unused mocap bodies wait
# Scene-camera render density (pixels per unit of normalized coordinate): its sub stream has 405 x 493 per
# axis; a little under keeps the render at 1136x712 for a soft, compressed stream.
SCENE_RENDER_FOCAL = 400.0
WORKSPACE = np.array([0.35, 0.0, 0.08])  # the middle of where the phantom lies, for keeping clutter out of view


@dataclass(frozen=True)
class WorldConfig:
    gap_m: float = 0.020
    # The arm's base over the table the phantom lies on: 0-4 cm on the travel case's extrusion, and on the
    # lab rig the table sits ~3.3 cm above the base.
    base_height_m: tuple[float, float] = (-0.06, 0.02)  # the lab rig's table is 3.4 cm over the base plane
    phantom_scale_along: tuple[float, float] = (0.92, 1.08)
    phantom_scale_across: tuple[float, float] = (0.9, 1.12)
    handless_prob: float = 0.5  # the rig's practice arm ends at the wrist; the demo's phantom has a hand
    focal_jitter: float = 0.015
    centre_jitter_px: float = 3.0
    mount_jitter_m: float = 0.0015
    mount_jitter_deg: float = 1.0
    max_people: int = 3
    max_objects: int = 4
    max_lights: int = 4
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    tabletop: TabletopConfig = field(default_factory=TabletopConfig)
    skin: SkinConfig = field(default_factory=SkinConfig)
    skin_seed: int | None = None  # review skin alone while holding the rest of the episode fixed
    appearance_seed: int | None = None  # override appearance while holding geometry/ink fixed for review
    ink_modes: dict = field(default_factory=lambda: {"lines": 0.35, "tattoos": 0.35, "both": 0.30})
    max_tattoos: int = 6
    held_out_designs: bool = False  # tattoo with the validation/test flash (evaluation) instead of the train flash
    min_ink_m: float = 0.08  # redraw the ink if less stroke than this is traceable
    scene_view: bool = True  # also render the third-person scene camera
    scene_mount_jitter_m: float = 0.01  # the scene camera's calibration spread and knocks
    scene_mount_jitter_deg: float = 1.5
    scene_focal_jitter: float = 0.03
    scene_centre_jitter_px: float = 6.0
    structures: tuple[int, int] = (2, 8)  # racks, shelves, cabinets, monitors beside and behind the arm
    max_gear: int = 16  # boxes and cylinders on the table
    max_sheets: int = 4
    max_cables: int = 5
    max_tags: int = 3
    # Real pixels (``realbank``), where a bank is set up: onto walls, rig clutter, the table and the forearm.
    real_assets: str | None = None  # the bank's directory (default: $TATBOT_TRAVEL_REAL_ASSETS)
    real_wall_prob: float = 0.5
    real_structure_prob: float = 0.4
    real_table_prob: float = 0.3
    real_ink_prob: float = 0.5
    max_real_ink: int = 3
    real_ink_only_prob: float = 0.0
    phantom_length_m: float = 0.63
    phantom_hand_pose: str = "open"  # open or closed; ignored for a handless phantom
    ink_theta_origin_rad: float = 0.0  # chart centre: back = 0, inner forearm = pi
    skin_rgb: tuple[float, float, float] | None = None  # legacy center override for skin.center_rgb
    room_assets: str | None = None  # private RGB-D room meshes and their textures
    selfview_assets: str | None = None  # private RGBA cradle layers on the real pixel grid
    selfview_gain: tuple[float, float] = (0.55, 1.5)
    wrist_robot_visible: bool = True  # a complete private self-view may replace every CAD visual


@dataclass
class World:
    model: mujoco.MjModel
    plan: RenderPlan
    intrinsics: Intrinsics
    phantom: Phantom
    ink: ink.InkGraph
    selfview: SelfViewLook
    mocap: dict[str, int]
    hand_offsets: list[np.ndarray]  # 4x4 phantom_from_hand, one per hand
    tabletop: list[TabletopItem] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    scene_plan: RenderPlan | None = None  # the scene camera's render and remap, when it is on
    scene_intrinsics: Intrinsics | None = None


def _rgba(c) -> tuple[float, float, float, float]:
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def _geom(mesh: str, material: str) -> str:
    return f'<geom type="mesh" mesh="{mesh}" material="{material}" contype="0" conaffinity="0" density="0"/>'


def _mocap(name: str, geoms: str) -> str:
    return f'<body name="{name}" mocap="true" pos="{HIDDEN[0]} {HIDDEN[1]} {HIDDEN[2]}">{geoms}</body>'


def _real_or(bank, kind: str, prob: float, rng: np.random.Generator, procedural, size: int = 256) -> np.ndarray:
    """A real crop of ``kind`` this often (when the bank has any), else the procedural texture."""
    if bank is not None and bank.has(kind) and rng.random() < prob:
        return bank.texture(rng, kind, size)
    return procedural(rng)


def _add_room(sb: SceneBuilder, rng: np.random.Generator, cfg: WorldConfig, bank=None,
              *, geometry_rng: np.random.Generator | None = None) -> dict:
    geometry_rng = rng if geometry_rng is None else geometry_rng
    real_table = bank is not None and bank.has("table") and rng.random() < cfg.real_table_prob
    look = sample_surface(rng, cfg.surface)
    texture = None
    rgba = _rgba(look["rgb"])
    if real_table or look["style"] != "plain":
        rgb = (bank.texture(rng, "table", 512) if real_table else
               textures.table_texture(rng, kind=look["style"], colour=np.asarray(look["rgb"]) * 255))
        sb.add_texture("table_tex", rgb)
        texture, rgba = "table_tex", (1, 1, 1, 1)
    if real_table:
        look["style"] = "photo"
        look["rgb"] = None
    repeat = float(rng.uniform(2.5, 4.0) if real_table else rng.uniform(1.0, 6.0))  # a real crop is ~0.3 m of mat
    sb.materials["table"] = Material(rgba, look["specular"], look["shininess"],
                                     texture=texture, texrepeat=(repeat, repeat) if texture else None)
    sb.world_xml.append('<geom name="table" type="box" size="0.9 1.1 0.02" pos="0.35 0 -0.02" material="table" '
                        'contype="0" conaffinity="0"/>')
    for i, (pos, size) in enumerate([((2.4, 0, 1.0), (0.05, 3.0, 1.6)), ((0, 2.6, 1.0), (3.0, 0.05, 1.6)),
                                     ((0, -2.6, 1.0), (3.0, 0.05, 1.6)), ((-2.2, 0, 1.0), (0.05, 3.0, 1.6))]):
        sb.add_texture(f"wall{i}_tex", _real_or(bank, "wrist_bg", cfg.real_wall_prob, rng, textures.wall_texture))
        sb.materials[f"wall{i}"] = Material((1, 1, 1, 1), 0.05, 0.1, texture=f"wall{i}_tex", texrepeat=(1, 1))
        shift = geometry_rng.uniform(-0.6, 0.6, 3) * np.array([1, 1, 0])
        p = np.asarray(pos) + shift
        if cfg.room_assets is not None:
            p *= 4  # fallback walls must sit behind the captured room's distant backing
        sb.world_xml.append(f'<geom type="box" pos="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" size="{size[0]} {size[1]} '
                            f'{size[2]}" material="wall{i}" contype="0" conaffinity="0"/>')
    floor = sample_surface(rng, cfg.surface)
    floor["style"] = "plain"
    sb.materials["floor"] = Material(_rgba(floor["rgb"]), floor["specular"], floor["shininess"])
    sb.world_xml.append('<geom name="floor" type="plane" pos="0 0 -0.75" size="6 6 0.1" material="floor" '
                        'contype="0" conaffinity="0"/>')
    top, bottom = rng.uniform(0.2, 1.0, 3), rng.uniform(0.05, 0.6, 3)
    sb.skybox = (tuple(top), tuple(bottom))
    return {"workspace_surface": look, "floor": floor}


def captured_floor_parts(obj: str, tolerance_m: float) -> tuple[str, str, int]:
    """Partition horizontal faces near the measured table plane; preserve all OBJ coordinates and UVs."""
    lines = obj.splitlines()
    vertices = np.asarray([list(map(float, line.split()[1:4])) for line in lines if line.startswith("v ")])
    faces = [line for line in lines if line.startswith("f ")]
    indices = np.asarray([[int(corner.split("/")[0]) - 1 for corner in line.split()[1:]] for line in faces])
    triangles = vertices[indices]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    horizontal = np.abs(normals[:, 2]) >= 0.8 * np.linalg.norm(normals, axis=1)
    floor = horizontal & (np.abs(triangles[..., 2]).max(axis=1) <= tolerance_m)
    header = "\n".join(line for line in lines if not line.startswith("f ")) + "\n"
    parts = [header + "\n".join(line for line, keep in zip(faces, floor, strict=True) if bool(keep) == selected) + "\n"
             for selected in (True, False)]
    return *parts, int(floor.sum())


def _add_captured_room(sb: SceneBuilder, root: str, cfg: WorldConfig) -> dict:
    """Static RGB-D geometry in world coordinates; the phantom and cradle were masked before meshing."""
    from pathlib import Path

    meshes = sorted(Path(root).glob("*.obj"))
    if not meshes:
        raise ValueError(f"no captured room meshes in {root}")
    replaced = {}
    for i, mesh in enumerate(meshes):
        name = f"captured_room{i}"
        obj = mesh.read_text()
        if cfg.surface.randomize_captured_floor:
            _, obj, count = captured_floor_parts(obj, cfg.surface.captured_floor_tolerance_m)
            if count:
                # The existing measured-height support plane supplies these
                # pixels without noisy depth fragments or photo UV seams.
                replaced[mesh.stem] = count
        if "\nf " not in obj:
            continue
        bgr = cv2.imread(str(mesh.with_suffix(".png")))
        if bgr is None:
            raise ValueError(f"missing room texture: {mesh.with_suffix('.png')}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sb.add_texture(name, rgb)
        sb.add_mesh_obj(name, obj)
        sb.materials[name] = Material((1, 1, 1, 1), 0, 0, emission=0.5, texture=name)
        sb.world_xml.append(_geom(name, name))
    return replaced


def _add_people(sb: SceneBuilder, rng: np.random.Generator, cfg: WorldConfig) -> None:
    n = int(rng.integers(0, cfg.max_people + 1))
    if n == 0:
        return
    vertices, faces = standing_body()
    sb.add_mesh_obj("person", obj_text(vertices, faces))
    for i in range(n):
        sb.materials[f"person{i}"] = Material(_rgba(textures.muted_colour(rng, 13, 230) / 255.0), 0.1, 0.2)
        angle = rng.uniform(-1.2, 1.2)
        dist = rng.uniform(1.1, 2.2)
        pos = np.array([dist * np.cos(angle), dist * np.sin(angle), -0.75])
        yaw = np.arctan2(-pos[1], -pos[0]) + np.pi / 2 + rng.normal(0, 0.4)
        q = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
        sb.world_xml.append(f'<body pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" quat="{q[0]:.4f} {q[1]:.4f} '
                            f'{q[2]:.4f} {q[3]:.4f}">{_geom("person", f"person{i}")}</body>')


def _add_objects(sb: SceneBuilder, rng: np.random.Generator, cfg: WorldConfig) -> None:
    for i in range(int(rng.integers(0, cfg.max_objects + 1))):
        r, az = rng.uniform(0.5, 0.85), rng.uniform(-1.4, 1.4)
        size = rng.uniform(0.02, 0.08, 3)
        kind = rng.choice(["box", "cylinder", "capsule"])
        sb.materials[f"object{i}"] = Material(_rgba(textures.muted_colour(rng, 5, 242) / 255.0),
                                              float(rng.uniform(0, 0.8)), 0.4)
        size_attr = f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}" if kind == "box" else f"{size[0]:.3f} {size[2]:.3f}"
        sb.world_xml.append(f'<geom type="{kind}" size="{size_attr}" pos="{r * np.cos(az):.3f} {r * np.sin(az):.3f} '
                            f'{size[2]:.3f}" euler="0 0 {rng.uniform(0, 3.14):.3f}" material="object{i}" '
                            'contype="0" conaffinity="0"/>')


def _clear_of_workspace(xy: np.ndarray) -> bool:
    """Off the part of the table where the phantom lies and the pen works (and off the arm's base)."""
    r, az = float(np.linalg.norm(xy)), float(np.degrees(np.arctan2(xy[1], xy[0])))
    return r > 0.2 and (r > 0.7 or abs(az) > 75.0)


def _in_front_of_workspace(pos: np.ndarray, scene_pose: np.ndarray) -> bool:
    """Between the scene camera and the workspace, where a tall thing would hide it."""
    cam = scene_pose[:3, 3]
    to_work, to_pos = WORKSPACE - cam, pos - cam
    cosine = float(to_work @ to_pos / (np.linalg.norm(to_work) * np.linalg.norm(to_pos)))
    return cosine > np.cos(np.radians(30.0)) and np.linalg.norm(to_pos) < np.linalg.norm(to_work)


def _spot(rng: np.random.Generator, r_range: tuple[float, float], ok) -> np.ndarray:
    for _ in range(30):
        r, az = rng.uniform(*r_range), rng.uniform(-np.pi, np.pi)
        xy = np.array([r * np.cos(az), r * np.sin(az)])
        if ok(xy):
            break
    return xy


def _on_table(xy: np.ndarray) -> bool:
    return -0.55 < xy[0] < 1.25 and -1.1 < xy[1] < 1.1


def _box(name: str, half, pos, yaw: float, material: str) -> str:
    return (f'<geom name="{name}" type="box" size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" euler="0 0 {yaw:.4f}" material="{material}" '
            'contype="0" conaffinity="0"/>')


def _dark_or_muted(rng: np.random.Generator, dark_prob: float) -> tuple[float, float, float, float]:
    if rng.random() < dark_prob:
        v = float(rng.uniform(0.02, 0.2))
        return (v, v, v * float(rng.uniform(0.95, 1.1)), 1.0)
    return _rgba(textures.muted_colour(rng, 5, 242) / 255.0)


def _structure_material(sb: SceneBuilder, rng: np.random.Generator, name: str, bank=None,
                        real_prob: float = 0.0) -> None:
    """Dark or muted, and often busy: a surface with blocks and stripes reads as shelved stuff at a distance;
    or the rig's own surroundings, from real pixels."""
    if bank is not None and bank.has("scene_bg") and rng.random() < real_prob:
        sb.add_texture(f"{name}_tex", bank.texture(rng, "scene_bg", 128))
        sb.materials[name] = Material((1, 1, 1, 1), float(rng.uniform(0, 0.3)), 0.3, texture=f"{name}_tex")
    elif rng.random() < 0.5:
        sb.add_texture(f"{name}_tex", textures.wall_texture(rng, shape=(128, 128)))
        dim = float(rng.uniform(0.25, 1.0))
        sb.materials[name] = Material((dim, dim, dim, 1), float(rng.uniform(0, 0.4)), 0.3, texture=f"{name}_tex")
    else:
        sb.materials[name] = Material(_dark_or_muted(rng, 0.65), float(rng.uniform(0, 0.5)), 0.3)


def _add_shelves(sb: SceneBuilder, rng: np.random.Generator, name: str, xy: np.ndarray, half: np.ndarray,
                 z0: float, yaw: float) -> None:
    """A rack: posts, two to five shelves, and small things on them."""
    sb.materials[name] = Material(_dark_or_muted(rng, 0.8), float(rng.uniform(0, 0.6)), 0.3)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    for k, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        post = xy + rot @ (np.array([sx, sy]) * half[:2])
        sb.world_xml.append(_box(f"{name}_post{k}", (0.008, 0.008, half[2]), (*post, z0 + half[2]), yaw, name))
    for k, z in enumerate(np.linspace(z0 + 0.02, z0 + 2 * half[2], int(rng.integers(2, 6)))):
        sb.world_xml.append(_box(f"{name}_shelf{k}", (half[0], half[1], 0.006), (*xy, z), yaw, name))
        for j in range(int(rng.integers(0, 4))):
            item = rng.uniform(0.02, 0.08, 3)
            spot = xy + rot @ (rng.uniform(-0.8, 0.8, 2) * half[:2])
            sb.materials[f"{name}_item{k}_{j}"] = Material(_dark_or_muted(rng, 0.5), float(rng.uniform(0, 0.6)),
                                                           0.3)
            sb.world_xml.append(_box(f"{name}_item{k}_{j}", item, (*spot, z + 0.006 + item[2]),
                                     float(rng.uniform(0, np.pi)), f"{name}_item{k}_{j}"))


def _add_rig(sb: SceneBuilder, rng: np.random.Generator, cfg: WorldConfig, scene_pose: np.ndarray,
             bank=None) -> None:
    """What stands around a rig, mostly dark: racks and cabinets beside and behind the arm, gear, paper,
    cables and tag boards on the table. The scene camera sees all of it, the wrist camera some."""
    for i in range(int(rng.integers(cfg.structures[0], cfg.structures[1] + 1))):
        xy = _spot(rng, (0.5, 1.5), lambda p: _clear_of_workspace(p) and not _in_front_of_workspace(
            np.append(p, 0.2), scene_pose))
        table = _on_table(xy)
        half = rng.uniform(0.06, 0.3, 3) * np.array([1.0, 1.0, 1.2 if table else 3.0])
        z0, yaw = (0.0 if table else -0.75), float(rng.uniform(0, np.pi))
        if rng.random() < 0.35:
            _add_shelves(sb, rng, f"structure{i}", xy, half, z0, yaw)
            continue
        _structure_material(sb, rng, f"structure{i}", bank, cfg.real_structure_prob)
        sb.world_xml.append(_box(f"structure{i}", half, (*xy, z0 + half[2]), yaw, f"structure{i}"))
    for i in range(int(rng.integers(0, cfg.max_gear + 1))):
        xy = _spot(rng, (0.2, 1.0), lambda p: _clear_of_workspace(p) and _on_table(p))
        half = rng.uniform(0.015, 0.09, 3)
        sb.materials[f"gear{i}"] = Material(_dark_or_muted(rng, 0.5), float(rng.uniform(0, 0.8)), 0.4)
        if rng.random() < 0.3:
            sb.world_xml.append(f'<geom name="gear{i}" type="cylinder" size="{half[0]:.4f} {half[2]:.4f}" pos="{xy[0]:.4f} '
                                f'{xy[1]:.4f} {half[2]:.4f}" material="gear{i}" contype="0" conaffinity="0"/>')
        else:
            sb.world_xml.append(_box(f"gear{i}", half, (*xy, half[2]), float(rng.uniform(0, np.pi)), f"gear{i}"))
    for i in range(int(rng.integers(0, cfg.max_cables + 1))):
        xy = _spot(rng, (0.25, 1.0), lambda p: _clear_of_workspace(p) and _on_table(p))
        radius, length = rng.uniform(0.002, 0.006), rng.uniform(0.08, 0.35)
        w, x, y, z = np.roll(Rotation.from_euler("ZY", [rng.uniform(0, np.pi), np.pi / 2]).as_quat(), 1)
        sb.materials[f"cable{i}"] = Material(_dark_or_muted(rng, 0.8), 0.2, 0.3)
        sb.world_xml.append(f'<geom name="cable{i}" type="capsule" size="{radius:.4f} {length:.4f}" pos="{xy[0]:.4f} {xy[1]:.4f} '
                            f'{radius:.4f}" quat="{w:.5f} {x:.5f} {y:.5f} {z:.5f}" material="cable{i}" contype="0" '
                            'conaffinity="0"/>')
    for i in range(int(rng.integers(0, cfg.max_tags + 1))):
        xy = _spot(rng, (0.2, 1.0), lambda p: _clear_of_workspace(p) and _on_table(p))
        sb.add_texture(f"tag{i}_tex", textures.tag_texture(rng))
        sb.materials[f"tag{i}"] = Material((1, 1, 1, 1), 0.1, 0.2, texture=f"tag{i}_tex")
        side = float(rng.uniform(0.025, 0.06))
        half = (side, side, side) if rng.random() < 0.6 else (side * 1.5, side * 1.5, 0.003)
        sb.world_xml.append(_box(f"tag{i}", half, (*xy, half[2]), float(rng.uniform(0, np.pi)), f"tag{i}"))


def _scene_camera_pose(rng: np.random.Generator, cfg: WorldConfig) -> np.ndarray:
    """The scene camera's optical frame in the arm base frame: the calibration, knocked a little."""
    pose = SCENE_CAMERA_POSE.copy()
    turn = Rotation.from_rotvec(np.radians(rng.normal(0.0, cfg.scene_mount_jitter_deg, 3))).as_matrix()
    pose[:3, :3] = pose[:3, :3] @ turn
    pose[:3, 3] += rng.normal(0.0, cfg.scene_mount_jitter_m, 3)
    return pose


def _add_lights(sb: SceneBuilder, rng: np.random.Generator, cfg: WorldConfig) -> dict:
    look = sample_lighting(rng, cfg.lighting, cfg.max_lights)
    for light in look["lights"]:
        pos, direction, colour = light["position"], light["direction"], light["diffuse"]
        shadow = "true" if light["shadow"] else "false"
        sb.lights_xml.append(
            f'<light pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" dir="{direction[0]:.3f} {direction[1]:.3f} '
            f'{direction[2]:.3f}" diffuse="{colour[0]:.3f} {colour[1]:.3f} {colour[2]:.3f}" '
            f'specular="0.2 0.2 0.2" castshadow="{shadow}" '
            f'cutoff="{light["cutoff"]:.3f}" exponent="{light["exponent"]:.3f}"/>')
    sb.headlight = tuple(look["headlight"])
    return look


def _add_robot_looks(sb: SceneBuilder, rng: np.random.Generator) -> None:
    sb.add_texture("chrome_tex", textures.chrome_texture(rng))
    cube = float(rng.uniform(0.45, 0.75))  # the scene camera's view of the tag cube: white with black tags
    sb.materials["cube"] = Material((cube, cube, cube, 1), 0.15, 0.1)
    sb.materials["chrome"] = Material((1, 1, 1, 1), float(rng.uniform(0.6, 1.0)), float(rng.uniform(0.6, 1.0)),
                                      texture="chrome_tex")
    shade = float(rng.uniform(0.05, 0.16))
    sb.materials["arm"] = Material((shade, shade, shade * 1.05, 1), float(rng.uniform(0.2, 0.5)), 0.4)
    white = float(rng.uniform(0.82, 0.97))
    sb.materials["shell"] = Material((white, white, white * 0.99, 1), 0.4, 0.5)


def _hand_offset(rng: np.random.Generator, phantom: Phantom, near_hand_end: bool) -> np.ndarray:
    """phantom_from_hand for one grip: palm against the side of the arm, fingers wrapping around it."""
    lo, hi = phantom.x_range
    if near_hand_end:
        x = min(rng.uniform(phantom.forearm[1] - 0.02, phantom.forearm[1] + 0.05), phantom.x_range[1] - 0.02)
    elif rng.random() < 0.15:
        x = rng.uniform(*phantom.forearm)  # a hand on the ink itself
    else:
        x = rng.uniform(lo + 0.03, lo + 0.12)
    phi = rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 2.4)
    radial = np.array([0.0, np.sin(phi), np.cos(phi)])
    tangent = np.array([0.0, np.cos(phi), -np.sin(phi)])
    centre = phantom.centre_at(x) + radial * (float(phantom.radius_at(x)) + 0.012)
    z = radial
    y = np.cross(z, tangent)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([tangent, y, z], axis=1)
    transform[:3, 3] = centre
    return transform


def _add_hands(sb: SceneBuilder, rng: np.random.Generator, phantom: Phantom) -> list[np.ndarray]:
    hand = build_hand()
    sb.add_mesh_obj("hand_skin", obj_text(hand.vertices, hand.skin_faces))
    sb.add_mesh_obj("hand_sleeve", obj_text(hand.vertices, hand.sleeve_faces))
    offsets = []
    for i in range(2):
        skin = textures.skin_colour(rng) / 255.0
        sb.materials[f"hand{i}_skin"] = Material(_rgba(skin), 0.2, 0.3)
        sb.materials[f"hand{i}_sleeve"] = Material(_rgba(rng.uniform(0.02, 0.9, 3)), 0.05, 0.1)
        sb.world_xml.append(_mocap(f"hand{i}", _geom("hand_skin", f"hand{i}_skin")
                                   + _geom("hand_sleeve", f"hand{i}_sleeve")))
        offsets.append(_hand_offset(rng, phantom, near_hand_end=(i == 1)))
    return offsets


def sample_ink(rng: np.random.Generator, phantom: Phantom, shell: SkinShell, cfg: WorldConfig, bank=None) -> list:
    """What is drawn on the arm: Sharpie lines, tattoos, or both; and often ballpoint cut from the rig's arm."""
    modes = list(cfg.ink_modes)
    mode = str(rng.choice(modes, p=np.array([cfg.ink_modes[m] for m in modes]) / sum(cfg.ink_modes.values())))
    items = []
    if cfg.real_ink_only_prob > 0 and bank is not None and bank.has("ink") and rng.random() < cfg.real_ink_only_prob:
        return [ink.photo_item(rng, shell, bank.ink(rng))
                for _ in range(int(rng.integers(1, cfg.max_real_ink + 1)))]
    if mode in ("lines", "both"):
        for _ in range(1 if rng.random() < 0.75 else 2):
            items.append(ink.line_item(rng, sample_chart_curve(rng, phantom), shell))
    if mode in ("tattoos", "both"):
        items += [ink.tattoo_item(rng, shell, cfg.held_out_designs)
                  for _ in range(int(rng.integers(1, cfg.max_tattoos + 1)))]
    if bank is not None and bank.has("ink") and rng.random() < cfg.real_ink_prob:
        items += [ink.photo_item(rng, shell, bank.ink(rng)) for _ in range(int(rng.integers(1, cfg.max_real_ink + 1)))]
    return items


def _add_phantom(sb: SceneBuilder, rng: np.random.Generator, phantom: Phantom, shell: SkinShell,
                 cfg: WorldConfig, skin_rng: np.random.Generator, bank=None) -> tuple[ink.InkGraph, list, dict]:
    """The practice arm, its forearm shell and the ink on it; returns the traceable strokes."""
    look = sample_skin(skin_rng, cfg.skin, cfg.skin_rgb)
    atlas = textures.skin_texture(skin_rng, base=np.asarray(look["rgb"]), properties=look)
    shell_skin = textures.skin_chart_texture(atlas, phantom.x_range, shell, cfg.ink_theta_origin_rad, ink.TEXTURE_SHAPE)
    sb.add_texture("skin_tex", atlas)
    for _ in range(8):
        items = sample_ink(rng, phantom, shell, cfg, bank)
        shell_tex, coverage = ink.compose(shell_skin, items, shell)
        graph = ink.ink_graph(coverage, shell)
        if graph.total_length >= cfg.min_ink_m:
            break
    sb.add_texture("shell_tex", shell_tex)
    specular, shininess = look["specular"], look["shininess"]
    sb.materials["skin"] = Material((1, 1, 1, 1), specular, shininess, texture="skin_tex")
    sb.materials["forearm"] = Material((1, 1, 1, 1), specular, shininess, texture="shell_tex")
    foam = float(rng.uniform(0.85, 0.98))
    sb.materials["foam"] = Material((foam, foam, foam * 0.97, 1), 0.05, 0.1)
    sb.add_mesh_obj("phantom_skin", obj_text(phantom.vertices, phantom.faces, phantom.uv))
    sb.add_mesh_obj("phantom_cap", obj_text(phantom.vertices, phantom.cap_faces))
    sb.add_mesh_obj("forearm_shell", shell.obj_text())
    geoms = _geom("phantom_skin", "skin") + _geom("phantom_cap", "foam") + _geom("forearm_shell", "forearm")
    sb.world_xml.append(_mocap("phantom", geoms))
    return graph, items, look


def _camera_jitter(sb_xml: str, rng: np.random.Generator, cfg: WorldConfig) -> str:
    """Perturb the wrist camera's pose in its mount (the CAD mount is nominal)."""
    rot = Rotation.from_quat([1.0, 0.0, 0.0, 0.0]) * Rotation.from_rotvec(
        np.radians(rng.normal(0.0, cfg.mount_jitter_deg, 3)))
    x, y, z, w = rot.as_quat()
    pos = rng.normal(0.0, cfg.mount_jitter_m, 3)
    old = f'<camera name="{CAMERA}" quat="0 1 0 0"'
    new = f'<camera name="{CAMERA}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}"'
    return sb_xml.replace(old, new)


def build_world(rng: np.random.Generator, cfg: WorldConfig | None = None) -> World:
    cfg = cfg or WorldConfig()
    appearance_seed = int(rng.integers(0, 2**63))
    appearance_rng = np.random.default_rng(appearance_seed if cfg.appearance_seed is None else cfg.appearance_seed)
    bank = realbank.load(cfg.real_assets)
    intr = Intrinsics.left_wrist().jittered(rng, cfg.focal_jitter, cfg.centre_jitter_px)
    plan = render_plan(intr)
    along, across = float(rng.uniform(*cfg.phantom_scale_along)), float(rng.uniform(*cfg.phantom_scale_across))
    handless = bool(rng.random() < cfg.handless_prob)
    phantom = build_phantom(length_m=cfg.phantom_length_m, handless=handless,
                            hand_pose=cfg.phantom_hand_pose).scaled(along, across)
    shell = chart_skin(phantom, theta_origin=cfg.ink_theta_origin_rad)
    sb = SceneBuilder(plan=plan, base_pos=(0.0, 0.0, float(rng.uniform(*cfg.base_height_m))), gap_m=cfg.gap_m)
    scene_intr = None
    if cfg.scene_view:
        scene_intr = Intrinsics.scene_camera().jittered(rng, cfg.scene_focal_jitter, cfg.scene_centre_jitter_px)
        sb.scene_plan = render_plan(scene_intr, focal=SCENE_RENDER_FOCAL)
        sb.scene_pose = _scene_camera_pose(rng, cfg)
        _add_rig(sb, rng, cfg, sb.scene_pose, bank)
    appearance = _add_room(sb, appearance_rng, cfg, bank, geometry_rng=rng)
    appearance["seed"] = appearance_seed if cfg.appearance_seed is None else cfg.appearance_seed
    if cfg.room_assets is not None:
        appearance["replaced_captured_floor_faces"] = _add_captured_room(sb, cfg.room_assets, cfg)
    tabletop, appearance["tabletop"] = add_tabletop(sb, rng, appearance_rng, cfg.tabletop, cfg.max_sheets)
    _add_people(sb, rng, cfg)
    _add_objects(sb, rng, cfg)
    appearance["lighting"] = _add_lights(sb, appearance_rng, cfg)
    _add_robot_looks(sb, appearance_rng)
    skin_seed = (int(np.random.SeedSequence([appearance["seed"], 1]).generate_state(1)[0])
                 if cfg.skin_seed is None else cfg.skin_seed)
    graph, items, appearance["skin"] = _add_phantom(sb, rng, phantom, shell, cfg, np.random.default_rng(skin_seed), bank)
    appearance["skin"]["seed"] = skin_seed
    appearance["skin"]["texture_mapping"] = "shared cylindrical atlas remapped into the canonical forearm chart"
    offsets = _add_hands(sb, rng, phantom)
    xml = _camera_jitter(sb.xml(), rng, cfg)
    model = mujoco.MjModel.from_xml_string(xml, sb.assets)
    if not cfg.wrist_robot_visible:
        for i, body in enumerate(model.geom_bodyid):
            name = model.body(int(body)).name or ""
            if name.startswith(f"{ARM}/"):
                model.geom_group[i] = SCENE_ONLY_GROUP  # kinematics and collision contracts stay intact
    mocap = {name: int(model.body(name).mocapid[0]) for name in ("phantom", "hand0", "hand1")}
    meta = {"body_surface": phantom.provenance, "appearance": appearance,
            "ink": [item.kind for item in items], "ink_length_m": graph.total_length, "handless": handless,
            "ink_strokes": len(graph.edges), "base_height_m": sb.base_pos[2], "intrinsics": intr.__dict__}
    if scene_intr is not None:
        meta["scene_intrinsics"] = scene_intr.__dict__
    return World(model=model, plan=plan, intrinsics=intr, phantom=phantom, ink=graph,
                 selfview=SelfViewLook.sample(rng, gain_range=cfg.selfview_gain, root=cfg.selfview_assets),
                 mocap=mocap, hand_offsets=offsets, tabletop=tabletop, meta=meta,
                 scene_plan=sb.scene_plan, scene_intrinsics=scene_intr)
