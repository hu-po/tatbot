"""Materialize and build the kinematic posed-body scene for ManiSkill."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tatbot_sim import tools
from tatbot_sim.inkmap.contracts import document_sha256, validate_scenario
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface, mesh_patch_from_scenario
from tatbot_sim.inkmap.program_target import ProgramTarget, render_program_target, write_program_target
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.repo import repo_root

SCENE_GEOMETRY_VERSION = 10
PATCH_VISUAL_OFFSET_M = 2e-4


@dataclass(frozen=True)
class BoxProxy:
    center: np.ndarray
    half_size: np.ndarray
    quaternion_wxyz: np.ndarray | None = None
    name: str = "support"


@dataclass(frozen=True)
class CapsuleProxy:
    name: str
    start: np.ndarray
    end: np.ndarray
    radius: float


@dataclass(frozen=True)
class ScenarioGeometry:
    root: Path
    body_obj: Path
    patch_obj: Path
    support_boxes: tuple[BoxProxy, ...]
    collision_capsules: tuple[CapsuleProxy, ...]
    surface: MeshPatchSurface
    target: ProgramTarget | None
    blank_skin_rgba: np.ndarray


def _write_obj(path: Path, vertices: np.ndarray, triangles_uv: np.ndarray | None = None) -> None:
    material = path.with_suffix(".mtl")
    material.write_text("newmtl skin\nKd 0.72 0.48 0.32\nKa 0.10 0.07 0.05\nNs 8\n")
    lines = [f"mtllib {material.name}", "usemtl skin"]
    flat = vertices.reshape(-1, 3)
    lines.extend(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in flat)
    if triangles_uv is not None:
        uv = triangles_uv.reshape(-1, 2)
        lines.extend(f"vt {point[0]:.9g} {point[1]:.9g}" for point in uv)
        lines.extend(f"f {i}/{i} {i + 1}/{i + 1} {i + 2}/{i + 2}" for i in range(1, len(flat) + 1, 3))
    else:
        lines.extend(f"f {i} {i + 1} {i + 2}" for i in range(1, len(flat) + 1, 3))
    path.write_text("\n".join(lines) + "\n")


def _scenario_support_boxes(scenario: dict) -> tuple[BoxProxy, ...]:
    """Legacy support metadata does not instantiate furniture or obstacles."""
    return ()


def _capsule_radius(name: str) -> float:
    if name in ("pelvis", "spine_lower", "spine_upper"):
        return 0.095
    if name in ("neck", "head"):
        return 0.075
    if name.startswith(("thigh", "upper_arm")):
        return 0.06
    if name.startswith(("shin", "forearm")):
        return 0.045
    if name.startswith("hand"):
        return 0.020
    if name.startswith("foot"):
        return 0.030
    return 0.035


def _collision_capsules(scenario: dict) -> tuple[CapsuleProxy, ...]:
    """Fit conservative clearance proxies directly to SOMA semantic regions.

    These proxies are a broad-phase robot-clearance aid, not a second rig or a
    skin/contact model. Their inputs are the same posed SOMA triangles and
    atlas face labels used by the scenario.
    """

    rig = load_body_rig()
    posed = rig.posed(
        scenario["pose"]["id"], np.asarray(scenario["pose"]["world_from_body"]),
    ).vertices.astype(np.float64)
    atlas = json.loads(
        (repo_root() / "web" / "inkmap" / "public" / "bodies" / "mhr-soma-v1.regions.json").read_text()
    )
    face_codes = np.asarray(atlas["faces"], dtype=np.int32)
    site_index = {name: index for index, name in enumerate(atlas["sites"])}
    groups = {
        "pelvis": (("hip", "love_handle", "glute"), None),
        "spine_lower": (("stomach", "lower_back"), None),
        "spine_upper": (("chest", "mid_back", "upper_back"), None),
        "neck": (("neck", "nape", "throat"), None),
        "head": (("face", "forehead", "crown"), None),
        "upper_arm.L": (("bicep", "tricep"), "left"),
        "upper_arm.R": (("bicep", "tricep"), "right"),
        "forearm.L": (("forearm",), "left"),
        "forearm.R": (("forearm",), "right"),
        "hand.L": (("hand", "palm"), "left"),
        "hand.R": (("hand", "palm"), "right"),
        "thigh.L": (("thigh",), "left"),
        "thigh.R": (("thigh",), "right"),
        "shin.L": (("shin", "calf"), "left"),
        "shin.R": (("shin", "calf"), "right"),
        "foot.L": (("foot_top", "sole"), "left"),
        "foot.R": (("foot_top", "sole"), "right"),
    }
    laterality_code = {None: (0, 1, 2), "left": (1,), "right": (2,)}
    capsules: list[CapsuleProxy] = []
    for name, (sites, laterality) in groups.items():
        codes = {
            site_index[site] * 4 + code
            for site in sites if site in site_index
            for code in laterality_code[laterality]
        }
        faces = np.flatnonzero(np.isin(face_codes, tuple(codes)))
        if not len(faces):
            continue
        points = posed[faces].reshape(-1, 3)
        center = points.mean(axis=0)
        covariance = np.cov(points - center, rowvar=False)
        axis = np.linalg.eigh(covariance)[1][:, -1]
        projection = (points - center) @ axis
        start = center + axis * np.quantile(projection, 0.08)
        end = center + axis * np.quantile(projection, 0.92)
        radial = np.linalg.norm((points - center) - np.outer(projection, axis), axis=1)
        radius = max(_capsule_radius(name), float(np.quantile(radial, 0.98)) + 0.005)
        if np.linalg.norm(end - start) > 1e-4:
            capsules.append(CapsuleProxy(name, start, end, radius))
    return tuple(capsules)


def materialize_scenario_geometry(scenario: dict, cache_root: Path | None = None) -> ScenarioGeometry:
    validate_scenario(scenario)
    digest = document_sha256(scenario)
    root = (cache_root or Path.home() / ".cache" / "tatbot" / "body-scenarios") / f"v{SCENE_GEOMETRY_VERSION}-{digest}"
    root.mkdir(parents=True, exist_ok=True)
    surface = mesh_patch_from_scenario(scenario)
    posed = surface.posed_vertices[0]
    body_obj = root / "body.obj"
    patch_obj = root / "tattoo-patch.obj"
    if not body_obj.exists():
        _write_obj(body_obj, posed)
    patch = surface.patches[0]
    patch_vertices = posed[patch.face_indices] + PATCH_VISUAL_OFFSET_M * surface.normals[0][patch.face_indices]
    uv = patch.triangles_uv.copy()
    uv[..., 0] = uv[..., 0] / surface.width_m + 0.5
    # Image row zero is chart negative-y, while OBJ v=1 addresses the image's
    # first row. This matches Surface.canvas_to_px and InkField exactly.
    uv[..., 1] = 0.5 - uv[..., 1] / surface.height_m
    if not patch_obj.exists():
        _write_obj(patch_obj, patch_vertices, uv)
    target = None
    skin_tone = "#b87a52"
    if scenario.get("schema_version") == 3:
        binding = scenario["program_binding"]
        bundle = binding["bundle"]
        placement_index = next(
            index
            for index, item in enumerate(bundle["placement_file"]["placements"])
            if item["id"] == binding["placement_id"]
        )
        placement = bundle["placement_file"]["placements"][placement_index]
        program = bundle["artworks"][placement["design_id"]]["program"]
        surface_placement = bundle["surface_placements"][placement_index]["placement"]
        tool = tools.registry().load_tool(bundle["request"]["tool_id"], repo_root())
        target = render_program_target(program, surface_placement, tool_width_m=tool.line_width_m)
        skin_tone = bundle["request"]["skin_tone"]
        write_program_target(root, target, skin_tone)
    skin_rgb = np.asarray(
        [int(skin_tone[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.uint8
    )
    blank_skin_rgba = np.empty((2, 2, 4), dtype=np.uint8)
    blank_skin_rgba[..., :3] = skin_rgb
    blank_skin_rgba[..., 3] = 255
    return ScenarioGeometry(
        root=root,
        body_obj=body_obj,
        patch_obj=patch_obj,
        support_boxes=_scenario_support_boxes(scenario),
        collision_capsules=_collision_capsules(scenario),
        surface=surface,
        target=target,
        blank_skin_rgba=blank_skin_rgba,
    )


def build_scenario_actors(scene, scenario: dict, num_envs: int):
    """Add the body visual, drawable patch to each subscene.

    Body capsules are deliberately *not* installed as physics collision
    shapes. They are conservative placement-search proxies (for rejecting
    robot/body and shaft/body overlap), not a skin surface: on the forearm the
    proxy can stand more than 20 mm proud of the rendered mesh and physically
    stop the TCP before it enters the sub-millimetre kinematic interaction
    band. Body episodes declare ``kinematic-contact-v1`` and use the exact
    posed triangles for deposition. Furniture is not instantiated.
    """

    import sapien
    geometry = materialize_scenario_geometry(scenario)
    bodies, patches = [], []
    for env_index in range(num_envs):
        body = scene.create_actor_builder()
        body.add_visual_from_file(str(geometry.body_obj))
        body.set_scene_idxs([env_index])
        bodies.append(body.build_kinematic(name=f"posed_body_{env_index}"))

        patch = scene.create_actor_builder()
        patch_material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.85)
        patch_material.set_base_color_texture(sapien.render.RenderTexture2D(
            array=geometry.blank_skin_rgba, format="R8G8B8A8Unorm", srgb=True,
            address_mode="edge",
        ))
        patch.add_visual_from_file(str(geometry.patch_obj), material=patch_material)
        patch.set_scene_idxs([env_index])
        patches.append(patch.build_kinematic(name=f"tattoo_patch_{env_index}"))

    return bodies, patches, geometry
