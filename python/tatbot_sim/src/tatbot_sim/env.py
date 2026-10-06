"""TatbotDraw-v0: a working surface in front of a WidowX AI that a tool marks.

Ink is a per-env PIGMENT FIELD (tatbot_sim.inkfield), not geometry: one float
per sheet texel, composited over the paper and uploaded into the pad's texture.
Whenever the resolved working TCP satisfies the tool's interaction contract,
the fitted tool's ``kind`` decides what happens to the field under the tip — a
pen adds pigment, the removal laser clears a fraction of it. That is the whole reason
for the representation: the earlier dot pool (kinematic cylinders teleported
under the tip, one per control step) was monotone-additive with no per-pixel
quantity to subtract from, so no tool could ever take ink away.

The field updates every control step, so what the env reports as ground truth
is exact. The TEXTURE refreshes every ``texture_refresh_steps`` steps instead,
because a texture upload costs ~1.5 ms per call almost regardless of its size:
at 30 Hz and real draw speeds the tip moves well under a line width between
refreshes, so strokes still read continuous and only the last millimetre of a
trail can lag (by less than the observation staleness LatencyDR already trains
across).

The TCP is the fitted tool's working point. Contact tools carry a small physical
tip collision and the flat substrate carries matching collision geometry; the
mark gate remains geometric and separately audited so solver tolerance can
never create ink in the air. Shaped surfaces still use that kinematic contact
contract until their collision mesh is qualified.

The surface is deliberately a *broader* distribution than the real sessions.
Its height is randomized through several centimetres rather than pinned to the
height the real paper happened to sit at, its orientation tilts a few degrees
off level (real tables and taped-down sheets are never perfectly flat), and
its texture, tint and the scene lighting move per environment, so the policy
meets many tables rather than one. The drawing plane is a true plane — point
plus normal — and the expert, the ink model and the floor clamp all work in
that plane rather than assuming a horizontal surface.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat, quat2mat

from tatbot_sim import capmesh, interaction, tasks, tools
from tatbot_sim.agent import TatbotWXAI
from tatbot_sim.calibration import world_from_follower_base
from tatbot_sim.config import MAX_TOOL_Z_CENTER, PAD_CENTER, DRConfig
from tatbot_sim.inkfield import InkField, laser_eta
from tatbot_sim.inkmap.contracts import load_scenario
from tatbot_sim.inkmap.mesh_patch_surface import MeshPatchSurface
from tatbot_sim.inkmap.scenario_scene import build_scenario_actors
from tatbot_sim.resolved import ResolvedConfig, resolve
from tatbot_sim.surface import (
    CylinderChart,
    DisplacedSurface,
    PlanarSurface,
    PlaneChart,
    random_height_field,
)
from tatbot_sim.textures import (
    BAND_SHELL_M,
    TEX_DIR,
    environment_face_sets,
    floor_textures,
    grid_paper_sheets,
    skin_sheets,
    write_surface_mesh,
)


def _invert_transform(transform):
    out = np.eye(4)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def amcrest_camera_configs(calibration_path, robot_world_path, scale=0.5):
    """Build the five real rig viewpoints in the sim robot's base frame.

    Calibration rotations use OpenCV optical axes (right, down, forward),
    while a SAPIEN camera pose uses (forward, left, up).  The column
    permutation below is therefore load-bearing; copying the calibration
    rotation directly points a rendered camera in the wrong direction.
    """
    bundle = json.loads(Path(calibration_path).expanduser().read_text())
    robot_world = json.loads(Path(robot_world_path).expanduser().read_text())
    # The calibration solve uses the dual-arm URDF root. This environment's
    # robot root is the follower base, so include its fixed rig mount before
    # expressing world cameras in the simulated base frame.
    world_from_base = world_from_follower_base(bundle, robot_world)
    base_from_world = _invert_transform(world_from_base)
    configs = []
    for name, entry in sorted(bundle["cameras"].items()):
        if not name.startswith("camera"):
            continue
        intr = entry["intrinsics"]
        width, height = round(int(intr["width"]) * scale), round(int(intr["height"]) * scale)
        intrinsic = np.array(
            [
                [float(intr["fx"]) * scale, 0.0, float(intr["cx"]) * scale],
                [0.0, float(intr["fy"]) * scale, float(intr["cy"]) * scale],
                [0.0, 0.0, 1.0],
            ]
        )
        pose = entry["world_from_camera"]
        world_from_camera_cv = np.eye(4)
        world_from_camera_cv[:3, :3] = np.asarray(pose["rotation"], float).reshape(3, 3)
        world_from_camera_cv[:3, 3] = np.asarray(pose["translation_m"], float)
        base_from_camera_cv = base_from_world @ world_from_camera_cv
        cv_rotation = base_from_camera_cv[:3, :3]
        sapien_rotation = np.column_stack([cv_rotation[:, 2], -cv_rotation[:, 0], -cv_rotation[:, 1]])
        configs.append(
            CameraConfig(
                uid=name,
                pose=sapien.Pose(p=base_from_camera_cv[:3, 3], q=mat2quat(sapien_rotation)),
                width=width,
                height=height,
                intrinsic=intrinsic,
                near=0.01,
                far=5.0,
            )
        )
    if len(configs) != 5:
        raise ValueError(f"fiducial benchmark needs five Amcrest calibrations, got {len(configs)}")
    return configs


def _close_rgb(a, b, tol: float = 0.02) -> bool:
    """Same colour, within loader round-tripping."""
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a[:3], b[:3], strict=False))


def sample_surface_profiles(rng, count: int, profile: str, radius_m) -> tuple[list[str], np.ndarray]:
    """Resolve one explicit profile per vectorized environment.

    Balanced means balanced in every batch when there are at least two slots;
    the odd slot and single-env batches alternate stochastically across scene
    rebuilds. Flat radii are represented by infinity in the vectorized chart,
    where infinity is handled exactly as a plane rather than as a huge-radius
    approximation.
    """
    if profile not in ("flat", "cylinder", "balanced"):
        raise ValueError(
            f"surface profile must be flat, cylinder, or balanced; got {profile!r}"
        )
    lo, hi = (float(radius_m[0]), float(radius_m[1]))
    if not np.isfinite([lo, hi]).all() or lo <= 0 or hi < lo:
        raise ValueError(f"cylinder_radius_m must be a positive ordered range, got {radius_m!r}")
    if profile == "balanced":
        n_cylinder = count // 2
        if count % 2 and rng.random() < 0.5:
            n_cylinder += 1
        profiles = ["cylinder"] * n_cylinder + ["flat"] * (count - n_cylinder)
        rng.shuffle(profiles)
    else:
        profiles = [profile] * count
    radii = np.full(count, np.inf, dtype=np.float32)
    curved = np.asarray([name == "cylinder" for name in profiles])
    radii[curved] = rng.uniform(lo, hi, int(curved.sum()))
    return profiles, radii


def add_palette_body(builder, scene, material) -> None:
    """The printed palette and its sticker, in the palette URDF's body frame.

    ``scene`` is a :class:`tatbot_sim.palette.PaletteScene`: the body mesh at
    the URDF's scale, and the tag where its ``palette_tag`` frame puts it --
    the in-plane turn included, which is what a camera reads the pose from.
    """
    builder.add_visual_from_file(
        str(tools.REPO / scene.mesh), scale=list(scene.mesh_scale), material=material)
    if scene.collision_mesh:
        builder.add_nonconvex_collision_from_file(
            str(tools.REPO / scene.collision_mesh), scale=list(scene.mesh_scale))
    builder.add_visual_from_file(
        str(tools.REPO / scene.tag_mesh),
        pose=sapien.Pose(p=list(scene.tag_xyz_m), q=euler2quat(*scene.tag_rpy, "sxyz")))


class TatbotDrawEnv(BaseEnv):
    """Drawing env for scripted data generation. No reward, no success."""

    SUPPORTED_REWARD_MODES = ["none"]
    agent: TatbotWXAI

    # Nominal surface placement, recovered from the real draw-square recordings:
    # forward kinematics on those joint trajectories puts link_6 at
    # (0.288, 0.003, 0.238) with its x-axis straight down, so the arm works
    # directly in front of itself rather than at the rig's design origin.
    PAD_CENTER = np.array(PAD_CENTER)
    PAD_TOP_Z = 0.028  # nominal only; per-episode height is randomized
    # Extent and thickness come from the SUBSTRATE the fitted tool works on
    # (config/substrates.yaml) — a 7.5 x 11 in gridded pad or an 85 mm paper
    # cylinder under the ballpoint, a 140 x 185 silicone skin under the laser
    # and the 3RL. Sizing the geometry and the texture from one record is what
    # keeps them from drifting apart.

    INTERACTION_MODEL = interaction.INTERACTION_MODEL
    CONTACT_ABOVE_TOLERANCE_M = interaction.CONTACT_ABOVE_TOLERANCE_M
    MAX_PENETRATION_M = interaction.MAX_PENETRATION_M
    PHYSICS_CONTACT_OFFSET_M = interaction.PHYSICS_CONTACT_OFFSET_M

    # Measured ceiling for holding the pen perpendicular to the surface: above
    # this the wrist cannot make the orientation at all (the IK residual does
    # not degrade, it fails outright; ~0.11 m over the pad centre, with a
    # little margin here). Sampling ranges are trimmed to fit underneath
    # rather than generating poses the arm cannot hold.
    MAX_TOOL_Z_CENTER = MAX_TOOL_Z_CENTER

    def __init__(
        self,
        *args,
        robot_uids="tatbot_wxai",
        contact_above_tolerance_m: float = CONTACT_ABOVE_TOLERANCE_M,
        max_penetration_m: float = MAX_PENETRATION_M,
        texture_refresh_steps: int = 3,
        num_textures: int = 8,
        dr: DRConfig | None = None,
        fiducial_calibration: str | None = None,
        fiducial_robot_world: str | None = None,
        fiducial_camera_scale: float = 0.5,
        scenario_path: str | None = None,
        ink_pixels_per_m: int | None = None,
        sensor_profile: str = 'deployment',
        config: ResolvedConfig | None = None,
        presentation_cameras=None,
        presentation_lighting=None,
        presentation_surfaces=None,
        **kwargs,
    ):
        self.presentation_cameras = presentation_cameras
        self.presentation_lighting = presentation_lighting
        self.presentation_surfaces = presentation_surfaces
        self.config = config or resolve(sensor_profile=sensor_profile, dr=dr, scenario_path=scenario_path)
        self.sensor_profile = self.config.sensor_profile
        scenario_path = self.config.scenario_path
        self.contact_above_tolerance_m = float(contact_above_tolerance_m)
        self.max_penetration_m = float(max_penetration_m)
        if not 0 <= self.contact_above_tolerance_m <= 0.0005:
            raise ValueError("contact_above_tolerance_m must be in [0, 0.0005]")
        if not 0 <= self.max_penetration_m <= 0.0005:
            raise ValueError("max_penetration_m must be in [0, 0.0005]")
        self.texture_refresh_steps = max(1, int(texture_refresh_steps))
        if ink_pixels_per_m is not None and not 4000 <= ink_pixels_per_m <= 24000:
            raise ValueError("ink resolution must be 4000–24000 pixels/m")
        self.ink_pixels_per_m = ink_pixels_per_m
        # what is in the gripper decides what the tool does to the field
        self.tool = self.config.tool
        self.substrate = self.config.substrate
        # ... and whether it carries ink between dips (scripts/lib/ink_spec.py)
        self.ink_policy = self.config.ink_policy
        self.pad_half_x = self.substrate.width_m / 2
        self.pad_half_y = self.substrate.height_m / 2
        # For a cylinder-shaped substrate this is the DIAMETER: the actor
        # origin sits a half-thickness below the crest, i.e. on the axis.
        self.pad_thickness = self.substrate.thickness_m
        self.substrate_is_cylinder = getattr(self.substrate, "shape", "pad") == "cylinder"
        # The writable sheet on a rigid cylinder is a paper-thin shell over the
        # textured fixture; extruding it by the diameter would push a second
        # solid out through the far side of the tube.
        self.sheet_thickness = BAND_SHELL_M if self.substrate_is_cylinder else self.pad_thickness
        self._tool_kind_warned = False
        self.num_textures = num_textures
        self.fiducial_calibration = fiducial_calibration
        self.fiducial_robot_world = fiducial_robot_world
        self.fiducial_camera_scale = fiducial_camera_scale
        self.body_scenario = load_scenario(scenario_path) if scenario_path else None
        if self.body_scenario and self.body_scenario["robot"]["tool_id"] != self.tool.tool_id:
            raise ValueError(
                f"scenario requires tool {self.body_scenario['robot']['tool_id']!r}, "
                f"active tool is {self.tool.tool_id!r}"
            )
        if self.body_scenario:
            urdf = tools.REPO / "urdf" / "tatbot.urdf"
            installed_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest()
            if self.body_scenario["robot"]["urdf_sha256"] != installed_sha256:
                raise ValueError("scenario robot URDF checksum does not match this checkout")
        # every randomization range lives in the DR tree (tatbot_sim.config);
        # the env carries no tuning literals of its own
        self.dr = self.config.dr
        # Where the substrate sits: the DR override when a recipe placed it
        # (see PadDR.center_xy), the class constant otherwise.
        self.pad_center = np.asarray(
            self.dr.pad.center_xy if self.dr.pad.center_xy is not None
            else self.PAD_CENTER, dtype=np.float64)
        self._reset_index = 0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def reset(self, seed=None, options=None):
        # Scene construction and episode placement have separate streams:
        # retaining an existing scene must not consume placement randomness.
        if self._reset_index == 0:
            # BaseEnv bootstraps with its own fixed seed; construction belongs
            # to our resolved scene, including clients that retain its assets.
            seed = self.config.seed
        elif seed is None:
            seed = self.config.seed + self._reset_index
        self._reset_index += 1
        self.episode_seed = tuple(int(value) for value in np.atleast_1d(seed))
        token = ",".join(map(str, self.episode_seed))
        self._rng = np.random.default_rng(self.config.seed_for(f'placement:{token}'))
        self._scene_rng = np.random.default_rng(self.config.seed_for(f'layout:{token}'))
        self._lighting_rng = np.random.default_rng(self.config.seed_for(f'lighting:{token}'))
        self._camera_seed = self.config.seed_for(f'camera:{token}')
        return super().reset(seed=seed, options=options)

    @property
    def _default_sim_config(self):
        # Millimetre-scale contact cannot use the old 10 mm broad-phase offset:
        # it made collision proximity larger than the entire drawing contract.
        return SimConfig(
            sim_freq=self.config.timing.physics_hz,
            control_freq=self.config.timing.control_hz,
            scene_config=SceneConfig(
                contact_offset=self.PHYSICS_CONTACT_OFFSET_M,
                solver_position_iterations=8,
                solver_velocity_iterations=1,
            ),
        )

    @property
    def _default_sensor_configs(self):
        if self.presentation_cameras is not None:
            return self.presentation_cameras(self)
        if self.fiducial_calibration or self.fiducial_robot_world:
            if not self.fiducial_calibration or not self.fiducial_robot_world:
                raise ValueError("fiducial calibration and robot-world paths must be supplied together")
            return amcrest_camera_configs(
                self.fiducial_calibration,
                self.fiducial_robot_world,
                self.fiducial_camera_scale,
            )
        return []  # the wrist cameras come from the agent

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.62, -0.42, 0.42], target=[0.29, 0, 0.05])
        return CameraConfig(
            "render_camera", pose=pose, width=960, height=720, fov=1.1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        self.agent = TatbotWXAI(
            self.scene, self._control_freq, self._control_mode,
            initial_pose=sapien.Pose(), config=self.config, camera_seed=self._camera_seed,
        )
        self._polish_ee_materials()

    def _polish_ee_materials(self):
        """Gloss the machine body and tip the way the real hardware looks.

        URDF material tags carry only color; the operator's spec (2026-08-22)
        is black glossy METAL body with rings and a white glossy PLASTIC tip,
        which needs metallic/roughness set on the loaded render materials.
        Runs at load, before the render pipeline bakes anything.
        """
        link = self.agent.robot.links_map.get("tattoo_pen")
        if link is None:
            return
        detail = self.tool.tip_detail or {}
        emitter_rgb = None
        if detail.get("kind") == "emitter":
            emitter_rgb = [float(v) for v in
                           str(detail.get("color", "0.20 0.45 1.00 1")).split()[:3]]
        self._emitter_materials = []
        for obj in link._objs:
            body = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                for part in shape.parts:
                    m = part.material
                    if emitter_rgb is not None and _close_rgb(m.base_color, emitter_rgb):
                        # The emitter window is not a surface to polish; it is
                        # a light. Collected here because this is the one place
                        # that walks the loaded materials, and matched on the
                        # datasheet's own colour rather than on part order,
                        # which the geometry is free to change.
                        self._emitter_materials.append(m)
                        continue
                    # every part of the pen is glossy (operator, 2026-08-22)
                    if m.base_color[0] > 0.8:  # white tip: glossy plastic
                        m.metallic, m.roughness = 0.0, 0.1
                    else:  # black machine body + metal rings
                        m.metallic, m.roughness = 0.85, 0.15
        if emitter_rgb is not None and not self._emitter_materials:
            print("[env] tool declares an emitter but no material matched its "
                  "colour — the laser will not flash")

    def _load_lighting(self, options: dict):
        """Every environment gets its own light rig — count, type, direction,
        colour temperature and intensity all vary, because the real rig might
        sit under studio panels, a desk lamp, or a window. ``scene_idxs``
        scopes each light to one sub-scene; without it a "randomized" batch
        shares a single rig. At most one shadow caster per env (each one is
        an extra shadow-map pass). Intensity ranges are calibrated with the
        environment map's image-based ambience stacked on top: the first cut
        blew ~1/3 of frames to pure white, and a sheet with an invisible
        ruling has no stencil to trace."""
        self.lighting_sample = {"mode": "presentation" if self.presentation_lighting else "default"}
        if self.presentation_lighting is not None:
            return self.presentation_lighting(self, options)
        if not self.dr.lighting.enabled:
            return super()._load_lighting(options)
        rng = self._lighting_rng
        lighting = self.dr.lighting
        amb = rng.uniform(*lighting.ambient)
        ambient = [amb, amb, amb * rng.uniform(0.9, 1.1)]
        self.scene.set_ambient_light(ambient)
        self.lighting_sample = {"mode": "randomized", "ambient": ambient, "lights": []}

        def tint(level):
            warmth = rng.uniform(-lighting.warmth, lighting.warmth)
            return [level * (1 + warmth), level, level * (1 - warmth)]

        for i in range(self.num_envs):
            idxs = [i]
            n_dir = int(rng.integers(0, lighting.max_directional + 1))
            n_point = int(rng.integers(0, lighting.max_point + 1))
            n_spot = int(rng.integers(0, lighting.max_spot + 1))
            if n_dir + n_point + n_spot == 0:
                n_dir = 1
            shadow_left = 1 if rng.random() < lighting.shadow_prob else 0
            for _ in range(n_dir):
                direction = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1.0, -0.2)]
                shadow = shadow_left > 0
                shadow_left -= shadow
                color = tint(rng.uniform(*lighting.directional_level))
                self.lighting_sample["lights"].append({"env": i, "type": "directional", "direction": direction, "color": color, "shadow": bool(shadow and self.enable_shadow)})
                self.scene.add_directional_light(
                    direction, color,
                    shadow=shadow and self.enable_shadow,
                    shadow_scale=5, shadow_map_size=2048, scene_idxs=idxs,
                )
            for _ in range(n_point):
                pos = [0.29 + rng.uniform(-0.6, 0.6), rng.uniform(-0.7, 0.7), rng.uniform(0.25, 1.0)]
                color = tint(rng.uniform(*lighting.point_level))
                self.lighting_sample["lights"].append({"env": i, "type": "point", "position": pos, "color": color})
                self.scene.add_point_light(
                    pos, color, scene_idxs=idxs
                )
            for _ in range(n_spot):
                pos = [0.29 + rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6), rng.uniform(0.4, 1.1)]
                aim = np.array([0.29 + rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.0])
                d = aim - np.array(pos)
                inner = rng.uniform(0.2, 0.7)
                direction = (d / np.linalg.norm(d)).tolist()
                outer = inner + rng.uniform(0.1, 0.6)
                color = tint(rng.uniform(*lighting.spot_level))
                self.lighting_sample["lights"].append({"env": i, "type": "spot", "position": pos, "direction": direction, "inner": inner, "outer": outer, "color": color})
                self.scene.add_spot_light(
                    pos, direction, inner, outer, color,
                    scene_idxs=idxs,
                )

    def _load_scene(self, options: dict):
        self.scene_seed = self.episode_seed
        rng = self._scene_rng
        if self.dr.background.enabled:
            # a procedural cube map per env: image-based ambience plus a
            # non-void background wherever no geometry covers the frame
            env_sets = (self.presentation_surfaces[1] if self.presentation_surfaces else environment_face_sets)(max(self.num_textures, 8))
            for sub in self.scene.sub_scenes:
                sub.set_environment_map_from_files(
                    *env_sets[int(rng.integers(len(env_sets)))]
                )
            # per-env floor: visual-only textured slab (nothing collides with
            # the ground; the arm base is fixed and pad/dots are kinematic)
            floors = (self.presentation_surfaces[0] if self.presentation_surfaces else floor_textures)(max(self.num_textures, 8))
            for env_idx in range(self.num_envs):
                fb = self.scene.create_actor_builder()
                fmat = sapien.render.RenderMaterial(
                    base_color=[1, 1, 1, 1],
                    roughness=float(rng.uniform(0.3, 0.95)),
                    specular=float(rng.uniform(0.0, 0.5)),
                )
                fmat.set_base_color_texture(sapien.render.RenderTexture2D(
                    floors[int(rng.integers(len(floors)))]))
                fb.add_box_visual(half_size=[2.0, 2.0, 0.01], material=fmat)
                fb.set_scene_idxs([env_idx])
                fb.initial_pose = sapien.Pose(p=[0, 0, -0.21])
                fb.build_kinematic(name=f"floor_{env_idx}")
        else:
            build_ground(self.scene, altitude=-0.2)
        if self.body_scenario is not None:
            self._load_body_scene(rng)
            return
        # A ruled substrate gets printed paper; a blank one gets silicone. The
        # planner sees the same shape of entry either way, so nothing above
        # here has to know which surface it is laying strokes on.
        if self.substrate.ruled:
            sheets = grid_paper_sheets(
                self.num_textures, self.substrate,
                wear_variants=self.dr.sheet.variants if self.dr.sheet.enabled else 0,
            )
        else:
            sheets = skin_sheets(self.num_textures, self.substrate)

        # One pad build per environment so the sheet and tint vary across the
        # batch. The ruled paper is a UV'd quad on the top face — box UVs wrap
        # all six faces and cannot carry it (see textures.py) — and each env
        # remembers its sheet's line geometry so strokes can trace the ruling.
        self.pad_sheets: list[dict] = []
        self.surface_profiles, self.surface_radius_m = sample_surface_profiles(
            rng, self.num_envs, self.dr.surface.profile, self.dr.surface.cylinder_radius_m
        )
        self.surface_height = self._sample_surface_height(rng)
        self.surface_has_contact_collision = bool(
            self.tool.contact
            and self.tool.contact_radius_m is not None
            and self.surface_height is None
            and all(profile == "flat" for profile in self.surface_profiles)
        )
        pad_actors = []
        fixture_actors = []
        for env_idx in range(self.num_envs):
            sheet = sheets[int(rng.integers(len(sheets)))]
            self.pad_sheets.append(sheet)
            if self.substrate_is_cylinder:
                # The rigid tube under the band: its own kinematic actor, so the
                # band's writable texture stays the pad's only textured shape
                # (_bind_sheet_textures binds the first one it finds). It is
                # posed with the pad every reset; both share the axis origin.
                fixture = self.scene.create_actor_builder()
                fixture.add_visual_from_file(sheet["fixture_obj"])
                fixture.set_scene_idxs([env_idx])
                fixture.initial_pose = sapien.Pose(p=[*self.pad_center, self.PAD_TOP_Z])
                fixture_actors.append(fixture.build_kinematic(name=f"fixture_{env_idx}"))
            builder = self.scene.create_actor_builder()
            tint = rng.uniform(0.82, 1.0)
            mat = sapien.render.RenderMaterial(
                base_color=[tint, tint * rng.uniform(0.95, 1.0), tint * rng.uniform(0.9, 1.0), 1.0],
                roughness=float(rng.uniform(0.55, 0.95)),
                specular=float(rng.uniform(0.02, 0.25)),
            )
            # A curved or displaced substrate is one solid: top, rim and
            # underside. A flat profile retains the cheaper box and UV quad.
            shaped = self.surface_profiles[env_idx] == "cylinder" or self.surface_height is not None
            if not shaped:
                builder.add_box_visual(
                    half_size=[self.pad_half_x, self.pad_half_y, self.pad_thickness / 2],
                    material=mat,
                )
                if self.surface_has_contact_collision:
                    builder.add_box_collision(
                        half_size=[self.pad_half_x, self.pad_half_y, self.pad_thickness / 2],
                    )
            builder.add_visual_from_file(
                self._sheet_mesh(env_idx, sheet),
                pose=sapien.Pose(p=[0, 0, self.pad_thickness / 2 + 2e-4]),
            )
            builder.set_scene_idxs([env_idx])
            builder.initial_pose = sapien.Pose(p=[*self.pad_center, self.PAD_TOP_Z])
            pad_actors.append(builder.build_kinematic(name=f"pad_{env_idx}"))
        self.pad = Actor.merge(pad_actors, name="pad")
        self.fixture = Actor.merge(fixture_actors, name="fixture") if fixture_actors else None
        self._load_palette(rng)
        self._bind_sheet_textures(pad_actors)

        # Ink and laser appearance, redrawn per build. Unlike the dot pool
        # these are PER ENV — a field costs nothing to vary env-to-env, where
        # a shared actor pool forced one width and colour on the whole batch.
        self._configure_ink_field(rng)

    def _configure_ink_field(self, rng):
        ink = self.dr.ink
        radius_range = ink.radius_m
        if self.body_scenario and self.body_scenario.get("schema_version") == 3:
            widths = {round(float(event["curve"]["width_m"]), 12)
                      for event in self.body_scenario["program_binding"]["ink_program"]["events"]
                      if event["kind"] == "stroke"}
            if len(widths) != 1:
                raise ValueError("typed episode requires one declared footprint width; mixed widths need per-stroke deposition")
            radius_range = (next(iter(widths)) / 2,) * 2
        lvl = rng.uniform(*ink.level, size=self.num_envs)
        ink_rgb = torch.as_tensor(
            np.stack([lvl, lvl, np.minimum(1.0, lvl * 1.3)], axis=1), dtype=torch.float32
        )
        self.ink_opacity = torch.as_tensor(
            rng.uniform(*ink.opacity, size=self.num_envs), dtype=torch.float32
        ).to(self.device)
        self.laser_clearance = torch.as_tensor(
            rng.uniform(*self.dr.laser.clearance, size=self.num_envs), dtype=torch.float32
        ).to(self.device)
        self._field_build = {
            "pen_radius_m": torch.as_tensor(
                rng.uniform(*radius_range, size=self.num_envs), dtype=torch.float32
            ),
            "laser_radius_m": torch.as_tensor(
                rng.uniform(*self.dr.laser.spot_radius_m, size=self.num_envs),
                dtype=torch.float32,
            ),
            "ink_rgb": ink_rgb,
        }
        self.ink_field = None  # built with the surface, at episode init
        self._robot_anchor_pose = sapien.Pose()

    def _load_body_scene(self, rng):
        """Scenario branch: full posed visual, local ink patch, and body proxies."""
        if self.body_scenario is None:
            raise RuntimeError("_load_body_scene requires self.body_scenario to be set")
        sheets = skin_sheets(self.num_textures, self.substrate)
        self.pad_sheets = [sheets[int(rng.integers(len(sheets)))] for _ in range(self.num_envs)]
        bodies, patches, geometry = build_scenario_actors(
            self.scene, self.body_scenario, self.num_envs,
        )
        self.body_actor = Actor.merge(bodies, name="posed_body")
        self.pad = Actor.merge(patches, name="tattoo_patch")
        self.fixture = None
        self._scenario_geometry = geometry
        self.surface_height = None
        self.surface_profiles = ["body_mesh"] * self.num_envs
        self.surface_radius_m = np.full(self.num_envs, np.nan, dtype=np.float32)
        self.surface_has_contact_collision = False
        self._load_palette(rng)
        self._bind_sheet_textures(
            patches,
            texture_size=(geometry.surface.cols, geometry.surface.rows),
            base_rgba=geometry.blank_skin_rgba if geometry.target is not None else None,
        )
        self._configure_ink_field(rng)

    def _bind_sheet_textures(
        self,
        pad_actors,
        texture_size: tuple[int, int] | None = None,
        base_rgba: np.ndarray | None = None,
    ):
        """Give each pad a WRITABLE copy of its sheet texture.

        The quad loaded from the sheet OBJ is the pad's only textured render
        shape (the box primitive carries none — box UVs wrap all six faces,
        see textures.py). Swapping its file-loaded texture for one built from
        an array is what lets the pigment field be composited in later;
        ``srgb=True`` is required to match the loader, whose default differs
        from the array constructor's and renders the paper dark if missed.

        Each pad is built with one scene index, so its Actor wraps a single
        Entity — and these ``_objs`` are Entities themselves, unlike the
        rigid-body components behind ``links_map`` in _polish_ee_materials.
        """
        self._sheet_tex = []
        if texture_size is None and self.ink_pixels_per_m is not None:
            texture_size = self._sheet_dims()[2:]
        bases = []
        for env_idx, actor in enumerate(pad_actors):
            entity = actor._objs[0]
            body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            material = next(
                (
                    part.material
                    for shape in body.render_shapes
                    for part in shape.parts
                    if part.material.get_base_color_texture() is not None
                ),
                None,
            )
            if material is None:
                raise RuntimeError(f"pad {env_idx} has no textured sheet quad")
            if base_rgba is None:
                bgr = cv2.imread(self.pad_sheets[env_idx]["png"])
                if texture_size is not None:
                    bgr = cv2.resize(bgr, texture_size, interpolation=cv2.INTER_AREA)
                rgba = np.ascontiguousarray(
                    np.concatenate(
                        [bgr[..., ::-1], np.full(bgr.shape[:2] + (1,), 255, np.uint8)], axis=-1
                    )
                )
            else:
                size = texture_size or (int(base_rgba.shape[1]), int(base_rgba.shape[0]))
                rgba = np.ascontiguousarray(cv2.resize(base_rgba, size, interpolation=cv2.INTER_NEAREST))
            # ``edge`` rather than SAPIEN's default ``repeat``: the body patch
            # mesh reaches past the field's raster (its UVs leave [0, 1]), and
            # a repeating texture stamped the drawn design across the whole
            # limb in every wrist camera (seen 2026-09-03). Paper and pad
            # quads sit inside [0, 1], so the mode changes nothing there.
            tex = sapien.render.RenderTexture2D(
                array=rgba, format="R8G8B8A8Unorm", srgb=True, address_mode="edge",
            )
            material.set_base_color_texture(tex)
            self._sheet_tex.append(tex)
            bases.append(rgba[..., :3].astype(np.float32) / 255.0)
        # the paper the field composites over, kept on device
        self._sheet_base = torch.as_tensor(np.stack(bases), dtype=torch.float32).to(self.device)
        self._sheet_rgba = None
        self._stencil_base = None
        self._stencil_coverage = None
        self._stencil_fraction = np.zeros(self.num_envs, dtype=np.float32)

    def _refresh_sheet_textures(self, force: bool = False):
        """Upload the composited sheet for every env whose field moved.

        Uploads cost per CALL rather than per byte, so envs that deposited
        nothing this interval (travel, hover, settle) are skipped outright.
        """
        if self.ink_field is None:
            return
        dirty = self.ink_field.dirty
        if not force and not bool(dirty.any()):
            return
        base = self._stencil_base if self._stencil_base is not None else self._sheet_base
        self._update_sheet_composite(base, force)
        for env_idx, tex in enumerate(self._sheet_tex):
            if force or bool(dirty[env_idx]):
                tex.upload(np.ascontiguousarray(self._sheet_rgba[env_idx]))
        self.ink_field.dirty.zero_()

    def _update_sheet_composite(self, base, force):
        """Keep the full byte image; recompute exactly the changed pigment rows."""
        ink = self.ink_field.ink_rgb
        # Inference tensors have no mutation counter. Preserve full blending
        # for those callers rather than trusting a cache that cannot be checked.
        untracked = torch.is_inference(base) or torch.is_inference(ink)
        key = None if untracked else (id(base), base._version, id(ink), ink._version)
        if force or untracked or self._sheet_rgba is None or self._sheet_composite_key != key:
            self._sheet_rgba = self.ink_field.composite_rgba(base).cpu().numpy()
            self._sheet_ink = self.ink_field.field.clone()
            self._sheet_composite_key = key
            # Retain the tensors so replacing a source cannot recycle its id.
            self._sheet_composite_sources = (base, ink)
            return
        for env_idx in range(self.num_envs):
            if bool(self.ink_field.dirty[env_idx]):
                self._composite_changed_rows(base, env_idx)

    def _composite_changed_rows(self, base, env_idx):
        current = self.ink_field.field[env_idx]
        previous = self._sheet_ink[env_idx]
        rows = (current != previous).any(dim=1).nonzero().flatten()
        if rows.numel() == 0:
            return
        first, last = rows[[0, -1]].cpu().tolist()
        band = slice(first, last + 1)
        region = (slice(env_idx, env_idx + 1), band, slice(None))
        rgba = self.ink_field.composite_rgba(base, region=region).cpu().numpy()[0]
        self._sheet_rgba[env_idx, band] = rgba
        previous[band].copy_(current[band])

    def set_stencil(self, coverage=None):
        """Render a transfer stencil below pigment without changing InkField.

        The blue-violet guide is appearance only. Its soft coverage remains a
        separate privileged label and never counts as deposited ink.
        """
        if coverage is None:
            self._stencil_base = None
            self._stencil_coverage = None
            self._stencil_fraction = np.zeros(self.num_envs, dtype=np.float32)
        else:
            stencil = torch.as_tensor(coverage, dtype=torch.float32, device=self.device)
            if stencil.shape != self._sheet_base.shape[:3]:
                raise ValueError(
                    f"stencil shape {tuple(stencil.shape)} differs from sheet {tuple(self._sheet_base.shape[:3])}"
                )
            stencil = stencil.clamp(0, 1)
            alpha = (0.24 * stencil).unsqueeze(-1)
            color = torch.tensor([0.18, 0.30, 0.78], dtype=torch.float32, device=self.device)
            self._stencil_base = self._sheet_base * (1 - alpha) + color * alpha
            self._stencil_coverage = stencil
            self._stencil_fraction = stencil.mean(dim=(1, 2)).detach().cpu().numpy()
        if self.ink_field is not None:
            self.ink_field.dirty[:] = True
            self._refresh_sheet_textures(force=True)

    @property
    def stencil_fraction(self) -> np.ndarray:
        """Stencil area as a fraction of the sheet; constant for the episode."""
        return self._stencil_fraction.copy()

    def stencil_visible_fraction(self, deposited=None) -> np.ndarray:
        """Stencil still showing after pigment covers it: mean(stencil x (1 - ink)).

        Falls as the drawing progresses, so it is the per-step label; the
        constant guide area is ``stencil_fraction``.
        """
        if self._stencil_coverage is None:
            return np.zeros(self.num_envs, dtype=np.float32)
        if deposited is None:
            if self.ink_field is None:
                raise RuntimeError("ink_field is unavailable for the stencil visibility label")
            deposited = self.ink_field.field
        deposited = torch.as_tensor(deposited, dtype=torch.float32, device=self.device).clamp(0, 1)
        if deposited.shape != self._stencil_coverage.shape:
            raise ValueError(
                f"deposited shape {tuple(deposited.shape)} differs from stencil {tuple(self._stencil_coverage.shape)}"
            )
        visible = self._stencil_coverage * (1 - deposited)
        return visible.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32)

    def preink(self, strokes_per_env):
        """Open the episode on a sheet that already carries these strokes.

        What a removal task needs: the laser's target has to exist before the
        first control step. It goes down through the same splat the pen uses,
        so pre-inked and drawn pigment are the same substance — a policy
        cannot learn to tell "already there" from "just drawn".
        """
        if self.ink_field is None:
            self.ink_field = InkField(
                self.num_envs, self.surface, device=self.device, **self._field_build
            )
        self.ink_field.rasterize(self.surface, strokes_per_env, self.ink_opacity)
        self._refresh_sheet_textures(force=True)

    def _surface_pose(self, b):
        fixed = self.config.surface_pose
        if fixed is not None:
            center = torch.tensor(fixed.position_m, dtype=torch.float32).repeat(b, 1)
            rotation = torch.tensor(quat2mat(fixed.quaternion_wxyz), dtype=torch.float32).repeat(b, 1, 1)
            quaternion = torch.tensor(fixed.quaternion_wxyz, dtype=torch.float32).repeat(b, 1)
            return center, rotation, quaternion
        pad_cfg = self.dr.pad
        xy = torch.rand(b, 2) * 2 * pad_cfg.xy_range - pad_cfg.xy_range
        yaw = torch.rand(b) * 2 * pad_cfg.yaw_range - pad_cfg.yaw_range
        z_range = pad_cfg.z_range or tuple(self.substrate.rest_z_m)
        lo, hi = z_range
        top_z = torch.rand(b) * (hi - lo) + lo
        # A few degrees of roll and pitch: the surface is a plane, not a
        # height. Rotation order Rz(yaw) @ Ry(pitch) @ Rx(roll).
        roll = torch.rand(b) * 2 * pad_cfg.tilt_range - pad_cfg.tilt_range
        pitch = torch.rand(b) * 2 * pad_cfg.tilt_range - pad_cfg.tilt_range
        rot = np.zeros((b, 3, 3), dtype=np.float32)
        quat = np.zeros((b, 4), dtype=np.float32)
        for i in range(b):
            rot[i] = euler2mat(float(roll[i]), float(pitch[i]), float(yaw[i]), "sxyz")
            quat[i] = euler2quat(float(roll[i]), float(pitch[i]), float(yaw[i]), "sxyz")
        rot_t = torch.as_tensor(rot)
        top_center = torch.zeros(b, 3)
        top_center[:, 0] = self.pad_center[0] + xy[:, 0]
        top_center[:, 1] = self.pad_center[1] + xy[:, 1]
        top_center[:, 2] = top_z
        return top_center, rot_t, torch.as_tensor(quat)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: like the upstream drawing envs, partial resets are unsupported
        # — the generator always resets whole batches. The field itself is
        # per-env and would take a partial reset; the assertion stays because
        # nothing downstream is written for it.
        if self.body_scenario is not None:
            self._initialize_body_episode(env_idx)
            return
        self._step_count = 0
        with torch.device(self.device):
            b = len(env_idx)
            assert b == self.num_envs, "TatbotDraw-v0 does not support partial resets"

            qpos = (
                torch.from_numpy(self.agent.keyframes["rest"].qpos)
                .float()
                .to(self.device)
                .repeat(b, 1)
            )
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose())
            self._robot_anchor_pose = sapien.Pose()

            top_center, rot_t, quat = self._surface_pose(b)
            # box centre sits half a thickness below the top face, along the normal
            p = top_center - rot_t[:, :, 2] * (self.pad_thickness / 2)
            self.pad_pose = Pose.create_from_pq(p, torch.as_tensor(quat))
            self.pad.set_pose(self.pad_pose)
            if self.fixture is not None:
                self.fixture.set_pose(self.pad_pose)

            self.pad_top_center = top_center  # (B, 3) top-face centre
            self._place_palette(b)
            self._reset_ink(b)
            self.pad_rot = rot_t  # (B, 3, 3) canvas frame; column 2 is the normal

        # The canvas moved, so the surface is rebuilt and the sheet starts
        # bare. Both live outside the torch.device block: they hold their own
        # device-placed tensors.
        self.surface = self._build_surface()
        if self.ink_field is None:
            self.ink_field = InkField(
                self.num_envs, self.surface, device=self.device, **self._field_build
            )
        else:
            self.ink_field.reset()
        self._refresh_sheet_textures(force=True)

    def _initialize_body_episode(self, env_idx: torch.Tensor):
        if self.body_scenario is None:
            raise RuntimeError("_initialize_body_episode requires self.body_scenario to be set")
        self._step_count = 0
        with torch.device(self.device):
            b = len(env_idx)
            assert b == self.num_envs, "TatbotDraw-v0 does not support partial resets"
            qpos = torch.from_numpy(self.agent.keyframes["rest"].qpos).float().to(self.device).repeat(b, 1)
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose())
            self._robot_anchor_pose = sapien.Pose()
            base = self._scenario_geometry.surface
            self.surface = MeshPatchSurface(
                [base.patches[0]] * b,
                [base.posed_vertices[0]] * b,
                device=self.device,
                width_m=base.width_m,
                height_m=base.height_m,
                cols=base.cols,
                rows=base.rows,
                normals=[base.normals[0]] * b,
            )
            source = self.body_scenario["placement"]["anchor"]
            face, bary = int(source["face"]), np.asarray(source["barycentric"])
            center = bary @ base.posed_vertices[0][face]
            normal = self.surface.base_normal_np(0)
            _, du, dv, _ = self.surface.frame(torch.zeros(b, 2, device=self.device))
            ex = du / du.norm(dim=1, keepdim=True)
            ez = torch.as_tensor(normal, dtype=torch.float32, device=self.device).expand(b, 3)
            ey = torch.linalg.cross(ez, ex)
            ey /= ey.norm(dim=1, keepdim=True)
            self.pad_top_center = torch.as_tensor(center, dtype=torch.float32, device=self.device).expand(b, 3)
            self.pad_rot = torch.stack([ex, ey, ez], dim=2)
            self._place_palette(b)
            self._reset_ink(b)
        if self.ink_field is None:
            self.ink_field = InkField(self.num_envs, self.surface, device=self.device, **self._field_build)
        else:
            self.ink_field.reset()
        self._refresh_sheet_textures(force=True)

    def _sample_surface_height(self, rng):
        """Optional per-env displacement over the flat/cylinder base profile.

        Drawn at scene build because the shape is baked into the sheet mesh —
        see SurfaceDR. Returns None rather than a grid of zeros so an ordinary
        flat profile keeps the cheaper PlanarSurface path.
        """
        cfg = self.dr.surface
        if not cfg.enabled:
            return None
        cols = int(cfg.grid_cols)
        rows = int(round(cols * self.substrate.height_m / self.substrate.width_m)) | 1
        n = self.num_envs
        feature_m = cfg.feature_m or tuple(self.substrate.surface_feature_m)
        max_slope_rad = cfg.max_slope_rad or tuple(self.substrate.surface_max_slope_rad)
        amplitude_m = cfg.amplitude_m or tuple(self.substrate.surface_amplitude_m)
        return random_height_field(
            rng, n, rows, cols,
            feature_m=rng.uniform(*feature_m, n),
            max_slope_rad=rng.uniform(*max_slope_rad, n),
            amplitude_m=rng.uniform(*amplitude_m, n),
            components=cfg.components,
            taper_frac=cfg.taper_frac,
            width_m=self.substrate.width_m, height_m=self.substrate.height_m,
        )

    def _sheet_mesh(self, env_idx: int, sheet: dict) -> str:
        """The sheet's visual: the flat quad, or this env's shaped surface.

        The mesh is evaluated through the Surface classes on a pad-local frame,
        so the triangles the cameras see are the same geometry the deposit gate
        and the tool orientation are computed from. Deriving the picture
        separately from a height formula is exactly how the two would drift.
        """
        profile = self.surface_profiles[env_idx]
        if profile == "flat" and self.surface_height is None:
            return sheet["obj"]
        if self.surface_height is None:
            cols = int(self.dr.surface.grid_cols)
            rows = int(round(cols * self.substrate.height_m / self.substrate.width_m)) | 1
            height = torch.zeros(1, rows, cols)
        else:
            rows, cols = self.surface_height.shape[1:]
            height = self.surface_height[env_idx][None]
        center = torch.zeros(1, 3)
        rot = torch.eye(3)[None]
        chart = (
            PlaneChart(center, rot)
            if profile == "flat"
            else CylinderChart(
                center,
                rot,
                torch.tensor([float(self.surface_radius_m[env_idx])]),
                self.dr.surface.cylinder_axis,
            )
        )
        local = DisplacedSurface(
            chart, height, *self._sheet_dims(),
        )
        us = torch.linspace(-self.pad_half_x, self.pad_half_x, cols)
        vs = torch.linspace(-self.pad_half_y, self.pad_half_y, rows)
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
        point, _, _, normal = local.env_view(0, uv.shape[0]).frame(uv)
        stem = Path(sheet["obj"]).with_name(
            f"profile_{profile}_{self.substrate.name}_{env_idx:02d}"
        )
        return write_surface_mesh(
            stem, Path(sheet["obj"]).stem,
            point.cpu().numpy(), normal.cpu().numpy(), rows, cols,
            thickness_m=self.sheet_thickness,
        )

    def _sheet_dims(self):
        """The substrate's extent and texture resolution, as Surface takes them."""
        sub = self.substrate
        ppm = self.ink_pixels_per_m or 0
        return (sub.width_m, sub.height_m, max(sub.texel_cols, int(np.ceil(sub.width_m * ppm))),
                max(sub.texel_rows, int(np.ceil(sub.height_m * ppm))))

    def _build_surface(self):
        center = self.pad_top_center.to(self.device)
        rot = self.pad_rot.to(self.device)
        dims = self._sheet_dims()
        if all(profile == "flat" for profile in self.surface_profiles) and self.surface_height is None:
            return PlanarSurface(center, rot, *dims)
        if self.surface_height is None:
            cols = int(self.dr.surface.grid_cols)
            rows = int(round(cols * self.substrate.height_m / self.substrate.width_m)) | 1
            height = torch.zeros(self.num_envs, rows, cols, device=self.device)
        else:
            height = self.surface_height.to(self.device)
        radii = torch.as_tensor(self.surface_radius_m, dtype=torch.float32, device=self.device)
        chart = (
            PlaneChart(center, rot)
            if bool(torch.isinf(radii).all())
            else CylinderChart(center, rot, radii, self.dr.surface.cylinder_axis)
        )
        return DisplacedSurface(chart, height, *dims)

    @property
    def canvas_frame_np(self):
        """Per-env canvas frame: top-face centre (B,3) and rotation (B,3,3).

        Column 2 of the rotation is the outward surface normal; the expert
        builds trajectories in this frame so drawing follows the tilted plane.
        """
        return self.pad_top_center.cpu().numpy(), self.pad_rot.cpu().numpy()

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
            # The PhysX GPU pipeline lets a fixed-base articulation's root
            # creep: measured 3.1 mm in +x/+z over 30 s on an IDLE arm,
            # quadratic in time, identical in every env, zero on the CPU
            # backend (RTX 3070). The tool then rides the
            # drift out of the 0.5 mm contact band on every later stroke
            # while its joint state still says it is on the sheet. Re-pin
            # the root where the episode placed it, every control step;
            # the per-step creep is a few microns and the apply below
            # pushes the correction before the next physics step.
            self.agent.robot.set_pose(self._robot_anchor_pose)

        tcp = self.agent.tcp.pose.p
        uv, dist, incidence = self.surface.project(tcp)
        touching = self._interaction_mask(dist)
        # Away at the palette the tip is nowhere near the sheet, but a planar
        # surface extends forever and a cap rim can sit inside the band of a
        # high-resting pad: the plan says which steps are dips, and none of
        # them marks.
        step = self._step_count
        if self._dip_mask is not None and step < self._dip_mask.shape[1]:
            touching = touching & ~self._dip_mask[:, step]
        inf = torch.full_like(dist, float("inf"))
        neg_inf = torch.full_like(dist, float("-inf"))
        self.interaction_min_m = torch.minimum(
            self.interaction_min_m, torch.where(touching, dist, inf))
        self.interaction_max_m = torch.maximum(
            self.interaction_max_m, torch.where(touching, dist, neg_inf))
        self.interaction_sum_m += torch.where(touching, dist, torch.zeros_like(dist))
        self.interaction_frames += touching.long()
        self._ink_step(tcp, touching, step)
        # Inactive frames break continuous deposition. Skipping this call
        # would connect the previous stroke to the next across a pen lift.
        self._apply_tool(uv, incidence, touching)

        self._pulse_emitter()
        self._step_count += 1
        if self._step_count % self.texture_refresh_steps == 0:
            self._refresh_sheet_textures()

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def _interaction_mask(self, distance: torch.Tensor) -> torch.Tensor:
        """Whether the resolved working point may affect the substrate."""
        return ((distance <= self.contact_above_tolerance_m)
                & (distance >= -self.max_penetration_m))

    # What each registered tool kind does to the pigment under its tip, from
    # the shared table in tatbot_sim.tasks -- the same one the (task, tool,
    # substrate) validator reads, so a run cannot be approved against one
    # notion of what the laser does and then rendered against another.
    FIELD_OPS = tasks.FIELD_OPS

    def _pulse_emitter(self):
        """Flash the laser's emitter window.

        A Q-switched pen fires in pulses, and a still blue dot reads as a lamp
        rather than a laser. Emission is RGBA on the loaded material and the
        path tracer treats it as an area light, so this throws real blue onto
        the skin for the frames it is bright.

        The period is deliberately not a whole number of control steps: a
        render may sample every Nth step (sim_cinematic --speed), and a pulse
        whose period divides the stride would be sampled at one fixed phase and
        come out constant. At 9 Hz against 30 Hz control the phase advances
        every frame for any small stride.
        """
        mats = getattr(self, "_emitter_materials", None)
        if not mats:
            return
        detail = self.tool.tip_detail or {}
        rgb = [float(v) for v in str(detail.get("color")).split()[:3]]
        hz = float(detail.get("hz", 9.0))
        peak = float(detail.get("peak", 22.0))
        phase = 2 * np.pi * hz * self._step_count / float(self.control_freq)
        # sharpened sine: mostly dark with a brief bright spike, which is what
        # a pulsed emitter looks like; a plain sine reads as a throbbing lamp
        level = 0.06 + 0.94 * (0.5 + 0.5 * np.sin(phase)) ** 4
        value = [c * peak * float(level) for c in rgb] + [1.0]
        for m in mats:
            m.emission = value

    def _apply_tool(self, uv, incidence, touching):
        """Let the fitted tool act on the pigment under the tip.

        Selecting on the registry's kind is what makes the laser a workspace
        change rather than an env change: TATBOT_TOOL_ID already swaps the
        rendered geometry, and this swaps what that geometry does.
        """
        if self.ink_field is None:
            return
        op = self.FIELD_OPS.get(self.tool.kind)
        if op is None and not self._tool_kind_warned:
            self._tool_kind_warned = True
            print(
                f"[env] tool kind {self.tool.kind!r} is not in FIELD_OPS — "
                "depositing like a pen; add it to TatbotDrawEnv.FIELD_OPS"
            )
        if op == "remove":
            self.ink_field.remove(
                self.surface, uv, laser_eta(self.laser_clearance, incidence), touching
            )
        else:
            self.ink_field.deposit_segment(self.surface, uv, self.ink_opacity * self._charge_factor(),
                                           touching)

    # --- the palette, and the ink the tool carries ---------------------------------

    def _load_palette(self, rng):
        """Freestanding CAD palette, independently placed per episode.

        Unmeasured physical rim heights use explicit simulation estimates.
        """
        self.palette = None
        self._cap_rims: dict[str, torch.Tensor] = {}
        self._dip_mask = None
        self._dip_credit = None
        self._palette_rotation = np.eye(3)
        self.palette_pose_source = None
        if not self.dr.palette.enabled:
            return
        from tatbot_sim import palette as sim_palette

        ink = tools.ink_registry()
        scene = self.config.palette_scene
        self.palette_pose_source, transform = sim_palette.base_transform(tools.REPO, scene)
        self._palette_rotation = np.asarray(transform[:3, :3], dtype=np.float64)
        if self.dr.palette.center_m is None:
            self.dr.palette.center_m = tuple(float(v) for v in transform[:3, 3])
        self._palette_slots = self.config.palette
        self._palette_layout = scene.rims
        self._palette_load = self.config.palette_load
        inks = ink.load_inks(tools.REPO)
        self._cap_mesh_dir = TEX_DIR / "inkcaps"
        actors = []
        wet: dict[str, list[float]] = {}   # slot -> ink rgb, for the disc actors below
        for env_idx in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            slab = sapien.render.RenderMaterial(
                base_color=[0.05, 0.05, 0.055, 1.0], roughness=0.6, specular=0.2)
            add_palette_body(builder, scene, slab)
            for slot_id, slot in self._palette_slots.items():
                off = self._palette_layout.get(slot_id)
                if off is None:
                    continue
                cap_mat = sapien.render.RenderMaterial(
                    base_color=[0.93, 0.93, 0.9, 1.0], roughness=0.35, specular=0.4)
                # A hollow cup (capmesh) — the rim is the frame, the cup hangs below.
                builder.add_visual_from_file(
                    str(capmesh.cap_mesh_path(self._cap_mesh_dir, slot.size)),
                    material=cap_mat,
                    pose=sapien.Pose(p=[off[0], off[1], off[2] + self.dr.palette.rim_above_tag_m]))
                load = self._palette_load.get(slot_id)
                if load is not None and not load.dry and load.ink_id in inks:
                    wet[slot_id] = [c / 255.0 for c in inks[load.ink_id].rgb]
            builder.set_scene_idxs([env_idx])
            builder.initial_pose = sapien.Pose(p=list(self.dr.palette.center_m))
            actors.append(builder.build_kinematic(name=f"palette_{env_idx}"))
        self.palette = Actor.merge(actors, name="palette")
        # The ink surface in each wet cap is its OWN kinematic actor (one per
        # env, merged per slot): a render shape's local pose is frozen once
        # attached, and the surface has to move — it drops as the cap drains
        # (operator, 2026-08-28) and it follows the rack's per-episode jitter.
        # What the dip plunges below is the same number (ink_spec.dip_plunge_m).
        self._ink_disc_actors: dict[str, Actor] = {}
        for slot_id, rgb in wet.items():
            slot = self._palette_slots[slot_id]
            per_env = []
            for env_idx in range(self.num_envs):
                b = self.scene.create_actor_builder()
                ink_mat = sapien.render.RenderMaterial(
                    base_color=[*rgb, 1.0], roughness=0.12, specular=0.7)
                b.add_cylinder_visual(
                    radius=slot.size.bore_diameter_m / 2 - 0.0001, half_length=0.0002, material=ink_mat,
                    pose=sapien.Pose(q=[0.7071068, 0.0, -0.7071068, 0.0]))
                b.set_scene_idxs([env_idx])
                b.initial_pose = sapien.Pose(p=[0.0, 0.0, -1.0])
                per_env.append(b.build_kinematic(name=f"ink_{slot_id}_{env_idx}"))
            self._ink_disc_actors[slot_id] = Actor.merge(per_env, name=f"ink_{slot_id}")
        self._cap_fill_ul = {s: float(self._palette_load[s].fill_ul) for s in self._palette_slots}

    def _update_ink_discs(self, slots=None) -> None:
        """Put every wet cap's ink surface where its env's fill says it is:
        the cap rim (rack pose included) plus capmesh's level for the fill."""
        if not getattr(self, "_ink_disc_actors", None) or not self._cap_rims:
            return
        for slot_id in (slots or self._ink_disc_actors):
            actor = self._ink_disc_actors.get(slot_id)
            if actor is None:
                continue
            slot = self._palette_slots[slot_id]
            rims = self._cap_rims[slot_id].clone()
            levels = [capmesh.ink_level_z(
                slot.size, self._cap_fills[i].get(slot_id, 0.0) if self._cap_fills else 0.0)
                for i in range(rims.shape[0])]
            normal = torch.as_tensor(self._palette_normals, dtype=rims.dtype, device=rims.device)
            level = torch.as_tensor(levels, dtype=rims.dtype, device=rims.device)
            rims += normal * level[:, None]
            quaternion = torch.as_tensor(self._palette_quats, dtype=rims.dtype, device=rims.device)
            actor.set_pose(Pose.create_from_pq(rims, quaternion))

    def _drain_cap(self, env_idx: int, slot_id: str, ul: float) -> None:
        """A dip took ``ul`` out of this env's cap: the fill is per env (each
        env drains its own rack), and the ink surface follows it down."""
        fills = self._cap_fills[env_idx]
        fills[slot_id] = max(0.0, fills.get(slot_id, 0.0) - ul)
        self._update_ink_discs([slot_id])

    def _place_palette(self, b: int):
        """Re-place the rack for the episode: a few millimetres and a few
        degrees of wander, the way a fixture that is lifted and put back
        moves. Records every cap rim in the world frame for the planner."""
        if self.palette is None:
            return
        pdr = self.dr.palette
        rng = self._rng
        centre = np.asarray(pdr.center_m, dtype=np.float32)[None, :].repeat(b, axis=0)
        centre[:, :2] += rng.uniform(-pdr.xy_jitter_m, pdr.xy_jitter_m, (b, 2))
        centre[:, 2] += rng.uniform(-pdr.z_jitter_m, pdr.z_jitter_m, b)
        yaw = rng.uniform(-pdr.yaw_jitter_rad, pdr.yaw_jitter_rad, b)
        # The rack's own orientation, then this episode's wander on top of it.
        # Compose the episode's yaw with the scene pose, including any tilt.
        # Cap rims and ink discs must follow that same orientation.
        matrices = np.stack([euler2mat(0.0, 0.0, float(y), "sxyz") @ self._palette_rotation
                             for y in yaw])
        rotations = np.stack([mat2quat(m) for m in matrices]).astype(np.float32)
        self._palette_quats = rotations
        self._palette_normals = matrices[:, :, 2].astype(np.float32)
        self.palette.set_pose(Pose.create_from_pq(torch.as_tensor(centre),
                                                  torch.as_tensor(rotations)))
        self._cap_rims = {}
        for slot_id, off in self._palette_layout.items():
            local = np.asarray(off, dtype=np.float64) + np.array([0, 0, pdr.rim_above_tag_m])
            point = centre + np.einsum("bij,j->bi", matrices, local)
            self._cap_rims[slot_id] = torch.as_tensor(point.astype(np.float32))

    def cap_rims_np(self) -> dict | None:
        """``{slot: (B, 3)}`` world rim centres for this episode, or None
        without a palette — what plan_batch needs to build dips."""
        if not self._cap_rims:
            return None
        return {k: v.cpu().numpy() for k, v in self._cap_rims.items()}

    def _reset_ink(self, b: int):
        """The tool opens the episode FULL and at the datasheet's capacity,
        nothing spent yet — a drawing episode is not a session (operator,
        2026-08-29). The planner overrides both per env through
        set_dip_schedule (config.InkDR.initial_charge_frac / capacity_scale)
        when dips are wanted, and a --task dip episode opens empty."""
        dev = self.device
        cap = float(self.ink_policy.charge_capacity_ul)
        self.ink_capacity_ul = torch.full((b,), cap, device=dev)
        self.ink_charge_ul = torch.full((b,), cap, device=dev)
        self.ink_used_ul = torch.zeros(b, device=dev)
        self.ink_contact_mm = torch.zeros(b, device=dev)
        self.ink_contact_s = torch.zeros(b, device=dev)
        self.ink_dips = torch.zeros(b, dtype=torch.int64, device=dev)
        self.interaction_min_m = torch.full((b,), float("inf"), device=dev)
        self.interaction_max_m = torch.full((b,), float("-inf"), device=dev)
        self.interaction_sum_m = torch.zeros(b, device=dev)
        self.interaction_frames = torch.zeros(b, dtype=torch.int64, device=dev)
        self._prev_tcp = None
        self._dip_mask = None
        self._dip_credit = None
        self._dip_slot: dict[tuple[int, int], str] = {}
        # every rack opens the episode as palette_load.yaml says it was poured
        self._cap_fills = [dict(getattr(self, "_cap_fill_ul", {})) for _ in range(b)]
        self._update_ink_discs()

    def set_dip_schedule(self, plan) -> None:
        """Hand the env the batch's dips (planning.BatchPlan): which EPISODE
        steps are spent at the palette, and on which step each dip's charge
        lands. Steps are counted from the episode start, approach included,
        the same clock _after_control_step keeps."""
        if getattr(plan, "ink_capacity_ul", None) is not None:
            self.ink_capacity_ul = torch.as_tensor(
                np.asarray(plan.ink_capacity_ul, dtype=np.float32), device=self.device)
        if getattr(plan, "ink_initial_ul", None) is not None:
            self.ink_charge_ul = torch.minimum(
                torch.as_tensor(np.asarray(plan.ink_initial_ul, dtype=np.float32), device=self.device),
                self.ink_capacity_ul)
        if plan.dip_mask is None:
            self._dip_mask = None
            self._dip_credit = None
            return
        b, t_draw = plan.dip_mask.shape
        total = plan.n_app + t_draw
        mask = torch.zeros(self.num_envs, total, dtype=torch.bool, device=self.device)
        credit = torch.zeros(self.num_envs, total, device=self.device)
        mask[:b, plan.n_app:] = torch.as_tensor(plan.dip_mask, device=self.device)
        self._dip_slot = {}
        for i in range(b):
            for step, dip in zip(plan.dip_credits[i], plan.dips[i], strict=True):
                credit[i, plan.n_app + step] += dip["charge_after_ul"] - dip["charge_before_ul"]
                self._dip_slot[(i, plan.n_app + step)] = dip["slot"]
        self._dip_mask = mask
        self._dip_credit = credit

    def _ink_step(self, tcp: torch.Tensor, touching: torch.Tensor, step: int) -> None:
        """Run the charge model for one control step: credit a dip landing
        now, then debit contact by distance and by time — the same
        arithmetic ink_spec.InkPolicy.stroke_ul does for the ledger."""
        pol = self.ink_policy
        if not pol.dips:
            self._prev_tcp = tcp.clone()
            return
        if self._dip_credit is not None and step < self._dip_credit.shape[1]:
            credit = self._dip_credit[:, step]
            landed = credit > 0
            if bool(landed.any()):
                self.ink_charge_ul = torch.minimum(self.ink_charge_ul + credit, self.ink_capacity_ul)
                self.ink_dips += landed.long()
                if pol.touches_stock:
                    for i in torch.nonzero(landed).flatten().tolist():
                        slot_id = self._dip_slot.get((i, step))
                        if slot_id is not None:
                            self._drain_cap(i, slot_id, float(credit[i]))
        if self._prev_tcp is None:
            moved_mm = torch.zeros_like(self.ink_charge_ul)
        else:
            moved_mm = torch.linalg.norm(tcp - self._prev_tcp, dim=1) * 1000.0
        self._prev_tcp = tcp.clone()
        dt = 1.0 / float(self.control_freq)
        debit = torch.where(
            touching, pol.deposit_ul_per_mm * moved_mm + pol.bleed_ul_per_s * dt,
            torch.zeros_like(moved_mm))
        taken = torch.minimum(self.ink_charge_ul, debit)
        self.ink_charge_ul = self.ink_charge_ul - taken
        self.ink_used_ul = self.ink_used_ul + taken
        self.ink_contact_mm = self.ink_contact_mm + torch.where(touching, moved_mm, torch.zeros_like(moved_mm))
        self.ink_contact_s = self.ink_contact_s + touching.float() * dt

    def _charge_factor(self) -> torch.Tensor:
        """Deposition scale from the charge: full at a fresh dip, InkDR.dry_floor
        at empty. 1 for a tool that carries no ink."""
        pol = self.ink_policy
        if not pol.dips or pol.charge_capacity_ul <= 0:
            return torch.ones_like(self.ink_opacity)
        frac = torch.clamp(self.ink_charge_ul / torch.clamp(self.ink_capacity_ul, min=1e-9), 0.0, 1.0)
        floor = float(self.dr.ink.dry_floor)
        return floor + (1.0 - floor) * frac

    def ink_episode_stats(self) -> dict:
        """What each episode spent, as numpy — for the dataset's meta/ink.json."""
        count = self.interaction_frames.clamp(min=1)
        mean = self.interaction_sum_m / count
        none = self.interaction_frames == 0
        return {
            "mode": self.ink_policy.mode,
            "charge_end_ul": self.ink_charge_ul.cpu().numpy(),
            "capacity_ul": self.ink_capacity_ul.cpu().numpy(),
            "used_ul": self.ink_used_ul.cpu().numpy(),
            "contact_mm": self.ink_contact_mm.cpu().numpy(),
            "contact_s": self.ink_contact_s.cpu().numpy(),
            "dips": self.ink_dips.cpu().numpy(),
            "interaction_frames": self.interaction_frames.cpu().numpy(),
            "interaction_min_m": torch.where(
                none, torch.full_like(self.interaction_min_m, float("nan")),
                self.interaction_min_m).cpu().numpy(),
            "interaction_mean_m": torch.where(
                none, torch.full_like(mean, float("nan")), mean).cpu().numpy(),
            "interaction_max_m": torch.where(
                none, torch.full_like(self.interaction_max_m, float("nan")),
                self.interaction_max_m).cpu().numpy(),
        }

    def evaluate(self):
        # pigment on the sheet: the ground truth a removal task is scored on,
        # and free now that ink is a quantity rather than a pile of actors
        if self.ink_field is None:
            return {"ink_coverage": torch.zeros(self.num_envs, device=self.device)}
        return {"ink_coverage": self.ink_field.coverage()}

    def _get_obs_extra(self, info: dict):
        return {"tcp_pose": self.agent.tcp.pose.raw_pose}
