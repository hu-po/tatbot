"""Generate scripted stroke episodes and write a LeRobot v3 dataset.

The pipeline in one line per stage: `plan_batch` samples WHAT to draw
(tatbot_sim.planning), the expert solves HOW (IK + noise + floor clamp), the
env renders, the sensor models corrupt (depth) and jitter (RGB response),
and the writer streams a schema-parity LeRobot dataset. Every randomization
range lives in the DR tree (tatbot_sim.config) and the FULL resolved
configuration is dumped into run_meta.json — a dataset records exactly which
distribution produced it.

Usually driven through `python -m tatbot_sim.factory <distribution>`, which
picks the tool and the preset recipe for one of the three datasets this
factory produces (tatbot_sim.distributions) — this module is the engine under
it, and stays directly callable for a run that is deliberately none of them.

Usage (on an x86_64 sim host):
    python -m tatbot_sim.factory paper-draw --num-episodes 64 --num-envs 16 \
        --out-dir ~/tatbot-sim/datasets/shapes-v0
    # any DR leaf is a CLI flag, e.g.:
    #   --dr.pad.tilt-range 0.15 --dr.lighting.ambient 0.02 0.3
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from wrist_cameras import common_stream_profile

from tatbot_sim import design_scene, interaction, tasks, tools
from tatbot_sim import judge as judging
from tatbot_sim.backends.maniskill import ManiSkillWorld
from tatbot_sim.carriage import CARRIAGE_JOINT
from tatbot_sim.config import ARTWORK_HORIZON_STEPS, DRConfig
from tatbot_sim.contact_force import (
    FORCE_MODEL as CONTACT_FORCE_MODEL,
)
from tatbot_sim.contact_force import (
    NOT_BENCH_CALIBRATED as CONTACT_FORCE_DISCLAIMER,
)
from tatbot_sim.episode import Episode
from tatbot_sim.expert import (
    StrokeExpert,
    highest_reachable_z,
    reachable_canvas_masks,
    reachable_height_ceiling,
    worst_reach_residual,
)
from tatbot_sim.inkmap.contracts import document_sha256
from tatbot_sim.lerobot_writer import LeRobotWriter, quantize_depth_codes
from tatbot_sim.observations import ObservationBuilder
from tatbot_sim.planning import (
    EPISODE_VARIANT_OUTCOMES,
    SceneTooLongError,
    apply_episode_variant,
    plan_batch,
    plan_tattoo_scenario,
)
from tatbot_sim.reference import CONTACT_REFERENCE_TOLERANCE_M, REACH_TOLERANCE_M
from tatbot_sim.repo import source_state
from tatbot_sim.temporal_labels import TemporalRecorder

EPISODE_START_MAX_LEAD_RAD = 0.1
"""Gross data-integrity bound, not a robot-motion acceptance threshold."""


MAX_CONSECUTIVE_SKIPS = 5
"""Give up after this many batches in a row refuse to fit the horizon. One
refusal is an unlucky scene draw and is skipped; a run where every draw refuses
is a recipe that cannot draw what it samples, and spinning on it would burn a
GPU all night to write nothing."""


@dataclass
class Args:
    out_dir: str
    tool_id: str | None = None
    substrate: str | None = None
    num_episodes: int = 64
    num_envs: int = 16
    horizon: int = ARTWORK_HORIZON_STEPS
    """Maximum control steps at 30 Hz; see config.ARTWORK_HORIZON_STEPS.
    Completed episodes end naturally; an over-budget artwork is never truncated."""
    seed: int = 0
    sensor_profile: str = "deployment"
    """Camera layout: deployment registry, or explicit legacy-two-view replay."""
    production_draw_config: str | None = None
    """Use the production Cartesian/C++ planner with this tatbot.draw-config/1.
    Initially supports nominal flat-paper ballpoint intent. Executes native
    400 Hz commands and samples configured camera cadence; no expert noise."""
    tool_calibration_jitter: bool = False
    """Let the named factory sample one persistent mount-frame tip offset for
    this shard from the measured calibration uncertainty. The factory applies
    it during explicit configuration resolution so URDF, IK and metadata share the same draw."""
    tool_calibration_scale: float = 1.0
    """Multiplier on the measured uncertainty radius. One spans the retained
    touch-off diagnostic; zero disables displacement while preserving metadata."""
    carriage_ik: bool = False
    """Solve the tool carriage as a seventh IK axis, the way the executor's own
    carriage_ik flag does (cpp/teleop/square_probe.cpp). Off pins it at rest and
    reproduces six-axis behaviour exactly. Opt-in: turning it on starts the arm
    at the carriage's 2 mm drawing bias, and every reach mask, clearance margin
    and accepted body scenario in this repo was qualified against the pinned
    pose."""
    sim_backend: str = "auto"
    task: str = "artwork"
    """Shared reviewed artwork by default; spiral is calibration. Explicit
    mix/language/maze/shapes remain legacy replay tasks. Erase removes artwork."""
    artwork_split: str = "train"
    """Whole artwork-family split: train, validation, test, or explicit all."""
    artwork_width_mm: float = 0.3
    """Nominal simulated footprint used by the artwork planner, not physical calibration."""
    artwork_pixels_per_m: int = 16000
    """Artwork pigment/texture resolution; 16 pixels/mm resolves the nominal 0.3 mm footprint."""
    design: str | None = None
    """A portable design (tatbot.inkmap-design/1 on a plane or cylinder chart,
    as Inkmap's Paper and Cylinder workspaces and `tatbot design place` export
    it) drawn in place of the shared collection, for --task artwork or erase.
    A plane design goes on a flat substrate, a cylinder design on the paper
    cylinder of the same radius (TATBOT_SUBSTRATE=paper_cylinder); the ink
    footprint is the design's own stroke width. See tatbot_sim.design_scene."""
    design_placement: str = "authored"
    """Where the design lands: "authored" keeps the anchor, rotation and mirror
    Inkmap saved, about the canvas centre, and refuses a placement the tool
    cannot reach; "sampled" recentres the artwork and draws an offset inside
    the reach envelope per env, the way the shared collection is placed."""
    squiggle_frac: float = 0.0
    """For --task mix: fraction of batches that draw squiggles instead of
    language scenes."""
    erase_frac: float = 0.0
    """For --task mix: fraction of batches that REMOVE a scene instead of
    drawing one. Defaults off — erase batches need a removal tool fitted, so
    turning this up on a pen-fitted rig is an error, not a silent mix."""
    dip_frac: float = 0.0
    """For --task mix: fraction of batches that are DIP episodes — hover,
    leave for the palette, charge the tool, come back, no stroke (the
    "dip" task family). Drawing batches never dip unless --dr.ink.dips is
    on; this is how dipping gets into a mix without every drawing opening
    at the palette."""
    supply: str = "wet"
    """Which palette load the run sees: "wet" (every right-arm cap full of
    --supply-ink; the default, because a simulator is not the bench and a
    batch should not be refused because nobody poured this morning), "bench"
    (config/palette_load.yaml as it is right now), or "dry" (every cap
    empty — a rehearsal tool still dips; a real needle is refused)."""
    supply_ink: str = "nighthawk_black"
    """The ink a --supply wet rack is filled with (config/inks.yaml id)."""
    dip_task_name: str = "dip {tool} into the {ink} ink cap."
    """Prompt for --task dip episodes; {tool} is the fitted tool's
    prompt_phrase and {ink} the chosen cap's ink ("empty" for a dry cap)."""
    min_reachable_frac: float = 0.15
    """Refuse to generate when less of a shaped substrate than this can be
    worked with the fitted tool held normal to it. Not a quality bar — a floor
    below which every scene in the run would be crowded into the same corner of
    the canvas, which is a dataset about one corner."""
    erase_passes: tuple[int, int] = (1, 12)
    """How many times an erase episode may retrace its scene. No longer
    sampled: the count is chosen per env so the episode lasts erase_seconds,
    and this is the range it is allowed to land in. A pass clears a fraction of
    what is under the beam, so a spread of counts is also what puts both
    partly-faded and nearly-clean sheets in the dataset."""
    erase_seconds: tuple[float, float] = (28.0, 60.0)
    """How long an erase episode should last. Measured from the operator's own
    laser-on-skin recordings (2026-08-26): five episodes spanning 28-60 s with
    a 39 s median. Sampling a scene and hoping the passes added up left the sim
    at a 14 s median, which no amount of domain randomisation would excuse --
    episode length is not a nuisance variable, it is what the demonstration
    looks like."""
    maze_horizon: int = 420
    """Horizon for the squiggle batches inside --task mix (language batches
    use --horizon as their cap)."""
    task_name: str = "draw a {size_mm}mm {shape} {tool} on the paper pad"
    """Per-episode task string for --task shapes: the frame of the real fm2
    recording ("draw a 6mm square using pen tip on the grid lines of the paper
    pad", no trailing period), sizes stated in mm from the sampled motif and
    the tool slot filled from the fitted datasheet. {shape} and {size_mm} are
    both available to overrides; unused slots are fine."""
    maze_task_name: str = "draw a continuous squiggle {tool} on the grid lines of the paper pad."
    """Task string for --task maze — the exact phrase of the real
    squiggle-grid-draw recordings, with the tool slot filled from the fitted
    tool's datasheet so a swap moves sim and real prompts together."""
    draw_clearance: float = interaction.WORKING_OFFSET_M
    """Resolved working-point offset from the surface, m. Contact-v1 is zero;
    retained as a CLI field so historical run_meta remains directly comparable."""
    require_qualified_geometry: bool = False
    """Opt into a strict qualification check. Offline simulation normally
    continues with nominal or synthetic tool geometry and records a warning;
    this flag is for a caller specifically requesting calibration evidence."""
    texture_refresh_steps: int = 3
    """Control steps between sheet-texture uploads. The pigment field itself
    updates every step, so recorded coverage is exact either way; this only
    sets how often the RENDER catches up. Uploads cost ~1.5 ms per env per
    call almost regardless of size, so 1 is measurably slower for a visual
    difference under a line width (see the plan doc's Phase 0)."""
    save_field_snapshots: bool = False
    """Write each episode's final pigment field as a greyscale PNG under
    meta/fields. Exact ground truth for what ended up on the sheet — what a
    scorer wants — but one extra image per episode, so it stays opt-in."""
    save_privileged_labels: bool = False
    """Write time-aligned synthetic tool/contact/intent/deposition state under
    meta/privileged, outside LeRobot observation/action features. Requires a
    texture refresh every step so RGB and deposition labels describe the same
    simulation step."""
    episode_variant: str = "blank-start"
    """Synthetic episode outcome. Non-nominal variants require --scenario and
    remain explicitly labeled failures even if they deposit some pigment."""
    occlusion_fraction: float = 0.15
    """Image-area fraction one declared occluder hides in every camera of an
    ``occluded`` episode. Comes from the pilot appearance; recorded exactly."""
    judge: bool = False
    """Score deposition episodes against their intended paths and record the
    result as ``drawing_score`` in run_meta. Expert generation establishes the
    simulator ceiling used to interpret later policy scores."""
    depth: bool = True
    """Record each wrist camera's depth as an <cam>_depth feature (mm, like
    the real D405s with use_depth)."""
    clamp_floor: bool = True
    """Clamp commanded steps so the needle never goes below the pad surface.
    Off reproduces the pre-clamp data for A/B auditing — do not train on it."""
    encoder_procs: int = 0
    """Video-encoder processes. 0 (default) = one encode thread per stream,
    in-process. > 0 = a worker-process pool; output bytes are identical, but
    measured on the sim node it is neutral at 16 envs and SLOWER above (streams
    serialize inside workers), so it stays opt-in for hosts where the
    tradeoff differs."""
    reconfigure_each_batch: bool = True
    """Rebuild scenes every batch so lighting, sheets, tints, floors,
    environment maps and camera jitter redraw per batch instead of once per
    env slot."""
    distribution: str | None = None
    """Which named recipe produced this dataset (tatbot_sim.distributions),
    set by the factory launcher. None means the run was assembled by hand from
    flags, which is still allowed — it just cannot claim to be one of the three
    distributions a training mix selects on."""
    keep_idle: bool = False
    """Keep episodes that never moved any pigment. By default an idle episode
    is dropped from the dataset and recorded under run_meta.dropped_episodes,
    because its prompt says "draw" while its sheet says nothing happened — a
    mislabelled demonstration, not data (2026-09-03)."""
    scenario: str | None = None
    """Compiled Inkmap tattoo-scenario JSON. When set, the full posed body,
    support fixture, collision proxies, exact SVG and surface trace replace
    random pad-scene sampling; normally supplied by `body-tattoo`."""
    dr: DRConfig = field(default_factory=DRConfig)
    """Every randomization range, one tree — see tatbot_sim.config."""


def _primitive_schedule(plan, surface, env_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest typed stroke for each intended contact step; -1 off-contact."""

    count = plan.targets.shape[1]
    primitive = np.full(count, -1, dtype=np.int32)
    layer = np.full(count, -1, dtype=np.int32)
    if plan.stroke_metadata is None:
        return primitive, layer
    intended_targets = plan.intended_targets if plan.intended_targets is not None else plan.targets
    intended_distance = np.sum(
        (intended_targets[env_index] - plan.surface_points[env_index])
        * plan.surface_normals[env_index],
        axis=1,
    )
    contact = intended_distance <= interaction.CONTACT_ABOVE_TOLERANCE_M
    best = np.full(count, np.inf)
    for stroke_index, path in enumerate(plan.paths[env_index]):
        points = np.asarray(path, dtype=np.float64)
        if len(points) > 256:
            points = points[np.linspace(0, len(points) - 1, 256).astype(int)]
        world, _ = surface.frame_np(env_index, points)
        distance = np.min(
            np.linalg.norm(plan.surface_points[env_index, :, None, :] - world[None, :, :], axis=2),
            axis=1,
        )
        choose = contact & (distance < best)
        primitive[choose] = stroke_index
        layer[choose] = int(plan.stroke_metadata[env_index][stroke_index]["layer_index"])
        best[choose] = distance[choose]
    return primitive, layer


def _timeline_intended_field(base_env, plan):
    if base_env.ink_field is None:
        raise RuntimeError("ink_field is unavailable for privileged labels")
    target = getattr(getattr(base_env, "_scenario_geometry", None), "target", None)
    if target is not None:
        resized = cv2.resize(
            target.coverage,
            (base_env.ink_field.cols, base_env.ink_field.rows),
            interpolation=cv2.INTER_AREA,
        )
        return torch.as_tensor(
            np.broadcast_to(resized, base_env.ink_field.field.shape),
            dtype=base_env.ink_field.field.dtype,
            device=base_env.ink_field.device,
        ).clone()
    return judging.intended_field_like(
        base_env.ink_field,
        base_env.surface,
        [judging.strokes_from_plan_paths(path) for path in plan.paths],
        base_env.ink_opacity,
    )


def _occlude_observations(frames: dict, fraction: float = 0.15) -> float:
    """Apply one declared rectangular camera occluder and return its exact area."""

    actuals: list[float] = []
    for camera, values in frames.items():
        height, width = values.shape[1:3]
        box_height = max(1, int(round(height * np.sqrt(fraction))))
        box_width = max(1, int(round(width * np.sqrt(fraction))))
        y0 = (height - box_height) // 2
        x0 = (width - box_width) // 2
        values = values.copy()
        values[:, y0 : y0 + box_height, x0 : x0 + box_width] = np.asarray(
            [20, 23, 28], dtype=values.dtype
        )
        frames[camera] = values
        actuals.append(box_height * box_width / (height * width))
    if not actuals:
        return 0.0
    if not np.allclose(actuals, actuals[0], rtol=0, atol=1e-9):
        raise RuntimeError("camera occluders do not share one labelable area fraction")
    return actuals[0]


def _configure_motion(args, config):
    if args.production_draw_config is None:
        return config, None
    import pen_path

    from tatbot_sim.production_plan import read_config
    from tatbot_sim.resolved import Timing

    settings = read_config(args.production_draw_config, config)
    if args.task not in ('artwork', 'spiral', 'maze', 'shapes') or args.episode_variant != 'blank-start':
        raise SystemExit('production-flat requires ordinary drawing intent without outcome perturbations')
    if any(args.dr.latency.obs_delay_steps):
        raise SystemExit('production-flat currently requires zero observation-delay steps')
    control_hz = round(1 / pen_path.C.period_s)
    return dataclasses.replace(config, timing=Timing(physics_hz=control_hz * 3, control_hz=control_hz)), settings


def _capture_profile(cameras, config, production):
    from tatbot_contracts.timing import SampleCadence

    sample_hz = int(cameras[0].fps)
    if sample_hz != cameras[0].fps:
        raise ValueError('dataset sampling currently requires an integer camera frequency')
    size = common_stream_profile(cameras, control_freq=sample_hz)
    cadence = SampleCadence(config.timing.control_hz, sample_hz)
    if not production and sample_hz != config.timing.control_hz:
        raise SystemExit('synthetic expert generation requires matching camera and control cadence')
    return sample_hz, size, cadence


def _validate_calibration_draw(args, geometry):
    delta_norm = float(np.linalg.norm(geometry.calibration_delta_m))
    if not args.tool_calibration_jitter and delta_norm > 1e-12:
        raise SystemExit(
            "a process-scoped tip calibration delta is set, but "
            "--tool-calibration-jitter is disabled; run through the named factory")
    uncertainty_limit = ((geometry.contact_uncertainty_m or 0.0)
                         * args.tool_calibration_scale)
    if args.tool_calibration_jitter and delta_norm > uncertainty_limit + 1e-12:
        raise SystemExit(
            f"tip calibration delta is {delta_norm * 1000:.3f} mm, outside the "
            f"recorded {uncertainty_limit * 1000:.3f} mm scaled uncertainty")


def _carriage_metadata(args, expert, production_cfg):
    if production_cfg is None:
        return {'carriage_ik': bool(args.carriage_ik),
                'carriage_policy': expert.carriage_policy.as_metadata() if expert.carriage_policy else None}
    import pen_path

    enabled = bool(production_cfg.get('path', {}).get('carriage_ik', False))
    return {'carriage_ik': enabled,
            'carriage_policy': {'planner': 'production-cartesian-cpp', 'enabled': enabled,
                                'constants_sha': pen_path.SHA, 'source': 'native-joint-reference'}}


def _validate_generation_args(args):
    if not np.isfinite(args.tool_calibration_scale) or args.tool_calibration_scale < 0:
        raise SystemExit("--tool-calibration-scale must be finite and non-negative")
    if args.save_privileged_labels and args.texture_refresh_steps != 1:
        raise SystemExit(
            "--save-privileged-labels requires --texture-refresh-steps 1 so RGB and "
            "deposition labels describe the same simulation step"
        )
    if args.episode_variant not in EPISODE_VARIANT_OUTCOMES:
        raise SystemExit(
            f"--episode-variant must be one of {', '.join(EPISODE_VARIANT_OUTCOMES)}"
        )
    if args.episode_variant != "blank-start" and not args.scenario:
        raise SystemExit("non-nominal episode variants require a compiled --scenario")
    if args.episode_variant == "occluded" and not 0.0 < args.occlusion_fraction < 1.0:
        raise SystemExit("--occlusion-fraction must lie strictly inside (0, 1) for an occluded episode")


def _select_batch_task(args, rng):
    if args.scenario:
        task = "body-tattoo"
    elif args.task == "mix":
        roll = rng.random()
        if roll < args.erase_frac:
            task = "erase"
        elif roll < args.erase_frac + args.squiggle_frac:
            task = "maze"
        elif roll < args.erase_frac + args.squiggle_frac + args.dip_frac:
            task = "dip"
        else:
            task = "language"
    else:
        task = args.task
    horizon = args.maze_horizon if task in ("maze", "shapes", "dip") else args.horizon
    return task, horizon


def _reachable_batch(args, expert, robot, idx_ik, base_env):
    q_now = robot.get_qpos()[:, idx_ik]
    slack = args.dr.pen_lean.max_off_base_rad
    masks = reachable_canvas_masks(
        expert, q_now, base_env.surface, args.draw_clearance, args.num_envs,
        max_off_base_rad=slack,
    )
    ceiling = reachable_height_ceiling(
        expert, q_now, base_env.surface, args.num_envs, max_off_base_rad=slack,
    )
    if masks is not None:
        frac = float(np.mean([m.fraction for m in masks]))
        if frac < args.min_reachable_frac:
            raise SystemExit(
                f"only {frac:.0%} of the {base_env.substrate.name} is reachable with "
                f"{base_env.tool.tool_id} held normal to it (need "
                f"{args.min_reachable_frac:.0%}). Lower the substrate, move it closer, "
                "or fit a tool with less protrusion — a run this constrained would "
                "crowd every scene into the same corner."
            )
    return masks, ceiling


@dataclass
class _BatchCapture:
    coverage_start: np.ndarray
    recorded_steps: np.ndarray
    capture_ticks: list[int]
    contact_simulated: bool
    effort_metadata: dict | None
    observation_occlusion: float
    temporal: TemporalRecorder | None
    obs_delay: np.ndarray


def _check_frame_zero(action, qpos, before_qpos, b):
    """Reject a reset that puts recorded state ahead of its first action."""
    action_np = action[:b].cpu().numpy()
    lead = np.abs(action_np - qpos)
    recorded_lead = float(lead.max())
    if recorded_lead > EPISODE_START_MAX_LEAD_RAD:
        env_index, joint_index = np.unravel_index(int(np.argmax(lead)), lead.shape)
        if before_qpos is None:
            raise AssertionError("frame-zero pre-step state was not captured")
        before_lead = torch.max(
            torch.abs(action[:b] - before_qpos)
        ).item()
        raise RuntimeError(
            "episode-start action/state lead exceeds the data-integrity "
            f"bound: {recorded_lead:.6f} rad > "
            f"{EPISODE_START_MAX_LEAD_RAD:.3f} rad. A reconfiguring reset "
            "may have replaced the articulation used for pose placement; "
            f"env={env_index}, joint={joint_index}, "
            f"before_step={before_lead:.6f} rad. Do not train on this run."
        )


def _append_timeline(args, temporal, timeline_target, primitive_schedules,
                     layer_schedules, base_env, plan, t, b, observation_occlusion):
    """Record time-aligned intent and deposition after a captured step."""
    if timeline_target is None or primitive_schedules is None or layer_schedules is None:
        raise AssertionError("privileged timeline was not initialized")
    tcp_pose = base_env.agent.tcp.pose.raw_pose[:b]
    tcp = tcp_pose[:, :3]
    _, contact_distance, contact_incidence = base_env.surface.project(tcp)
    touching = base_env._interaction_mask(contact_distance)
    draw_step = t - plan.n_app
    valid_step = 0 <= draw_step < plan.targets.shape[1]
    if valid_step and plan.dip_mask is not None:
        touching = touching & ~torch.as_tensor(
            plan.dip_mask[:b, draw_step], device=touching.device
        )
    target_world = np.zeros((b, 3), dtype=np.float32)
    surface_point = np.zeros((b, 3), dtype=np.float32)
    surface_normal = np.zeros((b, 3), dtype=np.float32)
    primitive = np.full(b, -1, dtype=np.int32)
    layer = np.full(b, -1, dtype=np.int32)
    target_valid = np.zeros(b, dtype=bool)
    if valid_step:
        intended_targets = plan.intended_targets if plan.intended_targets is not None else plan.targets
        target_world = intended_targets[:b, draw_step].astype(np.float32)
        surface_point = plan.surface_points[:b, draw_step].astype(np.float32)
        surface_normal = plan.surface_normals[:b, draw_step].astype(np.float32)
        intended_distance = np.sum(
            (target_world - surface_point) * surface_normal,
            axis=1,
        )
        target_valid = intended_distance <= interaction.CONTACT_ABOVE_TOLERANCE_M
        primitive = np.where(
            target_valid,
            primitive_schedules[:b, draw_step],
            -1,
        ).astype(np.int32)
        layer = np.where(
            target_valid,
            layer_schedules[:b, draw_step],
            -1,
        ).astype(np.int32)
    deposited = base_env.ink_field.field[:b]
    denominator = timeline_target[:b].sum(dim=(1, 2)).clamp_min(1e-12)
    remaining = torch.clamp(timeline_target[:b] - deposited, min=0).sum(
        dim=(1, 2)
    ) / denominator
    remaining = remaining.clamp(0, 1)  # bound reduction roundoff at an untouched target
    temporal.append(
        tool_pose_world=tcp_pose.detach().cpu().numpy(),
        contact_distance_m=contact_distance.detach().cpu().numpy(),
        contact_incidence=contact_incidence.detach().cpu().numpy(),
        pen_down=touching.detach().cpu().numpy(),
        target_world=target_world,
        target_valid=target_valid,
        surface_point_world=surface_point,
        surface_normal_world=surface_normal,
        primitive_index=primitive,
        layer_index=layer,
        progress=np.minimum(
            1.0,
            np.full(b, (t + 1), dtype=np.float32)
            / (plan.intended_lengths[:b] if plan.intended_lengths is not None else plan.lengths[:b]),
        ),
        deposited_coverage=deposited.mean(dim=(1, 2)).detach().cpu().numpy(),
        remaining_target_fraction=remaining.detach().cpu().numpy(),
        texture_synchronized=np.full(b, args.texture_refresh_steps == 1),
        stencil_visible_fraction=base_env.stencil_visible_fraction()[:b],
        observation_occlusion_fraction=np.full(
            b, observation_occlusion, dtype=np.float32
        ),
    )


def _delayed_observation(state, frames, depth, obs_hist, obs_delay, max_delay, b, cameras):
    """Pair current actions with the configured older camera observation."""
    if max_delay == 0:
        d_state, d_frames, d_depth = state, frames, depth
    else:
        # zero-copy pairing: hand the writer per-env VIEWS into the
        # history buffers instead of rebuilding batch arrays. The
        # writer only indexes frames[cam][i], history arrays are
        # never mutated, and the views keep them alive — while the
        # copying version moved ~80 MB per step and was the single
        # largest main-loop cost on the 4-core node (profiled).
        srcs = [obs_hist[max(0, len(obs_hist) - 1 - int(obs_delay[i]))]
                for i in range(b)]
        d_state = np.stack([srcs[i]["state"][i] for i in range(b)])
        d_frames = {c: [srcs[i]["frames"][c][i] for i in range(b)]
                    for c in cameras}
        d_depth = None if depth is None else {
            c: [srcs[i]["depth"][c][i] for i in range(b)] for c in cameras}
    return d_state, d_frames, d_depth


def _capture_recorders(args, base_env, plan, b, sample_hz):
    """Initialize the privileged recorders against the plan."""
    temporal = TemporalRecorder(b) if args.save_privileged_labels else None
    timeline_target = _timeline_intended_field(base_env, plan) if temporal is not None else None
    primitive_schedules = None
    layer_schedules = None
    if temporal is not None:
        schedules = [_primitive_schedule(plan, base_env.surface, i) for i in range(b)]
        primitive_schedules = np.stack([item[0] for item in schedules])
        layer_schedules = np.stack([item[1] for item in schedules])
    return temporal, timeline_target, primitive_schedules, layer_schedules


def _depth_frames(args, measurement, cameras, b):
    """Quantize captured depth on the device before copying it to the writer."""
    depth = None
    if args.depth:
        depth = {}
        for cam in cameras:
            d = measurement.depth_mm[cam]
            # quantize to 12-bit codes ON the GPU: leaves the encode
            # workers nothing but the codec call and halves the
            # device->host transfer (int16 fits 4095)
            depth[cam] = (
                quantize_depth_codes(d[:b].to(torch.float32))
                .to(torch.int16).cpu().numpy().astype(np.uint16)
            )
    return depth


def _write_eval_artifacts(out_dir, base_env, intended_fields, kept):
    artifacts: list[dict[str, str] | None] = [None] * len(kept)
    if intended_fields is None:
        return artifacts
    eval_dir = Path(out_dir) / "meta" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    drawn_np = base_env.ink_field.field.cpu().numpy()
    intended_np = intended_fields.cpu().numpy()
    for i, episode in enumerate(kept):
        if episode is None:
            continue
        intended_path = eval_dir / f"episode_{episode:06d}_intended.png"
        drawn_path = eval_dir / f"episode_{episode:06d}_drawn.png"
        overlay_path = eval_dir / f"episode_{episode:06d}_overlay.png"
        cv2.imwrite(str(intended_path), (255 * intended_np[i]).astype(np.uint8))
        cv2.imwrite(str(drawn_path), (255 * drawn_np[i]).astype(np.uint8))
        overlay = np.zeros((*drawn_np[i].shape, 3), dtype=np.uint8)
        overlay[..., 1] = (255 * intended_np[i]).astype(np.uint8)
        overlay[..., 2] = (255 * drawn_np[i]).astype(np.uint8)
        cv2.imwrite(str(overlay_path), overlay)
        artifacts[i] = {
            "intended": str(intended_path.relative_to(out_dir)),
            "drawn": str(drawn_path.relative_to(out_dir)),
            "overlay": str(overlay_path.relative_to(out_dir)),
        }
    return artifacts


def _episode_artwork(plan, base_env, i):
    # Erase episodes keep the program the sheet opened with: it is the
    # target against which removal is scored.
    extra = {}
    if plan.kinds[i] in ("language", "erase", "artwork", "spiral"):
        extra = {"program": plan.programs[i], "strokes_canvas_m": plan.paths[i]}
    elif plan.kinds[i] == "dip":
        extra = {"program": plan.programs[i]}
    if plan.kinds[i] in ("artwork", "spiral", "erase") and plan.programs[i]:
        prog = plan.programs[i]
        extra["artwork"] = {key: prog.get(key) for key in ("design_id", "source_sha256", "family", "split")}
    elif base_env.body_scenario is not None:
        from tatbot_sim.inkmap.collection import collection_entries
        design = base_env.body_scenario["design"]
        matched = next((e for e in collection_entries() if e["id"] == design["id"] and e["sha256"] == design["sha256"]), {})
        extra["artwork"] = {"design_id": design["id"], "source_sha256": design["sha256"],
            "family": matched.get("family", "calibration" if design["id"] == "spiral-v1" else None),
            "split": matched.get("split", "calibration" if design["id"] == "spiral-v1" else None)}
    return extra


def _capture_batch(args, runtime, base_env, plan, writer, expert, robot, idx7,
                   cameras, cadence, sample_hz, carriage_channel, rng, b):
    """Capture one planned batch, keeping action and observation timing together."""
    writer.open_batch(b, tasks=plan.tasks[:b])
    # non-zero once removal episodes open on a pre-inked sheet
    if base_env.ink_field is None:
        raise RuntimeError("ink_field is None after env reset")
    coverage_start = runtime.coverage_start.cpu().numpy()
    temporal, timeline_target, primitive_schedules, layer_schedules = (
        _capture_recorders(args, base_env, plan, b, sample_hz))

    # Deployment-timing DR: the async stack hands the policy stale
    # observations, so per env we pair obs from t-k with action_t. The
    # ring buffer holds the last max_delay+1 observation sets; episodes
    # open holding their first observation (real sessions start stale
    # too). Actions are never delayed — they are the labels.
    lat = args.dr.latency
    obs_delay = rng.integers(lat.obs_delay_steps[0],
                             lat.obs_delay_steps[1] + 1, args.num_envs)
    max_delay = int(obs_delay[:b].max())
    obs_hist: list[dict] = []

    # each episode records only until its own drawing ends (plus a short
    # settle tail), and the batch runs only as long as its longest
    # written episode. Padding every episode to the batch's longest was
    # 38% of language-batch steps spent on the arm holding motionless —
    # frames not worth generating (operator confirmed, 2026-08-25);
    # episodes are variable-length, like real recordings.
    recorded_steps = np.zeros(b, dtype=np.int64)
    capture_ticks = []
    contact_simulated = bool(base_env.surface_has_contact_collision)
    effort_metadata = None
    for t in range(runtime.horizon):
        if runtime.done:
            break
        before_qpos = robot.get_qpos()[:b, idx7] if t == 0 else None
        capture = cadence.due(t + 1)
        action, measurement, _ = runtime.step(capture=capture)
        qpos = measurement.qpos[:b].cpu().numpy()
        if t == 0:
            _check_frame_zero(action, qpos, before_qpos, b)
        if not capture:
            continue
        capture_ticks.append(t + 1)
        contact_simulated = measurement.contact.simulated
        effort_metadata = measurement.effort_metadata()
        state = measurement.state[:b].cpu().numpy()
        frames = {cam: measurement.rgb[cam][:b].cpu().numpy() for cam in cameras}
        observation_occlusion = 0.0
        if args.episode_variant == "occluded":
            observation_occlusion = _occlude_observations(frames, args.occlusion_fraction)
        if temporal is not None:
            _append_timeline(args, temporal, timeline_target, primitive_schedules,
                             layer_schedules, base_env, plan, t, b, observation_occlusion)
        depth = _depth_frames(args, measurement, cameras, b)
        obs_hist.append({"state": state, "frames": frames, "depth": depth})
        if len(obs_hist) > max_delay + 1:
            obs_hist.pop(0)

        d_state, d_frames, d_depth = _delayed_observation(
            state, frames, depth, obs_hist, obs_delay, max_delay, b, cameras)
        writer.add_steps(action[:b].cpu().numpy(), d_state, d_frames, d_depth,
                         active=[t < int(plan.lengths[i]) for i in range(b)])
        recorded_steps += t < plan.lengths[:b]
    return _BatchCapture(coverage_start, recorded_steps, capture_ticks,
                         contact_simulated, effort_metadata, observation_occlusion,
                         temporal, obs_delay)


def main(args: Args, *, config=None):
    from tatbot_sim.agent import TatbotWXAI
    from tatbot_sim.env import TatbotDrawEnv
    from tatbot_sim.resolved import Timing, resolve

    config = config or resolve(tool_id=args.tool_id, substrate_name=args.substrate,
        sensor_profile=args.sensor_profile, seed=args.seed, dr=args.dr,
        scenario_path=args.scenario, supply=(args.supply, args.supply_ink))
    config, production_cfg = _configure_motion(args, config)
    source_start = source_state()
    rng = np.random.default_rng(args.seed)
    _validate_generation_args(args)
    # Every task family this run can sample, checked against the fitted tool
    # and its substrate BEFORE the env is built. An erase episode with a pen
    # fitted would DRAW over its own target and write a dataset whose prompts
    # say "remove"; a language episode with the laser fitted would strip a
    # blank sheet while its prompts say "draw". Both look fine until someone
    # trains on them, so both are refused here rather than surfacing as the
    # engaged-episode warning after the run finishes.
    tool = config.tool
    substrate = config.substrate
    registry = tools.registry()
    workspace = config.workspace
    geometry = config.geometry
    _validate_calibration_draw(args, geometry)
    contact_eligible = geometry.contact_status == "pivot-calibrated"
    geometry_warnings = tools.geometry_warnings(tool, geometry)
    if tool.contact and not contact_eligible and args.require_qualified_geometry:
        raise SystemExit(
            f"{tool.tool_id!r} contact geometry is {geometry.contact_status!r}: "
            f"{geometry.contact_qualification_error or 'no pivot-calibrated TCP'}. "
            "--require-qualified-geometry requested a quality-gated fixed-point touch-off.")
    for warning in geometry_warnings:
        print(f"[generate] WARNING: {warning}", file=sys.stderr, flush=True)
    if tool.contact and abs(args.draw_clearance - interaction.WORKING_OFFSET_M) > 1e-9:
        raise SystemExit(
            f"{tool.tool_id!r} is a contact tool: its resolved working point must target "
            f"the surface ({interaction.WORKING_OFFSET_M:.4f} m), not "
            f"--draw-clearance {args.draw_clearance:.4f} m. Use approach/travel height "
            "for clearance; changing the drawing target recreates air-drawing data.")
    for task in tasks.active_tasks(args.task, args.erase_frac, args.squiggle_frac, args.dip_frac):
        try:
            tasks.validate_task(task, tool, substrate)
            tasks.validate_supply(task, tool, config.palette_load)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    # A dry-tool variant is intentionally a labeled failure. Validate the
    # nominal task/supply contract first, then empty the synthetic caps before
    # scene construction; ordinary generation may never bypass this check.
    if args.episode_variant == "dry-tool":
        config = config.with_supply('dry')
    design = design_scene.from_args(args, substrate, config=config)
    if args.task in ("artwork", "spiral", "erase") and not args.scenario:
        if not 0 < args.artwork_width_mm <= 2:
            raise SystemExit("artwork width must be in (0, 2] mm")
        args.dr.ink.radius_m = (design_scene.width_mm_for(design, args) / 2000,) * 2
    config = config.with_dr(args.dr)
    env = TatbotDrawEnv(
        config=config,
        num_envs=args.num_envs,
        obs_mode="rgbd" if args.depth else "rgb",
        control_mode="pd_joint_pos",
        sim_backend=args.sim_backend,
        texture_refresh_steps=args.texture_refresh_steps,
        reconfiguration_freq=1 if args.reconfigure_each_batch else 0,
        ink_pixels_per_m=args.artwork_pixels_per_m if args.task in ("artwork", "spiral", "erase") and not args.scenario else None,
    )
    base_env: TatbotDrawEnv = env.unwrapped
    camera_descriptions = base_env.agent.camera_descriptions
    cameras = tuple(camera.role for camera in camera_descriptions)
    sample_hz, image_size, cadence = _capture_profile(camera_descriptions, config, production_cfg is not None)
    contact_collision = bool(
        base_env.surface_has_contact_collision
        and base_env.body_scenario is None
    )
    interaction_model = interaction.model_for(collision=contact_collision)
    device = base_env.device
    expert = StrokeExpert(args.num_envs, device, config=config, noise=args.dr.noise, seed=config.seed_for("noise"),
                          carriage_ik=args.carriage_ik, control_hz=float(base_env.control_freq))
    carriage_channel = TatbotWXAI.joint_names.index(CARRIAGE_JOINT)
    world = ManiSkillWorld(env, config, expert)
    runtime = Episode(world, ObservationBuilder(config, args.num_envs, device))
    robot, idx7, idx_ik = world.robot, world.idx7, world.idx_ik

    # Can the fitted tool actually be held perpendicular over the sampled pad
    # heights? A tool that cannot does not fail — IK returns its best effort
    # and every episode marks tens of millimetres from where its own labels
    # say, which no downstream check would catch.
    # This coarse check covers the profile crest. Cylinders curve downward
    # from it; their changing normal and full drawable area are checked by the
    # per-batch reachable-canvas audit below.
    #
    # A compiled body scenario HAS NO PAD. ``_load_body_scene`` returns before
    # any sheet actor is built and the patch pose comes from the scenario
    # anchor alone, so ``dr.pad`` then describes geometry that is not in the
    # scene: probing it refused reachable body scenes over a phantom surface
    # (2026-09-08, a forearm scenario refused at 80.8 mm against a pad top the
    # scene never instantiated). The body path carries its own, stricter gate —
    # FK of the solved joint reference against every compiled target, below —
    # so skipping this one loses no coverage.
    if base_env.body_scenario is None:
        pad_z_range = base_env.dr.pad.z_range
        if pad_z_range is None:
            raise RuntimeError("pad z_range is not set")
        reach_z = pad_z_range
        worst_reach, worst_z = worst_reach_residual(
            expert, robot.get_qpos()[:, idx_ik], base_env.pad_center,
            reach_z, args.draw_clearance,
        )
        if worst_reach > REACH_TOLERANCE_M:
            ceiling = highest_reachable_z(
                expert, robot.get_qpos()[:, idx_ik], base_env.pad_center,
                reach_z, args.draw_clearance, REACH_TOLERANCE_M,
            )
            top = ceiling
            fix = (
                f"pass --dr.pad.z-range {pad_z_range[0]:.3f} {top:.3f}"
                if top is not None and top > pad_z_range[0] else
                "move the surface closer or revisit the tool's modelled working point"
            )
            raise SystemExit(
                f"{tool.tool_id!r} cannot reach the sampled drawing "
                f"envelope: IK is off by {worst_reach * 1000:.1f} mm with the pad top at "
                f"{worst_z:.3f} m (tolerance {REACH_TOLERANCE_M * 1000:.0f} mm). "
                f"To generate anyway, {fix} — or revisit the tool's grip point."
            )
        print(f"[generate] reach check: worst IK residual {worst_reach * 1000:.2f} mm "
              f"over pad z {pad_z_range}")
    else:
        print("[generate] reach check: skipped — a compiled body scenario has no pad; "
              "the exact per-target reference gate applies instead")

    writer = LeRobotWriter(args.out_dir, cameras=cameras, depth=args.depth,
                           fps=sample_hz, image_size=image_size,
                           task_name=args.task_name,
                           encoder_procs=max(0, args.encoder_procs))
    episode_log = []

    done = 0
    clamp_fracs: list[float] = []
    # A batch whose scene will not fit the horizon is unlucky, not fatal: it is
    # skipped and redrawn rather than taking the run down. Bounded, because a
    # recipe whose scenes NEVER fit would otherwise spin forever, and counted,
    # because silently skipping batches narrows the distribution while the
    # episode count still looks full.
    skipped: list[dict] = []
    # Episodes that ran but were not written: an unreachable joint reference
    # or a sheet nothing touched. Always recorded, so a dataset's episode
    # count is an honest count of demonstrations and not of attempts.
    dropped: list[dict] = []
    consecutive_skips = 0
    t_start = time.time()
    while done < args.num_episodes:
        b = min(args.num_envs, args.num_episodes - done)
        # env batch size is fixed; surplus envs in the last batch are discarded
        runtime.reset(seed=args.seed + done + 977 * len(skipped))
        robot, idx7, idx_ik = world.robot, world.idx7, world.idx_ik
        top_centers, rots = base_env.canvas_frame_np
        normals = rots[:, :, 2]

        task_b, horizon_b = _select_batch_task(args, rng)
        # Where can the fitted tool actually be held normal to THIS batch's
        # surface? On a cylinder the flanks ask the wrist for a lean it cannot
        # make, and a stroke laid across them is a label the arm quietly
        # misses — so strokes are placed only on ground the IK can reach.
        # Flat substrates need the mask too since the 2026-08-31 measured
        # ballpoint tip: its workable region is a diagonal band (low, near,
        # biased +y), and no rectangular pad placement sits wholly inside it —
        # stroke placement has to dodge the far/-y corner, which is what the
        # mask is for. On a healthy nominal tool it is all-true and costs one
        # batched solve.
        # One reach envelope drives both scene placement and travel height.
        # How high the tool can still be held. Travel and the opening
        # descent are the tallest poses an episode asks for, and over a
        # curved profile they are the ones the arm cannot make.
        masks, ceiling = _reachable_batch(args, expert, robot, idx_ik, base_env)

        try:
            if args.scenario:
                if base_env.body_scenario is None:
                    raise RuntimeError("body_scenario is None when scenario is set")
                plan = plan_tattoo_scenario(
                    rng, base_env.body_scenario, base_env.surface, config=config,
                    horizon=horizon_b, num_envs=args.num_envs,
                    dr=args.dr, draw_clearance=args.draw_clearance,
                    tool_ceiling=ceiling,
                )
            else:
                plan = plan_batch(
                    rng, base_env.pad_sheets, base_env.surface, config=config,
                    task=task_b, horizon=horizon_b, num_envs=args.num_envs,
                    dr=args.dr, draw_clearance=args.draw_clearance,
                    task_name=args.task_name, maze_task_name=args.maze_task_name,
                    erase_passes=args.erase_passes,
                    erase_seconds=args.erase_seconds,
                    reachable=masks,
                    tool_ceiling=ceiling,
                    cap_rims=base_env.cap_rims_np(),
                    artwork_split=args.artwork_split, artwork_width_mm=args.artwork_width_mm,
                    artwork_sampler=design_scene.sampler(design),
                    dip_task_name=args.dip_task_name,
                )
        except SceneTooLongError as e:
            skipped.append({"after_episodes": done, "task": task_b,
                            "steps_needed": e.needed, "steps_available": e.horizon})
            consecutive_skips += 1
            print(f"[generate] skipped a {task_b} batch: {e} "
                  f"({len(skipped)} skipped so far)", flush=True)
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                print(f"[generate] WARNING: {MAX_CONSECUTIVE_SKIPS} batches in a row "
                      f"would not fit — stopping at {done} episodes. Raise --horizon "
                      f"or narrow the scene style; this recipe cannot draw what it "
                      f"samples.", flush=True)
                break
            continue
        try:
            plan = apply_episode_variant(plan, args.episode_variant, tool_ceiling=ceiling)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        consecutive_skips = 0
        if production_cfg is not None:
            from tatbot_sim.production_plan import from_intent

            references_dir = Path(args.out_dir) / 'meta' / 'references' / f'batch-{done}-{len(skipped)}'
            plan = from_intent(plan, world, production_cfg, references_dir,
                               max_ticks=int(horizon_b * base_env.control_freq / Timing().control_hz),
                               artifact_root=Path(args.out_dir))
        stencil = _timeline_intended_field(base_env, plan) if args.episode_variant == "stencil-start" else None
        runtime.install(plan, count=b, clamp_floor=args.clamp_floor, stencil=stencil)
        quality = runtime.quality
        worst_mm, contact_mm = quality.residual_mm, quality.contact_mm
        dip_mm, dip_deg, dip_missed = quality.dip_lateral_mm, quality.dip_axis_deg, quality.dip_missed
        unreachable = {
            i for i in range(b)
            if worst_mm[i] > REACH_TOLERANCE_M * 1000
            or contact_mm[i] > CONTACT_REFERENCE_TOLERANCE_M * 1000
            or dip_missed[i]
        }
        if unreachable:
            detail = ", ".join(
                f"env {i}: residual {worst_mm[i]:.2f} mm, contact error {contact_mm[i]:.2f} mm"
                + (f", dip {dip_mm[i]:.2f} mm off the cap axis at {dip_deg[i]:.1f} deg"
                   if dip_missed[i] else "")
                for i in sorted(unreachable)
            )
            if len(unreachable) == b:
                if args.scenario and args.episode_variant == "blank-start":
                    raise SystemExit(
                        "compiled body tattoo is outside the exact IK envelope: "
                        f"{detail} above the {REACH_TOLERANCE_M * 1000:.0f} mm gate. "
                        "Recompile with a different --target-world-m or --patch-yaw-rad."
                    )
                if args.scenario:
                    raise SystemExit(
                        f"the {args.episode_variant} perturbation leaves the exact IK envelope "
                        f"although the intended tattoo is reachable: {detail} above the "
                        f"{REACH_TOLERANCE_M * 1000:.0f} mm gate. The answer key is unchanged; "
                        "choose a placement with more margin for this variant."
                    )
                skipped.append({"after_episodes": done, "task": task_b,
                                "reason": "ik_reference", "detail": detail,
                                "episode_variant": args.episode_variant})
                consecutive_skips += 1
                print(f"[generate] skipped a {task_b} batch: the joint reference misses "
                      f"its targets ({detail})", flush=True)
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    print(f"[generate] WARNING: {MAX_CONSECUTIVE_SKIPS} batches in a row "
                          f"were unreachable — stopping at {done} episodes.", flush=True)
                    break
                continue
            print(f"[generate] {len(unreachable)}/{b} episodes have an unreachable joint "
                  f"reference and will be dropped ({detail})", flush=True)
        capture = _capture_batch(args, runtime, base_env, plan, writer, expert, robot, idx7,
                                 cameras, cadence, sample_hz, carriage_channel, rng, b)
        coverage_start = capture.coverage_start
        recorded_steps = capture.recorded_steps
        capture_ticks = capture.capture_ticks
        contact_simulated = capture.contact_simulated
        effort_metadata = capture.effort_metadata
        observation_occlusion = capture.observation_occlusion
        temporal = capture.temporal
        obs_delay = capture.obs_delay

        # Pigment on the sheet, start and end. Ink is a measured quantity now
        # rather than a pile of actors, so every episode carries how much was
        # laid down (or, for a removal tool, how much was cleared) without any
        # extra machinery — the scoreboard the sim-eval harness wants.
        if base_env.ink_field is None:
            raise RuntimeError("ink_field is None after env reset")
        coverage_end = base_env.ink_field.coverage().cpu().numpy()
        ink_stats = runtime.statistics()
        engaged = [
            judging.engaged(plan.kinds[i], coverage_start[i], coverage_end[i],
                     dips=int(ink_stats["dips"][i]))
            for i in range(b)
        ]
        expected_outcomes = plan.expected_outcomes or ["nominal"] * b
        keep = [
            i not in unreachable
            and (engaged[i] or args.keep_idle or expected_outcomes[i] != "nominal")
            for i in range(b)
        ]
        kept = writer.close_batch(keep=keep)
        temporal_records: list[dict | None] = [None] * b
        if temporal is not None:
            temporal_records = temporal.write(
                Path(args.out_dir) / "meta" / "privileged",
                kept=kept,
                lengths=recorded_steps,
                stroke_metadata=plan.stroke_metadata,
                scenario_sha256=(
                    document_sha256(base_env.body_scenario)
                    if base_env.body_scenario is not None
                    else None
                ),
                variant_ids=plan.variant_ids,
                expected_outcomes=plan.expected_outcomes,
            )
        for i in range(b):
            if kept[i] is None:
                dropped.append({
                    "after_episodes": done, "env": i, "kind": plan.kinds[i],
                    "reason": ("dip_reference" if dip_missed[i]
                               else "ik_reference" if i in unreachable else "idle"),
                    **({"worst_residual_mm": float(worst_mm[i])} if i in unreachable else {}),
                    **({"dip_lateral_mm": float(dip_mm[i]),
                        "dip_axis_deg": float(dip_deg[i])} if dip_missed[i] else {}),
                })
        if all(index is None for index in kept):
            skipped.append({"after_episodes": done, "task": task_b,
                            "reason": "every episode dropped"})
            consecutive_skips += 1
            print(f"[generate] dropped every episode of a {task_b} batch "
                  f"({len(dropped)} dropped so far)", flush=True)
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                print(f"[generate] WARNING: {MAX_CONSECUTIVE_SKIPS} batches in a row "
                      f"produced no demonstration — stopping at {done} episodes.", flush=True)
                break
            continue
        drawing_scores = None
        intended_fields = None
        if args.judge:
            scorable = [kind not in ("erase", "dip") for kind in plan.kinds]
            intended_fields = judging.intended_field_like(
                base_env.ink_field,
                base_env.surface,
                [
                    judging.strokes_from_plan_paths(plan.paths[i]) if scorable[i] else []
                    for i in range(base_env.num_envs)
                ],
                base_env.ink_opacity,
            )
            drawing_scores = judging.score_fields(
                base_env.ink_field.field,
                intended_fields,
                texel_per_m=base_env.surface.texel_per_m,
                tolerance_m=float(base_env.ink_field.pen_radius_m.max()),
            )
        snap_dir: Path | None = None
        field_np: np.ndarray | None = None
        if args.save_field_snapshots:
            snap_dir = Path(args.out_dir) / "meta" / "fields"
            snap_dir.mkdir(parents=True, exist_ok=True)
            field_np = base_env.ink_field.field.cpu().numpy()
        eval_artifacts = _write_eval_artifacts(
            args.out_dir, base_env, intended_fields if args.judge else None, kept)
        for i in range(b):
            if kept[i] is None:
                continue
            sheet = base_env.pad_sheets[i]
            if snap_dir is not None and field_np is not None:
                cv2.imwrite(str(snap_dir / f"episode_{kept[i]:06d}.png"),
                            (255 * (1.0 - field_np[i])).astype(np.uint8))
            entry_extra = _episode_artwork(plan, base_env, i)
            episode_log.append({
                "episode": kept[i],
                "batch_runtime": runtime.metadata(),
                "capture_control_ticks": capture_ticks[:int(plan.lengths[i]) * sample_hz // config.timing.control_hz],
                "kind": plan.kinds[i],
                **entry_extra,
                "surface_z": float(top_centers[i][2]),
                "surface_point": [float(v) for v in top_centers[i]],
                "surface_normal": [float(v) for v in normals[i]],
                "surface_profile": base_env.surface_profiles[i],
                "surface_radius_m": (
                    float(base_env.surface_radius_m[i])
                    if np.isfinite(base_env.surface_radius_m[i]) else None
                ),
                **({"path_canvas_m": plan.paths[i]}
                   if plan.kinds[i] not in ("language", "erase", "dip", "artwork", "spiral") else {}),
                "approach_frames": plan.n_app,
                "steps_planned": int((plan.intended_lengths if plan.intended_lengths is not None else plan.lengths)[i]),
                "steps_executed": min(runtime.step_index, int(plan.lengths[i])),
                "frames_recorded": int(recorded_steps[i]),
                "obs_delay_steps": int(obs_delay[i]),
                # the solved joint reference against what it was asked for:
                # worst target miss, and worst pen-down height error after
                # contact settling (both gated before the episode ran)
                "reference": {
                    "worst_residual_mm": float(worst_mm[i]),
                    "contact_error_mm": float(contact_mm[i]),
                },
                **(
                    {
                        "drawing_score": {
                            **drawing_scores[i].as_dict(),
                            "artifacts": eval_artifacts[i],
                        },
                    }
                    if drawing_scores is not None and not drawing_scores[i].degenerate
                    else {}
                ),
                "ink_coverage_start": float(coverage_start[i]),
                "ink_coverage_end": float(coverage_end[i]),
                "engaged": engaged[i],
                "episode_outcome": {
                    "variant": plan.variant_ids[i] if plan.variant_ids is not None else "blank-start",
                    "expected": expected_outcomes[i],
                    "observed_engaged": engaged[i],
                    "success": bool(expected_outcomes[i] == "nominal" and engaged[i]),
                    "observation_occlusion_fraction": observation_occlusion,
                    "stencil_area_fraction": float(base_env.stencil_fraction[i]),
                    "stencil_visible_fraction": float(base_env.stencil_visible_fraction()[i]),
                },
                **(
                    {"privileged_timeline": temporal_records[i]}
                    if temporal_records[i] is not None
                    else {}
                ),
                "interaction": {
                    "model": interaction_model,
                    "frames": int(ink_stats["interaction_frames"][i]),
                    "distance_min_m": (float(ink_stats["interaction_min_m"][i])
                                       if np.isfinite(ink_stats["interaction_min_m"][i])
                                       else None),
                    "distance_mean_m": (float(ink_stats["interaction_mean_m"][i])
                                        if np.isfinite(ink_stats["interaction_mean_m"][i])
                                        else None),
                    "distance_max_m": (float(ink_stats["interaction_max_m"][i])
                                       if np.isfinite(ink_stats["interaction_max_m"][i])
                                       else None),
                },
                # the charge model's account of the episode (scripts/lib/
                # ink_spec.py): what the tool spent, where it dipped, and why
                "ink": {
                    "mode": ink_stats["mode"],
                    "used_ul": float(ink_stats["used_ul"][i]),
                    "contact_mm": float(ink_stats["contact_mm"][i]),
                    "contact_s": float(ink_stats["contact_s"][i]),
                    "charge_end_ul": float(ink_stats["charge_end_ul"][i]),
                    "capacity_ul": float(ink_stats["capacity_ul"][i]),
                    "charge_start_ul": (float(plan.ink_initial_ul[i])
                                        if plan.ink_initial_ul is not None else None),
                    "dips": (plan.dips[i] if plan.dips is not None else []),
                },
                "pen_tilt_deg_mean": float(np.degrees(
                    np.linalg.norm(plan.lean_profiles[i], axis=1).mean())),
                "pen_tilt_deg_max": float(np.degrees(
                    np.linalg.norm(plan.lean_profiles[i], axis=1).max())),
                # the ruling the strokes trace — everything a scorer needs
                "grid": {"pitch_m": sheet["pitch_m"], "xs": sheet["xs"], "ys": sheet["ys"]},
            })
        if args.clamp_floor:
            clamp_fracs.append(expert.clamped_fraction)
        done = writer.num_episodes
        rate = max(done, 1) * plan.episode_steps / (time.time() - t_start)
        print(f"[generate] {done}/{args.num_episodes} episodes, {rate:.0f} env-steps/s", flush=True)

    writer.finalize()
    # Stamp the tool, in the same shape a real recording carries — schema
    # parity is what lets sim and real datasets be mixed and audited together.
    spec = tool
    tool_meta_path = registry.write_dataset_tool_metadata(
        args.out_dir, spec, workspace,
        extra={"source": "sim", "resolved_config": config.metadata(),
               "observation_profile": runtime.observations.profile.metadata(),
               "sample_time": {**cadence.metadata(), "dataset_timestamp_origin": "first_capture",
                               "dataset_timestamps": "nominal-camera-grid",
                               "actual_sample_ticks": "run_meta.episodes[].capture_control_ticks"},
               "tip_placement": geometry.source,
               "tip_protrusion_m": float(np.linalg.norm(geometry.tcp_offset_m)),
               "calibrated_tip_offset_m": list(
                   registry.tip_offset_m(workspace, "right") or geometry.touch_offset_m),
               "calibration_delta_m": list(geometry.calibration_delta_m),
               "body_origin_m": list(geometry.body_origin_m),
               "body_rpy_rad": list(geometry.body_rpy_rad),
               "body_tip_offset_m": list(geometry.body_tip_offset_m),
               "tip_offset_m": list(geometry.touch_offset_m),
               "tcp_offset_m": list(geometry.tcp_offset_m),
               "tcp_in_body_m": list(geometry.tcp_in_body_m),
               "alignment_error_m": geometry.alignment_error_m,
               "substrate": base_env.substrate.name,
               "interaction_model": interaction_model,
               "working_offset_m": args.draw_clearance,
               "contact_above_tolerance_m": interaction.CONTACT_ABOVE_TOLERANCE_M,
               "max_penetration_m": interaction.MAX_PENETRATION_M,
               "physics_contact_offset_m": interaction.PHYSICS_CONTACT_OFFSET_M,
               # Which kinematic contract solved this shard. A moving carriage
               # channel and a held one are different demonstrations, and the
               # real follower position-holds the axis, so a dataset has to say
               # which one it is rather than leave a reader to infer it.
               "external_effort": effort_metadata or {
                   "model": CONTACT_FORCE_MODEL,
                   "simulated": bool(contact_simulated),
                   "disclaimer": CONTACT_FORCE_DISCLAIMER,
               },
               **_carriage_metadata(args, expert, production_cfg),
               "contact_collision": contact_collision,
               "geometry_basis": tools.geometry_basis(geometry),
               "qualification": "qualified" if contact_eligible else "development",
               "geometry_warnings": geometry_warnings,
               # next to the substrate rather than only in run_meta: this is
               # the file a training pipeline already reads to check tool and
               # feature parity, and "which of the three distributions is
               # this" is the same kind of question
               "distribution": args.distribution})
    source_end = source_state()
    run_meta = {
        "sensor_profile": {"name": args.sensor_profile, "cameras": [camera.as_dict() for camera in camera_descriptions]},
        "schema_version": 2,
        # the FULL resolved configuration: this dataset is self-describing
        "config": dataclasses.asdict(args),
        # the portable design drawn in place of the collection, digest and all
        "design": design_scene.summary(design),
        "tool": json.loads(tool_meta_path.read_text()),
        "software": {
            "repository": source_start["repository"],
            "revision_start": source_start["revision"],
            "revision_end": source_end["revision"],
            "dirty_start": source_start["dirty"],
            "dirty_end": source_end["dirty"],
        },
        "episodes": episode_log,
    }
    if args.judge:
        run_meta["evaluation"] = {
            "schema": "tatbot.sim-eval-input/1",
            "producer": "expert",
            "judge": "tatbot_sim.judge/1",
            "headline": "f1",
            "visual_artifacts": ["intended", "drawn", "overlay"],
            "disclaimer": (
                "Simulation evaluation is a screening result only. It does not "
                "authorize powered motion or human contact."
            ),
        }
    if base_env.body_scenario is not None:
        scenario = base_env.body_scenario
        if args.scenario is None:
            raise RuntimeError("args.scenario is None when body_scenario is present")
        run_meta["scenario"] = {
            "source_path": str(Path(args.scenario).expanduser().resolve()),
            "sha256": document_sha256(scenario),
            "trace_sha256": scenario["trace"]["sha256"],
            # Typed v3 scenarios carry body.id; schema-2 procedural suites
            # (``sim sample``) identify the body by model_spec_id instead.
            "body": scenario["body"].get("id", scenario["body"].get("model_spec_id")),
            "pose": scenario["pose"].get("id"),
            "placement": scenario["placement"].get("id"),
            "design": scenario["design"].get("id"),
        }
    # The ink story next to the tool story, inlined for the same reason
    # tool.json is: the palette load and the policy constants will change,
    # and this dataset has to stay readable after they do.
    ink_meta = tools.ink_registry().dataset_ink_metadata(
        spec, tools.REPO, load=config.palette_load, palette=config.palette)
    ink_meta["source"] = "sim"
    ink_meta["supply"] = {"kind": config.supply[0], "ink": config.supply[1]}
    ink_meta["ink_dr"] = dataclasses.asdict(args.dr.ink)
    ink_meta["palette_dr"] = dataclasses.asdict(args.dr.palette)
    ink_meta["episodes"] = [
        {"episode": e["episode"], **e["ink"]} for e in episode_log if "ink" in e]
    with open(Path(args.out_dir) / "meta" / "ink.json", "w") as f:
        json.dump(ink_meta, f, indent=2)
    if clamp_fracs:
        run_meta["floor_clamped_step_fraction"] = float(np.mean(clamp_fracs))
    # Always present, so "no skips" is a recorded fact rather than a missing key.
    run_meta["skipped_batches"] = skipped
    run_meta["dropped_episodes"] = dropped
    with open(Path(args.out_dir) / "meta" / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[generate] wrote {done} episodes to {args.out_dir}")
    if skipped:
        print(f"[generate] {len(skipped)} batch(es) skipped as unplannable — see "
              f"run_meta.skipped_batches", flush=True)
    if dropped:
        reasons = {}
        for item in dropped:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
        print(f"[generate] {len(dropped)} episode(s) dropped, not written: "
              + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
              + " — see run_meta.dropped_episodes", flush=True)
    idle = [e["episode"] for e in episode_log if not e["engaged"]]
    if idle:
        print(f"[generate] WARNING: {len(idle)}/{done} episodes never moved any pigment "
              f"and are mislabelled demonstrations (kept by --keep-idle) — exclude {idle}",
              flush=True)
    if clamp_fracs:
        print(f"[generate] floor clamp touched {100 * run_meta['floor_clamped_step_fraction']:.2f}% of commanded steps")
    runtime.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
