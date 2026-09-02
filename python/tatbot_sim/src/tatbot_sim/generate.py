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
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import tyro

import tatbot_sim  # noqa: F401  (registers agent + env)
from tatbot_sim import interaction, tasks, tools
from tatbot_sim.agent import TatbotWXAI
from tatbot_sim.config import DRConfig
from tatbot_sim.depth_noise import DepthCorruptor, RGBJitter
from tatbot_sim.env import TatbotDrawEnv
from tatbot_sim.expert import (
    StrokeExpert,
    highest_reachable_z,
    reachable_canvas_masks,
    reachable_height_ceiling,
    worst_reach_residual,
)
from tatbot_sim.inkmap.contracts import document_sha256
from tatbot_sim.lerobot_writer import LeRobotWriter, quantize_depth_codes
from tatbot_sim.planning import SceneTooLongError, plan_batch, plan_tattoo_scenario
from tatbot_sim.repo import source_state

CAMERAS = ("wrist_upper", "wrist_lower")
# How far the IK may miss before the envelope counts as unreachable. Well
# under a line width (2.2-4 mm): a miss at that scale still marks the sheet
# somewhere plausible, which is exactly what makes it dangerous.
REACH_TOLERANCE_M = 0.001
EPISODE_START_MAX_LEAD_RAD = 0.1
"""Gross data-integrity bound, not a robot-motion acceptance threshold."""


MAX_CONSECUTIVE_SKIPS = 5
"""Give up after this many batches in a row refuse to fit the horizon. One
refusal is an unlucky scene draw and is skipped; a run where every draw refuses
is a recipe that cannot draw what it samples, and spinning on it would burn a
GPU all night to write nothing."""


def _current_robot_and_joint_indices(base_env, ik_joint_names):
    """Resolve the live articulation after a possibly reconfiguring reset."""
    robot = base_env.agent.robot
    active_names = [joint.name for joint in robot.active_joints]
    try:
        idx7 = [active_names.index(name) for name in TatbotWXAI.joint_names]
        idx_ik = [active_names.index(name) for name in ik_joint_names]
    except ValueError as exc:
        raise RuntimeError(
            f"live articulation joints {active_names} do not satisfy the data contract"
        ) from exc
    return robot, idx7, idx_ik


@dataclass
class Args:
    out_dir: str
    num_episodes: int = 64
    num_envs: int = 16
    horizon: int = 420
    """Control steps per episode (30 Hz -> 14 s). For --task language this
    is the CAP: each batch's horizon is sized to its sampled scenes (shorter
    preferred — operator call), never above this. 900 = the 30 s ceiling."""
    seed: int = 0
    tool_calibration_jitter: bool = False
    """Let the named factory sample one persistent mount-frame tip offset for
    this shard from the measured calibration uncertainty. The factory applies
    it before process re-exec so URDF, IK and metadata share the same draw."""
    tool_calibration_scale: float = 1.0
    """Multiplier on the measured uncertainty radius. One spans the retained
    touch-off diagnostic; zero disables displacement while preserving metadata."""
    sim_backend: str = "auto"
    task: str = "mix"
    """What the expert does. "mix" (default): each batch flips a weighted
    coin between language scenes and squiggles — describable drawings are
    favoured over random walks (operator call, 2026-08-24), see
    ``squiggle_frac``. Use --horizon 900 with mix/language so scenes get the
    full 30 s budget. "maze": only squiggle walks on the printed 6 mm grid.
    "language": only scene programs from the drawing language. "shapes": the
    earlier closed-shape repertoire. "erase": the sheet OPENS with a scene
    already on it and the tool takes it off — needs a removal tool fitted."""
    squiggle_frac: float = 0.25
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
    task_name: str = field(default_factory=lambda: (
        "draw a {size_mm}mm {shape} "
        f"{tools.active_tool().prompt_phrase} on the paper pad"
    ))
    """Per-episode task string for --task shapes: the frame of the real fm2
    recording ("draw a 6mm square using pen tip on the grid lines of the paper
    pad", no trailing period), sizes stated in mm from the sampled motif and
    the tool slot filled from the fitted datasheet. {shape} and {size_mm} are
    both available to overrides; unused slots are fine."""
    maze_task_name: str = field(default_factory=lambda: (
        f"draw a continuous squiggle {tools.active_tool().prompt_phrase} "
        "on the grid lines of the paper pad."
    ))
    """Task string for --task maze — the exact phrase of the real
    squiggle-grid-draw recordings, with the tool slot filled from the fitted
    tool's datasheet so a swap moves sim and real prompts together."""
    draw_clearance: float = interaction.WORKING_OFFSET_M
    """Resolved working-point offset from the surface, m. Contact-v1 is zero;
    retained as a CLI field so historical run_meta remains directly comparable."""
    allow_provisional_geometry: bool = False
    """Generate a validation-only shard without a quality-gated pivot TCP.
    A touch-axis-inferred BODY is eligible when the fixed-point calibration
    qualifies the contact vector; the override is only for nominal or failed
    contact calibration and is stamped for the audit."""
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
    scenario: str | None = None
    """Compiled Inkmap tattoo-scenario JSON. When set, the full posed body,
    support fixture, collision proxies, exact SVG and surface trace replace
    random pad-scene sampling; normally supplied by `body-tattoo`."""
    dr: DRConfig = field(default_factory=DRConfig)
    """Every randomization range, one tree — see tatbot_sim.config."""


# A pigment change smaller than this is a rounding artifact, not a tool touching
# a sheet -- the smallest real stamp moves far more coverage than this.
ENGAGED_MIN_COVERAGE_DELTA = 1e-5


def _engaged(kind: str, start: float, end: float, dips: int = 0) -> bool:
    """Did the tool actually do to the sheet what the prompt says it did?

    An episode whose ink never moves is not a weak demonstration, it is a
    mislabelled one: it ships the same "remove the ink" sentence as the rest
    while showing the arm never touching the skin. A dip episode's claim is
    the dip, not a mark: it is engaged when its charge landed.
    """
    if kind == "dip":
        return dips > 0
    delta = float(end) - float(start)
    want = -1.0 if kind == "erase" else 1.0
    return want * delta > ENGAGED_MIN_COVERAGE_DELTA


def main(args: Args):
    source_start = source_state()
    rng = np.random.default_rng(args.seed)
    if not np.isfinite(args.tool_calibration_scale) or args.tool_calibration_scale < 0:
        raise SystemExit("--tool-calibration-scale must be finite and non-negative")
    # Every task family this run can sample, checked against the fitted tool
    # and its substrate BEFORE the env is built. An erase episode with a pen
    # fitted would DRAW over its own target and write a dataset whose prompts
    # say "remove"; a language episode with the laser fitted would strip a
    # blank sheet while its prompts say "draw". Both look fine until someone
    # trains on them, so both are refused here rather than surfacing as the
    # engaged-episode warning after the run finishes.
    tool = tools.active_tool()
    substrate = tools.active_substrate()
    registry = tools.registry()
    workspace = tools.workspace()
    geometry = tools.resolved_geometry(tool, workspace)
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
    contact_eligible = geometry.contact_status == "pivot-calibrated"
    if (tool.contact and not contact_eligible
            and not args.allow_provisional_geometry):
        raise SystemExit(
            f"{tool.tool_id!r} contact geometry is {geometry.contact_status!r}: "
            f"{geometry.contact_qualification_error or 'no pivot-calibrated TCP'}. "
            "Production contact generation needs a quality-gated fixed-point "
            "touch-off. Pass --allow-provisional-geometry only for a labelled "
            "validation shard.")
    if tool.contact and abs(args.draw_clearance - interaction.WORKING_OFFSET_M) > 1e-9:
        raise SystemExit(
            f"{tool.tool_id!r} is a contact tool: its resolved working point must target "
            f"the surface ({interaction.WORKING_OFFSET_M:.4f} m), not "
            f"--draw-clearance {args.draw_clearance:.4f} m. Use approach/travel height "
            "for clearance; changing the drawing target recreates air-drawing data.")
    try:
        tools.set_supply(args.supply, args.supply_ink)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for task in tasks.active_tasks(args.task, args.erase_frac, args.squiggle_frac, args.dip_frac):
        try:
            tasks.validate_task(task, tool, substrate)
            tasks.validate_supply(task, tool)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    env = gym.make(
        "TatbotDraw-v0",
        num_envs=args.num_envs,
        obs_mode="rgbd" if args.depth else "rgb",
        control_mode="pd_joint_pos",
        sim_backend=args.sim_backend,
        texture_refresh_steps=args.texture_refresh_steps,
        reconfiguration_freq=1 if args.reconfigure_each_batch else 0,
        dr=args.dr,
        scenario_path=args.scenario,
    )
    base_env: TatbotDrawEnv = env.unwrapped
    contact_collision = bool(
        tool.contact_radius_m is not None
        and base_env.pad_height is None
        and base_env.body_scenario is None
    )
    interaction_model = interaction.model_for(collision=contact_collision)
    device = base_env.device
    expert = StrokeExpert(args.num_envs, device, noise=args.dr.noise, seed=args.seed)
    ik_joint_names = expert.ik.chain.get_joint_parameter_names()
    robot, idx7, idx_ik = _current_robot_and_joint_indices(base_env, ik_joint_names)

    # Can the fitted tool actually be held perpendicular over the sampled pad
    # heights? A tool that cannot does not fail — IK returns its best effort
    # and every episode marks tens of millimetres from where its own labels
    # say, which no downstream check would catch.
    # A mound puts its summit above the pad heights this samples, so the check
    # has to cover the summit or it approves an envelope the run never uses.
    # It still measures the tool held STRAIGHT UP; what it cannot see is the
    # lean a shaped surface demands, which is what audit_reach is for.
    mound = (float(base_env.pad_height.max()) if base_env.pad_height is not None
             else float(base_env.substrate.mound_peak_m))
    pad_z_range = base_env.dr.pad.z_range
    if pad_z_range is None:
        raise RuntimeError("pad z_range is not set")
    reach_z = (pad_z_range[0], pad_z_range[1] + mound)
    worst_reach, worst_z = worst_reach_residual(
        expert, robot.get_qpos()[:, idx_ik], base_env.pad_center,
        reach_z, args.draw_clearance,
    )
    if worst_reach > REACH_TOLERANCE_M:
        ceiling = highest_reachable_z(
            expert, robot.get_qpos()[:, idx_ik], base_env.pad_center,
            reach_z, args.draw_clearance, REACH_TOLERANCE_M,
        )
        # the knob is the pad's height; the mound rides on top of it, so the
        # ceiling has to come back down by the mound before it is advice
        top = None if ceiling is None else ceiling - mound
        fix = (
            f"pass --dr.pad.z-range {pad_z_range[0]:.3f} {top:.3f}"
            if top is not None and top > pad_z_range[0] else
            f"even a flat-on-the-table skin is out of reach with a {mound * 1000:.0f} mm "
            "mound on it, so this is the mound or the tool rather than the pad height"
        )
        raise SystemExit(
            f"{tools.active_tool().tool_id!r} cannot reach the sampled drawing "
            f"envelope: IK is off by {worst_reach * 1000:.1f} mm with the pad top at "
            f"{worst_z:.3f} m (tolerance {REACH_TOLERANCE_M * 1000:.0f} mm). "
            f"To generate anyway, {fix} — or revisit the tool's grip point."
        )
    print(f"[generate] reach check: worst IK residual {worst_reach * 1000:.2f} mm "
          f"over pad z {pad_z_range}")

    writer = LeRobotWriter(args.out_dir, cameras=CAMERAS, depth=args.depth,
                           task_name=args.task_name,
                           encoder_procs=max(0, args.encoder_procs))
    corruptor = (
        DepthCorruptor(args.num_envs, device, cfg=args.dr.depth_noise, seed=args.seed)
        if args.depth and args.dr.corrupt_depth else None
    )
    rgb_jitter = RGBJitter(args.num_envs, device, cfg=args.dr.rgb, seed=args.seed)
    episode_log = []

    done = 0
    clamp_fracs: list[float] = []
    # A batch whose scene will not fit the horizon is unlucky, not fatal: it is
    # skipped and redrawn rather than taking the run down. Bounded, because a
    # recipe whose scenes NEVER fit would otherwise spin forever, and counted,
    # because silently skipping batches narrows the distribution while the
    # episode count still looks full.
    skipped: list[dict] = []
    consecutive_skips = 0
    t_start = time.time()
    while done < args.num_episodes:
        b = min(args.num_envs, args.num_episodes - done)
        # env batch size is fixed; surplus envs in the last batch are discarded
        env.reset(seed=args.seed + done + 977 * len(skipped))
        # Reconfiguration rebuilds the agent and its articulation. Refresh the
        # object and its joint maps before planning or setting qpos: setters on
        # the pre-reset object only update an orphaned articulation.
        robot, idx7, idx_ik = _current_robot_and_joint_indices(base_env, ik_joint_names)
        top_centers, rots = base_env.canvas_frame_np
        normals = rots[:, :, 2]

        if args.scenario:
            task_b = "body-tattoo"
        elif args.task == "mix":
            roll = rng.random()
            if roll < args.erase_frac:
                task_b = "erase"
            elif roll < args.erase_frac + args.squiggle_frac:
                task_b = "maze"
            elif roll < args.erase_frac + args.squiggle_frac + args.dip_frac:
                task_b = "dip"
            else:
                task_b = "language"
        else:
            task_b = args.task
        horizon_b = args.maze_horizon if task_b in ("maze", "shapes", "dip") else args.horizon
        # Where can the fitted tool actually be held normal to THIS batch's
        # surface? On a mound the flanks ask the wrist for a lean it cannot
        # make, and a stroke laid across them is a label the arm quietly
        # misses — so strokes are placed only on ground the IK can reach.
        # Flat substrates need the mask too since the 2026-08-31 measured
        # ballpoint tip: its workable region is a diagonal band (low, near,
        # biased +y), and no rectangular pad placement sits wholly inside it —
        # stroke placement has to dodge the far/-y corner, which is what the
        # mask is for. On a healthy nominal tool it is all-true and costs one
        # batched solve.
        ceiling = None
        q_now = robot.get_qpos()[:, idx_ik]
        slack = args.dr.pen_lean.max_off_base_rad
        masks = reachable_canvas_masks(
            expert, q_now, base_env.surface, args.draw_clearance, args.num_envs,
            max_off_base_rad=slack,
        )
        # How high the tool can still be held. Travel and the opening
        # descent are the tallest poses an episode asks for, and over a
        # mound they are the ones the arm cannot make.
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

        try:
            if args.scenario:
                if base_env.body_scenario is None:
                    raise RuntimeError("body_scenario is None when scenario is set")
                plan = plan_tattoo_scenario(
                    rng, base_env.body_scenario, base_env.surface,
                    horizon=horizon_b, num_envs=args.num_envs,
                    dr=args.dr, draw_clearance=args.draw_clearance,
                    tool_ceiling=ceiling,
                )
            else:
                plan = plan_batch(
                    rng, base_env.pad_sheets, base_env.surface,
                    task=task_b, horizon=horizon_b, num_envs=args.num_envs,
                    dr=args.dr, draw_clearance=args.draw_clearance,
                    task_name=args.task_name, maze_task_name=args.maze_task_name,
                    erase_passes=args.erase_passes,
                    erase_seconds=args.erase_seconds,
                    reachable=masks,
                    tool_ceiling=ceiling,
                    cap_rims=base_env.cap_rims_np(),
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
        consecutive_skips = 0
        if plan.preink is not None:
            base_env.preink(plan.preink)
        # the dips this batch makes: which steps are away at the palette (no
        # marking there) and when each charge lands on the tool
        base_env.set_dip_schedule(plan)

        # place the arm: solve the drawing start, then optionally start the
        # episode raised at the staged pose (the approach descends from it)
        q0 = robot.get_qpos()[:, idx_ik]
        q_start = expert.solve_pose(plan.targets[:, 0], q0, normals=plan.pen_normals[:, 0])
        full = robot.get_qpos().clone()
        full[:, idx_ik] = q_start
        robot.set_qpos(full)
        if plan.q_raised is not None:
            full = robot.get_qpos().clone()
            full[:, idx_ik] = torch.as_tensor(plan.q_raised, device=device)
            robot.set_qpos(full)
        if base_env.gpu_sim_enabled:
            base_env.scene._gpu_apply_all()
            base_env.scene.px.gpu_update_articulation_kinematics()
            base_env.scene._gpu_fetch_all()
        base_env.agent.controller.reset()

        # The pad surface is the hard floor for commanded steps: sim simulates
        # no contact, so an unclamped noise burst commands straight through the
        # pad and the data teaches exactly that (measured on-robot 2026-08-21).
        expert.reset(
            plan.targets,
            q_start,
            floor_plane=(plan.surface_points, plan.surface_normals)
            if args.clamp_floor else None,
            pen_normals=plan.pen_normals,
            approach_from=(plan.q_raised, plan.n_app) if plan.q_raised is not None else None,
        )
        if args.scenario:
            q_reference = expert.q_ref
            if q_reference is None:
                raise RuntimeError("expert did not produce a joint reference")
            solved = expert.ik.fk(q_reference.reshape(-1, 6))[:, :3, 3].reshape(
                args.num_envs, -1, 3,
            )
            desired = torch.as_tensor(plan.targets, dtype=solved.dtype, device=solved.device)
            residual = torch.linalg.norm(solved - desired, dim=-1)
            worst = float(residual.max())
            if worst > REACH_TOLERANCE_M:
                bad = int((residual > REACH_TOLERANCE_M).sum())
                raise SystemExit(
                    f"compiled body tattoo is outside the exact IK envelope: "
                    f"worst residual {worst * 1000:.2f} mm, {bad}/{residual.numel()} "
                    f"targets above the {REACH_TOLERANCE_M * 1000:.0f} mm gate. "
                    "Recompile with a different --target-world-m or --patch-yaw-rad."
                )
        if corruptor is not None:
            corruptor.reset()
        rgb_jitter.reset()
        writer.open_batch(b, tasks=plan.tasks[:b])
        # non-zero once removal episodes open on a pre-inked sheet
        if base_env.ink_field is None:
            raise RuntimeError("ink_field is None after env reset")
        coverage_start = base_env.ink_field.coverage().cpu().numpy()

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
        for t in range(int(plan.lengths[:b].max())):
            action = expert.act()
            before_qpos = robot.get_qpos()[:b, idx7] if t == 0 else None
            obs, _, _, _, _ = env.step(action)
            # jitter/corrupt run on the FULL batch (their per-env profiles are
            # (num_envs, ...)); everything is sliced to the b written episodes
            # in one place, before the device->host transfer
            qpos = obs["agent"]["qpos"][:b, idx7].cpu().numpy()
            if t == 0:
                action_np = action[:b].cpu().numpy()
                lead = np.abs(action_np - qpos)
                recorded_lead = float(lead.max())
                if recorded_lead > EPISODE_START_MAX_LEAD_RAD:
                    env_index, joint_index = np.unravel_index(int(np.argmax(lead)), lead.shape)
                    if before_qpos is None:  # narrowed by t == 0 above
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
            state = np.concatenate([qpos, np.zeros_like(qpos)], axis=1)  # ext_eff masked
            frames = {
                cam: rgb_jitter(obs["sensor_data"][cam]["rgb"])[:b].cpu().numpy()
                for cam in CAMERAS
            }
            depth = None
            if args.depth:
                depth = {}
                for cam in CAMERAS:
                    d = obs["sensor_data"][cam]["depth"]
                    if corruptor is not None:
                        d = corruptor(d)
                    # quantize to 12-bit codes ON the GPU: leaves the encode
                    # workers nothing but the codec call and halves the
                    # device->host transfer (int16 fits 4095)
                    depth[cam] = (
                        quantize_depth_codes(d[:b].to(torch.float32))
                        .to(torch.int16).cpu().numpy().astype(np.uint16)
                    )
            obs_hist.append({"state": state, "frames": frames, "depth": depth})
            if len(obs_hist) > max_delay + 1:
                obs_hist.pop(0)

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
                            for c in CAMERAS}
                d_depth = None if depth is None else {
                    c: [srcs[i]["depth"][c][i] for i in range(b)] for c in CAMERAS}
            writer.add_steps(action[:b].cpu().numpy(), d_state, d_frames, d_depth,
                             active=[t < int(plan.lengths[i]) for i in range(b)])

        writer.close_batch()
        # Pigment on the sheet, start and end. Ink is a measured quantity now
        # rather than a pile of actors, so every episode carries how much was
        # laid down (or, for a removal tool, how much was cleared) without any
        # extra machinery — the scoreboard the sim-eval harness wants.
        if base_env.ink_field is None:
            raise RuntimeError("ink_field is None after env reset")
        coverage_end = base_env.ink_field.coverage().cpu().numpy()
        ink_stats = base_env.ink_episode_stats()
        snap_dir: Path | None = None
        field_np: np.ndarray | None = None
        if args.save_field_snapshots:
            snap_dir = Path(args.out_dir) / "meta" / "fields"
            snap_dir.mkdir(parents=True, exist_ok=True)
            field_np = base_env.ink_field.field.cpu().numpy()
        for i in range(b):
            sheet = base_env.pad_sheets[i]
            if snap_dir is not None and field_np is not None:
                cv2.imwrite(str(snap_dir / f"episode_{done + i:06d}.png"),
                            (255 * (1.0 - field_np[i])).astype(np.uint8))
            entry_extra = {}
            # erase episodes are scene programs too — the program is what the
            # sheet OPENED with and what a scorer measures the clearing against
            if plan.kinds[i] in ("language", "erase"):
                entry_extra = {"program": plan.programs[i], "strokes_canvas_m": plan.paths[i]}
            elif plan.kinds[i] == "dip":
                entry_extra = {"program": plan.programs[i]}
            episode_log.append({
                "episode": done + i,
                "kind": plan.kinds[i],
                **entry_extra,
                "surface_z": float(top_centers[i][2]),
                "surface_point": [float(v) for v in top_centers[i]],
                "surface_normal": [float(v) for v in normals[i]],
                **({"path_canvas_m": plan.paths[i]}
                   if plan.kinds[i] not in ("language", "erase", "dip") else {}),
                "approach_frames": plan.n_app,
                "obs_delay_steps": int(obs_delay[i]),
                "ink_coverage_start": float(coverage_start[i]),
                "ink_coverage_end": float(coverage_end[i]),
                "engaged": _engaged(plan.kinds[i], coverage_start[i], coverage_end[i],
                                    dips=int(ink_stats["dips"][i])),
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
        done += b
        rate = done * plan.episode_steps / (time.time() - t_start)
        print(f"[generate] {done}/{args.num_episodes} episodes, {rate:.0f} env-steps/s", flush=True)

    writer.finalize()
    # Stamp the tool, in the same shape a real recording carries — schema
    # parity is what lets sim and real datasets be mixed and audited together.
    spec = tools.active_tool()
    tool_meta_path = registry.write_dataset_tool_metadata(
        args.out_dir, spec, workspace,
        extra={"source": "sim",
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
               "contact_collision": contact_collision,
               "provisional_geometry_override": bool(
                   args.allow_provisional_geometry and not contact_eligible),
               # next to the substrate rather than only in run_meta: this is
               # the file a training pipeline already reads to check tool and
               # feature parity, and "which of the three distributions is
               # this" is the same kind of question
               "distribution": args.distribution})
    source_end = source_state()
    run_meta = {
        "schema_version": 2,
        # the FULL resolved configuration: this dataset is self-describing
        "config": dataclasses.asdict(args),
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
    if base_env.body_scenario is not None:
        scenario = base_env.body_scenario
        if args.scenario is None:
            raise RuntimeError("args.scenario is None when body_scenario is present")
        run_meta["scenario"] = {
            "source_path": str(Path(args.scenario).expanduser().resolve()),
            "sha256": document_sha256(scenario),
            "trace_sha256": scenario["trace"]["sha256"],
            "body": scenario["body"]["id"],
            "pose": scenario["pose"]["id"],
            "placement": scenario["placement"]["id"],
            "design": scenario["design"]["id"],
        }
    # The ink story next to the tool story, inlined for the same reason
    # tool.json is: the palette load and the policy constants will change,
    # and this dataset has to stay readable after they do.
    ink_meta = tools.ink_registry().dataset_ink_metadata(spec, tools.REPO, load=tools.palette_load())
    ink_meta["source"] = "sim"
    ink_meta["supply"] = {"kind": tools.supply()[0], "ink": tools.supply()[1]}
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
    with open(Path(args.out_dir) / "meta" / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[generate] wrote {done} episodes to {args.out_dir}")
    if skipped:
        print(f"[generate] {len(skipped)} batch(es) skipped as unplannable — see "
              f"run_meta.skipped_batches", flush=True)
    idle = [e["episode"] for e in episode_log if not e["engaged"]]
    if idle:
        print(f"[generate] WARNING: {len(idle)}/{done} episodes never moved any pigment "
              f"and are mislabelled demonstrations — exclude {idle} or regenerate", flush=True)
    if clamp_fracs:
        print(f"[generate] floor clamp touched {100 * run_meta['floor_clamped_step_fraction']:.2f}% of commanded steps")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
