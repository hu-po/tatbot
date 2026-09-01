"""Batch planning: everything about WHAT a batch of episodes will draw.

One function, `plan_batch`, owns the sampling that used to live inline in
generate.main: the approach coin, the task branch (maze / language / shapes),
trajectory building, lean profiles, task strings, and the per-episode
metadata. The generator consumes the plan to produce data; preview and
replay tools (scripts/sim_preview.py) consume the *same* plan to render —
no more hand-mirrored RNG order in throwaway scripts, which broke twice in
one week.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from tatbot_sim import dipping, tasks, tools
from tatbot_sim.config import DRConfig
from tatbot_sim.env import TatbotDrawEnv
from tatbot_sim.language import sample_scene
from tatbot_sim.strokes import (
    MazeConfig,
    ShapeConfig,
    build_ee_trajectory,
    fit_strokes,
    overhead_steps,
    sample_maze,
    sample_shape,
)

# The follower's staged position — the six arm joints of the pose the real
# arm lifts to at connect. Read from config/trossen/tatbot.yaml (through
# tatbot_sim.tools) rather than copied: until 2026-08-30 a literal here
# matched the real episodes' t=0 pose only because nobody had changed either.
from tatbot_sim.tools import staged_pose as _staged_pose  # noqa: E402

STAGED_POSE = np.array(_staged_pose()[:6], dtype=np.float32)

# Longest single tracing pass an erase scene is sampled for. Kept small on
# purpose: a motif sized for a 40 s budget is one the reach mask has nowhere to
# put on a mound, and the sampler spends its retries placing nothing. Episode
# duration comes from how many times the scene is retraced, not from its size.
SCENE_PASS_CAP_S = 14.0


def sample_lean_profile(
    rng: np.random.Generator, t_len: int, max_tilt: float, keypoints: int = 4
) -> np.ndarray:
    """(T, 2) lean vector in the surface tangent plane, |v| = lean angle.

    Random keypoints smoothstep-blended over the path: continuous, C1, and
    wrap-free (interpolating tangent-plane vectors instead of azimuth angles
    dodges the 2-pi seam). Flicks and stipple later just supply a different
    profile through the same channel."""
    if max_tilt <= 0 or keypoints < 2:
        return np.zeros((t_len, 2))
    kp_t = np.linspace(0, t_len - 1, keypoints)
    ang = rng.uniform(0, 2 * np.pi, keypoints)
    mag = rng.uniform(0, max_tilt, keypoints)
    kp = np.stack([mag * np.cos(ang), mag * np.sin(ang)], axis=1)
    out = np.zeros((t_len, 2))
    for seg in range(keypoints - 1):
        a, bnd = kp_t[seg], kp_t[seg + 1]
        idx = np.arange(int(np.ceil(a)), int(np.floor(bnd)) + 1)
        u = (idx - a) / max(bnd - a, 1e-9)
        w = u * u * (3 - 2 * u)
        out[idx] = kp[seg] * (1 - w[:, None]) + kp[seg + 1] * w[:, None]
    return out




def lean_normals(normal: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Surface normal + lean profile (T,2) -> pen axes (T,3).

    ``normal`` is either one env's constant normal (3,) or the LOCAL normal at
    every step (T,3). That second form is the whole of what holding the tool
    perpendicular to a curved surface takes: lean stays exactly what it was, a
    controlled perturbation, but it is now a perturbation about the normal the
    tool actually meets rather than about one plane's.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n.ndim == 1:
        n = np.broadcast_to(n, (len(profile), 3))
    helper = np.where(
        np.abs(n[:, :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(n, helper)
    e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    theta = np.linalg.norm(profile, axis=1)  # (T,)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(theta[:, None] > 1e-12, profile / theta[:, None], 0.0)
    tangent = d[:, :1] * e1 + d[:, 1:] * e2
    v = np.cos(theta)[:, None] * n + np.sin(theta)[:, None] * tangent
    return v / np.linalg.norm(v, axis=1, keepdims=True)




def cap_lean(axes: np.ndarray, base_normal: np.ndarray, max_rad: float) -> np.ndarray:
    """Hold the tool no further than ``max_rad`` off the pad's own normal.

    A tool does not have to be exactly perpendicular to skin to work it, and on
    a mound's flanks exactly perpendicular is a pose the arm cannot make — so
    it returns a best effort tens of millimetres away and the labels never say
    so. Leaning as far as the wrist can and no further is the honest version of
    that trade: the surface still decides which WAY the tool tilts, the arm
    decides how far.
    """
    if max_rad is None or max_rad <= 0:
        return axes
    base = np.asarray(base_normal, dtype=np.float64)
    base = base / np.linalg.norm(base)
    dot = np.clip(axes @ base, -1.0, 1.0)
    lean = np.arccos(dot)
    over = lean > max_rad
    if not over.any():
        return axes
    out = axes.copy()
    keep = np.where(lean > 1e-9, max_rad / np.maximum(lean, 1e-9), 0.0)
    sin_l = np.sin(lean)
    safe = sin_l > 1e-9
    w0 = np.where(safe, np.sin((1 - keep) * lean) / np.where(safe, sin_l, 1.0), 1 - keep)
    w1 = np.where(safe, np.sin(keep * lean) / np.where(safe, sin_l, 1.0), keep)
    blend = w0[:, None] * base[None, :] + w1[:, None] * axes
    blend /= np.linalg.norm(blend, axis=1, keepdims=True)
    out[over] = blend[over]
    return out


def canvas_to_world(traj_xyz: np.ndarray, surface, env_index: int, clearance: float):
    """(T,3) canvas-frame trajectory -> world points and the local normals.

    This is the mapping step strokes.py was written to expect. The canvas
    frame's z is height above the drawing surface AT THAT POINT, so a stroke
    follows the shape rather than a plane, and travel keeps its clearance over
    whatever is underneath instead of cutting a chord across a rise.

    The surface points and normals come back too, because the planner needs
    exactly these to aim the tool and to give the floor clamp a tangent plane
    per step; asking the surface the same question twice is how the ink model,
    the orientation and the floor quietly stop agreeing.
    """
    points, normals = surface.frame_np(env_index, traj_xyz[:, :2])
    return points + (traj_xyz[:, 2:3] + clearance) * normals, points, normals




@dataclass
class BatchPlan:
    """What one batch of parallel episodes will draw, fully sampled."""

    n_app: int                      # approach frames (0 = no descent)
    q_raised: np.ndarray | None     # (B, 6) staged start poses when approaching
    draw_horizon: int               # steps in the drawing part
    targets: np.ndarray             # (B, T_draw, 3) world-frame needle targets
    pen_normals: np.ndarray         # (B, T_draw, 3) pen-axis directions
    surface_points: np.ndarray      # (B, T_draw, 3) the surface under each step
    surface_normals: np.ndarray     # (B, T_draw, 3) its outward normal there
    # ^ the floor clamp's tangent plane per step. One plane per episode is only
    # the surface while the surface is flat; on a shape it is a plane the tool
    # legitimately works below, and the clamp fights the trajectory.
    lean_profiles: list             # per-env (T_draw, 2) tangent-plane leans
    kinds: list                     # per-env task kind
    tasks: list                     # per-env prompt / task string
    paths: list                     # per-env intended stroke polyline(s)
    programs: list                  # per-env scene program (language) or None
    lengths: np.ndarray             # (B,) natural episode steps incl. approach
    # ^ where each drawing actually ENDS (plus a short settle tail); steps
    # beyond it are the batch holding for its longest episode, and the
    # generator can stop recording an episode there
    preink: list | None = None      # per-env Strokes the sheet OPENS with (erase)
    # Ink. dip_mask marks every drawing step spent away at the palette (the
    # env withholds deposition there); dip_credits is, per env, the drawing
    # step each dip's charge lands on; dips is the plan itself, per env, for
    # the dataset's meta/ink.json. All empty when the fitted tool never dips.
    dip_mask: np.ndarray | None = None      # (B, T_draw) bool
    dip_credits: list | None = None         # per-env list[int]
    dips: list | None = None                # per-env list[dict]
    # How the tool OPENS the episode (config.InkDR.initial_charge_frac /
    # capacity_scale, drawn per env here so the env and the planner agree):
    # the charge on it, and the capacity it was scaled to. None when the tool
    # carries no ink.
    ink_initial_ul: np.ndarray | None = None    # (B,)
    ink_capacity_ul: np.ndarray | None = None   # (B,)

    @property
    def episode_steps(self) -> int:
        return self.n_app + self.draw_horizon


class SceneTooLongError(RuntimeError):
    """A sampled scene does not fit the horizon, after every budget backoff.

    Typed because the caller's only sane response differs from every other
    planning failure: the batch is unlucky, not the run. ``generate`` skips it
    and redraws, where a bare RuntimeError took the whole process down and with
    it every episode already written -- one refusal at episode 96 of 144 cost
    the other 96 (measured, 2026-08-27).

    Carries the offending sizes so the skip can be reported rather than merely
    survived: a run that quietly skipped a third of its batches would be a
    narrower distribution wearing a full run's episode count.
    """

    def __init__(self, task: str, needed: int | None, horizon: int):
        self.task, self.needed, self.horizon = task, needed, horizon
        why = (f"{needed} steps needed, {horizon} available"
               if needed is not None else
               f"no scene could be placed at any budget, in {horizon} steps")
        super().__init__(f"{task} scene would not fit the horizon cap ({why})")


def plan_batch(
    rng: np.random.Generator,
    sheets: list[dict],
    surface,
    *,
    task: str,
    horizon: int,
    num_envs: int,
    dr: DRConfig,
    draw_clearance: float,
    task_name: str,
    maze_task_name: str,
    erase_passes: tuple[int, int] = (2, 4),
    erase_seconds: tuple[float, float] = (28.0, 60.0),
    reachable: list | None = None,
    tool_ceiling: float | None = None,
    style=None,
    cap_rims: dict | None = None,
    dip_task_name: str = "dip {tool} into the {ink} ink cap.",
) -> BatchPlan:
    """``reachable`` is one ReachMask per env: where the fitted tool can be held
    normal to the surface. Strokes are placed only there, because a stroke laid
    across ground the wrist cannot make is a label the arm quietly misses.

    ``style`` narrows what the scene sampler may draw (language.SceneStyle);
    None is the training draw.

    ``tool_ceiling`` is how high above the surface the tool can still be held.
    Travel and the opening descent are the tallest poses an episode asks for,
    and over a mound they are the ones the arm cannot make -- an episode that
    starts higher than this begins from a pose that was never solved, and the
    sequential solve carries that miss through everything after it.

    ``cap_rims`` is the env's per-env cap rim positions, ``{slot: (B,3)}``
    world frame (TatbotDrawEnv.cap_rims_np). With it, and a fitted tool
    whose ink policy dips, a task that deposits gets dips planned by the
    charge model (scripts/lib/ink_spec.plan_dips) and spliced into the
    world-frame trajectory at stroke boundaries (tatbot_sim.dipping). None,
    or a tool with ink.mode none, plans no dips."""
    inkctx = _ink_context(task, cap_rims, dr, num_envs, rng)
    ink_initial, ink_capacity = _ink_opening(inkctx, num_envs)
    # the canvas origin's height, which is what the start-height budget is
    # measured against; on a curved surface it is still the origin, not a peak
    tops = surface.origin_world_np()[:, 2]
    shape_cfg = ShapeConfig()
    maze_cfg = MazeConfig()
    if tool_ceiling is not None:
        lo, hi = shape_cfg.start_height_range
        shape_cfg = dataclasses.replace(
            shape_cfg,
            hover_height=min(shape_cfg.hover_height, tool_ceiling),
            start_height_range=(min(lo, tool_ceiling), min(hi, tool_ceiling)),
        )

    do_approach = rng.random() < dr.approach.prob
    n_app = int(rng.uniform(*dr.approach.duration_s) * 30) if do_approach else 0

    kinds, paths, programs = [], [], []
    preink: list | None = None
    if task in ("language", "erase"):
        lang_cfg = dataclasses.replace(shape_cfg, draw_speed_range=maze_cfg.draw_speed_range)
        # per-batch time budget, skewed short (min of two uniform draws:
        # "up to 30 s but opt for shorter" — operator call). Sampling the
        # budget per batch also keeps one long scene from stretching every
        # episode in the batch to the cap.
        cap_s = (horizon - n_app) / 30.0 - 0.5
        if cap_s <= 7.5:
            raise ValueError(
                f"--task language needs horizon >= {n_app + 240} (got {horizon}): "
                "the budget sampler draws from [7 s, cap] and an inverted range "
                "would silently misbehave"
            )
        # An erase episode traces its scene several times — a pass clears a
        # fraction of what is under the beam, so one pass never finishes the
        # job and a dataset of single passes would only show ink getting
        # fainter. Passes multiply the TRACING only; the descend-and-lift
        # overhead is paid once for the episode.
        verb = "erase" if task == "erase" else "draw"
        passes = 1  # for erase this is decided per env, once a pass has been measured
        target_s = 0.0
        if verb == "draw":
            # skewed short (min of two uniform draws: "up to 30 s but opt for
            # shorter" — operator call)
            budget_s = (cap_s * 0.96 if (style is not None and style.fill_budget)
                        else min(rng.uniform(7.0, cap_s), rng.uniform(7.0, cap_s)) * 0.92)
            scene_budget = budget_s
        else:
            # Erase budgets the SCENE, not the episode: give the sampler the
            # same 7 s floor the draw task uses and let the pass count set how
            # long the episode runs. Dividing the episode budget instead left
            # the scene near the cheapest motif's cost, where only one of the
            # twelve at its smallest size was affordable and roughly a third
            # of batches exhausted the sampler's retries.
            # An erase episode should last as long as the operator's do. The
            # recordings run 28-60 s; sampling a SCENE and hoping the passes
            # added up left the sim at a 14 s median, because the scene
            # sampler is deliberately skewed short. So sample the EPISODE
            # duration from the band the recordings occupy and let the pass
            # count decide how big a scene fits inside it.
            target_s = min(float(rng.uniform(*erase_seconds)), cap_s)
            if target_s <= 7.5:
                raise ValueError(
                    f"--task erase needs horizon >= {int((erase_seconds[0] + 0.5) * 30) + n_app} "
                    f"for a {erase_seconds[0]:.0f} s episode (got {horizon})"
                )
            # The scene is sampled for ONE pass and stays SMALL -- a motif
            # sized for a 40 s budget is one the reach mask has nowhere to put,
            # and the sampler exhausts its retries placing nothing. Duration
            # comes from the pass count, decided after measuring a pass below.
            one_pass_cap = min(SCENE_PASS_CAP_S, cap_s)
            scene_budget = min(rng.uniform(7.0, one_pass_cap),
                               rng.uniform(7.0, one_pass_cap)) * 0.92
        if verb == "erase":
            preink = []
        trajs = []
        dip_plans = []
        for i in range(num_envs):
            # the time budget is an estimate; if the realized build overruns
            # the hard cap anyway, resample the scene smaller rather than
            # truncating (a cut scene makes its own prompt a lie)
            budget_i = scene_budget
            # The refusal below has two causes and they need different fixes:
            # a scene that BUILT but overran, or a sampler that could never
            # place anything (the except path never binds traj). Stays None in
            # the second case so the error can say so.
            built_len = None
            for _ in range(6):
                try:
                    strokes, program = sample_scene(
                        rng, sheets[i], budget_i, verb=verb,
                        reachable=None if reachable is None else reachable[i],
                        style=style)
                except RuntimeError:
                    # Nothing affordable would fit where the tool can work. On
                    # a mound the reachable region is most of the skin but not
                    # a convenient shape, so a big motif can have nowhere to go
                    # -- ask for a smaller scene rather than abandoning the
                    # episode.
                    budget_i *= 0.8
                    continue
                traj = build_ee_trajectory(
                    strokes, rng, lang_cfg,
                    max_start_z=TatbotDrawEnv.MAX_TOOL_Z_CENTER - tops[i],
                )
                if verb == "erase":
                    # Retrace until the episode lasts as long as the operator's.
                    # Budgeting the scene and hoping the passes added up left
                    # the sim at a 14 s median against their 39, because the
                    # scene sampler under-fills its budget. Passes are the right
                    # knob anyway: retracing is what actually clears the ink.
                    #
                    # The marginal cost of a pass is MEASURED rather than
                    # modelled. Subtracting a nominal overhead from a one-pass
                    # build looks equivalent and is not: on a small scene the
                    # overhead is the larger term, the estimate collapses to its
                    # floor, and the pass count pins to its ceiling and overruns
                    # the horizon.
                    two = build_ee_trajectory(
                        strokes * 2, rng, lang_cfg,
                        max_start_z=TatbotDrawEnv.MAX_TOOL_Z_CENTER - tops[i],
                    )
                    per_pass = max(len(two) - len(traj), 1)
                    want = 1 + int(round((target_s * 30 - len(traj)) / per_pass))
                    passes = max(erase_passes[0], min(erase_passes[1], want))
                    # and it still has to fit the horizon, whatever the target
                    passes = max(1, min(passes,
                                        1 + (horizon - n_app - len(traj)) // per_pass))
                    traj = (traj if passes == 1 else two if passes == 2 else
                            build_ee_trajectory(
                                strokes * passes, rng, lang_cfg,
                                max_start_z=TatbotDrawEnv.MAX_TOOL_Z_CENTER - tops[i],
                            ))
                drawn = strokes * passes
                dip_plan_i = _plan_dips(inkctx, i, drawn, traj, surface, lang_cfg)
                built_len = len(traj) + _dip_budget(inkctx, i, dip_plan_i, drawn, surface, lang_cfg)
                if built_len <= horizon - n_app:
                    break
                budget_i *= 0.8
            else:
                raise SceneTooLongError(task, built_len, horizon - n_app)
            trajs.append(traj)
            dip_plans.append(dip_plan_i)
            program["passes"] = passes
            programs.append(program)
            kinds.append(task)
            # the scene ONCE: what the sheet opens with, and what a scorer
            # compares against — not the repeated tracing path
            paths.append([[[float(v) for v in pt] for pt in st.points] for st in strokes])
            if preink is not None:
                preink.append(strokes)
        worlds = [
            _world_with_dips(inkctx, i, traj.positions, traj.stroke_starts, dip_plans[i],
                             surface, draw_clearance)
            for i, traj in enumerate(trajs)
        ]
        draw_horizon = min(horizon - n_app, max(len(w.positions) for w in worlds) + 4)
        naturals = [min(len(w.positions) + 4, draw_horizon) for w in worlds]
        targets, surf_points, surf_normals, dip_mask, dip_credits, dips = _pack(
            worlds, draw_horizon, task, horizon - n_app)
    elif tasks.is_dip(task):
        # A dip episode: hover over the sheet, leave for the palette, charge,
        # come back to the same hover. No stroke, no mark — the tool opens
        # EMPTY (that is the point) and the one dip is a session start.
        if inkctx is None:
            raise ValueError(
                "--task dip needs a tool that dips and a palette in the scene "
                "(dr.palette.enabled); the laser never dips")
        ink_initial = np.zeros(num_envs, dtype=np.float32)
        worlds, naturals = [], []
        settle_n = max(1, int(round(shape_cfg.settle_time * 30)))
        for i in range(num_envs):
            lo, hi = shape_cfg.start_height_range
            z = float(rng.uniform(lo, hi))
            cap_z = TatbotDrawEnv.MAX_TOOL_Z_CENTER - tops[i]
            z = min(z, cap_z)
            xy = rng.uniform(-shape_cfg.center_range, shape_cfg.center_range, 2)
            hold = np.repeat(np.array([[xy[0], xy[1], z]], dtype=np.float32), settle_n, axis=0)
            need = [inkctx.ink.StrokeNeed(contact_mm=0.0, contact_s=0.0, ink_id=None)]
            plans = inkctx.ink.plan_dips(need, inkctx.policies[i], inkctx.palette, inkctx.load,
                                         arm="right", initial_charge_ul=0.0, tool_id=inkctx.tool_id)
            world = _world_with_dips(inkctx, i, hold, [0], plans, surface, draw_clearance)
            worlds.append(world)
            naturals.append(len(world.positions) + 4)
            slot = plans[0].slot_id if plans else None
            ink_id = plans[0].ink_id if plans else None
            ink = inkctx.inks.get(ink_id) if ink_id else None
            prompt = dip_task_name.format(
                tool=tools.active_tool().prompt_phrase,
                ink=(ink.display_name.lower().replace(" ink", "") if ink else "empty"))
            kinds.append("dip")
            paths.append([])
            programs.append({"prompt": prompt, "kind": "dip", "slot": slot, "ink": ink_id,
                             "hover_canvas_m": [float(v) for v in hold[0]]})
        draw_horizon = min(horizon - n_app, max(len(w.positions) for w in worlds) + 4)
        naturals = [min(n, draw_horizon) for n in naturals]
        targets, surf_points, surf_normals, dip_mask, dip_credits, dips = _pack(
            worlds, draw_horizon, task, horizon - n_app)
    else:
        draw_horizon = horizon - n_app
        naturals = []
        worlds = []
        sizes_mm: dict[int, int] = {}
        for i in range(num_envs):
            if task == "maze":
                traj_cfg = dataclasses.replace(shape_cfg, draw_speed_range=maze_cfg.draw_speed_range)
                # sample the speed FIRST so the walk length can be budgeted
                # against the horizon: ink steps ~ segments * pitch / speed,
                # and an unbudgeted walk was silently cut mid-stroke in ~10%
                # of episodes at the production horizon (drawn squiggle
                # shorter than the path recorded for the judge)
                speed = float(rng.uniform(*maze_cfg.draw_speed_range))
                avail = draw_horizon - overhead_steps(traj_cfg)
                max_seg = int(avail / 30.0 * speed / sheets[i]["pitch_m"]) - 1
                strokes = sample_maze(rng, sheets[i], maze_cfg, max_segments=max_seg,
                                      reachable=None if reachable is None else reachable[i])
                kind = "maze"
            else:
                kind, strokes, extent_m = sample_shape(rng, shape_cfg)
                sizes_mm[i] = int(round(extent_m * 1000))
                traj_cfg = shape_cfg
                speed = float(rng.uniform(*shape_cfg.draw_speed_range))
            # dips cost steps too: budget them out of the strokes' share of
            # the horizon before the fit, from the strokes as sampled (a fit
            # only ever shortens them, so it never needs MORE dips)
            dip_plan_i = _plan_dips(inkctx, i, strokes, None, surface, traj_cfg, speed=speed)
            dip_budget = _dip_budget(inkctx, i, dip_plan_i, strokes, surface, traj_cfg)
            # The budget is an estimate from the strokes as sampled; the fit
            # moves their starts and the splice measures the real distance,
            # so on an overrun take the measured excess off the strokes'
            # share and fit again rather than skipping the batch for a
            # handful of steps.
            world = None
            for _ in range(4):
                positions, strokes, natural, starts = fit_strokes(
                    strokes, rng, traj_cfg, speed, draw_horizon - dip_budget,
                    max_start_z=TatbotDrawEnv.MAX_TOOL_Z_CENTER - tops[i],
                    grid_walk=(kind == "maze"),
                )
                dip_plan_i = _plan_dips(inkctx, i, strokes, None, surface, traj_cfg, speed=speed)
                world = _world_with_dips(inkctx, i, positions[:natural], starts, dip_plan_i,
                                         surface, draw_clearance)
                if len(world.positions) <= draw_horizon:
                    break
                dip_budget += len(world.positions) - draw_horizon + 4
            if world is None:
                raise RuntimeError("Failed to build world trajectory")
            worlds.append(world)
            naturals.append(min(len(world.positions) + 4, draw_horizon))
            kinds.append(kind)
            paths.append([[float(v) for v in pt] for pt in strokes[0].points])
        programs = [None] * num_envs
        targets, surf_points, surf_normals, dip_mask, dip_credits, dips = _pack(
            worlds, draw_horizon, task, draw_horizon)

    lean_profiles = [
        sample_lean_profile(rng, draw_horizon, dr.pen_lean.max_rad, dr.pen_lean.keypoints)
        for _ in range(num_envs)
    ]
    # lean about the LOCAL normal at each step: perpendicular to the skin is
    # the behaviour being demonstrated, and the perturbation rides on top of it
    pen_normals = np.stack([
        cap_lean(lean_normals(surf_normals[i], lean_profiles[i]),
                 surface.base_normal_np(i), dr.pen_lean.max_off_base_rad)
        for i in range(num_envs)
    ])

    q_raised = None
    if do_approach:
        q_raised = STAGED_POSE[None, :] + rng.normal(
            0, dr.approach.pose_jitter_rad, (num_envs, 6)
        ).astype(np.float32)

    task_strings = [
        prog["prompt"] if prog is not None and kinds[i] in ("language", "erase", "dip")
        else (maze_task_name if kinds[i] == "maze"
              else task_name.format(shape=kinds[i], size_mm=sizes_mm.get(i, 0)))
        for i, prog in enumerate(programs)
    ]
    lengths = np.asarray([n_app + n for n in naturals], dtype=np.int64)
    return BatchPlan(n_app, q_raised, draw_horizon, targets, pen_normals,
                     surf_points, surf_normals,
                     lean_profiles, kinds, task_strings, paths, programs, lengths, preink,
                     dip_mask, dip_credits, dips, ink_initial, ink_capacity)


# --- ink: dips in the plan -------------------------------------------------------------

@dataclasses.dataclass
class _InkContext:
    """What plan_batch needs to put dips into a batch, resolved once."""

    ink: object                 # the ink_spec module
    policy: object              # InkPolicy of the fitted tool (datasheet)
    palette: dict               # slot -> PaletteSlot
    load: dict                  # slot -> SlotLoad
    cap_rims: dict              # slot -> (B, 3) world rim centres
    palette_dr: object          # PaletteDR
    tool_id: str
    inks: dict                  # ink_id -> Ink (per-ink dip overrides live here)
    policies: list              # per-env InkPolicy, capacity-scaled (InkDR.capacity_scale)
    initial_ul: np.ndarray      # (B,) charge the tool opens with (InkDR.initial_charge_frac)


def _ink_context(task: str, cap_rims, dr, num_envs: int, rng: np.random.Generator | None = None):
    """None unless this task deposits, the tool dips, the env placed a
    palette, and dips are wanted: InkDR.dips for a drawing task, always for
    ``--task dip``. Missing caps for a dipping tool that is asked to dip is a
    refusal, not a silent no-dip run: the dataset would say "3RL" and never
    show a dip."""
    req = tasks.TASK_REQUIREMENTS.get(task)
    if req is None or not req.needs_ink:
        return None
    if not (dr.ink.dips or tasks.is_dip(task)):
        return None
    policy = tools.active_ink_policy()
    if not policy.dips:
        return None
    if cap_rims is None:
        return None
    ink = tools.ink_registry()
    palette = ink.load_palette(tools.REPO)
    load = tools.palette_load()
    missing = [s for s in palette if s not in cap_rims]
    if missing:
        raise ValueError(f"env placed no cap for palette slot(s) {missing}")
    rng = rng if rng is not None else np.random.default_rng()
    scale = rng.uniform(dr.ink.capacity_scale[0], dr.ink.capacity_scale[1], num_envs)
    policies = [dataclasses.replace(policy, charge_capacity_ul=policy.charge_capacity_ul * float(k),
                                    uptake_ul=policy.uptake_ul * float(k)) for k in scale]
    frac = rng.uniform(dr.ink.initial_charge_frac[0], dr.ink.initial_charge_frac[1], num_envs)
    initial = np.asarray([p.charge_capacity_ul * float(f) for p, f in zip(policies, frac, strict=True)],
                         dtype=np.float32)
    return _InkContext(ink, policy, palette, load, cap_rims, dr.palette,
                       tools.active_tool().tool_id, ink.load_inks(tools.REPO), policies, initial)


def _ink_opening(ctx, num_envs: int):
    """How every env's tool opens the episode, for the env: charge and
    capacity. Without an ink context the tool is simply full — a drawing
    episode with dips off draws at full opacity."""
    policy = tools.active_ink_policy()
    if not policy.dips:
        return None, None
    if ctx is None:
        cap = np.full(num_envs, policy.charge_capacity_ul, dtype=np.float32)
        return cap.copy(), cap
    return ctx.initial_ul.copy(), np.asarray([p.charge_capacity_ul for p in ctx.policies], dtype=np.float32)


def _plan_dips(ctx, env_index: int, strokes, traj, surface, cfg, speed: float | None = None):
    if ctx is None:
        return []
    if speed is None:
        # the language path draws at a speed sampled inside build_ee_trajectory;
        # the middle of its range is the budget estimate, and the ledger
        # records what the env actually spent
        speed = float(np.mean(cfg.draw_speed_range))
    needs = dipping.stroke_needs(strokes, speed, cfg)
    return ctx.ink.plan_dips(needs, ctx.policies[env_index], ctx.palette, ctx.load, arm="right",
                             initial_charge_ul=float(ctx.initial_ul[env_index]),
                             tool_id=ctx.tool_id, inks=ctx.inks)


def _geometry(ctx, env_index: int, plan, cfg):
    slot = ctx.palette[plan.slot_id]
    # the cap as this dip finds it: real caps drain dip by dip, so a later
    # plunge goes deeper than the first (ink_spec.plan_dips carries it)
    fill = plan.cap_fill_ul if plan.cap_fill_ul else ctx.load[plan.slot_id].fill_ul
    # the ink's own dip (inks.yaml dip:) refines the datasheet's depth and dwell
    policy = ctx.ink.policy_with_ink(ctx.policies[env_index], ctx.inks.get(plan.ink_id) if plan.ink_id else None)
    return dipping.DipGeometry(
        rim_world=np.asarray(ctx.cap_rims[plan.slot_id][env_index], dtype=np.float64),
        plunge_m=ctx.ink.dip_plunge_m(policy, slot, fill),
        cap_depth_m=slot.size.depth_m,
        hover_m=ctx.palette_dr.hover_m,
        dwell_s=policy.dip_dwell_s,
        plunge_speed=ctx.palette_dr.plunge_speed,
        travel_speed=cfg.travel_speed,
        settle_time=cfg.settle_time,
    )


def _dip_budget(ctx, env_index: int, plans, strokes, surface, cfg) -> int:
    """Steps the dips will add, estimated from the hover above each stroke's
    start — the point the splice actually leaves from — plus a small margin
    for the rounding between this estimate and the built trajectory."""
    if not plans:
        return 0
    total = 0
    for plan in plans:
        start_xy = np.asarray(strokes[plan.before_stroke].points[:1], dtype=np.float32)
        pts, nms = surface.frame_np(env_index, start_xy)
        hover = pts[0] + cfg.hover_height * nms[0]
        total += dipping.dip_steps(hover, _geometry(ctx, env_index, plan, cfg), 1.0 / 30.0) + 12
    return total


def _world_with_dips(ctx, env_index: int, positions, stroke_starts, plans, surface, clearance):
    """Canvas trajectory -> world, with this env's dips spliced in."""
    tw, pts, nms = canvas_to_world(positions, surface, env_index, clearance)
    if ctx is None or not plans:
        return dipping.Spliced(tw, pts, nms, np.zeros(len(tw), dtype=bool), [], [])
    cfg = ShapeConfig()
    return dipping.splice(tw, pts, nms, stroke_starts, plans,
                          lambda plan: _geometry(ctx, env_index, plan, cfg), 1.0 / 30.0)


def _pack(worlds, draw_horizon: int, task: str, available: int):
    """Per-env spliced trajectories -> the batch arrays, padded by holding
    the final pose. A trajectory longer than the horizon is refused the way
    an unfittable scene is: the budget said the dips would fit, and cutting
    one mid-plunge is worse than redrawing the batch."""
    b = len(worlds)
    targets = np.zeros((b, draw_horizon, 3), dtype=np.float32)
    surf_points = np.zeros((b, draw_horizon, 3), dtype=np.float32)
    surf_normals = np.zeros((b, draw_horizon, 3), dtype=np.float32)
    dip_mask = np.zeros((b, draw_horizon), dtype=bool)
    credits, dips = [], []
    for i, w in enumerate(worlds):
        n = len(w.positions)
        if n > draw_horizon:
            raise SceneTooLongError(task, n, available)

        targets[i] = _padded(w.positions, draw_horizon)
        surf_points[i] = _padded(w.floor_points, draw_horizon)
        surf_normals[i] = _padded(w.floor_normals, draw_horizon)
        dip_mask[i] = _padded(w.dip_mask, draw_horizon, fill=False)
        credits.append(list(w.credit_steps))
        dips.append(list(w.dips))
    any_dips = any(dips)
    return (targets, surf_points, surf_normals,
            dip_mask if any_dips else None, credits if any_dips else None,
            dips if any_dips else None)



def _padded(a: np.ndarray, length: int, fill=None) -> np.ndarray:
    n = len(a)
    if n == length:
        return a
    tail = (np.repeat(a[-1:], length - n, axis=0) if fill is None
            else np.full((length - n,) + a.shape[1:], fill, dtype=a.dtype))
    return np.concatenate([a, tail])
