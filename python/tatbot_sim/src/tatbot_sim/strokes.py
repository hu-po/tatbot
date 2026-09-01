"""Minimal stroke abstraction for the sim data factory.

A stroke is a 2D polyline in the canvas frame (meters, origin at the pad
center, x/y in the pad plane). Shape generators emit lists of strokes;
``build_ee_trajectory`` turns them into a dense per-control-step end-effector
position trajectory (approach / draw / lift / travel segments included).

Deliberately tiny: no g-code, no surface meshes. When curved surfaces arrive,
canvas->world gains a mapping step and nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tatbot_sim.tools import active_tool


@dataclass
class Stroke:
    """One continuous pen-down polyline in the canvas frame."""

    points: np.ndarray  # (N, 2) xy in canvas frame, meters

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)


# fm2-measured pacing for the Lutin pen body (draw-square-fm2_20260831,
# drawing-phase tip speed p10/50/90 = 0.9/7.0/20.7 mm/s, travel p90 ~90 mm/s).
# The old (0.02, 0.05) band ran ~3x the operator. The laser is NOT in the
# family: its pacing was calibrated against its own real erase recordings
# (erase_seconds + measured pass cost), and slowing its strokes makes the
# smallest scene overrun the pass budget (measured: SceneTooLongError).
_LUTIN_PEN_BODY = ("lutin-ballpoint-dot", "lutin-3rl-bugpin")
_IS_LUTIN = active_tool().tool_id in _LUTIN_PEN_BODY
_DRAW_SPEED_RANGE = (0.004, 0.016) if _IS_LUTIN else (0.02, 0.05)
_TRAVEL_SPEED = 0.08 if _IS_LUTIN else 0.12
# MazeConfig's band also paces LANGUAGE scenes (plan_batch's lang_cfg borrows
# it), so it carries the same per-tool split: slowing it unconditionally made
# the laser's smallest erase scene overrun its pass budget.
_MAZE_SPEED_RANGE = (0.004, 0.014) if _IS_LUTIN else (0.012, 0.03)


@dataclass
class ShapeConfig:
    """Randomization ranges for the simple-shape generator."""

    kinds: tuple[str, ...] = ("square", "circle", "triangle", "line", "pentagon")
    # Sized to the arm's measured envelope over the real canvas: the tool
    # reaches every point within +/-60 mm of the pad centre, and up to ~60 mm
    # above it, with zero IK residual.
    size_range: tuple[float, float] = (0.02, 0.045)  # half-extent / radius, m
    center_range: float = 0.015  # max |xy| offset of shape center from pad center
    rotation: bool = True

    # Paced per fitted tool (module constants below): the Lutin pen body is
    # paced from the real fm2 teleop recording; the laser keeps the band its
    # own real recordings calibrated (via erase_seconds — slowing its strokes
    # would break the measured pass-count fit).
    draw_speed_range: tuple[float, float] = _DRAW_SPEED_RANGE
    travel_speed: float = _TRAVEL_SPEED  # m/s between strokes / from home
    hover_height: float = 0.025  # m above draw plane for travel between strokes
    # Every episode opens well clear of the surface and descends onto it, so
    # the policy sees an approach rather than starting already in contact.
    start_height_range: tuple[float, float] = (0.05, 0.10)
    approach_time: float = 0.5  # s for the vertical descend/lift ramps
    # brief holds around pen-down transitions: arriving at 0.12 m/s travel
    # speed and descending immediately inks a ~3 mm curl at the stroke start
    # while the servo is still swinging; the same happens in reverse at the
    # lift. Settling at hover before the descend and on the paper before the
    # lift lets the lateral motion die where it cannot mark the sheet.
    settle_time: float = 0.2


def _regular_polygon(n: int, size: float) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n + 1) + np.pi / n
    return np.stack([size * np.cos(ang), size * np.sin(ang)], axis=1)


def make_shape(kind: str, size: float, center: np.ndarray, angle: float) -> list[Stroke]:
    """Generate the strokes of a named shape, transformed into the canvas frame."""
    if kind == "square":
        pts = _regular_polygon(4, size * np.sqrt(2))
    elif kind == "triangle":
        pts = _regular_polygon(3, size)
    elif kind == "pentagon":
        pts = _regular_polygon(5, size)
    elif kind == "circle":
        pts = _regular_polygon(48, size)
    elif kind == "line":
        pts = np.array([[-size, 0.0], [size, 0.0]])
    else:
        raise ValueError(f"unknown shape kind: {kind}")
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, s], [-s, c]]) + center
    return [Stroke(pts)]


def sample_shape(rng: np.random.Generator, cfg: ShapeConfig) -> tuple[str, list[Stroke], float]:
    kind = cfg.kinds[rng.integers(len(cfg.kinds))]
    size = rng.uniform(*cfg.size_range)
    center = rng.uniform(-cfg.center_range, cfg.center_range, size=2)
    angle = rng.uniform(0, 2 * np.pi) if cfg.rotation else 0.0
    # the third element is the full extent, for prompts that state the size
    # the way the real recordings do ("draw a 6mm square ...")
    return kind, make_shape(kind, size, center, angle), 2.0 * size


@dataclass
class MazeConfig:
    """Squiggle paths traced along the sheet's printed 6 mm grid.

    The grid paper acts as a stencil: the path walks the lattice of rule
    intersections, so every stroke lies ON a printed line and quality is
    judged by how far the ink strays from the ruling. A self-avoiding walk
    with per-episode momentum samples a broad mix of stroke directions, run
    lengths and turn patterns — the base of data before richer line work
    (lettering, pixel art) arrives with its own prompts.
    """

    # keep nodes inside the arm's measured flat-reach envelope on the canvas
    reach: float = 0.06
    segments_range: tuple[int, int] = (15, 40)
    # slower than the shape task: 6 mm segments turn every quarter second at
    # shape speeds and the controller overshoots the corners off the ruling
    draw_speed_range: tuple[float, float] = _MAZE_SPEED_RANGE
    # chance of continuing straight at each node, sampled per episode from
    # this range: low = twisty scribble, high = long runs with few corners
    momentum_range: tuple[float, float] = (0.3, 0.85)


def sample_maze(
    rng: np.random.Generator, sheet: dict, cfg: MazeConfig,
    max_segments: int | None = None, reachable=None,
) -> list[Stroke]:
    """Self-avoiding lattice walk on the sheet's major-line intersections.

    ``sheet`` comes from textures.grid_paper_sheets: ``xs``/``ys`` are the
    major lines' canvas coordinates (metres, centred). Retries until a walk
    of at least the minimum segment count fits; a walk that corners itself
    early is kept if long enough. ``max_segments`` caps the walk so the ink
    time fits an episode's step budget — before this cap existed, ~10% of
    walks at the production horizon were silently cut mid-stroke, leaving
    the drawn squiggle shorter than the path recorded for the judge.
    """
    xs = [x for x in sheet["xs"] if abs(x) <= cfg.reach]
    ys = [y for y in sheet["ys"] if abs(y) <= cfg.reach]
    # Nodes the fitted tool cannot be held normal over are not walkable. On a
    # flat pad that is every node; on a mound the walk keeps to the ground it
    # can actually work.
    walkable = (np.ones((len(xs), len(ys)), dtype=bool) if reachable is None else
                np.array([[reachable.node_ok(x, y) for y in ys] for x in xs], dtype=bool))
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(
            f"sheet ruling too coarse for the reach envelope: {len(xs)}x{len(ys)} lines"
        )
    lo, hi = cfg.segments_range
    if max_segments is not None:
        hi = min(hi, max_segments)
        lo = min(lo, hi)
    for _ in range(50):
        want = int(rng.integers(lo, hi + 1))
        momentum = rng.uniform(*cfg.momentum_range)
        start = np.argwhere(walkable)
        if not len(start):
            raise ValueError("no reachable node on the ruling for the fitted tool")
        node = tuple(int(v) for v in start[rng.integers(len(start))])
        path = [node]
        used_edges: set[tuple] = set()
        last_dir = None
        while len(path) - 1 < want:
            i, j = path[-1]
            moves = []
            recent = set(path[-4:])
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + d[0], j + d[1]
                if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                    continue
                if not walkable[ni, nj]:
                    continue
                # no unit-cell circuits: the ~3 mm ink stroke nearly fills a
                # 6 mm cell, which reads as a blob rather than a line. Larger
                # loops stay legal and render cleanly.
                if (ni, nj) in recent:
                    continue
                edge = ((i, j), (ni, nj)) if (i, j) < (ni, nj) else ((ni, nj), (i, j))
                if edge in used_edges:
                    continue
                moves.append(((ni, nj), edge, d))
            if not moves:
                break
            straight = [m for m in moves if m[2] == last_dir]
            if straight and rng.random() < momentum:
                node, edge, last_dir = straight[0]
            else:
                turns = [m for m in moves if m[2] != last_dir] or moves
                node, edge, last_dir = turns[int(rng.integers(len(turns)))]
            used_edges.add(edge)
            path.append(node)
        if len(path) - 1 >= lo:
            pts = np.array([[xs[i], ys[j]] for i, j in path], dtype=np.float32)
            return [Stroke(pts)]
    raise RuntimeError("could not sample a squiggle walk — check reach vs grid pitch")


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at constant arc-length spacing ``step``."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < 1e-6:
        return points[:1]
    n = max(2, int(np.ceil(arc[-1] / step)) + 1)
    t = np.linspace(0, arc[-1], n)
    return np.stack([np.interp(t, arc, points[:, i]) for i in range(2)], axis=1)


@dataclass
class EETrajectory:
    """Dense EE position targets, one per control step, in the canvas frame."""

    positions: np.ndarray  # (T, 3) xyz; z is height above the draw plane
    pen_down: np.ndarray = field(default=None)  # (T,) bool, True while drawing
    # The step at which the trajectory begins travelling to stroke k, hovering
    # clear of the surface: the one place a dip (tatbot_sim.dipping) can be
    # spliced in without touching the stroke on either side.
    stroke_starts: list[int] = field(default_factory=list)

    def __len__(self):
        return len(self.positions)


def build_ee_trajectory(
    strokes: list[Stroke],
    rng: np.random.Generator,
    cfg: ShapeConfig,
    control_freq: float = 30.0,
    horizon: int | None = None,
    max_start_z: float | None = None,
    draw_speed: float | None = None,
) -> EETrajectory:
    """Chain strokes into a per-step EE trajectory: travel -> descend -> draw -> lift.

    Returns positions in the canvas frame with z=0 at the draw plane. The
    episode opens at a randomized 5-10 cm above the surface and descends, so
    approaching the skin is part of what the data teaches; pass ``max_start_z``
    to cap that opening height by what the arm can actually hold over this
    surface. ``draw_speed`` overrides the per-episode speed draw so a caller
    can budget stroke length against a step budget before building.

    If ``horizon`` is given, the trajectory is padded (holding the final
    hover pose) to exactly that many steps. A trajectory LONGER than the
    horizon raises: cutting it would end the episode mid-stroke with the
    drawn path shorter than the path the dataset records as ground truth —
    the caller must shorten the strokes instead (see planning._fit_strokes).
    """
    dt = 1.0 / control_freq
    if draw_speed is None:
        draw_speed = rng.uniform(*cfg.draw_speed_range)
    approach_steps = max(2, int(cfg.approach_time / dt))
    lo, hi = cfg.start_height_range
    if max_start_z is not None:
        hi = max(lo, min(hi, max_start_z))
    start_z = rng.uniform(lo, hi)

    pos: list[np.ndarray] = []
    down: list[bool] = []
    starts: list[int] = []

    def _extend(a: np.ndarray, b: np.ndarray, speed: float, pen: bool):
        dist = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(dist / (speed * dt))))
        for i in range(1, n + 1):
            pos.append(a + (b - a) * (i / n))
            down.append(pen)

    # open high above the surface, then descend to travel height
    cur = np.array([0.0, 0.0, start_z])
    pos.append(cur.copy())
    down.append(False)
    _extend(cur, np.array([0.0, 0.0, cfg.hover_height]), cfg.travel_speed, False)
    cur = np.array([0.0, 0.0, cfg.hover_height])

    for stroke in strokes:
        pts = _resample(stroke.points, draw_speed * dt)
        starts.append(len(pos) - 1)
        # travel at hover height, settle, then descend
        _extend(cur, np.array([*pts[0], cfg.hover_height]), cfg.travel_speed, False)
        for _ in range(max(1, int(cfg.settle_time / dt))):
            pos.append(np.array([*pts[0], cfg.hover_height]))
            down.append(False)
        for i in range(1, approach_steps + 1):
            z = cfg.hover_height * (1 - i / approach_steps)
            pos.append(np.array([*pts[0], z]))
            down.append(False)
        # draw
        for p in pts[1:]:
            pos.append(np.array([*p, 0.0]))
            down.append(True)
        # settle, then lift
        for _ in range(max(1, int(cfg.settle_time / dt))):
            pos.append(np.array([*pts[-1], 0.0]))
            down.append(True)
        # lift
        for i in range(1, approach_steps + 1):
            z = cfg.hover_height * (i / approach_steps)
            pos.append(np.array([*pts[-1], z]))
            down.append(False)
        cur = np.array([*pts[-1], cfg.hover_height])

    positions = np.stack(pos).astype(np.float32)
    pen_down = np.array(down, dtype=bool)
    if horizon is not None:
        if len(positions) > horizon:
            raise ValueError(
                f"trajectory needs {len(positions)} steps but the horizon is "
                f"{horizon}: shorten the strokes rather than cutting mid-stroke"
            )
        pad = horizon - len(positions)
        if pad:
            positions = np.concatenate([positions, np.repeat(positions[-1:], pad, axis=0)])
            pen_down = np.concatenate([pen_down, np.zeros(pad, dtype=bool)])
    return EETrajectory(positions=positions, pen_down=pen_down, stroke_starts=starts)


def overhead_steps(cfg: ShapeConfig, control_freq: float = 30.0) -> int:
    """Upper bound on the non-ink steps of a single-stroke episode: opening
    descent from the highest start, travel to the farthest stroke start in
    the reach envelope, both settles and both descend/lift ramps."""
    secs = (
        cfg.start_height_range[1] / cfg.travel_speed
        + 0.09 / cfg.travel_speed
        + 2 * cfg.settle_time
        + 2 * cfg.approach_time
    )
    return int(np.ceil(secs * control_freq)) + 4


def fit_strokes(
    strokes: list[Stroke],
    rng: np.random.Generator,
    cfg: ShapeConfig,
    speed: float,
    draw_horizon: int,
    max_start_z: float,
    grid_walk: bool,
) -> tuple[np.ndarray, list[Stroke], int, list[int]]:
    """Build the dense trajectory, shortening the strokes if the budgeted
    build still overruns (ceil rounding can add a few steps). Returns
    (positions padded to draw_horizon, strokes actually drawn, natural
    length before padding) — strokes and positions are kept consistent, so
    run_meta never records a longer path than was drawn, and the natural
    length lets the generator close the episode when the drawing ends
    instead of recording the padded hold. Grid walks shrink by dropping
    trailing nodes (stays on the ruling); shapes shrink toward their own
    centres (stays the named shape)."""
    for _ in range(30):
        traj = build_ee_trajectory(
            strokes, rng, cfg, max_start_z=max_start_z, draw_speed=speed
        )
        if len(traj) <= draw_horizon:
            pad = draw_horizon - len(traj)
            posn = traj.positions
            if pad:
                posn = np.concatenate([posn, np.repeat(posn[-1:], pad, axis=0)])
            return posn, strokes, len(traj), traj.stroke_starts
        if grid_walk:
            pts = strokes[0].points
            if len(pts) <= 3:
                break
            strokes = [Stroke(pts[:-2])]
        else:
            strokes = [
                Stroke((s.points - s.points.mean(0)) * 0.88 + s.points.mean(0))
                for s in strokes
            ]
    raise RuntimeError(f"could not fit strokes into a {draw_horizon}-step horizon")


def pacing_estimate(cfg: ShapeConfig, draw_speed_range: tuple[float, float]) -> tuple[float, float, float]:
    """(draw_speed, per-stroke overhead s, per-episode overhead s) for time
    budgeting — derived from the trajectory builder's own constants so the
    estimate cannot drift from the pacing it estimates (the drawing language
    used to carry a hand-mirrored copy).

    Draw speed uses the range's midpoint; per-stroke overhead is descend +
    lift plus both settle holds plus a travel allowance; episode overhead is
    the opening descent from the start height at travel speed plus a settle.
    """
    draw_speed = 0.5 * (draw_speed_range[0] + draw_speed_range[1])
    stroke_overhead = 2 * cfg.approach_time + 2 * cfg.settle_time + 0.4
    episode_overhead = cfg.start_height_range[1] / cfg.travel_speed + cfg.approach_time + 1.0
    return draw_speed, stroke_overhead, episode_overhead
