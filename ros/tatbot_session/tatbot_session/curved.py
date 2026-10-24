"""A curved page from the page setup's own measurements (page.surface.mode fitted; ros/README.md 4.3).

The gauge holds already stand the pen over the drawing's corners and centre, and each reads the skin under the tip
(gauge.measure curved: a quadric over the skin it sees) and keeps that skin (Frame.patch). Those patches, carried
into the arm's base by the hold's pose and pinned at the tip (the camera's mounting moves them, the tip's place does
not), are the surface's samples; the gauge's own points under the tips weigh most. tatbot_motion.surface fits the
height field over the page set up through them. A flat page gives a flat field, which plans as the plane did.

The wrist camera's mounting is turned a few degrees from its CAD pose, which turns every patch about its tip: it is
calibrated once from curved setups (`ros2 run tatbot_session gauge turn RUN... --write`, fit_turn) into
wrist-turn-<arm>.json and taken out of every hold's skin in the camera's frame (hold_patch); without that file each
setup fits it again (camera_tilt).

Nothing here moves the arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tatbot_session import geometry

# The overhead's watch of a curved page (page.track's follow_* in its stead): the overhead's match of a print on a
# raised surface scatters +-3-4 mm and +-5 deg pose to pose (the forearm, 2026-10-07; a flat sheet's, ~1 mm and whole
# degrees), and the flat page's 5.7 deg set one page up again after a single stroke on a still forearm. It follows
# nothing it would not set up again for: a 4 deg limit followed a still forearm's noise by 4.8 deg, a 4 mm one by
# 4.4 mm, and the swallow's last strokes went down that far off its first (2026-10-07); past 12 mm or 12 deg the
# forearm moved, and is set up again.
WATCH = {"follow_m": 0.012, "follow_rad": 0.21, "follow_max_m": 0.024, "follow_max_rad": 0.42}   # owner 2026-10-20: x2
# The wrist camera's rigid follow of a curved page (page.surface.rigid, RigidWatch, ArmExecutor._rigid_follow): the
# stencil read at hover stops and lifts against where the setup's holds read it, the whole surface moved with it.
RIGID = {"enabled": True,
         "follow_m": 0.002,       # a reading this far from the setup's is a move (one under it says the stencil stays)
         "confirm_m": 0.0015,     # ... confirmed by a second reading, at least apart_m away, on one rigid move within this
         "apart_m": 0.003,
         "spread_m": 0.010,       # readings spread this far apart give the move's turn too; else a slide alone
         "max_m": 0.010,          # the strokes moved farther than this since the holds: set up again (holds, field)
         "max_rad": 0.175,        # ... or turned more than this (10 deg; each doubled with every stop, 2026-10-20)
         "ambiguity_m": 0.001,    # a slid skin and a print turned round the arm read alike in print mm: past this
         "ambiguity_rad": 0.10,   # apart (or their normals this far) the strokes would go down on a guess: set up again
         "skin_m": 0.002,         # the hover's skin about the tip this far off the field (net of the pen's trim and
         "skin_stop_m": 0.0025,   # rise): a warning; its skin's part this far over it (deeper) or skin_stop_under_m
         "skin_stop_under_m": 0.005,   # under it: set up again (ArmExecutor._skin_watch: keys its way never count)
         "hover_frames": 5,       # depth frames at a curved hover stop, pooled (gauge.pool)
         # A stroke rides the skin its hover reads along its own path (skin_along, ride_profile), not one number: the
         # tiny heart's skin sank 1.2-1.3 mm on one lobe against the other under one field (2026-10-21)
         "profile_lower_m": 0.001,   # it goes at most this much under its plan where the skin reads under (owner)
         "profile_setup_m": 0.004,   # the skin along it this far off the field: a move, set the page up again (owner;
                                     # 2 mm until 2026-10-23, when a 2.35 mm read re-fit the page 4.5 deg off and
                                     # the stroke's plan met joint_3's limit on a snake the ride was following)
         "profile_near_m": 0.002,    # a path point's skin: the skin points within this of it on the page plane
         "profile_smooth_m": 0.003,  # ... their medians smoothed by a robust line over this much arc either side
         "profile_slope": 0.5,       # ... and its change along the path no steeper (raised to it): no step into skin
         "profile_keep_s": 60.0,     # where its own views read none, earlier strokes' frames this young: the heart's
                                     # skin drifted ~0.8 mm a minute (0.4 to 1.2 mm off level, 15:44:31-15:45:35)
         "profile_agree_m": 0.001}   # ... and only one whose skin agrees within this with the own views' where both
                                     # read (lent_off): that minute's drift plus its 0.03-0.10 mm scatter; past it
                                     # the skin moved more than it drifts, and those points ride the fallback
PATCH_POINTS = 1500       # each hold's surface points kept for the fit (a random subset, fixed seed)
POINT_WEIGHT = 50.0       # a gauge point's weight against one patch point's


def config(stack: dict) -> dict:
    """page.surface with its defaults; mode plane (the default) is the flat page every run used."""
    cfg = dict(stack["page"].get("surface") or {})
    cfg.setdefault("mode", "plane")
    cfg.setdefault("degree", 4)
    cfg.setdefault("max_loo_m", 0.0005)
    cfg.setdefault("reach_m", 0.020)
    cfg.setdefault("check", True)
    cfg.setdefault("frames", 15)
    cfg.setdefault("pool", 1)
    cfg.setdefault("track_correct", False)
    cfg.setdefault("turn_pen", False)
    cfg.setdefault("along_skin", False)
    cfg.setdefault("turned_checks", 2)
    for key, value in WATCH.items():
        cfg.setdefault(key, value)
    cfg["rigid"] = {**RIGID, **(cfg.get("rigid") or {})}
    return cfg


def in_plane(m) -> np.ndarray:
    """A 3 x 3 in-plane rigid move (page x, y) as the 4 x 4 in the page frame (z untouched)."""
    out = np.eye(4)
    out[:2, :2], out[:2, 3] = np.asarray(m, float)[:2, :2], np.asarray(m, float)[:2, 2]
    return out


def rigid_fit(src, dst, spread_m: float) -> np.ndarray:
    """The in-plane rigid move M (3 x 3) taking points src to dst (N, 2): a turn and a slide (Kabsch) when the points
    spread spread_m apart; closer together their turn is noise, and the move is their mean slide."""
    src, dst = np.asarray(src, float).reshape(-1, 2), np.asarray(dst, float).reshape(-1, 2)
    m = np.eye(3)
    a, b = src.mean(axis=0), dst.mean(axis=0)
    if len(src) >= 2 and np.max(np.linalg.norm(src - a, axis=1)) * 2 >= spread_m:
        u, _, vt = np.linalg.svd((src - a).T @ (dst - b))
        d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
        m[:2, :2] = vt.T @ np.diag([1.0, d]) @ u.T
    m[:2, 2] = b - m[:2, :2] @ a
    return m


def moved_by(m, xy) -> np.ndarray:
    return np.asarray(xy, float).reshape(-1, 2) @ np.asarray(m, float)[:2, :2].T + np.asarray(m, float)[:2, 2]


class RigidWatch:
    """Whether the stencil on a curved page has moved, from the wrist camera's readings of the tip on the print at hover
    stops and lifts. Each reading is a pair: where the pen's print point is (act), and where it would be had the stencil
    stayed (exp: the TCP's print point plus the residual the setup's gauge holds read there, which carries the arm's
    and the reading's own error at that place), both in print mm (uv) and on the page's base plane (xy).

    A reading within follow_m of the setup's says the stencil stays, and clears any held. One past it is held until a
    second, taken apart_m away at least (two frames from one place repeat one misread), agrees with it on one rigid
    move within confirm_m: then 'moved', and `step` is that move. The follow it drives moves the whole page surface
    rigidly with the skin (ArmExecutor._rigid_follow), never a stroke's points: the per-stroke field correction carried
    a 20 mm stroke 5 mm past the holds on 2026-10-08, its affine part reading the print's 15-20% stretch across a curve
    as a move."""

    def __init__(self, follow_m=RIGID["follow_m"], confirm_m=RIGID["confirm_m"], apart_m=RIGID["apart_m"],
                 spread_m=RIGID["spread_m"], **_):
        self.follow, self.confirm, self.apart, self.spread = float(follow_m), float(confirm_m), float(apart_m), float(spread_m)
        self.held: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []   # (act_xy, exp_xy, act_uv, exp_uv)
        self.agreed = 0   # how many of the newest held readings agree on one move (0: no move confirmed)

    def add(self, act_xy, exp_xy, act_uv, exp_uv) -> str:
        """'added' (the stencil where the setup read it), 'held' (moved, not yet confirmed) or 'moved'."""
        reading = tuple(np.asarray(v, float).reshape(2) for v in (act_xy, exp_xy, act_uv, exp_uv))
        if np.linalg.norm(reading[1] - reading[0]) < self.follow:
            self.clear()
            return "added"
        if self.held and np.linalg.norm(reading[0] - self.held[-1][0]) < self.apart:
            return "moved" if self.agreed else "held"   # the same place again: no second view
        self.held = [*self.held[-3:], reading]
        self.agreed = max((n for n in range(2, len(self.held) + 1) if self._agree(self.held[-n:])), default=0)
        return "moved" if self.agreed else "held"

    def _agree(self, readings) -> bool:
        act, exp = np.array([r[0] for r in readings]), np.array([r[1] for r in readings])
        m = rigid_fit(act, exp, self.spread)
        return bool(np.max(np.linalg.norm(moved_by(m, act) - exp, axis=1)) <= self.confirm)

    @property
    def confirmed(self) -> bool:
        return self.agreed >= 2

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """(the move on the base plane, the same move in print mm), each a 3 x 3 taking act to exp, from the readings
        that agree; the identity before a move is confirmed."""
        if not self.confirmed:
            return np.eye(3), np.eye(3)
        use = self.held[-self.agreed:]
        return tuple(rigid_fit([r[k] for r in use], [r[k + 1] for r in use], self.spread) for k in (0, 2))

    def clear(self) -> None:
        self.held, self.agreed = [], 0


def ambiguity(surface, plane_step, print_step, uv) -> tuple[float, float]:
    """How far apart two readings of one move put the print points uv (metres, and their normals in rad): the skin
    carried by plane_step on the base plane with the print on it (what the rigid follow draws on), or the skin where it
    was with the print slid over it by print_step in print mm. A forearm turned about its own axis (a roll) moves the
    print round the curve while its shape stays: the print's reading is the same, and only the holds can tell them
    apart. Along a cylinder's axis, or on a flat page, the two agree."""
    from tatbot_motion.surface import PageSurface

    uv = np.asarray(uv, float).reshape(-1, 2)
    carried = PageSurface(np.asarray(surface.base, float) @ in_plane(plane_step), surface.height, surface.chart)
    pb, nb = carried.lift(uv)
    pa, na = surface.lift(moved_by(print_step, uv))
    return (float(np.max(np.linalg.norm(pb - pa, axis=1))),
            float(np.max(np.arccos(np.clip(np.sum(nb * na, axis=1), -1.0, 1.0)))))


def disagree(read: dict) -> float:
    """How far a gauge reading's frames disagree: all of them; on a curved page (its record keeps each frame's
    height) the middle ones, the median's, a quarter of them (one at least) either end left out. On the forearm's
    silicone each frame scattered 0.2-0.5 mm (1 sigma) about the hold, so five frames' full spread passed 0.5 mm at 9
    of 14 holds whose middle three agreed within it (2026-10-07)."""
    heights = sorted(read.get("heights_m") or [])
    if len(heights) < 5:
        return float(read.get("spread_m", 0.0))
    cut = max(1, len(heights) // 4)
    return float(heights[-1 - cut] - heights[cut])


def rotation(vector) -> np.ndarray:
    """The rotation matrix of a rotation vector (axis times angle, rad)."""
    w = np.asarray(vector, float)
    angle = float(np.linalg.norm(w))
    if angle == 0.0:
        return np.eye(3)
    k = w / angle
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def turn_path(arm: str) -> Path:
    from tatbot_session import gauge

    return gauge.CALIB / f"wrist-turn-{arm}.json"


def load_turn(arm: str) -> np.ndarray | None:
    """The wrist camera's calibrated turn (a rotation vector in its frame, rad), or None before one is adopted."""
    path = turn_path(arm)
    return np.asarray(json.loads(path.read_text())["turn_cam_rad"], float) if path.exists() else None


def camera_pose(base_from_camera: np.ndarray, tip_cam, turn) -> np.ndarray:
    """The wrist camera's pose with its calibrated `turn` (about the tip, in its frame), from its CAD pose: what
    hold_patch does to a hold's skin, as a pose for rectifying its frames. The CAD pose with no turn."""
    if turn is None:
        return np.asarray(base_from_camera, float)
    tip, rot = np.asarray(tip_cam, float), rotation(turn)
    turned = np.eye(4)
    turned[:3, :3], turned[:3, 3] = rot, tip - rot @ tip
    return np.asarray(base_from_camera, float) @ turned


def sample_patch(frame, most: int = 0) -> np.ndarray:
    """A hold frame's surface points kept for the fit (camera frame): PATCH_POINTS of them, a fixed-seed subset."""
    if frame is None or frame.patch is None or not len(frame.patch):
        return np.zeros((0, 3))
    rng = np.random.default_rng(0)
    most = most or PATCH_POINTS
    return frame.patch if len(frame.patch) <= most else frame.patch[rng.choice(len(frame.patch), most,
                                                                                         replace=False)]


def hold_patch(frame, cal: dict, base_from_camera: np.ndarray, tcp: np.ndarray, turn=None,
               most: int = 0) -> np.ndarray:
    """A hold's surface points in the arm's base: the camera's frame carried by the hold's pose (FK and the camera's
    mounting), then moved so that the gauge's tip lands on the arm's tip, where the gauge's points are pinned. With
    the camera's calibrated `turn` (load_turn) the points are first turned by it about the tip, in the camera's frame."""
    pts = sample_patch(frame, most)
    if not len(pts):
        return np.zeros((0, 3))
    tip = np.asarray(cal["tip_cam_m"], float)
    if turn is not None:
        pts = (pts - tip) @ rotation(turn).T + tip
    rot, origin = base_from_camera[:3, :3], base_from_camera[:3, 3]
    shift = np.asarray(tcp, float)[:3, 3] - (rot @ tip + origin)
    return pts @ rot.T + origin + shift


def surface_normal_base(frame, base_from_camera: np.ndarray, turn=None) -> np.ndarray:
    """The gauge's surface normal under the tip, out of the skin, in the base (turned by the camera's `turn`)."""
    n = np.asarray(frame.n, float)
    return base_from_camera[:3, :3] @ (n if turn is None else rotation(turn) @ n)


def fit_turn(runs, degree: int = 4, reject_m: float = 0.0008, rounds: int = 4) -> np.ndarray:
    """The wrist camera's turn (a rotation vector in its frame, rad) that brings the skins of every run's holds
    together: camera_tilt's model in the camera's frame, one height field per run and one turn for all. A hold's
    point d (camera frame, from its tip) turned by w moves along the page normal by n_page . (w x d) = w . (d x m),
    m the page normal in the camera's frame at that hold. runs: [{"page": base_from_page, "holds": [{"patch_cam" (N,
    3), "tip_cam" (3,), "base_from_camera", "tcp" (4 x 4), "point" (base, or None)}]}]. A turn about the page normal
    moves no height, so the turn is fitted about the two axes square to the holds' mean normal (in the camera's
    frame) and is 0 about it."""
    from tatbot_motion.surface import _design, _terms

    normals = [np.asarray(h["base_from_camera"], float)[:3, :3].T @ np.asarray(run["page"], float)[:3, 2]
               for run in runs for h in run["holds"]]
    mean = np.mean(normals, axis=0)
    mean /= np.linalg.norm(mean)
    axes = np.linalg.svd(np.eye(3) - np.outer(mean, mean))[0][:, :2]   # the two axes square to the mean normal
    blocks, rhs, weights, columns = [], [], [], 0
    terms = _terms(degree)
    for run in runs:
        page = np.asarray(run["page"], float)
        rows, z, w, turn_rows = [], [], [], []
        for hold in run["holds"]:
            rot, origin = np.asarray(hold["base_from_camera"], float)[:3, :3], np.asarray(hold["base_from_camera"])[:3, 3]
            tip = np.asarray(hold["tip_cam"], float)
            shift = np.asarray(hold["tcp"], float)[:3, 3] - (rot @ tip + origin)
            local = _local(page, hold["patch_cam"] @ rot.T + origin + shift)
            m = rot.T @ page[:3, 2]
            rows.append(local)
            z.append(local[:, 2])
            w.append(np.ones(len(local)))
            turn_rows.append(np.cross(np.asarray(hold["patch_cam"], float) - tip, m))
            if hold.get("point") is not None:
                pt = _local(page, [hold["point"]])
                rows.append(pt)
                z.append(pt[:, 2])
                w.append(np.full(1, POINT_WEIGHT))
                turn_rows.append(np.zeros((1, 3)))
        local, z, w, turn_rows = (np.concatenate(v) for v in (rows, z, w, turn_rows))
        lo, hi = local[:, :2].min(axis=0), local[:, :2].max(axis=0)
        scale = max(*(hi - lo), 1e-3) / 2.0
        blocks.append((_design(local[:, :2], terms, scale), turn_rows))
        rhs.append(z)
        weights.append(w)
        columns += len(terms)
    a = np.zeros((sum(len(r) for r in rhs), columns + 2))
    row = col = 0
    for (field, turn_rows), z in zip(blocks, rhs, strict=True):
        a[row:row + len(z), col:col + field.shape[1]] = field
        a[row:row + len(z), columns:] = -turn_rows @ axes   # z + w . (d x m) = h  ->  h - w . (d x m) = z
        row, col = row + len(z), col + field.shape[1]
    z, w = np.concatenate(rhs), np.concatenate(weights)
    keep = np.ones(len(z), bool)
    coeffs = np.zeros(a.shape[1])
    for _ in range(rounds):
        sw = np.sqrt(w[keep])
        coeffs = np.linalg.lstsq(a[keep] * sw[:, None], z[keep] * sw, rcond=None)[0]
        r = z - a @ coeffs
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(reject_m, 3.0 * 1.4826 * mad)
    return axes @ coeffs[-2:]


def _local(page: np.ndarray, pts_base: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_base, float).reshape(-1, 3)
    return (np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(page).T)[:, :3]


def _turned(points: np.ndarray, tip: np.ndarray, tilt) -> np.ndarray:
    """Page-frame points turned about `tip` by the small rotation (tilt x, tilt y, 0) rad."""
    if not np.any(tilt):
        return points
    return (points - tip) @ rotation([tilt[0], tilt[1], 0.0]).T + tip


def untilt(page: np.ndarray, points_base, tip_base, tilt) -> np.ndarray:
    """Base points turned about the tip by a setup's fitted camera tilt (fit's camera_tilt_rad, `page`'s frame) as fit
    turns its holds' skin: a check's or a hover's skin, seen through the same camera, carries the same turn (a live
    forearm's check read 0.16 mm higher unturned, 2026-10-20). As they are without a tilt or a tip."""
    pts = np.asarray(points_base, float).reshape(-1, 3)
    if tilt is None or tip_base is None or not len(pts):
        return pts
    page = np.asarray(page, float)
    return _turned(_local(page, pts), _local(page, [tip_base])[0], tilt) @ page[:3, :3].T + page[:3, 3]


def camera_tilt(points, patches, tips, degree: int, extent=None, reject_m: float = 0.0008, rounds: int = 4):
    """The wrist camera's tilt (x, y rad, page frame) that brings the holds' skins together: each hold's patch is
    carried into the base by the camera's mounting and pinned at the tip, so a mounting turned by w turns every patch
    by w about its own tip, and two holds' skins then disagree where they overlap by w across their tips. Fitted with
    the height field, robustly (as fit_height): z + w_x dy - w_y dx = h(x, y), d from each point's tip. On the forearm
    (2026-10-07) the patches agreed on the skin's shape (radius 49-54 mm) and put its crest 3 mm off where the gauge
    points under the tips put it: a 3.4-3.7 deg turn about page y."""
    from tatbot_motion.surface import _design, _terms

    pts = np.concatenate([np.r_[p, patch] for p, patch in zip(points, patches, strict=True)])
    tip = np.concatenate([np.repeat(t[None], len(p) + len(patch), axis=0)
                          for t, p, patch in zip(tips, points, patches, strict=True)])
    w = np.concatenate([np.r_[np.full(len(p), POINT_WEIGHT), np.ones(len(patch))]
                        for p, patch in zip(points, patches, strict=True)])
    if extent is None:
        extent = (*pts[:, :2].min(axis=0), *pts[:, :2].max(axis=0))
    scale = max(extent[2] - extent[0], extent[3] - extent[1], 1e-3) / 2.0
    d = pts[:, :2] - tip[:, :2]
    a = np.column_stack([_design(pts[:, :2], _terms(degree), scale), -d[:, 1], d[:, 0]])
    keep = np.ones(len(pts), bool)
    coeffs = np.zeros(a.shape[1])
    for _ in range(rounds):
        sw = np.sqrt(w[keep])
        coeffs = np.linalg.lstsq(a[keep] * sw[:, None], pts[keep, 2] * sw, rcond=None)[0]
        r = pts[:, 2] - a @ coeffs
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(reject_m, 3.0 * 1.4826 * mad)
    return coeffs[-2:]


def _field(points, patches, degree: int, extent, offsets: bool, reject_m: float = 0.0008, rounds: int = 4):
    """(HeightField, each patch's own offset) through holds' points and patches (page frame), robustly as fit_height.
    With `offsets` every patch may sit its own height off the field and only the gauge points place it: each hold's
    skin gives the shape, its point the height (a hold's patch shares its pose's error with its point, which the
    pen at that pose shares too). Raises ValueError with too few points, or with offsets and no gauge point."""
    from tatbot_motion.surface import HeightField, _design, _terms, fit_height

    if not offsets:
        pts = np.concatenate([np.r_[p, patch] for p, patch in zip(points, patches, strict=True)])
        w = np.concatenate([np.r_[np.full(len(p), POINT_WEIGHT), np.ones(len(patch))]
                            for p, patch in zip(points, patches, strict=True)])
        return fit_height(pts, w, degree=degree, extent=extent), [0.0] * len(patches)
    if not sum(len(p) for p in points):
        raise ValueError("no gauge point: a field whose patches float needs one to place it")
    pts = np.concatenate([np.r_[p, patch] for p, patch in zip(points, patches, strict=True)])
    if len(pts) < 30:
        raise ValueError(f"{len(pts)} surface points: too few to fit a height field")
    w = np.concatenate([np.r_[np.full(len(p), POINT_WEIGHT), np.ones(len(patch))]
                        for p, patch in zip(points, patches, strict=True)])
    which = np.concatenate([np.r_[np.full(len(p), -1), np.full(len(patch), k)]
                            for k, (p, patch) in enumerate(zip(points, patches, strict=True))])
    if extent is None:
        extent = (*pts[:, :2].min(axis=0), *pts[:, :2].max(axis=0))
    scale = max(extent[2] - extent[0], extent[3] - extent[1], 1e-3) / 2.0
    terms = _terms(degree)
    a = np.c_[_design(pts[:, :2], terms, scale), (which[:, None] == np.arange(len(patches))[None, :]).astype(float)]
    keep = np.ones(len(pts), bool)
    coeffs = np.zeros(a.shape[1])
    for _ in range(rounds):
        sw = np.sqrt(w[keep])
        coeffs = np.linalg.lstsq(a[keep] * sw[:, None], pts[keep, 2] * sw, rcond=None)[0]
        r = pts[:, 2] - a @ coeffs
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        keep = np.abs(r) < max(reject_m, 3.0 * 1.4826 * mad)
    field = HeightField(degree, tuple(float(c) for c in coeffs[:len(terms)]), float(scale),
                        tuple(float(v) for v in extent))
    return field, [float(c) for c in coeffs[len(terms):]]


def fit(page: np.ndarray, holds: list[dict], cfg: dict, extent=None, bounds=None, turned: bool = False, drawing=None):
    """(HeightField over `page`, report) from holds [{"xy", "point" (base, or None), "patch" (N, 3 base), "tip" (base,
    or None)}]. A hold whose gauge point is None (its frames disagreed under the tip) gives its skin alone: the patch
    is the depth carried by the hold's pose, the point a quadric carried across the tip, and on the forearm's far
    flank the patches of the holds there were the field's only long baseline across the arm (2026-10-07).
    The wrist camera's turn: with `turned` the patches were turned by its calibration (hold_patch), and the turn left
    in them is fitted and reported, never applied; else (no calibration) it is fitted and taken out of each hold's
    point and patch about its tip (camera_tilt). With cfg["patch_offsets"] each patch sits its own height off the
    field and the gauge points place it (_field). The report holds the turn, each hold's skin's rms off the field
    and its leave-one-out residual (the field fitted without that hold, at its point: at a corner an
    extrapolation, so it is reported, never refused on; the check holds between corners are the field's test).
    `bounds` (half width, half height): skin points past the print are left out, the mat beyond a skin's edge among
    them. Its relief is along `drawing` ((N, 2) page points, the strokes'), else over the extent (relief_extent_m)."""
    points = [_local(page, [h["point"]]) if h.get("point") is not None else np.zeros((0, 3)) for h in holds]
    patches = [_local(page, h["patch"]) if len(h["patch"]) else np.zeros((0, 3)) for h in holds]
    if bounds is not None:
        patches = [patch[(np.abs(patch[:, 0]) <= bounds[0]) & (np.abs(patch[:, 1]) <= bounds[1])] for patch in patches]
    tilt = np.zeros(2)
    if all(h.get("tip") is not None for h in holds):
        tips = [_local(page, [h["tip"]])[0] for h in holds]
        tilt = camera_tilt(points, patches, tips, int(cfg["degree"]), extent)
        if not turned:
            points = [_turned(p, t, tilt) for p, t in zip(points, tips, strict=True)]
            patches = [_turned(patch, t, tilt) for patch, t in zip(patches, tips, strict=True)]
    degree, offsets = int(cfg["degree"]), bool(cfg.get("patch_offsets", False))
    field, shifts = _field(points, patches, degree, extent, offsets)
    loo = []
    for k in range(len(holds)):
        if not len(points[k]):
            loo.append(float("nan"))
            continue
        try:
            alone, _ = _field([p for j, p in enumerate(points) if j != k],
                              [patch for j, patch in enumerate(patches) if j != k], degree, field.extent, offsets)
            loo.append(float(points[k][0, 2] - alone(points[k][0, :2])[0]))
        except ValueError:
            loo.append(float("nan"))
    residual = [float(p[0, 2] - field(p[0, :2])[0]) if len(p) else float("nan") for p in points]
    patch_rms = [float(np.sqrt(np.mean((patch[:, 2] - shift - field(patch[:, :2])) ** 2))) if len(patch)
                 else float("nan") for patch, shift in zip(patches, shifts, strict=True)]
    grid = np.stack(np.meshgrid(np.linspace(field.extent[0], field.extent[2], 9),
                                np.linspace(field.extent[1], field.extent[3], 9)), -1).reshape(-1, 2)
    heights = field(grid)
    along, gauged = None if drawing is None else field(drawing), [float(p[0, 2]) for p in points if len(p)]
    report = {"degree": field.degree, "points": int(sum(len(p) + len(q) for p, q in zip(points, patches, strict=True))),
              "holds": len(holds), "gauged": int(sum(len(p) for p in points)), "residual_m": residual,
              "loo_m": loo, "patch_rms_m": patch_rms, "patch_offset_m": shifts, "relief_extent_m": float(np.ptp(heights)),
              "relief_m": float(np.ptp(heights if along is None else along)), "extent": list(field.extent),
              "drawing_m": None if along is None else [float(along.min()), float(along.max())],
              "gauged_m": gauged and [min(gauged), max(gauged)], "camera_tilt_rad": [float(v) for v in tilt],
              "turn": "calibrated" if turned else "fitted"}
    return field, report


RING_M = (0.008, 0.020)   # a hold's own skin about the tip's foot that it reads the field by (ring_miss)


def ring_miss(page: np.ndarray, field, patch_base: np.ndarray, point_base) -> float | None:
    """The median height of a hold's skin over `field` (page frame) in RING_M about its point: its raw depth carried
    into the base, never the point itself, which the gauge carries across the 8 mm under the pen. Seen at a grazing
    view (the forearm's far side, 45-56 deg) those points read 0.8-1.6 mm deep while the skin about them sat within
    0.5 mm of the field (2026-10-07). None with under 100 skin points there."""
    if patch_base is None or not len(patch_base):
        return None
    local = _local(page, patch_base)
    foot = _local(page, [point_base])[0]
    r = np.hypot(local[:, 0] - foot[0], local[:, 1] - foot[1])
    near = (r > RING_M[0]) & (r < RING_M[1])
    if near.sum() < 100:
        return None
    return float(np.median(local[near, 2] - field(local[near, :2])))


def check_miss(page: np.ndarray, field, patch_base, point_base, tip_base=None, tilt=None) -> tuple[float, float]:
    """(how far a check hold reads the skin off `field`: by the skin about its tip (ring_miss), else by its point; its
    point's own miss), its skin and point first turned by the setup's fitted camera tilt about its tip (untilt)."""
    patch, point = untilt(page, patch_base, tip_base, tilt), untilt(page, [point_base], tip_base, tilt)[0]
    local = _local(page, [point])[0]
    off = float(local[2] - field(local[:2])[0])
    ring = ring_miss(page, field, patch, point)
    return (off if ring is None else ring), off


PROFILE_POINTS = 8   # a path point is read with this many skin points near it at least (skin_along)


def skin_along(surface, skin_base, uv, near_m: float) -> np.ndarray:
    """(N,) how far the skin stands off `surface` along its normal at print points uv: the median over the skin points
    (base, as the hover's frame saw them, carried as a hold's patch) within near_m of each point on the page's plane;
    NaN where under PROFILE_POINTS read. No surface is fitted to the skin: each point's own skin against the field."""
    return _skin_at(surface, skin_base, surface.chart.to_plane(uv) if len(uv) else np.zeros((0, 2)), near_m)


def _over(surface, skin_base) -> tuple[np.ndarray, np.ndarray]:
    """(skin points' page-plane xy, each one's height off `surface` along its normal)."""
    local = _local(surface.base, np.asarray(skin_base, float).reshape(-1, 3))
    grad = surface.height.gradient(local[:, :2])
    return local[:, :2], (local[:, 2] - surface.height(local[:, :2])) / np.sqrt(1.0 + np.sum(grad ** 2, axis=1))


def _skin_at(surface, skin_base, xy, near_m: float) -> np.ndarray:
    out = np.full(len(xy), np.nan)
    if not len(np.reshape(skin_base, (-1, 3))) or not len(xy):
        return out
    at, over = _over(surface, skin_base)
    for k, p in enumerate(xy):
        near = np.sum((at - p) ** 2, axis=1) < near_m ** 2
        if near.sum() >= PROFILE_POINTS:
            out[k] = float(np.median(over[near]))
    return out


def lent_off(surface, own_base, lent_base, near_m: float) -> float | None:
    """How far a borrowed frame's skin stands off the stroke's own hover's skin where both read it (the median, over
    up to 300 of the own skin's points, of the borrowed skin within near_m less the own point's); None with under 20
    compared. The own views always read the 8-20 mm ring about the tip the skin watch reads."""
    own = np.asarray(own_base, float).reshape(-1, 3)
    own = own[np.linspace(0, len(own) - 1, min(300, len(own))).astype(int)] if len(own) else own
    xy, mine = _over(surface, own)
    diff = _skin_at(surface, lent_base, xy, near_m) - mine
    return float(np.nanmedian(diff)) if np.isfinite(diff).sum() >= 20 else None


def ride_profile(read, arc, fallback: float, rise: float, rigid: dict) -> tuple[np.ndarray, np.ndarray]:
    """(the skin's part over the field along a stroke's path, NaN unread: `read` (skin_along) less what the pen rides by
    already (`rise`, at each point or one for all: the surface check's rise_m and the turned checks' lift, turned_lift),
    at each point a robust line through the readings within profile_smooth_m of arc
    (a median there read a ramp's ends short, the needle that much deeper); the offset the stroke rides at each point:
    that part, never more than profile_lower_m under its plan, and where unread `fallback`, the ride by the hovers'
    readings about the tip (never under the plan); then raised wherever it would change faster than profile_slope
    along the path, so no step goes into the skin)."""
    read, arc = np.asarray(read, float), np.asarray(arc, float)
    seen, part, rise = np.isfinite(read), np.full(len(read), np.nan), np.broadcast_to(np.asarray(rise, float), read.shape)
    for k in np.flatnonzero(seen):   # a line through the readings about it, refitted without those 3 MADs off it
        w = seen & (np.abs(arc - arc[k]) <= float(rigid["profile_smooth_m"]))
        a, keep = np.c_[np.ones(w.sum()), arc[w] - arc[k]], np.ones(w.sum(), bool)
        for _ in range(2):
            fit = np.linalg.lstsq(a[keep], read[w][keep], rcond=None)[0] if keep.sum() >= 3 else [np.median(read[w]), 0]
            r = read[w] - a @ fit
            keep = np.abs(r) <= max(3.0 * 1.4826 * float(np.median(np.abs(r))), 1e-4)
        part[k] = float(fit[0]) - rise[k]
    ride = np.where(seen, np.maximum(np.nan_to_num(part), -float(rigid["profile_lower_m"])), max(fallback, 0.0))
    slope = float(rigid["profile_slope"])
    for order in (range(1, len(ride)), range(len(ride) - 2, -1, -1)):
        for k in order:
            j = k - 1 if order.step == 1 else k + 1
            ride[k] = max(ride[k], ride[j] - slope * abs(arc[k] - arc[j]))
    return part, ride


# The arm's error turns with its pose (a tool turned 8.6 deg further read a forearm ~1 mm higher, 2026-10-07): the
# holds stand square to the page, the strokes turned to the skin (turn_pen), and on a silicone skin round a bottle
# (~33 mm radius, 2026-10-24) every hover at the drawing's far end read the skin 1.4-1.8 mm over the holds' field, the
# other end's 0.0, where the holds read the same skin on it. A turned check (ArmExecutor._turned_checks) reads the skin
# at each end with the pen turned as the strokes will be; the strokes ride the lift they read, along the drawing.
def turned_lift(at, read, uv) -> np.ndarray:
    """(N,) how far the strokes ride over the field at print points uv by the turned checks' readings `read` (m over
    the field) at `at` (K, 2): linear along the line through two of them, held at either end beyond; one, from it to
    0 at its mirror through the page's centre; never under the field (never deeper)."""
    at, read, uv = np.asarray(at, float).reshape(-1, 2), np.asarray(read, float), np.atleast_2d(np.asarray(uv, float))
    if not len(at):
        return np.zeros(len(uv))
    if len(at) == 1:
        at, read = np.r_[at, -at], np.r_[read, 0.0]
    d = at[1] - at[0]
    s = np.clip((uv - at[0]) @ d / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.maximum(0.0, read[0] + s * (read[1] - read[0]))


class Turned:
    """A page surface the strokes ride turned_lift over (along its normal), on top of the pen's height: the arm's own
    error at the strokes' pose the turned checks read. Everything else is the surface's own (the holds', the skin
    watch's, the tracker's): only what the planners draw on lifts by it (ArmExecutor._on_surface)."""

    def __init__(self, surface, at, read):
        self.surface, self.at_xy, self.read = surface, np.asarray(at, float), np.asarray(read, float)

    def __getattr__(self, name):
        return getattr(self.surface, name)

    @property
    def flat(self) -> bool:
        return self.surface.flat and not np.any(self.read > 0.0)

    def at(self, uv) -> np.ndarray:
        return turned_lift(self.at_xy, self.read, uv)

    def lift(self, uv, offset_m: float = 0.0):
        points, normals = self.surface.lift(uv, offset_m)
        return points + self.at(uv)[:, None] * normals, normals

    def moved(self, base_from_page):
        return Turned(self.surface.moved(base_from_page), self.at_xy, self.read)


class Ridden:
    """A page surface whose strokes ride `extra` (m, along the normal) over it at the print points `path` (the nearest
    one's): a stroke's ride along its own skin (ride_profile), on top of the pen's height. Everything else is the
    surface's own, so only the strokes it is planned for lift by it (ArmExecutor._ridden)."""

    def __init__(self, surface, path, extra):
        self.surface, self.path, self.extra = surface, np.asarray(path, float), np.asarray(extra, float)

    def __getattr__(self, name):
        return getattr(self.surface, name)

    @property
    def flat(self) -> bool:   # a flat field ridden is planned as a surface, or the planners would take the plane
        return self.surface.flat and not np.any(self.extra)

    def at(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        return self.extra[np.argmin(np.sum((uv[:, None, :] - self.path[None, :, :]) ** 2, axis=2), axis=1)]

    def lift(self, uv, offset_m: float = 0.0):
        points, normals = self.surface.lift(uv, offset_m)
        return points + self.at(uv)[:, None] * normals, normals

    def moved(self, base_from_page):
        return Ridden(self.surface.moved(base_from_page), self.path, self.extra)


def skin_gap(surface, points) -> float | None:
    """How far skin points stand off `surface` along its normals, their median over at most 200 of them; None with
    under 50. Places a curved page's skin by the skin a frame sees about the tip (gauge.ring), not by its point."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) < 50:
        return None
    pts = pts[np.linspace(0, len(pts) - 1, min(200, len(pts))).astype(int)]
    return float(np.median([surface.height_over(p)[0] for p in pts]))


def coverage(program: dict | None, holds_xy, reach_m: float) -> float:
    """The farthest a stroke point lies from every hold (page metres): beyond the holds' patches the field is an
    extrapolation."""
    holds = np.asarray(holds_xy, float).reshape(-1, 2)
    pts = [np.asarray(op["points_m"], float).reshape(-1, 2) for op in (program or {}).get("ops", [])
           if op.get("op") == "stroke"]
    if not pts or not len(holds):
        return 0.0
    allp = np.concatenate(pts)
    return float(np.max(np.min(np.linalg.norm(allp[:, None, :] - holds[None, :, :], axis=2), axis=1)))


def layout(program: dict | None, clear_m, centre=(0.0, 0.0), reach_m: float = 0.020, inset_m=None) -> np.ndarray:
    """The holds of a curved page: the drawing's centre, then its box corners, inside the print's clear centre less
    its inset (geometry.hold_box: the print's inset_m, else 5 mm), so that their patches cover the drawing. The
    centre goes first: its skin gives the normals the corners are held along (ArmExecutor._first_skin). A drawing those five leave points of farther than `reach_m` from every
    hold gets the centre then a grid over its box spaced so none is (a 75 mm filler: 4 x 4; the corners and centre
    left it 37 mm from every hold and the draw refused, 2026-10-13)."""
    pts = [np.asarray(op["points_m"], float).reshape(-1, 2) for op in (program or {}).get("ops", [])
           if op.get("op") == "stroke"]
    hx, hy = geometry.hold_box(clear_m, inset_m)
    if not pts:
        lo, hi = np.array(centre) - 0.012, np.array(centre) + 0.012
    else:
        allp = np.concatenate(pts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
    mid = (lo + hi) / 2
    holds = np.array([mid, [lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    if pts and coverage(program, holds, reach_m) > reach_m:
        n = [max(2, int(np.ceil(extent / (reach_m * np.sqrt(2)))) + 1) for extent in hi - lo]
        grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1])), -1)
        grid = grid.reshape(-1, 2)
        holds = np.vstack([mid, grid[np.linalg.norm(grid - mid, axis=1) > 1e-9]])
    return np.clip(holds, [-hx, -hy], [hx, hy])


def check_points(holds_xy) -> np.ndarray:
    """The midpoints of the corner holds' edges, in turn round them: where a check hold reads the skin the field
    predicts. The corners are the four holds farthest from the holds' mean (with five, the centre is not one)."""
    c = np.asarray(holds_xy, float).reshape(-1, 2)
    if len(c) > 4:
        c = c[np.argsort(np.linalg.norm(c - c.mean(axis=0), axis=1))[-4:]]
    c = c[np.argsort(np.arctan2(*(c - c.mean(axis=0)).T[::-1]))]
    return (c + np.roll(c, -1, axis=0)) / 2
