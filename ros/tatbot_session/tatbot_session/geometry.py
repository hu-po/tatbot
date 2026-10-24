"""Page, touch and drift geometry for the orchestrator. Pure numpy, no ROS.

Every pose is a 4x4 matrix `<arm>/base_link <- X` in metres. The page frame (ros/README.md section 3):
origin at the page centre, x along stencil u (right), y toward the top of the print, z out of the
paper. The TCP's +z points along the tool toward the paper, so drawing holds tcp z = -page z.
"""
from __future__ import annotations

import json

import numpy as np
from tatbot_description.transforms import matrix_quat, quat_matrix, rpy_matrix  # noqa: F401
from tatbot_motion import timelaw


def translation(xy) -> np.ndarray:
    """4x4 translation by (x, y[, z])."""
    out = np.eye(4)
    out[: len(xy), 3] = xy
    return out


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the rotation between two poses."""
    return timelaw.rotation_angle(np.asarray(a)[:3, :3].T @ np.asarray(b)[:3, :3])


def page_moved(used: np.ndarray, newest: np.ndarray, translation_m: float, rotation_rad: float) -> bool:
    """True when the newest camera page pose left the one the plan used by more than a threshold."""
    shift = float(np.linalg.norm(np.asarray(newest)[:3, 3] - np.asarray(used)[:3, 3]))
    return shift > translation_m or rotation_angle(used, newest) > rotation_rad


# --- the overhead's markers: the sheet's moves without its pose ----------------------------------------------------
def page_marks(support_json: str) -> dict:
    """id -> (4, 3) a measured Page's marker corners as the overhead sees them: x/z and y/z of its anchor points in the
    world, the overhead's colour optical frame (tatbot_bridge.capture.WORLD_FRAME), so the ray through each corner,
    which no depth moves; and its range along that ray, from which marks_moved reads the sheet's rise (a plane over the
    whole sheet lifted it, up to 6.6 mm off the skin; each corner's own depth since 2026-10-19). {} without them; an id
    read twice in one reading is left out."""
    try:
        support = json.loads(support_json or "{}")
        ids = [int(i) for i in (support.get("markers") or {}).get("ids") or []]
        points = np.asarray(support.get("anchor_points_m") or [], float).reshape(-1, 4, 3)
    except (ValueError, TypeError, AttributeError):
        return {}
    if len(points) != len(ids):
        return {}
    return {i: np.c_[p[:, :2] / p[:, 2:], np.linalg.norm(p, axis=1)] for i, p in zip(ids, points, strict=True)
            if ids.count(i) == 1 and (p[:, 2] > 0).all()}


def marks_median(readings) -> dict:
    """Each marker's corners, each the median over the readings (page_marks) that saw it."""
    seen = {}
    for reading in readings:
        for i, corners in reading.items():
            seen.setdefault(i, []).append(corners)
    return {i: np.median(c, axis=0) for i, c in seen.items()}


def marks_moved(ref: dict, now: dict, printed: dict, up=None) -> tuple[float, float, np.ndarray, float, float, int] | None:
    """(slide m, turn rad, the move, rise m, bend m, markers) of a print between two readings of its markers (page_marks), on the skin
    in print metres: each common marker's corners now carried onto the print by the affine that takes its own corners
    at `ref` onto its printed square (`printed`: id -> (4, 2) page metres), so the skin's slope toward the camera there,
    curved or flat, scales that marker's move and no surface is fitted; then the rigid fit of where they were to where
    they are (curved.rigid_fit: the move, 3 x 3 in print metres), its slide at the markers read. The rise: the median
    corner's move in 3-D (its ray times its range) along `up`, the page's normal in the overhead's frame, else the
    corners' range less (0 for a reading kept without ranges). A sheet lifted L reads ~0.7 L of slide across the rays,
    and a 15 mm slide changes the ranges 4-6.5 mm on the rig's oblique view: the rise along the normal tells them
    apart. The bend: the farthest marker's centre off that rigid move (nan under three markers), a skin that changed
    shape. Markers: how many it shares with `ref`. None below two common markers: one stray decode is no reading."""
    from tatbot_session.curved import moved_by, rigid_fit

    common = sorted(set(ref) & set(now) & set(printed))
    if len(common) < 2:
        return None
    was, is_, rise = [], [], []
    for i in common:
        a = np.c_[ref[i][:, :2], np.ones(4)]
        m = np.linalg.lstsq(a, np.asarray(printed[i], float), rcond=None)[0]
        was.append(a @ m)
        is_.append(np.c_[now[i][:, :2], np.ones(4)] @ m)
        if min(ref[i].shape[1], now[i].shape[1]) > 2:
            rise += list((corners_3d(now[i]) - corners_3d(ref[i])) @ up if up is not None else ref[i][:, 2] - now[i][:, 2])
    was, is_ = np.concatenate(was), np.concatenate(is_)
    fit, c = rigid_fit(was, is_, 0.010), was.mean(axis=0)
    bend = np.linalg.norm((moved_by(fit, was) - is_).reshape(-1, 4, 2).mean(axis=1), axis=1).max() if len(common) > 2 else np.nan
    return (float(np.linalg.norm(fit[:2, :2] @ c + fit[:2, 2] - c)), float(np.arctan2(fit[1, 0], fit[0, 0])), fit,
            float(np.median(rise)) if rise else 0.0, float(bend), len(common))


def marks_extended(ref: dict, readings, printed: dict, up=None, slide_m: float = 0.001, turn_rad: float = 0.0175,
                   rise_m: float = 0.0025) -> dict:
    """`ref` (marks_median) with the print's markers it lacks that other `readings` saw while the sheet stood where
    `ref` reads it: each reading sharing two markers with it (marks_moved) within slide_m, turn_rad and rise_m, every
    new marker's corners the median over those readings, and again with the markers that joined until none does. Past
    the arm the overhead reads two or three of a curved print's markers, a different two now and then: a forearm's
    reference from its setup's newest readings (the arm over the print at its holds) kept markers 3 and 6 alone, and
    after a 45 mm move the readings were 6 and 16, or 9 and 16; one in ten shared two markers with it, and the hover
    judged nothing in the 10 s it waited for three (2026-10-24). The setup's earlier readings put 16 in it."""
    out = dict(ref)
    while True:
        more = {}
        for reading in readings:
            new = (set(reading) & set(printed)) - set(out)
            if new and (mv := marks_moved(out, reading, printed, up)) is not None and (
                    mv[0] <= slide_m and abs(mv[1]) <= turn_rad and abs(mv[3]) <= rise_m):
                for i in new:
                    more.setdefault(i, []).append(reading[i])
        if not more:
            return out
        out.update({i: np.median(c, axis=0) for i, c in more.items()})


# The overhead's watch of a person's skin by its markers (sheet_watch, ArmExecutor.refresh_page; page.track.watch). On
# three forearm sessions (2026-10-19/20, 1 Hz) breathing moved the print 0.1-0.3 mm; it swayed 0.4-0.9 mm (sd) over
# ~20 s, drifted 0.5-1.5 mm/min one way for 17 min at a time, and stepped 1-4 mm, most steps staying put. The page
# follows the drift by the follow_s median past follow_m at the drawing (follow_rad turned); it is set up again only
# for a move that lasts: the gate_s median off_m from where the page follows the sheet, rise_m along the page's normal
# or bent bend_m, held hold_s up to the newest reading, or a curved page's skin moved past total_m or total_rad from
# its holds, or one whose follow a print turned round the arm would put roll_m under the skin (ArmExecutor._rolled).
# Replayed with the page following before each stroke (it lags the sheet up to follow_m), 3.5 mm held 10 s fired on no
# still silicone, the left forearm or 31 references of a lively right one (3 mm on 3 of them); a 5 mm step fires in
# 11-20 s. The owner doubled every stop (2026-10-20): off_m, bend_m, total_*, roll_m twice those. Not rise_m, depth: a
# hover whose gauge reads no skin has no other bound, and the 3RL's needles reach 4 mm from a tube 5 mm over the field.
WATCH = {"follow_s": 20.0, "follow_m": 0.001, "follow_rad": 0.0175, "gate_s": 10.0, "hold_s": 10.0, "off_m": 0.007,
         "rise_m": 0.0025, "bend_m": 0.003, "total_m": 0.020, "total_rad": 0.10, "roll_m": 0.001}


def window(timed: list, span: float, least: int = 3) -> list:
    """The moves of `timed` ((t, move), oldest first) in the newest `span` s ending at a reading that holds `least`."""
    for k in range(len(timed) - 1, least - 2, -1):
        moves = [m for t, m in timed[:k + 1] if t >= timed[k][0] - span]
        if len(moves) >= least:
            return moves
    return []


def median_move(moves) -> tuple[np.ndarray, float, float] | None:
    """(the move, 3 x 3, its slide and turn each their median; the median rise; the median bend, 0 under three) of
    marks_moved readings; None without one."""
    if not moves:
        return None
    turn, bends, out = np.median([np.arctan2(m[2][1, 0], m[2][0, 0]) for m in moves]), [m[4] for m in moves], np.eye(3)
    out[:2, :2], out[:2, 2] = [[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]], np.median([m[2][:2, 2] for m in moves], 0)
    return out, float(np.median([m[3] for m in moves])), float(np.nanmedian(bends)) if np.isfinite(bends).sum() > 2 else 0.0


def apart(a, b, points) -> float:
    """How far apart two in-plane moves (3 x 3) put any of `points` (N, 2)."""
    from tatbot_session.curved import moved_by

    return float(np.max(np.linalg.norm(moved_by(a, points) - moved_by(b, points), axis=1)))


def sheet_watch(timed: list, points, followed, total, cfg=WATCH) -> tuple[str | None, str]:
    """Whether a sheet moved since its setup by the overhead's marker readings (timed: (t, marks_moved), oldest first,
    since setup): ("fire", why) when its follow_s median passes `total` (m at the drawing's `points`, rad), or one of
    WATCH's moves has held hold_s up to the newest reading: the gate_s median off_m at the drawing from `followed` (the
    move the page has followed, 3 x 3: a move that lasts stays off the page until it is set up again, however long
    ago it came), past rise_m or bend_m; ("pending", why) while it has not held that long; else (None, ""): one that
    went back is no move. A gap holds the state: a gate_s window under three readings decides nothing."""
    now = median_move(window(timed, cfg["follow_s"]))
    if now is not None:
        far, turn = apart(now[0], np.eye(3), points), float(np.arctan2(now[0][1, 0], now[0][0, 0]))
        if far > total[0] or abs(turn) > total[1]:
            return "fire", f"{far * 1e3:.1f} mm, {np.degrees(turn):+.1f} deg from its setup"
    since, why = None, ""
    for k in range(len(timed) - 1, -1, -1):   # back from the newest reading while the move stays
        t = timed[k][0]
        gate = [m for tt, m in timed[:k + 1] if tt >= t - cfg["gate_s"]]
        if len(gate) < 3:
            continue
        gate = median_move(gate)
        off = apart(gate[0], followed, points)
        found = ", ".join(text for text, past in ((f"{off * 1e3:.1f} mm off the page", off > cfg["off_m"]),
                                                  (f"{gate[1] * 1e3:+.1f} mm along its normal", abs(gate[1]) > cfg["rise_m"]),
                                                  (f"bent {gate[2] * 1e3:.1f} mm", gate[2] > cfg["bend_m"])) if past)
        if not found:
            break
        since, why = t, why or found
    held = None if since is None else timed[-1][0] - since
    return (None, "") if held is None else ("fire" if held >= cfg["hold_s"] else "pending", f"{why}, held {held:.0f} s")


def corners_3d(corners) -> np.ndarray:
    """(4, 3) page_marks corners as points in the overhead's frame: each ray, unit, times its range."""
    rays = np.c_[corners[:, :2], np.ones(len(corners))]
    return rays / np.linalg.norm(rays, axis=1, keepdims=True) * corners[:, 2:3]


# --- touches and the page plane --------------------------------------------------------------
def touch_points(spread_m: float, centre=(0.0, 0.0)) -> np.ndarray:
    """Three page-frame (x, y) touch points about `centre`: (-s, -s), (s, -s), (0, s) from it.

    Near the drawing on purpose: points spread over the clear centre landed on the printed border
    within a few mm of registration error (bench 2026-09-26), and a flat page needs no wider base.
    A touch leaves a dot, so it belongs under the drawing it measures, not in a neighbour's place.
    """
    s = float(spread_m)
    return np.array([[-s, -s], [s, -s], [0.0, s]]) + np.asarray(centre, dtype=float)


def inner_edges(clear_m) -> dict:
    """The print's clear centre's edge per side in the page frame, metres: x of the left and right sides, y of the
    bottom and top."""
    hx, hy = clear_m[0] / 2.0, clear_m[1] / 2.0
    return {"left": -hx, "right": hx, "bottom": -hy, "top": hy}


def off_print(program: dict, page: dict) -> str | None:
    """Why the program's lines, their width included, leave the clear centre of the print `page` (the effective
    stack.yaml page: its clear_m) and would draw on its border; None when they stay inside."""
    edges = inner_edges(page["clear_m"])
    tools = {r["id"]: r.get("tool") or {} for r in program.get("resources", [])}
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for op in program.get("ops", []):
        if op.get("op") == "stroke":
            tool = tools.get(op.get("resource_id"), {})
            radius = max(tool.get("line_width_m") or 0.0, op.get("generation_width_m") or 0.0) / 2.0
            points = np.asarray(op["points_m"], float)[:, :2]
            lo, hi = np.minimum(lo, points.min(axis=0) - radius), np.maximum(hi, points.max(axis=0) + radius)
    if not np.isfinite(lo).all():
        return None
    clear_lo, clear_hi = np.array([edges["left"], edges["bottom"]]), np.array([edges["right"], edges["top"]])
    if (lo >= clear_lo - 1e-9).all() and (hi <= clear_hi + 1e-9).all():
        return None

    def mm(a, b):
        return f"x {a[0] * 1000:.1f} to {b[0] * 1000:.1f}, y {a[1] * 1000:.1f} to {b[1] * 1000:.1f} mm"

    return (f"the drawing ({mm(lo, hi)}) leaves the clear centre of the print ({mm(clear_lo, clear_hi)}; page "
            f"{page['size_m'][0] * 1000:g} x {page['size_m'][1] * 1000:g} mm from {page.get('geometry', 'stack.yaml')}): "
            "place it inside, or draw on a larger print")


def program_box(program: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """(centre, half-size) of a program's stroke points in the page frame, metres; None without strokes."""
    points = [p[:2] for op in (program or {}).get("ops", []) if op.get("op") == "stroke" for p in op["points_m"]]
    if not points:
        return None
    lo, hi = np.min(points, axis=0), np.max(points, axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def touch_layout(program: dict | None, spread_m: float) -> np.ndarray:
    """The touch points for a program: about its drawing's centre, the spread shrunk to stay inside a
    small drawing (at most 0.8 of its smaller half-size, never under 4 mm: a single line has no
    height); about the page centre without strokes."""
    box = program_box(program)
    if box is None:
        return touch_points(spread_m)
    centre, half = box
    return touch_points(min(float(spread_m), max(0.8 * float(np.min(half)), 0.004)), centre)


HOLD_INSET_M = 0.005   # the gauge holds stay this far inside a print's clear centre unless it sets its own (inset_m)


def hold_box(clear_m, inset_m=None) -> tuple[float, float]:
    """The half-sizes of the box the gauge holds keep to: the print's clear centre less `inset_m` (the print's own,
    page.inset_m; HOLD_INSET_M without one). A 45 x 63 mm forearm print's 32 mm centre less 5 mm left a 30 mm
    drawing's corner holds 4 mm inside its corners; it sets 2.5 mm."""
    inset = HOLD_INSET_M if inset_m is None else float(inset_m)
    return clear_m[0] / 2 - inset, clear_m[1] / 2 - inset


def gauge_layout(program: dict | None, spread_m: float, clear_m, inset_m=None) -> np.ndarray:
    """The wrist gauge's holds on a flat page: the corners of a square spread_m about the drawing's centre, and the
    centre, inside the print's clear centre less its inset (hold_box). The gauge leaves no dot, so unlike
    touch_layout it need not stay under the drawing, and five points name a bad one where four cannot."""
    box = program_box(program)
    centre = np.zeros(2) if box is None else box[0]
    hx, hy = hold_box(clear_m, inset_m)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0]]) * spread_m + centre
    return np.clip(corners, [-hx, -hy], [hx, hy])


def fit_plane(points, normal_hint) -> tuple[np.ndarray, float]:
    """Least-squares plane through >= 3 points: (unit normal n, d) with n.p + d = 0, n along normal_hint."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise ValueError("a plane needs three touches")
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    normal = vt[-1]
    if float(normal @ np.asarray(normal_hint, dtype=float)) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return normal, float(-normal @ centroid)


def touched_page(camera: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """base_from_page from the camera's x, y and yaw and the touched plane's height and tilt.

    Origin: the camera origin projected onto the plane. z: the plane normal (oriented like the camera's
    page z by fit_plane). x: the camera's page x projected onto the plane.
    """
    camera = np.asarray(camera, dtype=float)
    n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    origin = camera[:3, 3] - (n @ camera[:3, 3] + d) * n
    x = camera[:3, 0] - (camera[:3, 0] @ n) * n
    x /= np.linalg.norm(x)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2], out[:3, 3] = x, np.cross(n, x), n, origin
    return out


def page_used(camera: np.ndarray, correction: np.ndarray, trim_xy) -> np.ndarray:
    """The page pose a plan uses: camera · (camera-at-touch⁻¹ · touched) · T(trim).

    `correction` = inv(camera_at_touch) @ touched carries the touched height and tilt rigidly with the
    page, so a page that moved after the touches is re-planned in its new camera pose.
    """
    return np.asarray(camera) @ np.asarray(correction) @ translation(list(trim_xy))


def tool_down_pose(base_from_page: np.ndarray, xy, height_m: float, current_base_from_tcp: np.ndarray) -> np.ndarray:
    """TCP pose over page point xy at height_m along page z: tcp z = -page z, tcp x = the current tcp
    x projected onto the page plane (no wrist spin)."""
    page = np.asarray(base_from_page, dtype=float)
    z = -page[:3, 2]
    x = np.asarray(current_base_from_tcp)[:3, 0] - (np.asarray(current_base_from_tcp)[:3, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = page[:3, 0]
    x = x / np.linalg.norm(x)
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2] = x, np.cross(z, x), z
    out[:3, 3] = (page @ np.array([xy[0], xy[1], height_m, 1.0]))[:3]
    return out


# --- resume ---------------------------------------------------------------
def arc_at(traj, index: int, tip_measured=None) -> float:
    """The stroke arc reached by sample `index`: the drawn sample nearest the measured tip at or before
    `index` (the time lag makes the measured tip the better witness), else the last finite arc so far,
    else the plan's first finite arc (nothing drawn yet: resume where this plan started)."""
    arc = np.asarray(traj.arc_m, dtype=float)
    upto = np.arange(len(arc)) <= index
    drawn = upto & np.isfinite(arc)
    if not drawn.any():
        finite = arc[np.isfinite(arc)]
        return float(finite[0]) if finite.size else 0.0
    if tip_measured is not None:
        idx = np.flatnonzero(drawn)
        dist = np.linalg.norm(np.asarray(traj.tip)[idx] - np.asarray(tip_measured), axis=1)
        return float(arc[idx[int(np.argmin(dist))]])
    return float(arc[drawn][-1])


def stroke_length(points_m, closed: bool = False) -> float:
    """Polyline length (m); a closed stroke returns to its first point (as tatbot_motion draws it)."""
    return timelaw.polyline_length(np.vstack([points_m, points_m[:1]]) if closed and len(points_m) > 1 else points_m)


def pose_moved(q_latched, q_now, fk) -> tuple[float, float]:
    """(max revolute joint change rad, tip change m) between the latched and the current joints."""
    joint = float(np.max(np.abs(np.asarray(q_now)[:6] - np.asarray(q_latched)[:6])))
    tip = float(np.linalg.norm(fk(q_now)[:3, 3] - fk(q_latched)[:3, 3]))
    return joint, tip


def tip_force_along_tool(kin, q, tau) -> float:
    """The tip force along the tcp +z axis (toward the paper) that the arm joints' external torques
    imply for a point force at the tcp: tau = Jv^T f (tatbot_hardware tip_force_along_tool)."""
    _, rot, jac = kin.evaluate(np.asarray(q, dtype=float))
    jv = jac[:3, :6]
    f = np.linalg.solve(jv @ jv.T + 1e-9 * np.eye(3), jv @ np.asarray(tau, dtype=float)[:6])
    return float(f @ rot[:, 2])


def first_contact(heights, forces, *, window_m: float = 0.005, air_from_m: float = 0.0025, rise_n: float = 0.6,
                  stiffness_n_m=(800.0, 3000.0), nominal_n_m: float = 1500.0, min_samples: int = 10) -> dict | None:
    """Where a guarded descent met the page, from the tip height along the page normal (kinematics)
    and the tip force along the tool (joint torques), sample by sample down to the trip.

    Only the final approach counts: the samples since the tip was last window_m above the trip (the
    hold and fast leg before carry other biases). The give of the arm and EE mount under load makes
    the force rise linearly from first contact, F = base + k (h_contact - h); base is the median
    force more than air_from_m above the trip. The unbroken run over base + rise_n that ends at the
    trip is fitted; a stiffness outside stiffness_n_m (the in-air force wanders ~+-0.8 N, bench
    2026-09-26) falls back to the give model, offset = (F at the trip - base) / nominal_n_m.
    Returns {"offset_m": h_contact - h_trip, "stiffness_n_m", "base_n", "rise_n", "method": "fit" |
    "model", "samples"}, or None without enough in-air samples (then the trip pose stands).
    """
    h = np.asarray(heights, dtype=float)
    f = np.asarray(forces, dtype=float)
    ok = np.isfinite(h) & np.isfinite(f)
    h, f = h[ok], f[ok]
    if len(h) < 2 * min_samples:
        return None
    high = np.flatnonzero(h > h[-1] + window_m)
    if len(high):
        h, f = h[high[-1] + 1:], f[high[-1] + 1:]
    air = h > h[-1] + air_from_m
    if air.sum() < min_samples:
        return None
    base = float(np.median(f[air]))
    rise = float(np.mean(f[-min(5, len(f)):]) - base)
    # The ramp is the unbroken run over base + rise_n that ends at the trip: an in-air wander over
    # the threshold before it is not contact.
    below = np.flatnonzero(f <= base + rise_n)
    ramp = np.zeros(len(h), dtype=bool)
    ramp[(below[-1] + 1) if len(below) else 0:] = True
    if ramp.sum() >= min_samples:
        slope, intercept = np.polyfit(h[ramp], f[ramp], 1)
        k = -float(slope)
        if stiffness_n_m[0] <= k <= stiffness_n_m[1]:
            return {"offset_m": float((intercept - base) / k - h[-1]), "stiffness_n_m": k, "base_n": base,
                    "rise_n": rise, "method": "fit", "samples": int(ramp.sum())}
    if rise <= 0:
        return None
    return {"offset_m": rise / nominal_n_m, "stiffness_n_m": nominal_n_m, "base_n": base, "rise_n": rise,
            "method": "model", "samples": int(ramp.sum())}


def fiducial_contact_time(times, gap, *, min_after: int = 8, min_rate: float = 0.00005, gain: float = 1.5):
    """When the EE met the page, from the gap between the overhead-tracked EE fiducial and the same
    frame by FK of the encoders, along the page normal, sample by sample through a guarded descent.

    Before contact the gap is flat (a constant registration offset plus ~1 mm noise); once the tip is
    on the page the EE slows while the encoders keep descending, and the arm's and mount's give opens
    the gap: gap = g0 + v max(0, t - tc). The hinge tc is the grid point minimising the squared error
    (g0, v by least squares). Returns {"t_c", "rate_m_s", "after", "gain"} when the hinge beats a flat
    line by `gain` with at least `min_after` samples after it and v over `min_rate`, else None."""
    t, gap = np.asarray(times, dtype=float), np.asarray(gap, dtype=float)
    if len(t) <= 2 * min_after:   # exactly twice min_after leaves no hinge to try (TypeError on 2026-10-03)
        return None
    flat = float(np.sum((gap - gap.mean()) ** 2))
    best = None
    for tc in t[min_after:-min_after]:
        x = np.maximum(0.0, t - tc)
        a = np.c_[np.ones_like(t), x]
        coef, *_ = np.linalg.lstsq(a, gap, rcond=None)
        sse = float(np.sum((gap - a @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tc), float(coef[1]))
    sse, tc, rate = best
    after = int(np.sum(t > tc))
    if rate < min_rate or after < min_after or flat < gain * sse:
        return None
    return {"t_c": tc, "rate_m_s": rate, "after": after, "gain": flat / max(sse, 1e-12)}


def measured_age(measured_at: float | None, now: float) -> float | None:
    """Seconds since the page was last measured, or None if this node never saw it measured."""
    return None if measured_at is None else max(0.0, now - measured_at)


def stale_page(info: dict, max_lost_s: float) -> str | None:
    """Why a camera page pose is too stale to start from, or None. A fixed or measured page is current;
    a lost one stands for its last measured pose, which is refused once older than `max_lost_s` (or when
    this node never saw it measured, as after a stack restart)."""
    if info.get("source") != "lost":
        return None
    age = info.get("measured_age_s")
    if age is None:
        return "the page is lost and has not been measured since the stack started"
    if age > max_lost_s:
        return f"the page has been lost for {age:.0f} s (page.max_lost_s {max_lost_s:g}): its last measured pose is stale"
    return None
