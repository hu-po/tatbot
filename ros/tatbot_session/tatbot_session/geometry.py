"""Page, touch and drift geometry for the orchestrator. Pure numpy, no ROS.

Every pose is a 4x4 matrix `<arm>/base_link <- X` in metres. The page frame (ros/README.md section 3):
origin at the page centre, x along stencil u (right), y toward the top of the print, z out of the
paper. The TCP's +z points along the tool toward the paper, so drawing holds tcp z = -page z.
"""
from __future__ import annotations

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


# --- touches and the page plane --------------------------------------------------------------
def touch_points(spread_m: float, centre=(0.0, 0.0)) -> np.ndarray:
    """Three page-frame (x, y) touch points about `centre`: (-s, -s), (s, -s), (0, s) from it.

    Near the drawing on purpose: points spread over the clear centre landed on the printed border
    within a few mm of registration error (bench 2026-09-26), and a flat page needs no wider base.
    A touch leaves a dot, so it belongs under the drawing it measures, not in a neighbour's place.
    """
    s = float(spread_m)
    return np.array([[-s, -s], [s, -s], [0.0, s]]) + np.asarray(centre, dtype=float)


SIDES = {"left": (0, -1.0), "right": (0, 1.0), "bottom": (1, -1.0), "top": (1, 1.0)}   # (axis, sign)


def inner_edges(clear_m, inner_edges_m=None) -> dict:
    """The print's nominal inner border edge per side in the page frame, metres: x of the left and
    right sides, y of the bottom and top. page.inner_edges_m (from the installed print's settings.json,
    config.print_page) gives them for a print whose border does not start at the clear centre's edge
    (a coded print's knots sit on the lattice junctions by their centres, so its innermost ink is up to
    a knot radius off that edge, by a different amount on each side); without it they are +-clear_m / 2."""
    hx, hy = clear_m[0] / 2.0, clear_m[1] / 2.0
    if not inner_edges_m:
        return {"left": -hx, "right": hx, "bottom": -hy, "top": hy}
    try:
        out = {side: float(inner_edges_m[side]) for side in SIDES}
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError(f"page.inner_edges_m {inner_edges_m!r}: needs left, right, bottom and top") from err
    if not all(0.0 < sign * out[side] < clear_m[axis] for side, (axis, sign) in SIDES.items()):
        raise ValueError(f"page.inner_edges_m {out}: page frame metres, left and bottom below 0, right and top above")
    return out


def off_print(program: dict, page: dict) -> str | None:
    """Why the program's lines, their width included, leave the clear centre of the print `page` (the effective
    stack.yaml page: its clear_m and inner_edges_m) and would draw on its border; None when they stay inside."""
    edges = inner_edges(page["clear_m"], page.get("inner_edges_m"))
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


def height_plane(points, normal) -> tuple[np.ndarray, float]:
    """The plane with the given unit normal at the touches' median height: (n, d) with n.p + d = 0.
    Three touches 24 mm apart pin a height well and a tilt badly (+-1 mm is +-2.4 deg; bench
    2026-09-26 fits swung 7-13 deg), so the camera's tilt is kept and the touches set the height."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return n, float(-np.median(np.asarray(points, dtype=float) @ n))


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
