"""Close-up ink inspection: the pen held low over the drawing, the wrist D405 photographing the paper about its tip,
each frame put on the page through the wrist gauge, and every planned stroke scored for the ink along it.

The draw's far views (inspect.aim, ~240 mm out at ~50 deg) resolve ~2.5 px/mm, and their CAD camera pose puts views
several millimetres apart: a 0.5 mm line of light ink does not show in them. Held 12 mm over the paper the camera
is ~170 mm from it (~3.9 px/mm), and the gauge pins its pose where it looks: the pen's tip is a fixed point in the
camera's frame (gauge.py), so the CAD pose turned until the paper's measured normal is the page's, then moved until
that point is the arm's tip, is right about the tip, which is where a hold keeps the page (HOLD_RADIUS_M).
"""
from __future__ import annotations

import numpy as np

from tatbot_session import gauge

HOLD_RADIUS_M = 0.025    # page within this of a hold's tip is kept: farther, the paper recedes and a turn error grows
INK_DELTA_E = 25.0       # distance in OpenCV's 8-bit Lab from the paper's local colour that reads as ink
BACKGROUND_MM = 3.0      # the paper's local colour: the median over a square this wide (a 0.5 mm line is not it)
SAMPLE_M = 0.0005        # a stroke is scored at points this far apart
PEN_HUE = gauge.PEN_HUE  # the pen's teal is masked out of every frame before it is put on the page


def _turn(a, b) -> np.ndarray:
    """The least rotation taking unit vector a onto unit vector b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def anchored(base_from_cam, tcp, frame: gauge.Frame, cal: dict, page_normal) -> np.ndarray:
    """The camera's pose with its tilt and its place from the gauge: base_from_cam (CAD) turned the least so that
    the paper's normal measured in `frame` is the page's, then moved so that the gauge's contact point (the fitted
    tip less its offset along the normal) lands on the arm's tip `tcp` (base)."""
    rot = np.asarray(base_from_cam, float)[:3, :3]
    up = np.asarray(page_normal, float)
    if up @ (rot @ frame.n) < 0:   # both toward the camera
        up = -up
    rot = _turn(rot @ frame.n, up) @ rot
    contact = np.asarray(cal["tip_cam_m"], float) - float(cal["offset_m"]) * frame.n
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, np.asarray(tcp, float) - rot @ contact
    return out


def on_paper(base_from_page, base_from_cam, frame: gauge.Frame) -> np.ndarray:
    """base_from_page moved along its normal onto the paper this frame's depth measures (through the camera pose
    base_from_cam): the draw's page height is minutes old and the arm's tip height drifts (2 mm over an hour on
    2026-10-03), which at the camera's ~48 deg off the normal moves every rectified point ~2 mm."""
    page = np.asarray(base_from_page, float).copy()
    paper = (base_from_cam @ [*frame.c, 1.0])[:3]
    page[:3, 3] += float((paper - page[:3, 3]) @ page[:3, 2]) * page[:3, 2]
    return page


def in_clear(window, px_per_mm: float, clear_m, margin_m: float = 0.001) -> np.ndarray:
    """The page image pixels inside the print's clear centre (`clear_m`, the print's page) less margin_m:
    outside it the stencil's printed border would read as ink."""
    from tatbot_session import inspect as ins

    hx, hy = (v / 2 - margin_m for v in clear_m)
    gx, gy = ins.page_grid(window, px_per_mm)
    return (np.abs(gx) <= hx) & (np.abs(gy) <= hy)


def holds(program: dict, step_m: float, limit: int = 12) -> list[np.ndarray]:
    """Page points to hold the pen over: the drawing's centre and a step to each side of it along the page's axes
    (the pen hides the paper on one side of its tip), and a grid of `step_m` over a larger drawing."""
    pts = np.array([p for op in program.get("ops", []) if op.get("op") == "stroke" for p in op["points_m"]], float)
    if not len(pts):
        return []
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    out = [centre + d for d in ([0.0, 0.0], [step_m, 0.0], [-step_m, 0.0], [0.0, step_m], [0.0, -step_m])]
    xs, ys = (np.arange(lo[i], hi[i] + step_m / 2, step_m) for i in (0, 1))
    if len(xs) > 2 or len(ys) > 2:
        out += [np.array([x, y]) for x in xs for y in ys]
    kept: list[np.ndarray] = []
    for p in out:
        if all(np.linalg.norm(p - k) > step_m / 3 for k in kept):
            kept.append(p)
    return kept[:limit]


def extent(program: dict, margin_m: float = HOLD_RADIUS_M) -> tuple[float, float, float, float]:
    """The page window the close-up images cover: the drawing's box and `margin_m` about it."""
    pts = np.array([p for op in program.get("ops", []) if op.get("op") == "stroke" for p in op["points_m"]], float)
    lo, hi = pts.min(axis=0) - margin_m, pts.max(axis=0) + margin_m
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def pen_mask(image_bgr) -> np.ndarray:
    """The pen's teal in a camera frame, grown a little: it is no paper."""
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    teal = ((hsv[..., 0] >= PEN_HUE[0]) & (hsv[..., 0] <= PEN_HUE[1]) & (hsv[..., 1] > 90)).astype(np.uint8)
    return cv2.dilate(teal, np.ones((9, 9), np.uint8)).astype(bool)


def ink_score(page_bgr, valid, px_per_mm: float) -> np.ndarray:
    """Each page pixel's Lab distance from the paper's local colour (0 off the views): ink of any colour, and
    the arm's soft shadows drop out with the local median."""
    import cv2

    lab = cv2.cvtColor(np.asarray(page_bgr, np.uint8), cv2.COLOR_BGR2LAB)
    if valid.any():
        lab[~valid] = np.median(lab[valid], axis=0).astype(np.uint8)
    k = int(round(BACKGROUND_MM * px_per_mm)) | 1
    background = cv2.medianBlur(lab, max(5, k))
    score = np.linalg.norm(lab.astype(np.float32) - background.astype(np.float32), axis=2)
    score[~valid] = 0.0
    return score


def _samples(program: dict) -> list[tuple[dict, np.ndarray]]:
    """Each stroke op and its page points every SAMPLE_M along it."""
    out = []
    for op in program.get("ops", []):
        if op.get("op") != "stroke":
            continue
        pts = np.asarray(op["points_m"] + ([op["points_m"][0]] if op.get("closed") else []), float)
        parts = [pts[:1]]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / SAMPLE_M)))
            parts.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
        out.append((op, np.concatenate(parts)))
    return out


def _score(samples, near, valid, extent_m, px_per_mm: float, shift=(0.0, 0.0)):
    """Per stroke (seen, inked) boolean arrays with every sample moved by `shift` (page metres)."""
    from tatbot_session import inspect as ins

    h, w = valid.shape
    out = []
    for _, pts in samples:
        px = np.round(ins.to_image_px(pts + np.asarray(shift, float), extent_m, px_per_mm)).astype(int)
        inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        seen = np.zeros(len(px), bool)
        seen[inside] = valid[px[inside, 1], px[inside, 0]]
        inked = seen.copy()
        inked[seen] = near[px[seen, 1], px[seen, 0]] > 0
        out.append((seen, inked))
    return out


def coverage(program: dict, score, valid, extent_m, px_per_mm: float, *, tol_m: float = 0.0015,
             threshold: float = INK_DELTA_E, search_m: float = 0.008) -> dict:
    """Per stroke and per resource: the share of its points the views saw (`seen`) and of those the share with
    ink within tol_m of the plan (`ink`, None unseen); and the page shift within search_m that finds the most
    ink (`best_shift_m`, with the shares there), which names ink drawn off its plan."""
    import cv2

    ink_px = (score > threshold).astype(np.uint8)
    r = max(1, int(round(tol_m * 1000 * px_per_mm)))
    near = cv2.dilate(ink_px, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    samples = _samples(program)
    # the search scores on the ink itself, not widened by tol_m: widened, every shift within tol_m of the true one
    # ties, and the least of them is taken
    sharp = cv2.dilate(ink_px, np.ones((3, 3), np.uint8))
    steps = np.arange(-search_m, search_m + 1e-9, 0.0005)
    best, best_ink = (0.0, 0.0), -1
    for dx in steps:
        for dy in steps:
            ink = sum(int(i.sum()) for _, i in _score(samples, sharp, valid, extent_m, px_per_mm, (dx, dy)))
            if ink > best_ink or (ink == best_ink and np.hypot(dx, dy) < np.hypot(*best)):
                best, best_ink = (float(dx), float(dy)), ink

    def rows(shift):
        strokes, per = [], {}
        for (op, _), (seen, inked) in zip(samples, _score(samples, near, valid, extent_m, px_per_mm, shift), strict=True):
            row = {"id": op.get("id"), "resource_id": op.get("resource_id"), "points": int(len(seen)),
                   "seen": float(seen.mean()), "ink": float(inked.sum() / seen.sum()) if seen.any() else None}
            strokes.append(row)
            agg = per.setdefault(row["resource_id"], [0, 0])
            agg[0], agg[1] = agg[0] + int(seen.sum()), agg[1] + int(inked.sum())
        return strokes, {k: {"seen_points": s, "ink": (i / s if s else None)} for k, (s, i) in per.items()}

    strokes, resources = rows((0.0, 0.0))
    shifted, shifted_resources = rows(best)
    return {"strokes": strokes, "resources": resources, "threshold": threshold, "tol_m": tol_m,
            "best_shift_m": list(best), "at_best_shift": {"strokes": shifted, "resources": shifted_resources}}


def overlay(page_bgr, result: dict, program: dict, extent_m, px_per_mm: float) -> np.ndarray:
    """The close-up page with each planned stroke drawn green where it found ink (over half its seen points),
    red where it did not, grey where it was not seen."""
    import cv2

    from tatbot_session import inspect as ins

    out = np.asarray(page_bgr, np.uint8).copy()
    found = {s["id"]: s for s in result["strokes"]}
    for op in program.get("ops", []):
        if op.get("op") != "stroke":
            continue
        s = found.get(op.get("id"), {})
        colour = (128, 128, 128) if s.get("ink") is None else ((0, 200, 0) if s["ink"] > 0.5 else (0, 0, 255))
        pts = op["points_m"] + ([op["points_m"][0]] if op.get("closed") else [])
        px = np.round(ins.to_image_px(pts, extent_m, px_per_mm)).astype(np.int32)
        cv2.polylines(out, [px], False, colour, 1, cv2.LINE_AA)
    return out
