"""Predicted appearance of a tatbot-program: its strokes as ink on a substrate, and what it costs.

This is a prediction, labelled with the weakest ink evidence it used. It is not the compile preview
(`tatbot_ink.preview`), which colours strokes by their display rgb to show order.

Each stroke is drawn at its deposited width (the tool's measured `line_width_m`, else a stated width, else
the generation width). A dipped ink fades over the last quarter of its `mm_per_dip` to 40 % strength, so
the dip schedule shows in the picture. A stroke's `deposition` (one value, or one per point for a whip's
ramp) scales its coverage linearly, an assumption no ink has measured. A tool with a `magnum` row (`row_deg` on the
page, `row_mm` end to end, `thickness_mm`) sweeps that row instead: wide across it, thin along it, square at the
ends. Skin spreads ink: `spread_mm` blurs coverage before compositing.

A compiled program carries its dose (`stats.dose`, tatbot_ink.dose: needle pokes per millimetre from the planner's
time law). Each stroke then darkens by it rather than by how often the pen passed alone: its first and last
millimetre, its middle and its turns each as that many steady passes, and a spot at each end where the pen came
down and lifted with the machine running. That a poke darkens as a share of a pass does is an assumption the lab
has yet to measure; a program without the record (a canvas preview) is drawn by its passes as before.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import linear_to_srgb
from .inks import look, weakest
from .raster import add_stroke, composite
from .timing import minutes

# Display colours for substrates, not measurements. Skin tones are a light-to-deep ladder of six.
SUBSTRATES = {
    "paper": (246, 244, 239),
    "silicone": (233, 206, 186),
    "skin-1": (246, 221, 203), "skin-2": (234, 199, 172), "skin-3": (214, 170, 135),
    "skin-4": (183, 135, 99), "skin-5": (138, 94, 66), "skin-6": (88, 58, 42),
}
DEFAULT_WIDTH_MM = 0.6
FADE_START, FADE_FLOOR = 0.75, 0.4
RAMP_STEP_MM = 0.25   # a deposition ramp is drawn in steps finer than a line's width, so it shows no bands
DOSE_END_MM = 1.0     # a stroke's start and end in its dose record: its first and last millimetre (tatbot_motion.dose)
DOSE_TURN_MM = 0.25   # a turn's dose is drawn this far each side of its vertex...
DOSE_TURN_DEG = 30.0  # ...where the stroke turns at least this much


@dataclass
class Rendering:
    srgb: np.ndarray            # (H, W, 3) in 0..1, row 0 at the top of the page
    coverage: dict[str, np.ndarray]
    report: dict


def _substrate(name_or_rgb) -> tuple[float, float, float]:
    rgb = SUBSTRATES.get(name_or_rgb, name_or_rgb) if isinstance(name_or_rgb, str) else name_or_rgb
    if isinstance(rgb, str):
        raise ValueError(f"unknown substrate {name_or_rgb!r}; have {', '.join(SUBSTRATES)}")
    return tuple(c / 255.0 for c in rgb)


def _fade(since_mm: np.ndarray, mm_per_dip: float | None) -> np.ndarray:
    if not mm_per_dip:
        return np.ones_like(since_mm)
    start = FADE_START * mm_per_dip
    t = np.clip((since_mm - start) / (mm_per_dip - start), 0.0, 1.0)
    return 1.0 - (1.0 - FADE_FLOOR) * t


def _densify(pts_mm: np.ndarray, step_mm: float = 1.0) -> np.ndarray:
    """Split segments longer than `step_mm`, so a long straight hatch line fades along its length."""
    if len(pts_mm) < 2:
        return pts_mm
    out = [pts_mm[:1]]
    for a, b in zip(pts_mm[:-1], pts_mm[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_mm)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


def _arc_mm(pts_mm: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts_mm, axis=0), axis=1))])


def _deposition(op: dict, n: int) -> np.ndarray:
    """The stroke's deposition at each of its `n` points: one value or one per point, 1 when it has none."""
    dep = np.asarray(op.get("deposition", 1.0), float)
    if dep.ndim and dep.shape != (n,):
        raise ValueError(f"stroke {op.get('id', '?')}: {dep.size} deposition values for {n} points")
    return np.clip(np.broadcast_to(dep, (n,)), 0.0, 1.0)


def _dose(program: dict, op: dict, pts_mm: np.ndarray, turns_mm: np.ndarray, width_mm: float):
    """(each segment's pokes over the steady dose, the pokes on the spot at the stroke's two ends in steady passes
    across its mark) from the program's dose record; (None, (0, 0)) for a program or a stroke without one."""
    record = (program.get("stats") or {}).get("dose") or {}
    row = (record.get("strokes") or {}).get(op.get("id"))
    if row is None:
        return None, (0.0, 0.0)
    start, middle, end, corner, _, down, up = row
    steady = record["steady_per_mm"]
    dwell = (down / (steady * width_mm), up / (steady * width_mm))
    if middle is None or len(pts_mm) < 2:   # a dot: all of it on the spot
        return None, dwell
    arc = _arc_mm(pts_mm)
    at, zone = (arc[:-1] + arc[1:]) / 2, min(DOSE_END_MM, arc[-1] / 2)
    pokes = np.full(len(at), float(middle))
    if corner is not None and len(turns_mm):
        pokes[np.abs(at[:, None] - turns_mm[None, :]).min(axis=1) <= DOSE_TURN_MM] = corner
    pokes[at <= zone], pokes[at >= arc[-1] - zone] = start, end
    return pokes / steady, dwell


def _turns_mm(pts_mm: np.ndarray) -> np.ndarray:
    """The arc lengths (mm) of a polyline's vertices that turn DOSE_TURN_DEG or more."""
    seg = np.diff(pts_mm, axis=0)
    length = np.linalg.norm(seg, axis=1)
    if len(seg) < 2 or not (length > 0).all():
        return np.zeros(0)
    unit = seg / length[:, None]
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)))
    return np.cumsum(length)[:-1][turn >= DOSE_TURN_DEG]


def _width_mm(resource: dict, op: dict, width_mm: float | None) -> float:
    measured = (resource.get("tool") or {}).get("line_width_m")
    if measured:
        return measured * 1000.0
    if width_mm:
        return width_mm
    return (op.get("generation_width_m") or DEFAULT_WIDTH_MM / 1000.0) * 1000.0


def render_program(program: dict, catalog: dict, substrate="silicone", px_per_mm: float = 10.0,
                   width_mm: float | None = None, spread_mm: float = 0.0, fade: bool = True) -> Rendering:
    width_m, height_m = program["page"]["size_m"]
    shape = (int(round(height_m * 1000 * px_per_mm)), int(round(width_m * 1000 * px_per_mm)))
    sub_name = substrate if isinstance(substrate, str) else "custom"
    resources = {r["id"]: r for r in program["resources"]}
    coverage: dict[str, np.ndarray] = {}
    since = dict.fromkeys(resources, 0.0)
    contact = dict.fromkeys(resources, 0.0)
    partial = dict.fromkeys(resources, 0.0)
    for op in program["ops"]:
        rid = op.get("resource_id")
        if op["op"] == "dip":
            since[rid] = 0.0
        if op["op"] != "stroke":
            continue
        res = resources[rid]
        pts_mm = np.asarray(op["points_m"], float) * 1000.0
        dep = _deposition(op, len(pts_mm))
        mm_per_dip = (res.get("dip") or {}).get("mm_per_dip") if (res.get("tool") or {}).get("ink") == "dip" else None
        ramp = np.ptp(dep) > 0
        dosed = op.get("id") in ((program.get("stats") or {}).get("dose") or {}).get("strokes", {})
        turns = _turns_mm(pts_mm) if dosed else np.zeros(0)
        if (fade and mm_per_dip) or ramp or dosed:
            dense = _densify(pts_mm, RAMP_STEP_MM if ramp or dosed else 1.0)
            dep, pts_mm = np.interp(_arc_mm(dense), _arc_mm(pts_mm), dep), dense
        seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1) if len(pts_mm) > 1 else np.zeros(1)
        along = since[rid] + np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        dep = (dep[:-1] + dep[1:]) / 2 if len(dep) > 1 else dep
        opacity = (_fade(along, mm_per_dip) if fade else 1.0) * dep
        since[rid] += float(seg.sum())
        contact[rid] += float(seg.sum())
        partial[rid] += float(seg[dep < 1.0].sum())
        px = np.column_stack([(pts_mm[:, 0] + width_m * 500) * px_per_mm, (height_m * 500 - pts_mm[:, 1]) * px_per_mm])
        magnum = (res.get("tool") or {}).get("magnum")
        if magnum:   # its row swept along the line, held at the page angle row_deg (y down in pixels)
            a = np.radians(magnum["row_deg"])
            width, row = magnum["thickness_mm"], magnum["row_mm"] * px_per_mm * np.array([np.cos(a), -np.sin(a)])
        else:
            width, row = _width_mm(res, op, width_mm), None
        pokes, dwell = (None, (0.0, 0.0)) if magnum else _dose(program, op, pts_mm, turns, width)
        add_stroke(coverage.setdefault(rid, np.zeros(shape)), px, width * px_per_mm, opacity, row, pokes, dwell)
    if spread_mm > 0:
        from scipy.ndimage import gaussian_filter
        coverage = {rid: gaussian_filter(c, spread_mm * px_per_mm) for rid, c in coverage.items()}
    looks = {rid: look(catalog, resources[rid]["ink_id"], sub_name) for rid in coverage}
    dose = (program.get("stats") or {}).get("dose")
    linear = composite(_substrate(substrate), [(looks[rid], c) for rid, c in coverage.items()])
    report = {
        "substrate": sub_name, "px_per_mm": px_per_mm, "spread_mm": spread_mm, "fade": fade,
        "inks": {rid: {"ink_id": lk.ink_id, "source": lk.source, "contact_mm": round(contact[rid], 1),
                       "partial_mm": round(partial[rid], 1),
                       "area_mm2": round(float(coverage[rid].sum()) / px_per_mm ** 2, 1)}
                 for rid, lk in looks.items()},
        "weakest_source": weakest(looks.values()),
        "dose": {key: dose[key] for key in ("model", "rate_hz", "ends", "steady_per_mm")} if dose else None,
        "time": minutes(program),
    }
    return Rendering(linear_to_srgb(linear), coverage, report)
