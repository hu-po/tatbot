"""A cell's numbers from its difference: what its technique asks, each with the floor it was read against.

Straight parallel strokes (line, carry, solid, whip) are read in the stroke frame: u along a stroke from its start,
v across. The measures, all darkening in L* unless named otherwise:
- per_line: each stroke's mean within LINE_HALF_MM of it, away from its ends;
- carry (carry): how far each dip went, the stroke lengths from the dip summed to where the level falls below half
  the cycle's peak (strong) and below a quarter of it and twice the floor (visible);
- along: the strokes' darkening in tenths of their length from the start, and first over last (a fade);
- bands: across the strokes, the strongest periods in each quarter of their length (the dip cycle shows as bands
  as many strokes wide as one dip carries);
- speckle: the share of the ring 1.5-8 mm around the strokes darkened by more than SPECKLE_DL;
- past_end: the darkening in the millimetre beyond the strokes' far end;
- ink_width_mm: per stroke, the width of opaque ink holding the same ink (the integral of `c` across it): a line
  narrower than the pixels still has one;
- flood: what a fresh dip lays down before the load settles. The ink width over the first FLOOD_MM of each stroke
  that follows a dip, against the strokes' steady width beyond it and against the starts of the strokes that follow
  no dip (every stroke starts on a needle that dwells, so a start is wider than a middle with or without a dip),
  and the speckle thrown ahead of where the pen came down.
Corners are read by their ends against their middles. Every technique's strokes are also read by where the pen
was on each (stroke_ends): the millimetre where it came down, the millimetre where it lifted and the millimetre
about each turn, each against the strokes' middles, since the pen dwells differently at the three (the dose model,
tatbot_ink.dose). The measures are 2026-11-05's skin tests' readers, made general; docs/lab.md has how closely these
reproduce them.
"""
from __future__ import annotations

import numpy as np

from tatbot_lab.read.diff import KEEP_OUT_MM, Diff
from tatbot_lab.read.frames import Frame

LINE_HALF_MM = 0.45     # a stroke's own band
END_MM = 0.6            # stroke ends left out of its band
ZONE_MM = 1.0           # a stroke's start, its stop and each turn: this much of its line (the dose model's millimetre)
TURN_DEG = 30.0         # a vertex turning at least this much is a turn
RING_MM = (1.5, 8.0)    # the speckle ring about the strokes' box
SPECKLE_DL = 8.0
BAND_PERIODS_MM = (0.625, 6.7)
FLOOD_MM = 3.0          # the first millimetres of a stroke read for a fresh dip's excess
AHEAD_MM = (0.5, 4.0)   # the strip ahead of a stroke's start read for what the loaded needles threw


def _r(v, n=2):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _straight(strokes: list[np.ndarray], frame: Frame):
    """Each straight stroke's (u0, u1, v) in the frame, from the first stroke's start."""
    start = strokes[0][0]
    out = []
    for s in strokes:
        a, b = s[0] - start, s[-1] - start
        ua, ub = a @ frame.along, b @ frame.along
        out.append((min(ua, ub), max(ua, ub), float((a @ frame.across + b @ frame.across) / 2), ua <= ub))
    return start, out


def per_line(d: Diff, u: np.ndarray, v: np.ndarray, lines) -> list[float | None]:
    out = []
    for u0, u1, vk, _ in lines:
        m = d.valid & (np.abs(v - vk) <= LINE_HALF_MM) & (u >= u0 + END_MM) & (u <= u1 - END_MM)
        out.append(_r(d.dl[m].mean(), 1) if m.sum() > 5 else None)
    return out


def _crossing(vals: np.ndarray, cum: np.ndarray, lengths: list[float], above) -> tuple[int, float]:
    """The last stroke whose level is `above` its threshold, and how far ink got: the length drawn to that stroke's
    end, plus the share of the next stroke the level takes to fall to the threshold (linear between the two)."""
    hits = [i for i, v in enumerate(vals) if np.isfinite(v) and above(v)]
    if not hits:
        return 0, 0.0
    k = hits[-1]
    mm = float(cum[k])
    if k + 1 < len(vals) and np.isfinite(vals[k + 1]):
        threshold = above.threshold
        share = (vals[k] - threshold) / max(vals[k] - vals[k + 1], 1e-9)
        mm += min(max(share, 0.0), 1.0) * lengths[k + 1]
    return k + 1, mm


class _Above:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, v: float) -> bool:
        return v > self.threshold


def carry(levels: list, lengths: list[float], noise: float) -> dict:
    """How far one dip carried: strokes in drawing order, `levels` their darkening, `lengths` their mm. Strong is
    over half the peak (the 90th percentile of the strokes), visible over a quarter of it and twice the floor; each
    distance runs to where the level crosses its threshold, so a stroke at the threshold does not jump it by a
    whole stroke (2026-11-05's readers counted whole strokes: their 257 and 296 mm are ±20 mm)."""
    vals = np.array([np.nan if v is None else v for v in levels], float)
    if not np.isfinite(vals).any():
        return {"peak": None, "strong_mm": None, "visible_mm": None, "strong_lines": 0, "visible_lines": 0}
    peak = float(np.nanpercentile(vals, 90))
    cum = np.cumsum(lengths)
    strong_lines, strong = _crossing(vals, cum, lengths, _Above(0.5 * peak))
    seen_lines, seen = _crossing(vals, cum, lengths, _Above(max(0.25 * peak, 2 * noise)))
    return {"peak": _r(peak, 1), "strong_mm": _r(strong, 0), "visible_mm": _r(seen, 0), "strong_lines": strong_lines,
            "visible_lines": seen_lines}


def carry_cycles(levels: list, lengths: list[float], noise: float, dips_mm: list[float]) -> list[dict]:
    """`carry` for each dip cycle: strokes go to the cycle their middle falls in, and each cycle is read from its
    dip. `outlasted` is a cycle whose last stroke still holds over half its peak: the dip carried at least its
    length, and the cycle cannot say how much further."""
    cum = np.cumsum(lengths)
    middles = cum - np.asarray(lengths) / 2
    starts = sorted({0.0, *dips_mm})
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else float("inf")
        members = [i for i, m in enumerate(middles) if start <= m < end]
        if not members:
            continue
        result = carry([levels[i] for i in members], [lengths[i] for i in members], noise)
        result.update(from_mm=_r(start, 1), length_mm=_r(sum(lengths[i] for i in members), 0),
                      outlasted=result["strong_lines"] == len(members))
        out.append(result)
    return out


def along(d: Diff, u, v, lo_v, hi_v, length) -> dict:
    inside = d.valid & (v >= lo_v) & (v <= hi_v)
    prof = []
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        m = inside & (u >= a * length) & (u <= b * length)
        prof.append(float(d.dl[m].mean()) if m.sum() else float("nan"))
    return {"tenths": [_r(p, 1) for p in prof], "first_over_last": _r(prof[0] / max(prof[-1], 0.5), 2)}


def bands(d: Diff, u, v, lo_v, hi_v, length, px_per_mm: float) -> list[dict]:
    """The strongest periods across the strokes, per quarter of their length. The frame's columns run across the
    strokes, so each column is one place across them: the profile is the columns' means."""
    step = 1.0 / px_per_mm
    cols = np.flatnonzero((v[0] >= lo_v) & (v[0] <= hi_v))
    out = []
    for qa, qb in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
        rows = np.flatnonzero((u[:, 0] >= qa * length) & (u[:, 0] <= qb * length))
        block, ok = d.dl[np.ix_(rows, cols)], d.valid[np.ix_(rows, cols)]
        with np.errstate(invalid="ignore"):
            pr = np.where(ok.any(axis=0), (block * ok).sum(axis=0) / ok.sum(axis=0), np.nan)
        if len(rows) == 0 or np.isnan(pr).any() or len(pr) < 61:
            out.append({"quarter": [qa, qb], "periods_mm": [], "power": []})
            continue
        pr = (pr - np.convolve(pr, np.ones(41) / 41, "same"))[20:-20]
        f = np.abs(np.fft.rfft(pr * np.hanning(len(pr))))
        fr = np.fft.rfftfreq(len(pr), step)
        s = (fr > 1 / BAND_PERIODS_MM[1]) & (fr < 1 / BAND_PERIODS_MM[0])
        top = np.argsort(f[s])[::-1][:3]
        out.append({"quarter": [qa, qb], "periods_mm": [_r(1 / fr[s][i]) for i in top],
                    "power": [_r(f[s][i], 0) for i in top], "median_power": _r(np.median(f[s]), 0)})
    return out


def ink_width(d: Diff, u, v, lines, pitch_mm: float | None) -> list[float | None]:
    half = min(1.5, pitch_mm / 2) if pitch_mm else 1.5
    out = []
    for u0, u1, vk, _ in lines:
        m = d.valid & (np.abs(v - vk) <= half) & (u >= u0 + END_MM) & (u <= u1 - END_MM)
        if m.sum() < 20:
            out.append(None)
            continue
        station = np.round(u[m] * 10).astype(int)   # 0.1 mm apart along the stroke
        station -= station.min()
        total, n = np.bincount(station, np.clip(d.c[m], -1, 1)), np.bincount(station)
        widths = total[n > 0] / n[n > 0] * (2 * half)   # the band's mean share of light taken, times its width
        out.append(_r(float(np.median(widths)), 3))
    return out


def _start(d: Diff, u, v, line, half: float) -> tuple[float | None, float | None, float | None]:
    """One stroke from where the pen came down: (ink width over its first FLOOD_MM, ink width over the rest less its
    far end, the share of the strip ahead of its start darkened by more than SPECKLE_DL)."""
    u0, u1, vk, forward = line
    s = (u - u0) if forward else (u1 - u)   # mm along the stroke from its start
    band = d.valid & (np.abs(v - vk) <= half)

    def width(m):
        return float(np.clip(d.c[m], -1, 1).mean() * 2 * half) if m.sum() >= 10 else None
    ahead = band & (s <= -AHEAD_MM[0]) & (s >= -AHEAD_MM[1])
    return (width(band & (s >= 0) & (s <= FLOOD_MM)), width(band & (s > FLOOD_MM) & (s <= u1 - u0 - END_MM)),
            float((d.dl[ahead] > SPECKLE_DL).mean()) if ahead.sum() >= 10 else None)


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def flood(d: Diff, u, v, lines, lengths: list[float], dips_mm: list[float], pitch_mm: float | None) -> dict:
    """A fresh dip's excess: each stroke that follows a dip, read at its start, against the steady line and against
    the other strokes' starts. `over_steady` near 1 and `over_other_starts` near 1 are a dip that floods nothing."""
    half = min(1.5, pitch_mm / 2) if pitch_mm else 1.5
    begins = np.cumsum(lengths) - np.asarray(lengths)
    after_dip = {int(np.argmin(np.abs(begins - mm))) for mm in dips_mm if np.abs(begins - mm).min() < 0.05}
    reads = [_start(d, u, v, line, half) for line in lines]
    steady = _median(r[1] for r in reads)
    others = _median(r[0] for i, r in enumerate(reads) if i not in after_dip)

    def ratio(value, base):
        return _r(value / base, 2) if value is not None and base and base > 1e-3 else None
    dips = [{"from_mm": _r(begins[i], 1), "start_width_mm": _r(reads[i][0], 3), "over_steady": ratio(reads[i][0], steady),
             "ahead_speckle": _r(reads[i][2], 3)} for i in sorted(after_dip)]
    firsts = _median(reads[i][0] for i in after_dip)
    return {"window_mm": FLOOD_MM, "steady_width_mm": _r(steady, 3), "other_starts_width_mm": _r(others, 3),
            "dips": dips, "over_steady": ratio(firsts, steady), "over_other_starts": ratio(firsts, others),
            "ahead_speckle": _r(_median(reads[i][2] for i in after_dip), 3)}


def straight(d: Diff, frame: Frame, strokes: list[np.ndarray], technique: str, settings: dict,
             dips_mm: list[float] = ()) -> dict:
    start, lines = _straight(strokes, frame)
    u, v = frame.uv(start)
    length = max(u1 for _, u1, _, _ in lines)
    lo_v, hi_v = min(vk for *_, vk, _ in lines), max(vk for *_, vk, _ in lines)
    lengths = [u1 - u0 for u0, u1, _, _ in lines]
    block = (u >= -KEEP_OUT_MM) & (u <= length + KEEP_OUT_MM) & (v >= lo_v - KEEP_OUT_MM) & (v <= hi_v + KEEP_OUT_MM)
    ring = d.valid & ~block & (u >= -RING_MM[1]) & (u <= length + RING_MM[1]) & (v >= lo_v - RING_MM[1]) \
        & (v <= hi_v + RING_MM[1])
    inside = d.valid & (u >= 0) & (u <= length) & (v >= lo_v - LINE_HALF_MM) & (v <= hi_v + LINE_HALF_MM)
    lines_dl = per_line(d, u, v, lines)
    out = {"dL_mean": _r(d.dl[inside].mean(), 1),
           "da_light": _r(d.da[inside & d.light].mean(), 1) if (inside & d.light).sum() > 20 else None,
           "db_light": _r(d.db[inside & d.light].mean(), 1) if (inside & d.light).sum() > 20 else None,
           "per_line": lines_dl, "ink_width_mm": ink_width(d, u, v, lines, settings.get("pitch_mm")),
           "speckle": {"share": _r((d.dl[ring] > SPECKLE_DL).mean(), 3), "mean": _r(d.dl[ring].mean(), 2)},
           "along": along(d, u, v, lo_v, hi_v, length),
           "past_end": _r(d.dl[d.valid & (u > length) & (u <= length + 1.0) & (v >= lo_v) & (v <= hi_v)].mean(), 1)}
    out["stroke_ends"] = stroke_ends(d, frame, strokes)
    if len(dips_mm):
        out["flood"] = flood(d, u, v, lines, lengths, list(dips_mm), settings.get("pitch_mm"))
    if technique == "carry":
        out["carry"] = carry_cycles(lines_dl, lengths, d.noise, list(dips_mm))
    if technique in ("whip", "solid", "carry"):
        out["bands"] = bands(d, u, v, lo_v, hi_v, length, frame.px_per_mm)
    return out


def _along(pts: np.ndarray, stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's distance from a stroke's polyline and the arc length (mm from its first point) of the nearest
    point on it."""
    best, arc_at, arc = np.full(pts.shape[:-1], np.inf), np.zeros(pts.shape[:-1]), 0.0
    for a, b in zip(stroke[:-1], stroke[1:], strict=True):
        ab = b - a
        length = float(np.linalg.norm(ab))
        t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        dist = np.linalg.norm(pts - (a + t[..., None] * ab), axis=-1)
        nearer = dist < best
        best, arc_at = np.where(nearer, dist, best), np.where(nearer, arc + t * length, arc_at)
        arc += length
    return best, arc_at


def stroke_ends(d: Diff, frame: Frame, strokes: list[np.ndarray]) -> dict:
    """Darkening over each stroke's first ZONE_MM (where the pen came down: `starts`), its last (where it lifted:
    `stops`) and ZONE_MM about each vertex turning TURN_DEG or more (`turns`), against the rest of the strokes'
    lines (`middles`), and each over the middles. Strokes are in drawing order, each from its first point. A stroke
    under three zones long has no middle and counts in none; None where a zone has too few pixels."""
    x, y = frame.page()
    pts = np.stack([x, y], axis=-1)
    zones = {name: np.zeros(x.shape, bool) for name in ("starts", "stops", "turns", "middles")}
    for stroke in strokes:
        seg = np.diff(stroke, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        total = float(lengths.sum())
        if total < 3 * ZONE_MM:
            continue
        dist, arc = _along(pts, stroke)
        near = d.valid & (dist <= LINE_HALF_MM)
        unit = seg / np.maximum(lengths, 1e-12)[:, None]
        turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)))
        at = np.cumsum(lengths)[:-1][turn >= TURN_DEG]
        turning = np.zeros(x.shape, bool)
        for s in at[(at > ZONE_MM) & (at < total - ZONE_MM)]:
            turning |= np.abs(arc - s) <= ZONE_MM / 2
        start, stop = arc <= ZONE_MM, arc >= total - ZONE_MM
        zones["starts"] |= near & start
        zones["stops"] |= near & stop
        zones["turns"] |= near & turning & ~start & ~stop
        zones["middles"] |= near & ~start & ~stop & ~turning
    level = {name: float(d.dl[mask].mean()) if mask.sum() >= 20 else None for name, mask in zones.items()}
    out = {name: _r(value, 1) for name, value in level.items()}
    for name in ("starts", "stops", "turns"):
        both = level[name] is not None and level["middles"] is not None
        out[f"{name}_over_middles"] = _r(level[name] / max(level["middles"], 0.5), 2) if both else None
    return out


def corners(d: Diff, frame: Frame, strokes: list[np.ndarray]) -> dict:
    """Darkening at every stroke end and turn (within END_MM) against the strokes' middles, all together; and the
    starts, the stops and the turns apart (stroke_ends)."""
    x, y = frame.page()
    pts = np.stack([x, y], axis=-1)
    near_line = np.zeros(x.shape, bool)
    for s in strokes:
        for a, b in zip(s[:-1], s[1:], strict=True):
            ab = b - a
            t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            near_line |= np.linalg.norm(pts - (a + t[..., None] * ab), axis=-1) <= LINE_HALF_MM
    vertices = np.vstack(strokes)
    near_vertex = np.min(np.linalg.norm(pts[..., None, :] - vertices, axis=-1), axis=-1) <= END_MM
    ends = d.valid & near_line & near_vertex
    mids = d.valid & near_line & ~near_vertex
    e, m = d.dl[ends].mean(), d.dl[mids].mean()
    return {"dL_mean": _r(d.dl[d.valid & near_line].mean(), 1), "ends": _r(e, 1), "middles": _r(m, 1),
            "ends_over_middles": _r(e / max(m, 0.5), 2), "stroke_ends": stroke_ends(d, frame, strokes)}
