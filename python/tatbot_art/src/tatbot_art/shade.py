"""Whip and pendulum shading: a tone image as strokes whose deposition follows the tone along them.

Tone is darkness, 1 - luminance: 1 is full ink, 0 none. Strokes run along one direction, from dark to light (the
tone's mean descent unless an angle is given), `pitch_mm` apart, and each point's deposition is the tone under it,
so the predicted render's coverage follows the image. Where the tone is under `floor` no stroke runs.

- whip: one stroke per run of tone, begun at its darker end, fading as the tone does. Every whip pays its own
  descent and lift on the robot.
- pendulum: neighbouring runs joined into one zigzag where their ends lie within `join_mm`, its turns at the light
  end lifted out (deposition 0) and at the dark end kept in, so a region pays one descent and lift.

Deposition never rises along a stroke faster than 1 per `rise_mm`: the needle goes in no steeper than the planner's
skin ride (a 0.5 slope over an assumed 2 mm lift), so a pendulum's swings toward a steep dark edge reach it lighter.
A stroke's start is its descent, which goes straight to depth, so a steep climb there is trimmed, not ramped.
Deposition below 1 is a lift on the robot: `artwork` freezes the strokes for `tatbot ros compile`, which draws them
with a tool whose datasheet records its fade_lift_mm; the render previews them.

The cartridge sets the mark (tatbot_art.cartridge). A round liner's strokes follow the tone's descent and lie its
width apart. A magnum's run across its row, the way nearer the descent, a swath apart: the robot
holds one heading per page, so a tone whose descent runs along the row cannot be whipped with that magnum on that
page, and `off_descent_deg` says how far the strokes stray from it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

TECHNIQUES = ("whip", "pendulum")
ADAPTER = "tatbot-art-shade/1"   # an acquired-artwork adapter (tatbot_contracts.artwork.ACQUIRED_ADAPTERS)
MIN_RUN_MM = 1.0   # shorter runs are dots, which tatbot ros compile drops beside a line (tatbot_ink.place MIN_STROKE_M)


@dataclass(frozen=True)
class Shading:
    technique: str = "whip"
    angle_deg: float | None = None   # stroke direction, dark to light, counter-clockwise from +x; None: from the tone
    pitch_mm: float | None = None    # between neighbouring strokes; None: the mark's width, so tone 1 is solid
    step_mm: float = 0.25            # tone sampled this finely along a stroke
    floor: float = 0.05              # tone under this draws nothing
    rise_mm: float = 4.0             # deposition rises from 0 to 1 over at least this much path
    join_mm: float | None = None     # a pendulum joins runs whose ends are this close; None: 3 mm or two pitches
    tol_mm: float = 0.02             # simplification: points within this of the line kept (deposition 0.02 counts as it)


def tone_from_image(path, width_mm: float, px_per_mm: float) -> np.ndarray:
    """(H, W) darkness in 0..1, row 0 at the top, `width_mm` wide at `px_per_mm`; transparency is paper."""
    from PIL import Image

    from .image import on_paper

    with Image.open(path) as im:
        grey = on_paper(im).convert("L")
        w = max(1, int(round(width_mm * px_per_mm)))
        grey = grey.resize((w, max(1, int(round(grey.height * w / grey.width)))), Image.Resampling.LANCZOS)
    return 1.0 - np.asarray(grey, float) / 255.0


def _sampler(tone: np.ndarray, px_per_mm: float):
    """Tone at page points (mm, origin at the image centre, y up), bilinear, 0 outside."""
    h, w = tone.shape

    def at(pts_mm: np.ndarray) -> np.ndarray:
        cols = (pts_mm[:, 0] + w / (2 * px_per_mm)) * px_per_mm - 0.5
        rows = (h / (2 * px_per_mm) - pts_mm[:, 1]) * px_per_mm - 0.5
        return map_coordinates(tone, [rows, cols], order=1, mode="constant", cval=0.0)
    return at


def descent_angle(tone: np.ndarray, floor: float = 0.05) -> float:
    """The direction (deg, y up) from dark to light: against the tone's mean gradient inside where it inks (its
    hard edges eroded away, or they would outweigh the fade); 0 when flat."""
    g_rows, g_cols = np.gradient(gaussian_filter(tone, 1.0))
    inked = binary_erosion(tone >= floor, iterations=3)
    if not inked.any():
        inked = tone >= floor
    if not inked.any():
        return 0.0
    gx, gy = float(g_cols[inked].mean()), float(-g_rows[inked].mean())
    return 0.0 if math.hypot(gx, gy) < 1e-6 else math.degrees(math.atan2(-gy, -gx))


def runs(tone: np.ndarray, px_per_mm: float, s: Shading, angle_deg: float) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Per scan line (in order across the image), its runs of tone at or over the floor, each (points mm, tone),
    sampled every step_mm along the stroke direction."""
    at = _sampler(tone, px_per_mm)
    a = math.radians(angle_deg)
    d, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    reach = math.hypot(*tone.shape) / (2 * px_per_mm) + s.pitch_mm
    t = np.arange(-reach, reach + s.step_mm, s.step_mm)
    lines = []
    for offset in np.arange(-reach, reach + s.pitch_mm, s.pitch_mm):
        pts = offset * n + t[:, None] * d
        value = at(pts)
        inked = np.flatnonzero(value >= s.floor)
        found = []
        if len(inked):
            for chunk in np.split(inked, np.flatnonzero(np.diff(inked) > 1) + 1):
                if (len(chunk) - 1) * s.step_mm >= MIN_RUN_MM:
                    found.append((pts[chunk], np.clip(value[chunk], 0.0, 1.0)))
        lines.append(found)
    return [line for line in lines if line]


def start_in(pts: np.ndarray, dep: np.ndarray, rise_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """A stroke without its leading samples while the tone climbs faster than 1 per `rise_mm` (an edge's
    antialiasing): it descends where the climb levels off, straight to that depth, instead of ramping in."""
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    i = 0
    while i < len(dep) - 2 and dep[i + 1] - dep[i] > ds[i] / rise_mm:
        i += 1
    return pts[i:], dep[i:]


def limit_rise(pts: np.ndarray, dep: np.ndarray, rise_mm: float) -> np.ndarray:
    """`dep` lowered wherever it would rise faster along `pts` than 1 per `rise_mm`."""
    out = dep.astype(float).copy()
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] + ds[i - 1] / rise_mm)
    return out


def simplify(pts: np.ndarray, dep: np.ndarray, tol_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Douglas-Peucker over (x, y, deposition), deposition scaled so 0.02 of it weighs as tol_mm."""
    xyz = np.column_stack([pts, dep * (tol_mm / 0.02)])
    keep = np.zeros(len(xyz), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xyz) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        seg, rel = xyz[hi] - xyz[lo], xyz[lo + 1:hi] - xyz[lo]
        u = np.clip((rel @ seg) / (seg @ seg), 0.0, 1.0) if seg @ seg > 0 else np.zeros(len(rel))
        dist = np.linalg.norm(rel - u[:, None] * seg, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol_mm:
            stack += [(lo, lo + 1 + k), (lo + 1 + k, hi)]
            keep[lo + 1 + k] = True
    return pts[keep], dep[keep]


def _darker_first(pts: np.ndarray, dep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    third = max(1, len(dep) // 3)
    return (pts, dep) if dep[:third].mean() >= dep[-third:].mean() else (pts[::-1], dep[::-1])


def whips(lines) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_darker_first(pts, dep) for line in lines for pts, dep in line]


def pendulums(lines, at, s: Shading) -> list[tuple[np.ndarray, np.ndarray]]:
    """Runs chained line by line into zigzags: each next run is the one on the following line with an end within
    join_mm of the chain's end, entered from that end. A join at the light end is lifted out (both ends 0); one
    through untoned paper (a notch between two dark ends) is refused and starts a new chain."""
    pool = [list(line) for line in lines]
    chains = []
    for k in range(len(pool)):
        while pool[k]:
            pts, dep = _darker_first(*pool[k].pop(0))
            parts_p, parts_d, line = [pts], [dep], k
            while line + 1 < len(pool):
                end, end_dep = parts_p[-1][-1], parts_d[-1][-1]
                best = None
                for i, (p, _dep) in enumerate(pool[line + 1]):
                    for flip in (False, True):
                        start = p[-1] if flip else p[0]
                        gap = float(np.linalg.norm(start - end))
                        if gap <= s.join_mm and (best is None or gap < best[0]):
                            best = (gap, i, flip)
                if best is None:
                    break
                _, i, flip = best
                p, q = pool[line + 1][i]
                p, q = (p[::-1], q[::-1]) if flip else (p, q)
                light = max(end_dep, q[0]) <= 2 * s.floor
                if not light and at(((end + p[0]) / 2)[None])[0] < s.floor:
                    break
                pool[line + 1].pop(i)
                if light:
                    parts_d[-1] = np.r_[parts_d[-1][:-1], 0.0]
                    q = np.r_[0.0, q[1:]]
                parts_p.append(p)
                parts_d.append(q)
                line += 1
            chains.append((np.vstack(parts_p), np.concatenate(parts_d)))
    return chains


def stroke_angle(descent_deg: float, cartridge) -> float:
    """A round mark follows the tone's descent; a magnum runs across its row, whichever way across is nearer it."""
    across = cartridge.across_deg()
    if across is None:
        return descent_deg
    angle = across if math.cos(math.radians(descent_deg - across)) >= 0 else across + 180.0
    return (angle + 180.0) % 360.0 - 180.0


def generate(tone: np.ndarray, px_per_mm: float, s: Shading, cartridge) -> tuple[list[dict], dict]:
    """The strokes (page mm: points and deposition) a `cartridge` (tatbot_art.cartridge.Cartridge) draws, and a
    summary of how they were made: the angle, the tone's descent and the mark at that angle."""
    if s.technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {', '.join(TECHNIQUES)}")
    descent = descent_angle(tone, s.floor)
    angle = stroke_angle(descent, cartridge) if s.angle_deg is None else float(s.angle_deg)
    pitch = s.pitch_mm or cartridge.swath_mm(angle)
    s = replace(s, pitch_mm=pitch, join_mm=s.join_mm or max(3.0, 2.0 * pitch))
    lines = runs(tone, px_per_mm, s, angle)
    raw = whips(lines) if s.technique == "whip" else pendulums(lines, _sampler(tone, px_per_mm), s)
    strokes = []
    for pts, dep in raw:
        pts, dep = start_in(pts, dep, s.rise_mm)
        pts, dep = simplify(pts, limit_rise(pts, dep, s.rise_mm), s.tol_mm)
        strokes.append({"points_mm": pts, "deposition": dep})
    off = abs((angle - descent + 180.0) % 360.0 - 180.0)
    return strokes, {**asdict(s), "angle_deg": round(angle, 2), "descent_deg": round(descent, 2),
                     "off_descent_deg": round(off, 1), "runs": sum(map(len, lines)), "mark": cartridge.describe(angle)}


def artwork(strokes: list[dict], size_mm: tuple[float, float], ink_id: str, info: dict, *, name: str,
            source_sha256: str, identifier: str) -> dict:
    """The strokes as an acquired artwork `tatbot ros compile` places and draws (tatbot.inkmap-artwork/2): canvas
    metres from its lower left, y up, each path at the mark's width with its deposition per point. Its recipe digest
    binds the settings and mark that made it (generate's `info`) to the tone image's bytes (`source_sha256`)."""
    from tatbot_contracts.artwork import freeze_artwork
    from tatbot_contracts.canonical import canonical_digest
    from tatbot_contracts.paths import freeze_program

    w, h = size_mm

    def canvas(x, y):   # page mm about the centre to canvas metres from the lower left, inside it
        return [round(min(max(float(x) + w / 2, 0.0), w) / 1000.0, 9), round(min(max(float(y) + h / 2, 0.0), h) / 1000.0, 9)]

    elements = [{"id": f"shade-{k + 1}", "kind": "path", "closed": False, "fill": False,
                 "width_m": round(info["mark"]["width_mm"] / 1000.0, 7),
                 "deposition": [round(float(d), 4) for d in st["deposition"]],
                 "points_m": [canvas(x, y) for x, y in st["points_mm"]]} for k, st in enumerate(strokes)]
    geometry = {"canvas_m": {"width": w / 1000.0, "height": h / 1000.0}, "inks": [{"id": ink_id, "color_srgb": [0, 0, 0]}],
                "layers": [{"id": "shade", "ink_id": ink_id, "elements": elements}], "negative_space_masks": []}
    program, _ = freeze_program(geometry, source_sha256=source_sha256, name=name, adapter=ADAPTER)
    source = {"kind": "imported", "identifier": identifier, "license": None, "attribution": None, "generation": None}
    recipe = canonical_digest({"adapter": ADAPTER, "source_sha256": source_sha256, "shading": info})
    return freeze_artwork(program, name=name, source=source,
                          conversion={"adapter": ADAPTER, "recipe_sha256": recipe, "chord_error_m": info["tol_mm"] / 1000.0})


def preview_program(strokes: list[dict], size_mm: tuple[float, float], ink_id: str, mark: dict) -> dict:
    """The strokes as a program `tatbot art render` draws with the `mark` (generate's summary's: a round mark's
    width, a magnum's row), timed by the planner's own estimate. Not executable: it has no format or page
    measurement; `artwork` is what compiles."""
    from tatbot_motion import load_motion
    from tatbot_motion.estimate import estimate_duration

    ops = [{"op": "stroke", "id": f"s{k:04d}", "resource_id": ink_id, "generation_width_m": mark["width_mm"] / 1000.0,
            "points_m": (st["points_mm"] / 1000.0).tolist(), "deposition": np.round(st["deposition"], 4).tolist()}
           for k, st in enumerate(strokes)]
    tool = {"id": mark["tool_id"], **({"magnum": mark["magnum"]} if mark["magnum"] else {})}
    motion = load_motion()
    travel = sum(float(np.linalg.norm(b["points_mm"][0] - a["points_mm"][-1])) for a, b in zip(strokes, strokes[1:],
                                                                                              strict=False)) / 1000.0
    timing = estimate_duration(ops, float(motion["tip_speed"]["draw_default_m_s"]), travel, motion)
    return {"page": {"size_m": [size_mm[0] / 1000.0, size_mm[1] / 1000.0]},
            "resources": [{"id": ink_id, "ink_id": ink_id, "tool": tool}], "ops": ops,
            "stats": {"strokes": len(ops), "travel_m": round(travel, 4), "time_estimate": timing}}
