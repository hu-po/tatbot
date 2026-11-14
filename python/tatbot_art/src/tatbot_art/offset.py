"""Offset fill: an ink's regions drawn as passes one mark apart along their edges, made here rather than by DBV3.

DBV3 lays three to four passes over each spot of a band of line art. ECS packs about 4 mm of line per mm² whatever the
pen width, its edge trace goes round both edges of a band out and back, and the edge field breaks its lines into 1-2 mm
dashes. For a chicano lettering piece on the silicone print (2026-11-10), that came to 17 m of line where about 5 m
covers it. This fill is the single-pass baseline for regions, as centreline-trace/1 is for a line drawing's skeleton:

1. **Rings.** The region's edge moved in by `inset_mm` (default half a pitch, so the outer pass's mark ends on the
   edge; less overlaps it), then by one `pitch_mm` more each time. Each is one closed pass where what it encloses
   is at least a pitch across.
2. **Middles.** A band the rings leave bare, under two marks wide or a serif, is drawn once along its middle. The
   middle is the whole region's skeleton, so it runs on unbroken where the band narrows under a mark.
3. **Gaps.** Paper still bare past one-pixel fringes, such as the core between two rings or a strip between two
   passes, gets its own skeleton.

A skeleton pixel's branches are counted round it, so a pixel staircase is one line, not a run of junctions. Branches
are joined where they continue straightest, and spurs shorter than a pitch are pruned. Paths are in canvas mm from
the lower left, y up, ordered nearest-first. A magnum lays its row only across a stroke, so the fill takes round marks
only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from .separate import _disk

ADAPTER = "tatbot-art-offset/1"   # an acquired-artwork adapter (tatbot_contracts.artwork.ACQUIRED_ADAPTERS)
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


@dataclass(frozen=True)
class Offset:
    pitch_mm: float                 # between neighbouring passes: the mark's width
    inset_mm: float | None = None   # the outer pass from the region's edge; None: half a pitch
    tol_mm: float = 0.05            # simplification: points within this of the line are dropped
    smooth_mm: float = 0.25         # each pixel path averaged over this either side of a point
    min_mm: float = 1.0             # shorter paths are dots, which tatbot ros compile drops beside a line
    sliver_mm2: float = 0.3         # an uncovered patch smaller than this is left to the ink's spread


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: an 8-connected skeleton of `mask`, one pixel across."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            # P2..P9 clockwise from north, each as the neighbour's value at every pixel
            p = [np.roll(img, (-dy, -dx), (0, 1)) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
            count = sum(p)
            turns = sum((p[k] == 0) & (p[(k + 1) % 8] == 1) for k in range(8))
            a, b = (p[0] * p[2] * p[4], p[2] * p[4] * p[6]) if step == 0 else (p[0] * p[2] * p[6], p[0] * p[4] * p[6])
            kill = (img == 1) & (count >= 2) & (count <= 6) & (turns == 1) & (a == 0) & (b == 0)
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _branches(skel: np.ndarray) -> np.ndarray:
    """How many branches leave each skeleton pixel: the 0-to-1 steps round its eight neighbours. A staircase pixel,
    touching three neighbours of one line, counts 2; an end 1; a junction 3 or more."""
    img = np.pad(skel.astype(np.uint8), 1)
    ring = [np.roll(img, (-dy, -dx), (0, 1)) for dy, dx in NEIGHBOURS]
    steps = sum((ring[k] == 0) & (ring[(k + 1) % 8] == 1) for k in range(8)).astype(np.int8)
    return (steps * img)[1:-1, 1:-1]


# Orthogonal steps first: on a staircase they visit every pixel, where a diagonal would skip the corner one.
STEPS = sorted(NEIGHBOURS, key=lambda d: abs(d[0]) + abs(d[1]))


def _ahead(cur, prev, on: set, nodes: set, visited: set):
    """The next pixel from `cur`: a node one step away first (it ends the walk), else an unwalked pixel."""
    near = [(cur[0] + dy, cur[1] + dx) for dy, dx in STEPS]
    for q in near:
        if q in nodes and q != prev:
            return q
    return next((q for q in near if q in on and q not in visited and q not in nodes), None)


def _walk(start, first, on: set, nodes: set, visited: set) -> list:
    """Pixels from `start` through `first` to the next node, or to where the line ends."""
    path, prev, cur = [start], start, first
    while cur is not None:
        path.append(cur)
        if cur in nodes:
            break
        visited.add(cur)
        prev, cur = cur, _ahead(cur, prev, on, nodes, visited)
    return path


def _segments(skel: np.ndarray, branches: np.ndarray) -> list[np.ndarray]:
    """The skeleton as pixel paths (row, col) between its nodes, an end or a junction, and its loops; each pixel
    other than a node is walked once."""
    on = set(zip(*np.nonzero(skel), strict=True))
    nodes = {p for p in on if branches[p] != 2}
    visited: set[tuple] = set()
    out = [np.array([node]) for node in nodes if branches[node] == 0]
    for node in sorted(nodes):
        for dy, dx in STEPS:
            q = (node[0] + dy, node[1] + dx)
            if q in on and q not in visited and q not in nodes:
                out.append(np.array(_walk(node, q, on, nodes, visited)))
    for p in sorted(on - nodes - visited):   # loops: no node on them; each closes on its first pixel
        if p not in visited:
            visited.add(p)
            path = _walk(p, _ahead(p, None, on, nodes, visited), on, nodes, visited)
            out.append(np.array(path + [p] if len(path) > 2 else path))
    return out


def _length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0


def _heading(path: np.ndarray, at_start: bool, reach: int = 6) -> np.ndarray:
    """The unit direction a path leaves its end by, looking `reach` pixels in."""
    a, b = (path[0], path[min(reach, len(path) - 1)]) if at_start else (path[-1], path[max(-reach - 1, -len(path))])
    v = (b - a).astype(float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _links(segs: list[np.ndarray], free) -> dict[tuple[int, int], tuple[int, int]]:
    """Each segment end (k, 0 at its start or 1 at its end) at a junction, paired with the end there it continues
    straightest into, where they turn less than 60 degrees."""
    at: dict[tuple, list[tuple[int, int]]] = {}
    for k, s in enumerate(segs):
        if not np.array_equal(s[0], s[-1]):
            for e, p in ((0, s[0]), (1, s[-1])):
                if not free(p):
                    at.setdefault(tuple(p), []).append((k, e))
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for touching in at.values():
        pairs = sorted(((-float(_heading(segs[a], ea == 0) @ _heading(segs[b], eb == 0)), (a, ea), (b, eb))
                        for i, (a, ea) in enumerate(touching) for b, eb in touching[i + 1:] if a != b), reverse=True)
        for straight, x, y in pairs:   # straight: 1 when they continue straight on through the junction
            if straight > 0.5 and x not in link and y not in link:
                link[x], link[y] = y, x
    return link


def _chain(segs: list[np.ndarray], link: dict, k: int, used: set) -> np.ndarray:
    """Segment `k` and those linked on from it, walked from its free end where it has one, as one pixel path."""
    enter = 1 if (k, 0) in link and (k, 1) not in link else 0
    parts, cur = [], (k, enter)
    while cur is not None and cur[0] not in used:
        used.add(cur[0])
        s = segs[cur[0]] if cur[1] == 0 else segs[cur[0]][::-1]
        parts.append(s[1:] if parts else s)
        cur = link.get((cur[0], 1 - cur[1]))
    return np.concatenate(parts)


def _prune_and_join(segments: list[np.ndarray], branches: np.ndarray, spur_px: float) -> list[np.ndarray]:
    """Drop spurs (a free end within `spur_px` of a junction), then join segments through each junction where they
    continue straightest, so a stroke runs on through a crossing instead of lifting at it."""
    def free(p):
        return branches[tuple(p)] <= 1

    segs = [s for s in segments if not ((free(s[0]) != free(s[-1])) and _length(s) < spur_px)]
    link = _links(segs, free)
    used: set[int] = set()
    open_first = [k for k in range(len(segs)) if (k, 0) not in link or (k, 1) not in link]   # then rings of links
    return [_chain(segs, link, k, used) for k in [*open_first, *range(len(segs))] if k not in used]


def _skeleton_paths(mask: np.ndarray, spur_px: float, keep: np.ndarray | None = None) -> list[tuple[np.ndarray, bool]]:
    """`mask`'s skeleton as paths, only where `keep` is set: the middle of a band is taken from the whole band, so it
    runs on unbroken where the band narrows under a mark."""
    skel = thin(mask) if keep is None else thin(mask) & keep
    branches = _branches(skel)
    segs = _prune_and_join(_segments(skel, branches), branches, spur_px)
    return [(s, len(s) > 2 and np.array_equal(s[0], s[-1])) for s in segs]


def _rings(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [(c[:, 0, ::-1], True) for c in contours if len(c) > 2]   # (x, y) -> (row, col)


def _simplify(path: np.ndarray, closed: bool, tol_px: float, smooth_px: int) -> np.ndarray:
    """The pixel path averaged over `smooth_px` pixels, which takes out a skeleton's and a contour's one-pixel
    staircase (its ends stay put), then simplified to within `tol_px`."""
    pts = path.astype(float)
    if len(pts) > 2 * smooth_px + 1 and smooth_px > 0:
        k = np.ones(2 * smooth_px + 1) / (2 * smooth_px + 1)
        if closed:
            wrap = np.vstack([pts[-smooth_px:], pts, pts[:smooth_px]])
            pts = np.column_stack([np.convolve(wrap[:, i], k, mode="valid") for i in (0, 1)])
        else:
            inner = np.column_stack([np.convolve(pts[:, i], k, mode="valid") for i in (0, 1)])
            pts = np.vstack([pts[:smooth_px], inner, pts[-smooth_px:]])
    if len(pts) < 3:
        return pts
    return cv2.approxPolyDP(pts[:, ::-1].astype(np.float32).reshape(-1, 1, 2), tol_px, closed)[:, 0, ::-1].astype(float)


def raster(paths: list[dict], shape: tuple[int, int], px_per_mm: float, width_mm: float) -> np.ndarray:
    """What the paths' marks cover, `width_mm` wide, on a canvas of `shape` pixels."""
    img = np.zeros(shape, np.uint8)
    h = shape[0]
    thick = max(1, round(width_mm * px_per_mm))
    for p in paths:
        pts = np.column_stack([p["points_mm"][:, 0] * px_per_mm - 0.5, h - p["points_mm"][:, 1] * px_per_mm - 0.5])
        pts = np.rint(pts * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], p["closed"], 1, thickness=thick, lineType=cv2.LINE_8, shift=4)
        for x, y in (pts[0, 0], pts[-1, 0]):
            cv2.circle(img, (int(x), int(y)), thick // 2, 1, -1, shift=4)
    return img.astype(bool)


def _order(paths: list[dict]) -> list[dict]:
    """Nearest-first from the canvas origin: each path started at its end (or a closed one at its point) nearest the
    last path's end, so travel between them stays short."""
    left, out, at = list(paths), [], np.zeros(2)
    while left:
        best = None
        for k, p in enumerate(left):
            pts = p["points_mm"]
            if p["closed"]:
                i = int(np.argmin(np.linalg.norm(pts - at, axis=1)))
                d, how = float(np.linalg.norm(pts[i] - at)), i
            else:
                d0, d1 = float(np.linalg.norm(pts[0] - at)), float(np.linalg.norm(pts[-1] - at))
                d, how = (d0, None) if d0 <= d1 else (d1, -1)
            if best is None or d < best[0]:
                best = (d, k, how)
        _, k, how = best
        p = left.pop(k)
        pts = p["points_mm"]
        if p["closed"]:
            pts = np.roll(pts[:-1] if np.array_equal(pts[0], pts[-1]) else pts, -how, axis=0)
        elif how == -1:
            pts = pts[::-1]
        out.append({**p, "points_mm": pts})
        at = pts[0] if p["closed"] else pts[-1]
    return out


def generate(region: np.ndarray, px_per_mm: float, s: Offset) -> tuple[list[dict], dict]:
    """The passes over `region` (bool, row 0 at the top, `px_per_mm`): paths {points_mm, closed, kind} in canvas mm
    from the lower left, y up, and a summary of how they were made and what they cover."""
    if s.pitch_mm <= 0:
        raise ValueError("an offset fill needs a pitch above 0 mm")
    inset = s.pitch_mm / 2 if s.inset_mm is None else s.inset_mm
    h = region.shape[0]
    d = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1] / px_per_mm   # mm to the region's edge
    half = _disk(max(1, round(s.pitch_mm * px_per_mm / 2)))
    spur_px = s.pitch_mm * px_per_mm
    tol_px = s.tol_mm * px_per_mm

    def to_mm(found, kind):
        out = []
        for pix, closed in found:
            pts = _simplify(pix, closed, tol_px, round(s.smooth_mm * px_per_mm))
            mm = np.column_stack([(pts[:, 1] + 0.5) / px_per_mm, (h - pts[:, 0] - 0.5) / px_per_mm])
            if closed or _length(mm) >= s.min_mm:
                out.append({"points_mm": mm, "closed": bool(closed and len(mm) > 2), "kind": kind})
        return out

    def uncovered(paths):
        """The region the paths' marks leave bare, past one-pixel fringes along its edge and specks. A strip two
        pixels across stays: the paper left between two passes down a band just over two marks wide shows."""
        bare = ndi.binary_opening(region & ~raster(paths, region.shape, px_per_mm, s.pitch_mm),
                                  structure=np.ones((2, 2), bool))
        labels, n = ndi.label(bare)
        if not n:
            return bare
        sizes = ndi.sum(bare, labels, index=np.arange(1, n + 1)) / px_per_mm ** 2
        return np.isin(labels, 1 + np.flatnonzero(sizes >= s.sliver_mm2))

    paths, level = [], inset
    while (inner := d >= level).any():   # each level's ring, where what it encloses is a pitch or more across
        paths += to_mm(_rings(ndi.binary_opening(inner, structure=half)), "ring")
        level += s.pitch_mm
    if (bare := uncovered(paths)).any():   # a band too narrow for a ring: its middle, from the whole region's skeleton
        paths += to_mm(_skeleton_paths(region, spur_px, keep=ndi.binary_dilation(bare, structure=_disk(1))), "middle")
    if (bare := uncovered(paths)).any():   # what is left, such as the core between two rings: its own skeleton
        paths += to_mm(_skeleton_paths(bare, spur_px), "gap")
    paths = _order(paths)
    covered = raster(paths, region.shape, px_per_mm, s.pitch_mm)
    area = float(region.sum()) / px_per_mm ** 2
    line = sum(_length(np.vstack([p["points_mm"], p["points_mm"][:1]]) if p["closed"] else p["points_mm"]) for p in paths)
    kinds = {k: sum(1 for p in paths if p["kind"] == k) for k in ("ring", "middle", "gap")}
    return paths, {**asdict(s), "inset_mm": round(inset, 4), "paths": len(paths), "by_kind": kinds,
                   "line_mm": round(line, 1), "area_mm2": round(area, 1),
                   "line_per_mm2": round(line / area, 3) if area else None,
                   "covered": round(float((covered & region).sum() / max(1, region.sum())), 4),
                   "outside_mm2": round(float((covered & ~region).sum()) / px_per_mm ** 2, 1)}


def artwork(paths: list[dict], size_mm: tuple[float, float], pen_id: str, colour_srgb, info: dict, *, name: str,
            source_sha256: str, identifier: str) -> dict:
    """The passes as an acquired artwork `tatbot ros compile` places and draws (tatbot.inkmap-artwork/2): canvas
    metres from its lower left, y up, each path a pitch wide. Its recipe digest binds the settings (generate's
    `info`) to the region source's bytes (`source_sha256`). `pen_id` names the layer's ink as a DBV3 pen would, so
    a tatbot-inks/3 binding finds it the same way."""
    from tatbot_contracts.artwork import freeze_artwork
    from tatbot_contracts.canonical import canonical_digest
    from tatbot_contracts.paths import freeze_program

    w, h = size_mm

    def canvas(x, y):
        return [round(min(max(float(x), 0.0), w) / 1000.0, 9), round(min(max(float(y), 0.0), h) / 1000.0, 9)]

    elements = [{"id": f"offset-{k + 1}", "kind": "path", "closed": p["closed"], "fill": False,
                 "width_m": round(info["pitch_mm"] / 1000.0, 7), "deposition": 1.0,
                 "points_m": [canvas(x, y) for x, y in p["points_mm"]]}
                for k, p in enumerate(paths)]
    rgb = [round(min(max(float(c), 0.0), 1.0), 6) for c in colour_srgb]
    geometry = {"canvas_m": {"width": w / 1000.0, "height": h / 1000.0}, "inks": [{"id": pen_id, "color_srgb": rgb}],
                "layers": [{"id": "offset", "ink_id": pen_id, "elements": elements}], "negative_space_masks": []}
    program, _ = freeze_program(geometry, source_sha256=source_sha256, name=name, adapter=ADAPTER)
    source = {"kind": "imported", "identifier": identifier, "license": None, "attribution": None, "generation": None}
    settings = {k: v for k, v in info.items() if k in Offset.__dataclass_fields__}
    recipe = canonical_digest({"adapter": ADAPTER, "source_sha256": source_sha256, "offset": settings})
    conversion = {"adapter": ADAPTER, "recipe_sha256": recipe, "chord_error_m": float(info["tol_mm"]) / 1000.0}
    return freeze_artwork(program, name=name, source=source, conversion=conversion)


def mark_mm(tool_id: str) -> float:
    """A round cartridge's mark (tatbot_art.cartridge): the default pitch. A magnum lays its row only across a stroke,
    so offset passes, which turn every way, are refused for one."""
    from . import cartridge

    c = cartridge.load(tool_id)
    if c.kind == "magnum":
        raise ValueError(f"{tool_id} is a magnum: offset passes turn every way, and it lays its row only across them")
    return round(c.swath_mm(0.0), 3)

