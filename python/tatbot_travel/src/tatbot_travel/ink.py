"""Ink on the forearm: Sharpie lines and tatbot flash tattoos, and the strokes they make.

The demo traces anything on the skin that is not skin: a red or black Sharpie
line, or the linework of tattoos (the practice arm is ideally covered in tatbot
designs). Every ink item is rasterised in millimetres -- a line as a stroked
polyline, acquired artwork through its shared metric review renderer -- and
remapped into the skin shell's texture, so ink wraps the arm the way a decal
would. The traced strokes come from the same texture: the ink mask is thinned
to its centrelines and split into a graph of strokes between ends and
junctions, each lifted to the skin through the shell. What the camera sees and
what the expert follows are therefore one object.
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from tatbot_travel.lines import DrawnLine
from tatbot_travel.shell import SkinShell

TEXTURE_SHAPE = (1024, 1280)  # rows around the arm (theta), columns along it (x)
RASTER_MM = 0.1  # millimetres per pixel of an item's own raster
# Marker and ballpoint lines; the lab's practice arm carries purple-blue ballpoint scribbles.
# The rig's practice arm carries violet and blue ballpoint: most ink here is that, the rest marker and flash.
SHARPIE = {"black": ((22, 22, 26), 0.2), "red": ((175, 30, 38), 0.1), "blue": ((38, 52, 150), 0.3),
           "purple": ((92, 58, 150), 0.4)}
TATTOO_INKS = [((18, 18, 22), 0.35), ((25, 40, 120), 0.2), ((150, 25, 30), 0.07), ((30, 90, 50), 0.05),
               ((80, 35, 110), 0.28), ((95, 60, 35), 0.05)]
BALLPOINT = [((70, 50, 140), 0.5), ((95, 60, 150), 0.3), ((40, 55, 145), 0.2)]  # violet, purple, blue
PROCEDURAL_SHARE = 0.6  # of tattoos: designs drawn fresh (designs.py) rather than from the library
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---- texture <-> chart ------------------------------------------------------------------------
def texel_chart(shell: SkinShell, shape=TEXTURE_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Chart coordinates of texel centres: x per column, theta per row (GL: row 0 is v = 1)."""
    h, w = shape
    x = shell.x[0] + (shell.x[-1] - shell.x[0]) * np.arange(w) / (w - 1)
    theta = np.pi - 2.0 * np.pi * np.arange(h) / (h - 1)
    return x, theta


def chart_to_texel(shell: SkinShell, x, theta, shape=TEXTURE_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    cols = (np.asarray(x) - shell.x[0]) / (shell.x[-1] - shell.x[0]) * (w - 1)
    rows = (np.pi - np.asarray(theta)) / (2.0 * np.pi) * (h - 1)
    return cols, rows


# ---- ink items ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class InkItem:
    """A raster of ink coverage in its own millimetre frame, placed on the arm's metric chart."""

    alpha: np.ndarray  # (h, w) float32 coverage, RASTER_MM per pixel
    centre: tuple[float, float]  # (x m, arc m) of the raster's centre on the chart
    angle: float  # rotation of the raster on the chart (rad)
    colour: tuple[float, float, float]
    kind: str  # "sharpie-<colour>", a design id, or "procedural"
    opacity: float = 1.0  # how dark the ink shows; faded ink is still ink to trace


@lru_cache(maxsize=1)
def design_manifest() -> list[dict]:
    from tatbot_sim.inkmap.collection import collection_entries

    return list(collection_entries())


@lru_cache(maxsize=1)
def library() -> list[dict]:
    """Reviewed acquired artwork only; preview files are derived representations."""
    return design_manifest()


@lru_cache(maxsize=64)
def design_raster(design_id: str, size_mm: float) -> np.ndarray:
    """A flash design's ink coverage at RASTER_MM per pixel, its longer side ``size_mm``."""
    import cairosvg
    from PIL import Image
    from tatbot_contracts.artwork import canvas_m
    from tatbot_contracts.paths import render_paths
    from tatbot_sim.inkmap.collection import artwork_record

    entry = next(d for d in library() if d["id"] == design_id)
    record = artwork_record(entry)
    if not np.isclose(size_mm, max(canvas_m(record)) * 1000, rtol=0, atol=1e-6):
        raise ValueError("physical resizing requires DBV3 regeneration at the requested size")
    svg = render_paths({key: record["program"][key] for key in ("canvas_m", "inks", "layers", "negative_space_masks")})
    px = max(8, int(round(size_mm / RASTER_MM)))
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=px)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")).astype(np.float32) / 255.0
    darkness = 1.0 - rgba[..., :3].mean(axis=-1)
    alpha = rgba[..., 3] * np.clip(darkness * 1.4, 0, 1)
    if alpha.shape[0] > alpha.shape[1]:  # keep the longer side as requested
        scale = px / alpha.shape[0]
        alpha = cv2.resize(alpha, (max(8, int(alpha.shape[1] * scale)), px), interpolation=cv2.INTER_AREA)
    return alpha


def outline_version(alpha: np.ndarray, pen_mm: float) -> np.ndarray:
    """The design as a pen would draw it: its centrelines at a pen's width, not its fills."""
    from skimage.morphology import skeletonize

    skeleton = skeletonize(alpha > 0.5).astype(np.uint8)
    radius = max(1, int(round(0.5 * pen_mm / RASTER_MM)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(skeleton, kernel).astype(np.float32)


def line_item(rng: np.random.Generator, chart: np.ndarray, shell: SkinShell) -> InkItem:
    """A Sharpie line along a chart curve, rasterised in the arm's metric chart."""
    metric = np.stack([chart[:, 0], chart[:, 1] * shell.radius_at(chart[:, 0])], axis=1) * 1000.0  # mm
    width_mm = float(rng.uniform(0.8, 3.0))
    lo = metric.min(axis=0) - width_mm - 1.0
    size = metric.max(axis=0) + width_mm + 1.0 - lo
    raster = np.zeros((int(size[1] / RASTER_MM) + 1, int(size[0] / RASTER_MM) + 1), np.float32)
    pts = np.round((metric - lo) / RASTER_MM * 16).astype(np.int32)  # (x, arc) -> (col, row), 4 fractional bits
    thickness = max(1, int(round(width_mm / RASTER_MM)))
    cv2.polylines(raster, [pts], False, 1.0, thickness=thickness, lineType=cv2.LINE_AA, shift=4)
    name = str(rng.choice(list(SHARPIE), p=[w for _, w in SHARPIE.values()]))
    colour = np.array(SHARPIE[name][0], np.float32) * rng.uniform(0.85, 1.15)
    centre = (lo + 0.5 * size) / 1000.0
    return InkItem(alpha=raster, centre=(float(centre[0]), float(centre[1])), angle=0.0,
                   colour=tuple(colour), kind=f"sharpie-{name}", opacity=float(rng.uniform(0.85, 1.0)))


def designs(held_out: bool = False) -> list[dict]:
    """Flash to draw from: train-split artwork and the (unsplit) preview pieces, or the held-out artwork."""
    splits = {"validation", "test"} if held_out else {"train", None}
    return [d for d in library() if d.get("usage") in ("artwork", "preview") and d.get("split") in splits]


def tattoo_size_mm(rng: np.random.Generator, entry: dict) -> float:
    """A design's longer side, around the size tatbot tattoos it at."""
    if "artwork" in entry:
        from tatbot_contracts.artwork import canvas_m

        return float(max(canvas_m(entry["artwork"])) * 1000)
    if "size_range_mm" in entry:
        lo, hi = entry["size_range_mm"]
        return float(rng.uniform(0.8 * lo, 1.4 * hi))
    return float(max(entry["default_size_mm"]) * rng.uniform(0.7, 1.2))


def _flash(rng: np.random.Generator, held_out: bool) -> tuple[np.ndarray, str]:
    """A design's coverage and name: drawn fresh, or from the library (held-out designs only when asked)."""
    if not held_out and rng.random() < PROCEDURAL_SHARE:
        from tatbot_travel import designs as procedural_flash

        return procedural_flash.procedural(rng, float(rng.uniform(15.0, 90.0)), RASTER_MM), "procedural"
    choices = designs(held_out)
    entry = choices[int(rng.integers(len(choices)))]
    alpha = design_raster(entry["id"], round(tattoo_size_mm(rng, entry)))
    if rng.random() < 0.5:
        alpha = outline_version(alpha, pen_mm=float(rng.uniform(0.6, 1.4)))
    return alpha, entry["id"]


def tattoo_item(rng: np.random.Generator, shell: SkinShell, held_out: bool = False) -> InkItem:
    """A tattoo: fresh or library flash at a tattoo's size, sometimes faded."""
    alpha, kind = _flash(rng, held_out)
    if rng.random() < 0.5:
        alpha = alpha[:, ::-1].copy()
    inks = [c for c, _ in TATTOO_INKS]
    colour = np.array(inks[rng.choice(len(inks), p=[w for _, w in TATTOO_INKS])], np.float32)
    x = rng.uniform(shell.x[0] + 0.02, shell.x[-1] - 0.02)
    arc = rng.normal(0.0, 0.8) * float(shell.radius_at(x))  # mostly on the back of the forearm
    opacity = float(rng.uniform(0.45, 1.0)) if rng.random() < 0.3 else 1.0
    return InkItem(alpha=alpha, centre=(float(x), float(arc)), angle=float(rng.uniform(-np.pi, np.pi)),
                   colour=tuple(colour * rng.uniform(0.85, 1.15)), kind=kind, opacity=opacity)


def photo_item(rng: np.random.Generator, shell: SkinShell, alpha_px: np.ndarray) -> InkItem:
    """Ink cut from a frame of the rig's practice arm (``realbank``), in ballpoint colours at a tattoo's size."""
    scale = float(rng.uniform(20.0, 70.0)) / (max(alpha_px.shape) * RASTER_MM)
    alpha = cv2.resize(alpha_px, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    colours = [c for c, _ in BALLPOINT]
    colour = np.array(colours[rng.choice(len(colours), p=[w for _, w in BALLPOINT])], np.float32)
    x = rng.uniform(shell.x[0] + 0.02, shell.x[-1] - 0.02)
    arc = rng.normal(0.0, 0.8) * float(shell.radius_at(x))  # mostly on the back of the forearm
    return InkItem(alpha=alpha.astype(np.float32), centre=(float(x), float(arc)),
                   angle=float(rng.uniform(-np.pi, np.pi)), colour=tuple(colour * rng.uniform(0.85, 1.15)),
                   kind="photo", opacity=float(rng.uniform(0.6, 1.0)))


# ---- composing ink into the shell texture --------------------------------------------------------
def _remap_item(item: InkItem, shell: SkinShell, shape=TEXTURE_SHAPE) -> tuple[np.ndarray, tuple]:
    """Coverage of one item in texture space, cropped to its bounding box: (alpha, (r0, r1, c0, c1))."""
    h, w = item.alpha.shape
    half = 0.5 * np.hypot(h, w) * RASTER_MM / 1000.0
    cx, carc = item.centre
    radius = float(shell.radius_at(cx))
    c0, r1 = chart_to_texel(shell, cx - half, (carc - half) / radius, shape)
    c1, r0 = chart_to_texel(shell, cx + half, (carc + half) / radius, shape)
    r0, r1 = int(max(0, np.floor(r0) - 2)), int(min(shape[0], np.ceil(r1) + 3))
    c0, c1 = int(max(0, np.floor(c0) - 2)), int(min(shape[1], np.ceil(c1) + 3))
    if r0 >= r1 or c0 >= c1:
        return np.zeros((0, 0), np.float32), (0, 0, 0, 0)
    xs, thetas = texel_chart(shell, shape)
    gx, gt = np.meshgrid(xs[c0:c1], thetas[r0:r1])
    dx = gx - cx
    darc = gt * shell.radius_at(gx) - carc
    cos, sin = np.cos(-item.angle), np.sin(-item.angle)
    u = (cos * dx - sin * darc) * 1000.0 / RASTER_MM + 0.5 * w
    v = (sin * dx + cos * darc) * 1000.0 / RASTER_MM + 0.5 * h
    alpha = cv2.remap(item.alpha, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return alpha, (r0, r1, c0, c1)


def compose(skin: np.ndarray, items: list[InkItem], shell: SkinShell) -> tuple[np.ndarray, np.ndarray]:
    """Skin texture with the ink items painted in; returns (RGB uint8 texture, ink coverage)."""
    texture = skin.astype(np.float32)
    coverage = np.zeros(skin.shape[:2], np.float32)
    for item in items:
        alpha, (r0, r1, c0, c1) = _remap_item(item, shell, skin.shape[:2])
        if alpha.size == 0:
            continue
        a = alpha[..., None] * item.opacity
        texture[r0:r1, c0:c1] = texture[r0:r1, c0:c1] * (1 - a) + np.asarray(item.colour, np.float32) * a
        coverage[r0:r1, c0:c1] = np.maximum(coverage[r0:r1, c0:c1], alpha)
    return np.clip(texture, 0, 255).astype(np.uint8), coverage


# ---- the stroke graph -----------------------------------------------------------------------------
@dataclass
class InkGraph:
    """Traceable strokes: edges between ends and junctions of the thinned ink."""

    edges: list[DrawnLine]
    ends: list[tuple[int, int]]  # (node at arclength 0, node at the far end) per edge
    nodes: dict[int, list[tuple[int, bool]]] = field(default_factory=dict)  # node -> (edge, starts here)
    points: np.ndarray = field(init=False)  # every ink point, phantom frame
    normals: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)  # (edge, sample) per point

    def __post_init__(self):
        self.nodes = {}
        for e, (a, b) in enumerate(self.ends):
            self.nodes.setdefault(a, []).append((e, True))
            self.nodes.setdefault(b, []).append((e, False))
        self.points = np.vstack([e.points for e in self.edges]) if self.edges else np.zeros((0, 3))
        self.normals = np.vstack([e.normals for e in self.edges]) if self.edges else np.zeros((0, 3))
        self.index = (np.vstack([np.stack([np.full(len(e.points), k), np.arange(len(e.points))], axis=1)
                                  for k, e in enumerate(self.edges)]) if self.edges else np.zeros((0, 2), int))

    @property
    def total_length(self) -> float:
        return float(sum(e.length for e in self.edges))

    def surface_arrays(self) -> dict[str, np.ndarray]:
        """Static tracing geometry with recoverable canonical skin addresses."""
        if any(e.face_indices is None or e.barycentric is None for e in self.edges):
            raise ValueError("generated ink strokes must retain SOMA surface addresses")
        return {"stroke_offsets": np.r_[0, np.cumsum([len(e.points) for e in self.edges])],
                "face_indices": (np.concatenate([e.face_indices for e in self.edges])
                                 if self.edges else np.zeros(0, dtype=np.int32)),
                "barycentric": (np.concatenate([e.barycentric for e in self.edges])
                                if self.edges else np.zeros((0, 3))),
                "points_m": self.points, "normals": self.normals,
                "chart_points": (np.concatenate([e.chart for e in self.edges])
                                 if self.edges else np.zeros((0, 2)))}

    def nearest(self, local_point: np.ndarray) -> tuple[int, float]:
        """(edge, arclength) of the ink point nearest a point in the phantom frame."""
        k = int(np.argmin(np.linalg.norm(self.points - local_point, axis=1)))
        e, i = self.index[k]
        return int(e), float(self.edges[e].arclength[i])

    def nearest_other(self, local_point: np.ndarray, exclude: set[int]) -> tuple[int, float, float]:
        """Nearest ink point on an edge not in ``exclude``: (edge, arclength, distance)."""
        mask = ~np.isin(self.index[:, 0], list(exclude))
        if not mask.any():
            return -1, 0.0, np.inf
        dist = np.linalg.norm(self.points[mask] - local_point, axis=1)
        k = int(np.argmin(dist))
        e, i = self.index[mask][k]
        return int(e), float(self.edges[e].arclength[i]), float(dist[k])

    def continuations(self, edge: int, at_far_end: bool) -> list[tuple[int, bool]]:
        """Edges leaving the node at one end of ``edge``: (edge, forward along it).

        Only the way in is excluded, so a closed loop continues into itself.
        """
        node = self.ends[edge][1 if at_far_end else 0]
        arrived = (edge, not at_far_end)
        return [option for option in self.nodes[node] if option != arrived]


class _SkeletonWalk:
    """Walks thinned ink pixel by pixel between its nodes (ends and junctions, labelled > 0)."""

    def __init__(self, skel: np.ndarray, labels: np.ndarray):
        self.skel, self.labels = skel, labels
        self.visited = np.zeros_like(skel, dtype=bool)

    def neighbours(self, r: int, c: int):
        h, w = self.skel.shape
        for dr, dc in _NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and self.skel[rr, cc]:
                yield rr, cc

    def _step(self, r: int, c: int, prev: tuple[int, int]) -> tuple[int, int] | None:
        """The next pixel along: a node if one is adjacent, else an unwalked pixel."""
        onward = None
        for rr, cc in self.neighbours(r, c):
            if (rr, cc) == prev:
                continue
            if self.labels[rr, cc]:
                return rr, cc
            if not self.visited[rr, cc]:
                onward = (rr, cc)
        return onward

    def walk(self, r: int, c: int, prev: tuple[int, int], start: int) -> tuple[list, int]:
        """From ``prev`` through (r, c) to the next node: (pixels, that node's label)."""
        path = [prev, (r, c)]
        while self.labels[r, c] == 0:
            self.visited[r, c] = True
            onward = self._step(r, c, path[-2])
            if onward is None:
                return path, start  # a closed loop walked all the way round
            r, c = onward
            path.append(onward)
        return path, int(self.labels[r, c])

    def from_nodes(self) -> tuple[list, list]:
        """Every path leaving a node."""
        paths, ends = [], []
        for r0, c0 in zip(*np.nonzero(self.labels), strict=True):
            for r, c in self.neighbours(r0, c0):
                if not self.visited[r, c] and self.labels[r, c] == 0:
                    path, end = self.walk(r, c, (r0, c0), int(self.labels[r0, c0]))
                    paths.append(path)
                    ends.append((int(self.labels[r0, c0]), end))
        return paths, ends

    def loops(self) -> tuple[list, list]:
        """Closed loops have no node: label a pixel on each and walk round from it."""
        paths, ends = [], []
        label = int(self.labels.max()) + 1
        for r, c in zip(*np.nonzero(self.skel & (self.labels == 0)), strict=True):
            if self.visited[r, c]:
                continue
            self.visited[r, c] = True
            self.labels[r, c] = label
            onward = next(((rr, cc) for rr, cc in self.neighbours(r, c) if not self.visited[rr, cc]), None)
            if onward is not None:
                path, _ = self.walk(*onward, (r, c), label)
                paths.append(path)
                ends.append((label, label))
            label += 1
        return paths, ends


def _skeleton_graph(mask: np.ndarray) -> tuple[list, list]:
    """Pixel skeleton -> (paths as pixel sequences, their end-node ids)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    count = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    labels, _ = ndimage.label(skel & (np.where(skel, count, 0) != 2), structure=np.ones((3, 3)))
    walker = _SkeletonWalk(skel, labels)
    paths, ends = walker.from_nodes()
    loop_paths, loop_ends = walker.loops()
    return paths + loop_paths, ends + loop_ends


def _path_to_edge(path, shell: SkinShell, shape, colour: str) -> DrawnLine | None:
    xs, thetas = texel_chart(shell, shape)
    pix = np.asarray(path, dtype=float)
    chart = np.stack([np.interp(pix[:, 1], np.arange(len(xs)), xs),
                      np.interp(pix[:, 0], np.arange(len(thetas)), thetas)], axis=1)
    if len(chart) >= 5:  # soften the pixel staircase
        kernel = np.ones(5) / 5.0
        padded = np.pad(chart, ((2, 2), (0, 0)), mode="edge")
        inner = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1)
        chart = np.vstack([chart[:1], inner[1:-1], chart[-1:]])
    points, normals = shell.lift(chart[:, 0], chart[:, 1])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-3:
        return None
    target = np.linspace(0.0, s[-1], max(2, int(s[-1] / 0.001) + 1))
    chart = np.stack([np.interp(target, s, chart[:, k]) for k in range(2)], axis=1)
    # Interpolating world points can leave a curved triangle mesh. Resolve the
    # resampled chart again through the same canonical skin addresses instead.
    points, normals = shell.lift(chart[:, 0], chart[:, 1])
    faces, bary = shell.addresses(chart[:, 0], chart[:, 1])
    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return DrawnLine(chart=chart, points=points, normals=normals, arclength=arclength,
                     width_m=0.0, colour=colour, face_indices=faces, barycentric=bary, surface=shell)


def ink_graph(coverage: np.ndarray, shell: SkinShell, *, min_spur_m=0.004, min_edge_m=0.006,
              min_link_m=0.002, min_piece_m=0.015) -> InkGraph:
    """The traceable strokes of the ink in ``coverage`` (the shell texture's ink alpha)."""
    paths, ends = _skeleton_graph(coverage > 0.5)
    degree = Counter(node for pair in ends for node in pair)
    merged: dict[int, int] = {}

    def root(node: int) -> int:
        while merged.get(node, node) != node:
            node = merged[node]
        return node

    edges, kept = [], []
    for path, (a, b) in zip(paths, ends, strict=True):
        edge = _path_to_edge(path, shell, coverage.shape, "ink")
        length = 0.0 if edge is None else edge.length
        free_ends = int(degree[a] == 1) + int(degree[b] == 1)
        if free_ends == 0 and length < min_link_m:
            # Thinning leaves false junctions at staircase corners, joined by links a pixel or
            # two long: merge their ends rather than cut the stroke apart.
            merged[root(a)] = root(b)
            continue
        if edge is None or (free_ends == 1 and length < min_spur_m) or (free_ends == 2 and length < min_edge_m):
            continue  # a thinning spur off a stroke's end, or a speck
        edges.append(edge)
        kept.append((a, b))
    ends = [(root(a), root(b)) for a, b in kept]
    for a, b in ends:  # join each stroke's two ends to find the connected pieces of ink
        merged[root(a)] = root(b)
    piece = Counter()
    for edge, (a, _) in zip(edges, ends, strict=True):
        piece[root(a)] += edge.length
    keep = [k for k, (a, _) in enumerate(ends) if piece[root(a)] >= min_piece_m]  # no specks, dots or eyes
    return InkGraph(edges=[edges[k] for k in keep], ends=[ends[k] for k in keep])
