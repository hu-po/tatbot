"""Coded flower-of-life stencil frame: one petal arc per lattice edge, its side is the bit.

The frame keeps the flower-of-life hex lattice. Every lattice edge between two
junctions carries exactly one of its two petal arcs: a 60-degree arc of the
circle centred on one of the edge's two third vertices, so it bulges toward
the other. The bulge side is one bit, and it is read by comparing ink on the
two candidate arcs, so a washed-off edge reads as an erasure rather than as the
other bit. Six arcs meet at every junction and all of them leave along
directions 30 degrees off the lattice edges, which makes the junction a
six-fold keypoint whatever the bits are.

The bits come from a SHA-256 stream seeded by the print ID and are then
improved so that every window (the edges within two lattice steps of a
junction) differs from every other window, under all twelve lattice rotations
and reflections, in at least `window_distance` edges. There is no separate
print-ID grid: the whole code is the ID.

Lattice coordinates are axial (q, r) with page position
`centre + spacing*(q + r/2, r*sqrt(3)/2)` in millimetres (y down). `DIRS[k]` is
the step at page angle 60*k degrees. An edge is `(q, r, k)` with k in 0..2 from
node (q, r) to (q, r)+DIRS[k]; bit +1 means the arc bulges toward
(q, r)+DIRS[k+1] (the circle centred on (q, r)+DIRS[k-1]), -1 toward
(q, r)+DIRS[k-1].
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEME = "tatbot.stencil-coded-fol/1"
VERSION = "coded-fol-1"
DIRS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
SQRT3 = math.sqrt(3)
# Chord fractions at which a reader compares the two candidate arcs: clear of the junctions.
SAMPLE_T = (.3, .4, .5, .6, .7)
# The default is the bead-and-teardrop design chosen in the style search: full petals, the
# bit a filled stretch of each petal, beads with clear halos on a 6.5 mm lattice.
DEFAULTS = {"width_mm": 100., "height_mm": 150., "margin_mm": 5., "frame_mm": 14., "spacing_mm": 6.5,
            "stroke_mm": .3, "knot_mm": 2.2, "dpi": 300, "window_radius": 3, "optimize_rounds": 200,
            "arcs": "full", "bit": "teardrop", "knot": "disk", "ornament": "none", "ornament_rate": .6,
            "halo_mm": 1.5, "teardrop_from": .3, "teardrop_to": .45, "seed_at": .36}
# Style axes. The decoder needs a blob on every junction and one two-state mark per edge;
# everything else is free for the artwork.
STYLES = {"arcs": ("single", "full"), "bit": ("side", "teardrop", "seed"), "knot": ("disk", "ring", "dot-ring"),
          "ornament": ("none", "dots", "rings")}
# Mark placement along the chord, from the edge's end: stored in coded.json so the reader follows.
MARK_SHAPE = ("teardrop_from", "teardrop_to", "seed_at")


def print_id_for(seed):
    """Deterministic 24-hex print ID for a bench seed (a real print mints a fresh one)."""
    return hashlib.sha256(f"tatbot-coded-print:{seed}".encode()).hexdigest()[:24]


class BitStream:
    def __init__(self, key):
        self.key = f"{SCHEME}\0{key}".encode()
        self.counter = 0

    def bit(self):
        digest = hashlib.sha256(self.key+b"\0"+self.counter.to_bytes(8, "big")).digest()
        self.counter += 1
        return 1 if digest[0] & 1 else -1


# --- lattice transforms (shared with the decoder) ---------------------------------------------

def transforms():
    """The twelve lattice symmetries as (sigma, m, matrix): k -> sigma*k+m, node -> matrix @ node."""
    rotate = ((0, -1), (1, 1))
    reflect = ((1, 1), (0, -1))

    def mul(a, b):
        return tuple(tuple(sum(a[i][t]*b[t][j] for t in range(2)) for j in range(2)) for i in range(2))

    out = []
    for sigma, base in ((1, ((1, 0), (0, 1))), (-1, reflect)):
        matrix = base
        for m in range(6):
            out.append((sigma, m, matrix))
            matrix = mul(rotate, matrix)
    return out


def map_edge(q, r, k, bit, sigma, m, matrix, offset=(0, 0), reflect_flips=True):
    """Image of edge (q, r, k) with `bit` under a lattice symmetry plus offset, canonicalised.

    Reversing an edge flips both kinds of bit. A reflection swaps an edge's two sides, so it
    flips a `side` bit but not a `teardrop` bit (which end of the petal is filled)."""
    qq = matrix[0][0]*q+matrix[0][1]*r+offset[0]
    rr = matrix[1][0]*q+matrix[1][1]*r+offset[1]
    kk = (sigma*k+m) % 6
    bit = bit*sigma if reflect_flips else bit
    if kk >= 3:
        qq, rr, kk, bit = qq+DIRS[kk][0], rr+DIRS[kk][1], kk-3, -bit
    return qq, rr, kk, bit


def hex_distance(dq, dr):
    return (abs(dq)+abs(dr)+abs(dq+dr))//2


def window_slots(radius):
    """Relative canonical edges with both ends within `radius` steps of the centre node."""
    slots = []
    for dq in range(-radius, radius+1):
        for dr in range(-radius, radius+1):
            if hex_distance(dq, dr) > radius:
                continue
            for k in range(3):
                eq, er = dq+DIRS[k][0], dr+DIRS[k][1]
                if hex_distance(eq, er) <= radius:
                    slots.append((dq, dr, k))
    return slots


# --- geometry ---------------------------------------------------------------------------------

class Layout:
    """Band geometry: which junctions and edges the frame draws."""

    _centres: dict = {}  # geometry -> searched lattice phase; the search is pure and costs 96 builds

    def __init__(self, width_mm, height_mm, margin_mm, frame_mm, spacing_mm, stroke_mm, centre_mm=None, **_):
        self.w, self.h, self.m, self.f = float(width_mm), float(height_mm), float(margin_mm), float(frame_mm)
        self.s, self.stroke = float(spacing_mm), float(stroke_mm)
        key = (self.w, self.h, self.m, self.f, self.s, self.stroke)
        if centre_mm is None and key not in Layout._centres:
            # The lattice phase that fits the most whole edges into the band: row alignment
            # decides whether a 9-10 mm band holds one triangle row or two.
            phases = [(fx/8, fy/12) for fy in range(12) for fx in range(8)]
            Layout._centres[key] = max(((self.w/2+fx*self.s, self.h/2+fy*self.s*SQRT3) for fx, fy in phases),
                                       key=lambda c: len(self._build(c)[1]))
        if centre_mm is None:
            centre_mm = Layout._centres[key]
        self.centre = (float(centre_mm[0]), float(centre_mm[1]))
        self.nodes, self.edges = self._build(self.centre)

    def _build(self, centre):
        self.centre = centre
        clear = self.stroke/2+.05
        reach_q = int(self.w/self.s)+4
        reach_r = int(self.h/(self.s*SQRT3/2))+4
        nodes = sorted((q, r) for r in range(-reach_r, reach_r+1) for q in range(-reach_q-reach_r, reach_q+reach_r+1)
                       if self.in_band(*self.position(q, r), clear))
        node_set = set(nodes)
        edges = []
        for q, r in nodes:
            for k in range(3):
                other = (q+DIRS[k][0], r+DIRS[k][1])
                if other in node_set and all(self.in_band(x, y, clear) for bit in (1, -1)
                                             for x, y in self.arc(q, r, k, bit, samples=12)):
                    edges.append((q, r, k))
        # A junction with no drawn edge is not a keypoint.
        used = {(q, r) for q, r, _ in edges} | {(q+DIRS[k][0], r+DIRS[k][1]) for q, r, k in edges}
        return [n for n in nodes if n in used], edges

    def position(self, q, r):
        return (self.centre[0]+self.s*(q+r/2), self.centre[1]+self.s*r*SQRT3/2)

    def in_band(self, x, y, clear=0.):
        m, f = self.m, self.f
        outer = m+clear <= x <= self.w-m-clear and m+clear <= y <= self.h-m-clear
        inner = m+f-clear < x < self.w-m-f+clear and m+f-clear < y < self.h-m-f+clear
        return outer and not inner

    def arc(self, q, r, k, bit, samples=None):
        """Page-mm points along the edge's arc for `bit`, from node (q, r) to its k neighbour."""
        cq, cr = DIRS[(k-bit) % 6]
        cx, cy = self.position(q+cq, r+cr)
        px, py = self.position(q, r)
        qx, qy = self.position(q+DIRS[k][0], r+DIRS[k][1])
        start = math.atan2(py-cy, px-cx)
        delta = (math.atan2(qy-cy, qx-cx)-start+math.pi) % math.tau-math.pi
        steps = samples or max(8, math.ceil(abs(delta)*self.s/.09))
        return [(cx+self.s*math.cos(start+delta*i/steps), cy+self.s*math.sin(start+delta*i/steps))
                for i in range(steps+1)]


# --- code -------------------------------------------------------------------------------------

def window_table(layout, radius, reflect_flips=True):
    """Per (centre, symmetry) window, row 12*c+g: the page edge index behind each slot (-1 when
    the page draws no edge there) and the sign that maps the page bit into the window's frame."""
    slots = window_slots(radius)
    index = {edge: i for i, edge in enumerate(layout.edges)}
    edge_ids, signs = [], []
    for cq, cr in layout.nodes:
        for sigma, m, matrix in transforms():
            mapped = [map_edge(dq, dr, k, 1, sigma, m, matrix, (cq, cr), reflect_flips) for dq, dr, k in slots]
            edge_ids.append([index.get((q, r, k), -1) for q, r, k, _ in mapped])
            signs.append([sign for *_, sign in mapped])
    return np.array(edge_ids, np.int64), np.array(signs, np.int64)


def _values(table, bits, rows=None):
    edge_ids, signs = table if rows is None else (table[0][rows], table[1][rows])
    return np.where(edge_ids >= 0, bits[np.maximum(edge_ids, 0)]*signs, 0).astype(np.float32)


def _agreement(a, b):
    """agree[i, j]: slots where windows a[i] and b[j] both draw an edge with the same bit."""
    return (a > 0).astype(np.float32) @ (b > 0).T.astype(np.float32) + \
        (a < 0).astype(np.float32) @ (b < 0).T.astype(np.float32)


def window_margins(layout, bits, radius, table=None, reflect_flips=True):
    """Per window centre (identity symmetry): its size (drawn edges) and its minimum distance to
    every other (centre, symmetry) window, counted as the window's edges that the other window
    does not explain. A window read with `distance - 1` edges erased still names one hypothesis."""
    table = window_table(layout, radius, reflect_flips) if table is None else table
    values = _values(table, bits)
    identity = np.arange(0, len(values), 12)
    size = (values[identity] != 0).sum(1)
    dist = size[:, None]-_agreement(values[identity], values)
    dist[np.arange(len(identity)), identity] = np.inf
    return size.astype(int), dist.min(1).astype(int)


def build_code(layout, print_id, radius=2, rounds=400, reflect_flips=True):
    """Seeded bits, then greedy flips that raise the worst window's relative distance. Deterministic."""
    stream = BitStream(print_id)
    bits = np.array([stream.bit() for _ in layout.edges], np.int64)
    if rounds <= 0 or not layout.edges:
        return bits
    table = window_table(layout, radius, reflect_flips)
    identity = np.arange(0, len(table[0]), 12)
    containing = [np.flatnonzero((table[0] == e).any(1)) for e in range(len(layout.edges))]
    values = _values(table, bits)
    size = (values[identity] != 0).sum(1).astype(np.float32)
    agree = _agreement(values[identity], values)
    self_mask = np.zeros_like(agree, bool)
    self_mask[np.arange(len(identity)), identity] = True

    def score(agree):
        rel = np.where(self_mask, np.inf, size[:, None]-agree).min(1)/np.maximum(size, 1)
        return (float(rel.min()), -int((rel <= rel.min()+1e-6).sum()), float(rel.mean())), rel

    best, rel = score(agree)
    for _ in range(rounds):
        worst = np.argsort(rel, kind="stable")[:3]
        candidates = sorted({int(e) for row in worst for e in table[0][identity[row]] if e >= 0})
        improved = False
        for e in candidates:
            bits[e] = -bits[e]
            cols = containing[e]
            rows = np.flatnonzero(np.isin(identity, cols))
            trial = agree.copy()
            new_values = _values(table, bits, cols)
            values_new = values.copy()
            values_new[cols] = new_values
            trial[:, cols] = _agreement(values_new[identity], new_values)
            trial[rows] = _agreement(values_new[identity[rows]], values_new)
            score_trial, rel_trial = score(trial)
            if score_trial > best:
                best, rel, agree, values, improved = score_trial, rel_trial, trial, values_new, True
                break
            bits[e] = -bits[e]
        if not improved:
            break
    return bits


# --- artwork ----------------------------------------------------------------------------------

def settings_for(**overrides):
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("dpi", "window_radius", "optimize_rounds"):
        settings[key] = int(settings[key])
    if "bit" in overrides and "arcs" not in overrides:
        settings["arcs"] = "single" if settings["bit"] == "side" else "full"
    return settings


def validate(settings):
    for name in ("width_mm", "height_mm", "spacing_mm", "stroke_mm", "frame_mm"):
        value = settings[name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if settings["spacing_mm"] < 3 or settings["stroke_mm"] > settings["spacing_mm"]/6:
        raise ValueError("spacing must be >= 3 mm and the stroke at most a sixth of it")
    if min(settings["width_mm"], settings["height_mm"]) <= 2*(settings["margin_mm"]+settings["frame_mm"]):
        raise ValueError("frame must leave a nonempty centre")
    if settings["frame_mm"] < settings["spacing_mm"]:
        raise ValueError("frame must be at least one lattice spacing wide")
    if not 1 <= settings["window_radius"] <= 3:
        raise ValueError("window radius must be 1-3")
    _validate_style(settings)


def _validate_style(settings):
    for axis, choices in STYLES.items():
        if settings[axis] not in choices:
            raise ValueError(f"{axis} must be one of {', '.join(choices)}")
    if not 0 < settings["teardrop_from"] < settings["teardrop_to"] <= .5 or not .1 < settings["seed_at"] < .5:
        raise ValueError("teardrop_from < teardrop_to <= 0.5 and 0.1 < seed_at < 0.5 along the chord")
    if (settings["bit"] == "side") != (settings["arcs"] == "single"):
        raise ValueError("a side bit draws one arc per edge (arcs=single); teardrop and seed need both (arcs=full)")


def generate(seed, output, *, print_id=None, **overrides):
    """Write stencil.png, stencil.svg, settings.json, tracking.json and coded.json; return output."""
    from PIL import Image, ImageDraw
    from PIL import __version__ as pillow_version
    settings = settings_for(**overrides)
    validate(settings)
    print_id = print_id or print_id_for(seed)
    layout = Layout(**settings)
    reflect_flips = settings["bit"] == "side"
    bits = build_code(layout, print_id, settings["window_radius"], settings["optimize_rounds"], reflect_flips)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scale = settings["dpi"]/25.4
    size = (round(layout.w*scale), round(layout.h*scale))
    canvas = Image.new("1", size, 1)
    pen = _Pen(ImageDraw.Draw(canvas), scale, layout.stroke)
    for (q, r, k), bit in zip(layout.edges, bits, strict=True):
        arcs = [layout.arc(q, r, k, int(bit))] if settings["arcs"] == "single" else \
            [layout.arc(q, r, k, 1), layout.arc(q, r, k, -1)]
        for points in arcs:
            pen.line(_trim(points, settings["knot_mm"]/2+settings["halo_mm"]) if settings["halo_mm"] > 0 else points)
        if settings["bit"] == "teardrop":
            pen.polygon(_teardrop(arcs, int(bit), settings["teardrop_from"], settings["teardrop_to"]))
        elif settings["bit"] == "seed":
            (x0, y0), (x1, y1) = arcs[0][0], arcs[0][-1]
            t = settings["seed_at"] if bit > 0 else 1-settings["seed_at"]
            pen.disk(x0+t*(x1-x0), y0+t*(y1-y0), .065*layout.s)
    _ornaments(pen, layout, settings, print_id)
    _knots(pen, layout, settings)
    png_path, svg_path = output/"stencil.png", output/"stencil.svg"
    canvas.save(png_path, dpi=(settings["dpi"], settings["dpi"]), optimize=False)
    svg_path.write_text(pen.svg(layout)+"\n", encoding="utf-8")
    sizes, margins = window_margins(layout, bits, settings["window_radius"], reflect_flips=reflect_flips)
    code = {"scheme": SCHEME, "print_id": print_id, "geometry": {k: settings[k] for k in
            ("width_mm", "height_mm", "margin_mm", "frame_mm", "spacing_mm", "stroke_mm", "knot_mm")},
            "style": {axis: settings[axis] for axis in (*STYLES, *MARK_SHAPE)},
            "centre_mm": list(layout.centre), "window_radius": settings["window_radius"],
            "nodes": [list(n) for n in layout.nodes],
            "edges": [[q, r, k, int(b)] for (q, r, k), b in zip(layout.edges, bits, strict=True)],
            "window": _window_summary(sizes, margins)}
    code_path = output/"coded.json"
    code_path.write_text(json.dumps(code, separators=(",", ":"))+"\n")
    histogram = canvas.histogram()
    record = dict(settings, seed=str(seed), generator_version=VERSION, pixels=list(size),
                  pillow_version=pillow_version, physical_instance_id=None, instance_mark=None,
                  coded_print_id=print_id, coded_scheme=SCHEME,
                  clear_center_mm=[layout.w-2*(layout.m+layout.f), layout.h-2*(layout.m+layout.f)],
                  border_inner_mm=border_inner_mm(~np.asarray(canvas, bool), scale, layout),
                  black_fraction=round(histogram[0]/(size[0]*size[1]), 5),
                  junctions=len(layout.nodes), coded_edges=len(layout.edges), window=code["window"])
    record["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (png_path, svg_path, code_path)}
    record["artwork_svg_sha256"] = record["files"]["stencil.svg"]
    (output/"settings.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    from stencil_reference import export
    export(output/"settings.json")
    return output


def border_inner_mm(ink, scale, layout, span=.85):
    """The border's inner ink edges [x0, y0, x1, y1] in page mm: per side, the innermost ink
    across the middle `span` of that side of the clear centre, which is what a border fit
    along that side finds. Knots sit on the junctions by their centres, so they reach up to a
    knot radius past the nominal clear centre, and the lattice phase makes that uneven between
    opposite sides. A side without ink keeps its nominal edge."""
    inner = layout.m+layout.f
    nominal = [inner, inner, layout.w-inner, layout.h-inner]
    ys, xs = np.nonzero(ink)
    x, y = (xs+.5)/scale, (ys+.5)/scale
    cx, cy = (nominal[0]+nominal[2])/2, (nominal[1]+nominal[3])/2
    beside_x = np.abs(y-cy) <= span*(nominal[3]-nominal[1])/2     # beside the left and right sides
    beside_y = np.abs(x-cx) <= span*(nominal[2]-nominal[0])/2     # beside the top and bottom sides
    half = .5/scale
    found = [x[beside_x & (x < cx)].max()+half if (beside_x & (x < cx)).any() else None,
             y[beside_y & (y < cy)].max()+half if (beside_y & (y < cy)).any() else None,
             x[beside_x & (x > cx)].min()-half if (beside_x & (x > cx)).any() else None,
             y[beside_y & (y > cy)].min()-half if (beside_y & (y > cy)).any() else None]
    return [round(float(nominal[i] if v is None else v), 3) for i, v in enumerate(found)]


class _Pen:
    """Draws the same shapes into the 1-bit raster and the SVG."""

    def __init__(self, draw, scale, stroke):
        self.draw, self.scale, self.stroke = draw, scale, stroke
        self.width = max(1, round(stroke*scale))
        self.elements = []

    def line(self, points, stroke=None):
        stroke = stroke or self.stroke
        width = max(1, round(stroke*self.scale))
        pixel = [(x*self.scale, y*self.scale) for x, y in points]
        self.draw.line(pixel, fill=0, width=width, joint="curve")
        for x, y in (pixel[0], pixel[-1]):
            self.draw.ellipse((x-width/2, y-width/2, x+width/2, y+width/2), fill=0)
        coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in points)
        self.elements.append(f'<polyline points="{coords}" fill="none" stroke="black" stroke-width="{stroke:g}" '
                             'stroke-linecap="round" stroke-linejoin="round"/>')

    def polygon(self, points):
        self.draw.polygon([(x*self.scale, y*self.scale) for x, y in points], fill=0)
        coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in points)
        self.elements.append(f'<polygon points="{coords}" fill="black"/>')

    def disk(self, x, y, radius):
        s = self.scale
        self.draw.ellipse(((x-radius)*s, (y-radius)*s, (x+radius)*s, (y+radius)*s), fill=0)
        self.elements.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{radius:g}" fill="black"/>')

    def ring(self, x, y, radius):
        steps = max(24, math.ceil(math.tau*radius/.09))
        self.line([(x+radius*math.cos(math.tau*i/steps), y+radius*math.sin(math.tau*i/steps))
                   for i in range(steps+1)])

    def svg(self, layout):
        return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout.w:g}mm" '
                          f'height="{layout.h:g}mm" viewBox="0 0 {layout.w:g} {layout.h:g}">',
                          '<rect width="100%" height="100%" fill="white"/>', *self.elements, "</svg>"])


def _trim(points, radius):
    """An arc without its ends inside `radius` of the junctions it joins: a clear halo that
    keeps each knot a round, isolated blob once the transfer spreads."""
    (x0, y0), (x1, y1) = points[0], points[-1]
    return [(x, y) for x, y in points if math.hypot(x-x0, y-y0) > radius and math.hypot(x-x1, y-y1) > radius]


def _teardrop(arcs, bit, start, stop):
    """The filled part of a petal: between the two arcs, from `start` to `stop` of the way along,
    at the edge's start (bit +1) or its end (bit -1)."""
    first, second = arcs
    n = len(first)-1
    lo, hi = round(start*n), round(stop*n)
    if bit < 0:
        lo, hi = n-hi, n-lo
    return first[lo:hi+1]+second[lo:hi+1][::-1]


def _knots(pen, layout, settings):
    radius = settings["knot_mm"]/2
    if radius <= 0:
        return
    for q, r in layout.nodes:
        x, y = layout.position(q, r)
        if settings["knot"] == "disk":
            pen.disk(x, y, radius)
        elif settings["knot"] == "ring":
            pen.ring(x, y, radius-layout.stroke/2)
        else:
            pen.disk(x, y, radius/2.2)
            pen.ring(x, y, radius-layout.stroke/2)


def _ornaments(pen, layout, settings, print_id):
    """Uncoded decoration at triangle centres, chosen by a hash of the print ID."""
    if settings["ornament"] == "none":
        return
    nodes = set(layout.nodes)
    for q, r in layout.nodes:
        for a, b in ((DIRS[0], DIRS[1]), (DIRS[1], DIRS[2])):
            corners = [(q, r), (q+a[0], r+a[1]), (q+b[0], r+b[1])]
            if not all(c in nodes for c in corners):
                continue
            key = hashlib.sha256(f"{print_id}:{corners}".encode()).digest()[0]/255
            if key >= settings["ornament_rate"]:
                continue
            x, y = (sum(v)/3 for v in zip(*(layout.position(*c) for c in corners), strict=True))
            if not layout.in_band(x, y, .8):
                continue
            if settings["ornament"] == "dots":
                pen.disk(x, y, .06*layout.s)
            else:
                pen.ring(x, y, .11*layout.s)


def _window_summary(sizes, margins):
    rel = margins/np.maximum(sizes, 1)
    return {"edges_p50": int(np.median(sizes)), "edges_min": int(sizes.min()),
            "distance_min": int(margins.min()), "distance_p50": float(np.median(margins)),
            "erasable_fraction_min": round(float(((margins-1)/np.maximum(sizes, 1)).min()), 3),
            "erasable_fraction_p50": round(float(np.median((margins-1)/np.maximum(sizes, 1))), 3),
            "relative_distance_min": round(float(rel.min()), 3)}


def load_code(directory):
    """The coded lattice written beside an artwork (coded.json), or None."""
    path = Path(directory)/"coded.json"
    if not path.is_file():
        return None
    code = json.loads(path.read_text())
    if code.get("scheme") != SCHEME:
        raise ValueError(f"unsupported coded stencil scheme in {path}")
    return code
