"""Separation: a flat-colour image into per-(ink, fill) DBV3 sources on one canvas.

1. Fit the image to the drawn width and whiten its paper (`image.prepare`).
2. Give every pixel the nearest colour the load can draw over white, paper included (`gamut.achievable`,
   CIEDE2000). A two-ink mix must beat the nearest single ink by `mix_penalty` to be chosen.
3. Clean: open each colour fill so none is thinner than one line, and hand specks under `min_area_mm2`
   to the nearest surviving colour. The darkest ink and paper are not opened: a black outline thinner
   than the tool is still drawn (as a traced edge, one line wide), and a white gap is the artist's.
4. Per ink, a coverage map from each region's recipe, at the ink's hatch angle in the load.
5. Per (ink, fill), a greyscale source, eroded half a line inside its regions:
   - `hatch`: tints and mix layers, the coverage through the measured tone curve, drawn as parallel
     streamlines at the ink's angle, which is what lets two inks mix optically;
   - `solid`: regions the ink covers fully, for ECS to pack along their contours. Streamlines seed only
     beside earlier streamlines, so in narrow or ragged shapes they fill the cores and miss the rest
     (34% coverage on traditional black shadows, against 80% packed with a fifth of the strokes);
   - `edges`: the darkest ink's solid regions, traced, when the style outlines.
   With the offset fill the solid regions stay whole and none is traced: its passes are set in from each
   region's own edge.

DBV3 makes every path from these sources (`acquire`), but for the offset fill's (`offset`). Nothing here draws a line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .color import cielab, delta_e00, oklab
from .gamut import Swatch, achievable, angles
from .tone import ToneCurve

# How a style treats its darkest ink.
# - outline: trace its solid regions' edges. A relief print's mass edge is its outline; a geometric line is
#   drawn once along its middle, not traced on both sides.
# - lines: the source is line art. Its dark lines are kept through the resample and widened to one drawn
#   line. A relief print's white gouges are the detail there, so linocut keeps neither.
STYLES = {"flash": {"outline": True, "lines": True}, "geometric": {"outline": False, "lines": True},
          "linocut": {"outline": False, "lines": False}}
WHITE = np.ones(3)
MIN_LAYER_MM2 = 2.0  # a layer whose source is smaller than this gives DBV3 nothing to draw
# The style skills' limits at the drawn size: a colour region narrower than FINE_MM reads as outline (and a geometric
# cell that narrow holds no fill); a paper gap narrower than GAP_MM is filled by ink spread.
FINE_MM = 2.0
GAP_MM = 1.0


@dataclass
class Layer:
    ink_id: str
    fill: str               # "hatch", "solid" or "edges"
    source: np.ndarray      # uint8 greyscale, 255 = no ink
    angle_deg: float        # hatch angle, counter-clockwise from +x with y up
    area_mm2: float


@dataclass
class Separation:
    target: np.ndarray      # the assigned colours over white, (H, W, 3) sRGB
    labels: np.ndarray      # index into swatches, -1 = paper
    swatches: list[Swatch]
    layers: list[Layer]
    px_per_mm: float
    assigned: np.ndarray    # the labels before cleaning: what the image asked for at this size


def candidates(looks, width_mm: float, mixes: bool) -> list[Swatch]:
    swatches = achievable(list(looks), WHITE, width_mm)
    return swatches if mixes else [sw for sw in swatches if len(sw.layers) == 1]


def assign(rgb: np.ndarray, swatches: list[Swatch], mix_penalty: float = 4.0) -> np.ndarray:
    """Nearest drawable colour per pixel (-1 = paper), solved once per distinct 6-bit colour."""
    lab_c = cielab(np.array([WHITE, *[sw.srgb for sw in swatches]]))
    penalty = np.array([0.0, *[mix_penalty * (len(sw.layers) - 1) for sw in swatches]])
    q = np.clip(np.rint(rgb * 63), 0, 63).astype(np.int32)
    keys = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
    uniq, inverse = np.unique(keys.ravel(), return_inverse=True)
    ulab = cielab(np.stack([(uniq >> 12) & 63, (uniq >> 6) & 63, uniq & 63], axis=-1) / 63.0)
    best = np.empty(len(uniq), int)
    for s in range(0, len(uniq), 2048):
        best[s:s + 2048] = (delta_e00(ulab[s:s + 2048, None, :], lab_c[None]) + penalty).argmin(axis=1)
    return best[inverse].reshape(rgb.shape[:2]) - 1


def _disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def clean(labels: np.ndarray, darkest: set[int], line_px: float, min_area_px: float) -> np.ndarray:
    """Drop what the tool cannot draw, then give each dropped pixel its nearest kept colour."""
    keep = np.zeros(labels.shape, bool)
    for k in np.unique(labels):
        mask = labels == k
        opened = mask if (k < 0 or k in darkest) else ndi.binary_opening(mask, structure=_disk(max(1, round(line_px / 2))))
        comp, n = ndi.label(opened)
        if n:
            sizes = ndi.sum(opened, comp, index=np.arange(1, n + 1))
            opened &= np.isin(comp, np.flatnonzero(sizes >= min_area_px) + 1)
        keep |= opened
    if not keep.any():
        return labels
    rows, cols = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    return labels[rows, cols]


def widen(labels: np.ndarray, swatches: list[Swatch], darkest: str, line_px: float) -> np.ndarray:
    """Grow the darkest ink's solid regions to at least one drawn line across."""
    solid = [k for k, sw in enumerate(swatches) if len(sw.layers) == 1 and sw.layers[0].ink_id == darkest
             and sw.layers[0].coverage >= 0.99]
    radius = round((line_px - 1) / 2)
    if not solid or radius < 1:
        return labels
    grown = ndi.binary_dilation(np.isin(labels, solid), structure=_disk(radius))
    return np.where(grown & ~np.isin(labels, solid), solid[0], labels)


def coverage(labels: np.ndarray, swatches: list[Swatch], ink_ids) -> dict[str, np.ndarray]:
    cov = {i: np.zeros(labels.shape) for i in ink_ids}
    for k, sw in enumerate(swatches):
        region = labels == k
        if region.any():
            for lay in sw.layers:
                cov[lay.ink_id][region] = lay.coverage
    return cov


def ink_order(looks) -> list[str]:
    """Lightest ink first, so darker ink covers its ragged edges; black last."""
    return [lk.ink_id for lk in sorted(looks, key=lambda lk: -float(oklab(np.asarray(lk.srgb))[0]))]


def layers_for(cov: dict[str, np.ndarray], looks, tone: ToneCurve, line_px: float, outline: bool,
               px_per_mm: float, offset: bool = False) -> list[Layer]:
    """Per (ink, fill), its source. With `offset`, each ink's solid regions are kept whole for the offset fill
    (tatbot_art.offset), whose outer pass is set in from the region's own edge, so they are neither eroded nor
    traced."""
    order = ink_order(looks)
    ang = angles(looks)
    half = _disk(max(1, round(line_px / 2)))  # a traced edge covers half a line inside, so fills start there
    solid_at = float(tone.coverage(tone.solid_grey))
    out = []

    def add(ink, fill, region, grey):
        source = np.where(region, grey, 255).astype(np.uint8)
        if (source < 255).sum() / px_per_mm ** 2 >= MIN_LAYER_MM2:
            out.append(Layer(ink, fill, source, ang[ink], float(region.sum()) / px_per_mm ** 2))

    for ink in order:
        c = cov[ink]
        solid = c >= solid_at
        edged = outline and ink == order[-1] and solid.any() and not offset
        add(ink, "hatch", ndi.binary_erosion((c > 0) & ~solid, structure=half), tone.grey(c))
        thin_lines = ink == order[-1] and not outline  # a band one line wide is drawn as one pass
        whole = thin_lines or offset
        add(ink, "solid", solid if whole else ndi.binary_erosion(solid, structure=half), tone.solid_grey)
        if edged:
            add(ink, "edges", solid, 0)
    return out


def separate(rgb: np.ndarray, looks, *, style: str, tone: ToneCurve, px_per_mm: float, line_width_mm: float,
             mixes: bool = True, mix_penalty: float = 4.0, min_area_mm2: float = 0.5, offset: bool = False) -> Separation:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; have {', '.join(STYLES)}")
    looks = list(looks)
    swatches = candidates(looks, line_width_mm, mixes)
    labels = assigned = assign(rgb, swatches, mix_penalty)
    darkest = ink_order(looks)[-1]
    solid_darkest = {k for k, sw in enumerate(swatches)
                     if any(lay.ink_id == darkest and lay.coverage >= 0.99 for lay in sw.layers)}
    line_px = line_width_mm * px_per_mm
    labels = clean(labels, solid_darkest, line_px, min_area_mm2 * px_per_mm ** 2)
    if STYLES[style]["lines"]:
        labels = widen(labels, swatches, darkest, line_px)
    cov = coverage(labels, swatches, [lk.ink_id for lk in looks])
    target = np.where(labels[..., None] < 0, WHITE, np.array([sw.srgb for sw in swatches] or [WHITE])[labels])
    return Separation(target, labels, swatches,
                      layers_for(cov, looks, tone, line_px, STYLES[style]["outline"], px_per_mm, offset), px_per_mm,
                      assigned)


def usage(sep: Separation) -> list[dict]:
    """Each colour the separation uses, with its share of the inked area."""
    inked = max(1, int((sep.labels >= 0).sum()))
    ids, counts = np.unique(sep.labels[sep.labels >= 0], return_counts=True)
    rows = [{**sep.swatches[k].as_dict(), "area_mm2": round(float(n) / sep.px_per_mm ** 2, 1),
             "share": round(float(n) / inked, 3)} for k, n in zip(ids, counts, strict=True)]
    return sorted(rows, key=lambda r: -r["area_mm2"])


def _narrow(mask: np.ndarray, width_px: float, outside: bool = False) -> np.ndarray:
    """The part of `mask` narrower than `width_px`: what opening it with a disk that wide takes away. `outside` is
    what lies beyond the canvas, so paper running off its edge is not narrow there."""
    radius = max(1, round(width_px / 2))
    pad = radius + 1
    grown = np.pad(mask, pad, constant_values=outside)
    opened = ndi.binary_opening(grown, structure=_disk(radius))[pad:-pad, pad:-pad]
    return mask & ~opened


def detail(sep: Separation, darkest: str, fine_mm: float = FINE_MM, gap_mm: float = GAP_MM) -> dict:
    """What the drawn size leaves of the design, per ink: its area, the part of it in colour regions narrower than
    `fine_mm` (for the darkest ink, mostly its lines), and what cleaning took from it (fills thinner than a line,
    specks, colour under a widened line). `gaps_mm2` is the paper between inks narrower than `gap_mm`."""
    px2 = sep.px_per_mm ** 2
    rows: dict[str, dict[str, float]] = {}
    pad = round(fine_mm * sep.px_per_mm)
    for k, box in enumerate(ndi.find_objects(sep.labels + 1)):   # colour k's bounding box; paper (0) has none
        if box is None:
            continue
        crop = tuple(slice(max(0, s.start - pad), s.stop + pad) for s in box)
        mask = sep.labels[crop] == k
        thin = int(_narrow(mask, fine_mm * sep.px_per_mm).sum())
        for lay in sep.swatches[k].layers:
            row = rows.setdefault(lay.ink_id, {"area": 0, "fine": 0})
            row["area"] += int(mask.sum())
            row["fine"] += thin
    uses: dict[str, list[int]] = {}
    for k, sw in enumerate(sep.swatches):
        for lay in sw.layers:
            uses.setdefault(lay.ink_id, []).append(k)
    inks = {}
    for ink, ks in uses.items():
        dropped = int((np.isin(sep.assigned, ks) & ~np.isin(sep.labels, ks)).sum())
        row = rows.get(ink, {"area": 0, "fine": 0})
        if row["area"] or dropped:
            inks[ink] = {"area_mm2": round(row["area"] / px2, 1), "under_mm2": round(row["fine"] / px2, 1),
                         "share": round(row["fine"] / row["area"], 3) if row["area"] else None,
                         "dropped_mm2": round(dropped / px2, 1)}
    gaps = _narrow(sep.labels < 0, gap_mm * sep.px_per_mm, outside=True)
    return {"fine_mm": fine_mm, "gap_mm": gap_mm, "darkest": darkest, "inks": inks,
            "gaps_mm2": round(float(gaps.sum()) / px2, 1)}
