"""The achievable set: the colours a palette load can draw, each with its hatch recipe and its line cost.

An ink alone gives tints by line pitch (coverage = deposited width / pitch). Two inks hatched at their own
angles give optical mixes. Area colour follows a Yule-Nielsen modified Neugebauer model in linear sRGB
reflectance over the Demichel primaries (substrate, ink i, ink j, the i-over-j overlap), with `n` = 2
until the swatch card fits it. A translucent overlap filters (Ri * Rj / Rsub); an opaque one covers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from .color import cielab, delta_e00, linear_to_srgb, srgb_to_linear

# Screen-angle practice: each ink of a load keeps its own hatch angle, at least 15 degrees apart from the
# next, black on 45 when present. Index = position in the load.
SCREEN_ANGLES = (45.0, -45.0, 0.0, 90.0, 15.0, -75.0)
LEVELS = (1 / 3, 2 / 3, 1.0)


@dataclass(frozen=True)
class Layer:
    ink_id: str
    angle_deg: float
    coverage: float
    pitch_mm: float


@dataclass
class Swatch:
    layers: tuple[Layer, ...]
    srgb: np.ndarray
    line_mm_per_cm2: dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return "+".join(f"{lay.ink_id}@{round(lay.coverage * 100)}" for lay in self.layers) or "substrate"

    @property
    def hex(self) -> str:
        return "#" + "".join(f"{int(round(c * 255)):02x}" for c in np.clip(self.srgb, 0, 1))

    def as_dict(self) -> dict:
        return {"id": self.id, "hex": self.hex,
                "layers": [lay.__dict__ for lay in self.layers],
                "line_mm_per_cm2": {k: round(v, 1) for k, v in self.line_mm_per_cm2.items()}}


def angles(looks) -> dict[str, float]:
    """Hatch angle per ink, black-like inks first onto 45 degrees."""
    order = sorted(looks, key=lambda lk: sum(lk.srgb))
    return {lk.ink_id: SCREEN_ANGLES[i % len(SCREEN_ANGLES)] for i, lk in enumerate(order)}


def mix(substrate_srgb, parts, n: float = 2.0) -> np.ndarray:
    """Area colour of up to two (InkLook, coverage) parts over a substrate, as sRGB."""
    sub = srgb_to_linear(np.asarray(substrate_srgb, float))
    if not parts:
        return linear_to_srgb(sub)
    if len(parts) > 2:
        raise ValueError("a mix overlaps at most two inks")
    def laid_over(base, lk):
        ink = srgb_to_linear(np.asarray(lk.srgb, float))
        return ink if lk.opaque else base * ink

    (li, a), *rest = parts
    ri = laid_over(sub, li)
    if not rest:
        prim = [(1 - a, sub), (a, ri)]
    else:
        (lj, b), = rest
        prim = [((1 - a) * (1 - b), sub), (a * (1 - b), ri), ((1 - a) * b, laid_over(sub, lj)),
                (a * b, laid_over(ri, lj))]
    total = sum(w * np.power(np.clip(r, 0, 1), 1 / n) for w, r in prim)
    return linear_to_srgb(total ** n)


def achievable(looks, substrate_srgb, width_mm: float = 0.6, levels=LEVELS, n: float = 2.0,
               min_delta_e: float = 2.0) -> list[Swatch]:
    """Singles at each level and every two-ink pair at each level pair, pruned to distinct colours.

    Cheaper recipes (less line per cm², hence fewer dips) win when two land within `min_delta_e` CIEDE2000.
    """
    by_id = {lk.ink_id: lk for lk in looks}
    ang = angles(looks)

    def layer(ink_id, cov):
        return Layer(ink_id, ang[ink_id], cov, width_mm / cov)

    recipes = [(layer(i, c),) for i, c in product(by_id, levels)]
    for i, j in combinations(by_id, 2):
        recipes += [(layer(i, a), layer(j, b)) for a, b in product(levels, levels)]
    swatches = []
    for rec in recipes:
        srgb = mix(substrate_srgb, [(by_id[lay.ink_id], lay.coverage) for lay in rec], n)
        swatches.append(Swatch(rec, srgb, {lay.ink_id: 100.0 / lay.pitch_mm for lay in rec}))
    swatches.sort(key=lambda s: sum(s.line_mm_per_cm2.values()))
    kept: list[Swatch] = []
    kept_lab = np.zeros((0, 3))
    for sw in swatches:
        lab = cielab(sw.srgb)
        if len(kept) and float(delta_e00(kept_lab, lab).min()) < min_delta_e:
            continue
        kept.append(sw)
        kept_lab = np.vstack([kept_lab, lab])
    return kept
