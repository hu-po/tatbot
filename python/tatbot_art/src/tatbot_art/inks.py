"""What each catalogued ink looks like, with the evidence behind it.

`config/inks.yaml` keeps `rgb` as the display colour. An optional `appearance` list records observations
(`srgb`, `source`, `substrate`, `date`); the strongest source for the substrate wins, then one recorded
for `any` substrate, then `rgb` itself as `nominal`. A render carries the weakest source it used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Weakest first: a render is only as trustworthy as the weakest ink in it.
SOURCES = ("nominal", "vendor-photo", "photo", "swatch")
CATALOG = Path(__file__).resolve().parents[4] / "config" / "inks.yaml"


@dataclass(frozen=True)
class InkLook:
    ink_id: str
    name: str
    srgb: tuple[float, float, float]
    source: str
    opaque: bool


def load_catalog(path: Path | str | None = None) -> dict:
    data = yaml.safe_load(Path(path or CATALOG).read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def look(catalog: dict, ink_id: str, substrate: str = "any") -> InkLook:
    try:
        entry = catalog[ink_id]
    except KeyError:
        raise ValueError(f"unknown ink {ink_id!r}; have {', '.join(sorted(catalog))}") from None
    best = None
    for obs in entry.get("appearance") or ():
        if obs.get("source") not in SOURCES:
            raise ValueError(f"{ink_id}: appearance source {obs.get('source')!r} not in {SOURCES}")
        if obs.get("substrate", "any") not in (substrate, "any"):
            continue
        rank = (SOURCES.index(obs["source"]), obs.get("substrate", "any") == substrate)
        if best is None or rank > best[0]:
            best = (rank, obs)
    rgb, source = (best[1]["srgb"], best[1]["source"]) if best else (entry["rgb"], "nominal")
    return InkLook(ink_id=ink_id, name=entry.get("display_name", ink_id),
                   srgb=tuple(float(c) / 255.0 for c in rgb), source=source,
                   opaque=entry.get("viscosity") == "opaque")


def weakest(looks) -> str:
    return min((lk.source for lk in looks), key=SOURCES.index, default="nominal")
