"""Inklang body sites for sim prompts — the language side of *where*.

The lexicon is ``config/inkmap/sites.json`` (shared with the Inkmap web app
and the placement schema): 45 leaf sites, laterality and aspect rules, and a
``geometry`` class per site (``flat`` / ``wrap`` / ``crease``) — the thing the
surface primitive can condition on long before a full body exists in sim.

This module is a *provider*, deliberately not wired into the prompt frame:
``language.sample_scene`` owns the training distribution, and adding a site
slot there changes what every future dataset says. The wiring is one line —
append :func:`site_phrase` to the surface slot and record :func:`as_meta`
in the program — behind a DR flag that defaults off, whenever the training
side wants it. Pure stdlib + numpy-free; mirrors the TypeScript realizer in
``web/inkmap/src/core/lang.ts`` for the site part of the sentence.
"""

from __future__ import annotations

import json
from typing import Any

from tatbot_sim.repo import repo_root

_LEXICON_PATH = repo_root() / "config" / "inkmap" / "sites.json"

with open(_LEXICON_PATH) as _f:
    LEXICON: dict[str, Any] = json.load(_f)

INKLANG_VERSION: str = LEXICON["inklang"]
SITES: dict[str, dict[str, Any]] = LEXICON["sites"]
GEOMETRY_CLASSES = ("flat", "wrap", "crease")


def site_phrase(site_id: str, laterality: str | None = None,
                aspect: str | None = None, level: str | None = None) -> str:
    """The canonical noun phrase for a site — ``left upper inner forearm``.

    Mirrors the TS realizer's site part exactly: ``[laterality] [level]
    [aspect] <name>``, with laterality only when the site is sided.
    """
    site = SITES[site_id]
    if laterality in ("left", "right") and site["laterality"] == "midline":
        raise ValueError(f"{site_id} is a midline site; {laterality!r} does not apply")
    if aspect is not None and aspect not in site.get("aspects", []):
        raise ValueError(f"aspect {aspect!r} not allowed on {site_id}")
    if level is not None and level not in LEXICON["levels"]:
        raise ValueError(f"unknown level {level!r}")
    parts = []
    if laterality in ("left", "right"):
        parts.append(laterality)
    if level is not None:
        parts.append(level)
    if aspect is not None:
        parts.append(aspect)
    parts.append(site["name"])
    return " ".join(parts)


def sample_site(rng, geometry: str | None = None,
                with_refinement: bool = False) -> dict[str, Any]:
    """Draw a site (optionally restricted to one geometry class), uniform over
    the lexicon, with laterality for sided sites and, when asked, a sampled
    level/aspect refinement. ``rng`` is a ``numpy.random.Generator`` or
    anything with ``choice``/``random``.

    Returns ``{"id", "laterality", "aspect", "level", "geometry", "phrase"}``
    — ``phrase`` drops straight into a prompt's surface slot, the rest into
    ``run_meta``'s program.
    """
    if geometry is not None and geometry not in GEOMETRY_CLASSES:
        raise ValueError(f"unknown geometry class {geometry!r}")
    ids = sorted(sid for sid, s in SITES.items() if geometry is None or s["geometry"] == geometry)
    sid = ids[int(rng.integers(len(ids))) if hasattr(rng, "integers") else int(rng.random() * len(ids))]
    site = SITES[sid]
    laterality = None
    if site["laterality"] == "sided" or (site["laterality"] == "any" and rng.random() < 0.5):
        laterality = "left" if rng.random() < 0.5 else "right"
    aspect = None
    level = None
    if with_refinement:
        aspects = site.get("aspects", [])
        if aspects and rng.random() < 0.5:
            aspect = aspects[int(rng.integers(len(aspects))) if hasattr(rng, "integers") else 0]
        if site["geometry"] != "crease" and rng.random() < 0.35:
            level = "upper" if rng.random() < 0.5 else "lower"
    return {
        "id": sid,
        "laterality": laterality,
        "aspect": aspect,
        "level": level,
        "geometry": site["geometry"],
        "phrase": site_phrase(sid, laterality, aspect, level),
    }


def as_meta(site: dict[str, Any]) -> dict[str, Any]:
    """The run_meta record for a sampled site (drops the derived phrase)."""
    return {
        "id": site["id"],
        "laterality": site["laterality"],
        "aspect": site["aspect"],
        "level": site["level"],
        "lexicon": INKLANG_VERSION,
    }
