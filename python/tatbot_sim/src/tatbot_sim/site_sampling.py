"""Distribution-only sampling of structured tattoo sites.

InkLang parsing, realization, and grounding belong to the TypeScript reference
core. This module reads its lexicon only to draw structured training inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tatbot_sim.repo import repo_root

_LEXICON_PATH = repo_root() / "config" / "inkmap" / "sites.json"
LEXICON: dict[str, Any] = json.loads(_LEXICON_PATH.read_text())
INKLANG_VERSION: str = LEXICON["inklang"]
SITES: dict[str, dict[str, Any]] = LEXICON["sites"]


@dataclass(frozen=True)
class SiteChoice:
    id: str
    laterality: str | None
    aspect: str | None = None
    level: str | None = None

    def as_inklang_site(self, *, region_uv: list[float] | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "laterality": self.laterality,
            "aspect": self.aspect,
            "level": self.level,
            **({"region_uv": region_uv} if region_uv is not None else {}),
        }


def site_choices(site_ids: tuple[str, ...]) -> tuple[SiteChoice, ...]:
    choices = []
    for site_id in site_ids:
        try:
            site = SITES[site_id]
        except KeyError as exc:
            raise ValueError(f"unknown site {site_id!r}") from exc
        lateralities = ("left", "right") if site["laterality"] == "sided" else (None,)
        choices.extend(SiteChoice(site_id, laterality) for laterality in lateralities)
    return tuple(choices)
