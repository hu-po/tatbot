"""Structured site distribution sampling; no duplicate InkLang realizer."""

from __future__ import annotations

import pytest
from tatbot_sim import site_sampling as sites


def test_lexicon_loads_and_is_the_locked_59():
    assert len(sites.SITES) == 59
    assert sites.INKLANG_VERSION == "0.3"


def test_unknown_sites_are_rejected_and_sided_ones_come_in_pairs():
    with pytest.raises(ValueError):
        sites.site_choices(("flux_capacitor",))
    sided = next(site_id for site_id, site in sites.SITES.items() if site["laterality"] == "sided")
    assert [c.laterality for c in sites.site_choices((sided,))] == ["left", "right"]
