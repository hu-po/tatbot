"""The inklang site provider: phrases match the web realizer, sampling is
lexicon-complete, and the geometry classes the surface primitive conditions on
are exactly the lexicon's. Stdlib + numpy only — no SAPIEN, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from tatbot_sim import sites


def test_lexicon_loads_and_is_the_locked_45():
    assert len(sites.SITES) == 59
    assert sites.INKLANG_VERSION == "0.3"


def test_phrases_mirror_the_web_realizer():
    assert sites.site_phrase("knee_ditch", "left") == "left knee ditch"
    assert sites.site_phrase("forearm", "left", aspect="inner", level="upper") == "left upper inner forearm"
    assert sites.site_phrase("sternum") == "sternum"
    assert sites.site_phrase("foot_top", "right") == "right top of the foot"


def test_lies_are_rejected():
    with pytest.raises(ValueError):
        sites.site_phrase("sternum", "left")  # midline
    with pytest.raises(ValueError):
        sites.site_phrase("sternum", aspect="inner")  # no aspects on the sternum
    with pytest.raises(ValueError):
        sites.site_phrase("forearm", level="sideways")


def test_sampling_covers_the_lexicon_and_respects_geometry():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(2000):
        s = sites.sample_site(rng, with_refinement=True)
        seen.add(s["id"])
        assert s["geometry"] in sites.GEOMETRY_CLASSES
        # The phrase must always re-derive from the parts (truthfulness).
        assert s["phrase"] == sites.site_phrase(s["id"], s["laterality"], s["aspect"], s["level"])
    assert seen == set(sites.SITES), "uniform sampling should cover all 59 sites in 2000 draws"
    for geometry in sites.GEOMETRY_CLASSES:
        s = sites.sample_site(rng, geometry=geometry)
        assert s["geometry"] == geometry


def test_meta_is_json_ready_and_versioned():
    rng = np.random.default_rng(1)
    m = sites.as_meta(sites.sample_site(rng, with_refinement=True))
    assert set(m) == {"id", "laterality", "aspect", "level", "lexicon"}
    assert m["lexicon"] == "0.3"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
