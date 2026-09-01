"""Inklang lexicon guards: the Python side of the language contract.

The TypeScript core (web/inkmap/src/core/lang.ts) owns parsing and realizing;
these tests keep the lexicon files themselves honest for every non-web
consumer (sim prompts, eval judges), so a web-only edit cannot silently break
the robot side. Stdlib only.
"""
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config" / "inkmap"

with open(CONFIG / "sites.json") as f:
    SITES_LEX = json.load(f)
with open(CONFIG / "styles.json") as f:
    STYLES_LEX = json.load(f)
with open(CONFIG / "placement.schema.json") as f:
    PLACEMENT_SCHEMA = json.load(f)

ASPECTS = set(SITES_LEX["aspects"])
# The ANSI/NIST-NCIC base sites inklang leaves are allowed to refine.
NCIC_SITES = {
    "ABDOMEN", "ANKLE", "ARM", "UPPER ARM", "FOREARM", "BACK", "BREAST",
    "BUTTOCKS", "CALF", "CHEEK", "CHEST", "CHIN", "EAR", "ELBOW", "FACE",
    "FINGER", "FOOT", "GROIN", "HAND", "HEAD", "HIP", "KNEE", "LEG",
    "NECK", "SHOULDER", "THIGH", "TOE", "WRIST", "NOSE", "FOREHEAD",
}


def norm(phrase: str) -> str:
    return " ".join(phrase.lower().replace("-", " ").split())


def test_lexicon_versions_agree():
    assert SITES_LEX["inklang"] == STYLES_LEX["inklang"] == "0.3"


def test_levels_are_a_separate_slot():
    # Since 0.2 "upper"/"lower"/"mid" are levels, never aspects.
    assert set(SITES_LEX["levels"]) == {"upper", "lower", "mid"}
    assert not ({"upper", "lower", "mid"} & ASPECTS)


def test_region_atlases_match_the_lexicon():
    # Every shipped atlas speaks the current lexicon and covers all 45 sites.
    bodies_dir = Path(__file__).resolve().parents[2] / "web" / "inkmap" / "public" / "bodies"
    atlases = sorted(bodies_dir.glob("*.regions.json"))
    assert len(atlases) >= 2, "expected an atlas per shipped body"
    for path in atlases:
        with open(path) as f:
            atlas = json.load(f)
        assert atlas["inklang"] == SITES_LEX["inklang"], path.name
        assert set(atlas["sites"]) == set(SITES_LEX["sites"]), f"{path.name}: sites differ from the lexicon"
        n_sites = len(atlas["sites"])
        assert all(v == -1 or 0 <= (v >> 2) < n_sites for v in atlas["faces"]), path.name


def test_59_leaf_sites_locked():
    # Decision 2026-08-31 (v0.3): 59 leaf sites. Growing the lexicon is a
    # version bump, not a drive-by edit.
    assert len(SITES_LEX["sites"]) == 59


def test_parents_and_anchor_hints_are_valid():
    for sid, s in SITES_LEX["sites"].items():
        if "parent" in s:
            assert s["parent"] in SITES_LEX["sites"], f"{sid}: unknown parent {s['parent']}"
            assert "parent" not in SITES_LEX["sites"][s["parent"]], f"{sid}: parents must be top-level leaves"
        if "anchor" in s:
            assert s["anchor"] in ("centroid", "extremum_front", "extremum_back"), f"{sid}: {s['anchor']}"


def test_sites_well_formed():
    for sid, s in SITES_LEX["sites"].items():
        assert s["laterality"] in ("sided", "midline", "any"), sid
        assert s["geometry"] in ("flat", "wrap", "crease"), sid
        assert s["ncic_site"] in NCIC_SITES, f"{sid}: {s['ncic_site']}"
        assert s["group"] in ("head", "arm", "torso_front", "torso_back", "leg"), sid
        for a in s.get("aspects", []):
            assert a in ASPECTS, f"{sid}: {a}"


def test_zones_reference_real_sites():
    for zid, z in SITES_LEX["zones"].items():
        assert z["laterality"] in ("sided", "midline"), zid
        for m in z["members"]:
            assert m in SITES_LEX["sites"], f"zone {zid}: {m}"


def test_compound_aliases_are_consistent():
    for phrase, tgt in SITES_LEX["compound_aliases"].items():
        site = SITES_LEX["sites"].get(tgt["site"])
        assert site is not None, phrase
        assert "aspect" in tgt or "level" in tgt, f'"{phrase}" refines nothing'
        if "aspect" in tgt:
            assert tgt["aspect"] in site.get("aspects", []), (
                f'"{phrase}" implies aspect {tgt["aspect"]} that {tgt["site"]} does not allow'
            )
        if "level" in tgt:
            assert tgt["level"] in SITES_LEX["levels"], f'"{phrase}": unknown level {tgt["level"]}'
            assert site["geometry"] != "crease", f'"{phrase}": levels make no sense on a crease site'


def test_no_phrase_resolves_to_two_things():
    seen: dict[str, str] = {}

    def claim(phrase: str, owner: str) -> None:
        k = norm(phrase)
        assert seen.get(k, owner) == owner, f'"{k}": {seen[k]} vs {owner}'
        seen[k] = owner

    for sid, s in SITES_LEX["sites"].items():
        for p in [sid, s["name"], *s.get("aliases", [])]:
            claim(p, f"site:{sid}")
    for zid, z in SITES_LEX["zones"].items():
        for p in [zid, z["name"], *z.get("aliases", [])]:
            claim(p, f"zone:{zid}")
    for table in ("styles", "techniques", "colors"):
        for tid, e in STYLES_LEX[table].items():
            for p in [tid, e["name"], *e["aliases"]]:
                claim(p, f"term:{tid}")
    # Aspect and laterality words are grammar, not names: they must not
    # collide with any site/style phrase or each other.
    grammar = set(ASPECTS)
    for aliases in SITES_LEX["aspects"].values():
        grammar.update(aliases)
    for lid, aliases in SITES_LEX["laterality_words"].items():
        grammar.add(lid)
        grammar.update(aliases)
    for w in grammar:
        assert norm(w) not in seen, f'grammar word "{w}" collides with {seen.get(norm(w))}'


def test_style_axes_sizes():
    assert len(STYLES_LEX["styles"]) == 23
    assert len(STYLES_LEX["techniques"]) == 7
    assert len(STYLES_LEX["colors"]) == 5
    defaults = [t for t, e in STYLES_LEX["techniques"].items() if e.get("default")]
    assert defaults == ["machine"]


def test_placement_schema_is_v3_with_site_language():
    assert PLACEMENT_SCHEMA["properties"]["schema_version"]["const"] == 3
    placement = PLACEMENT_SCHEMA["properties"]["placements"]["items"]["properties"]
    assert "site" in placement and "language" in placement
    # Additive: the v2 required set must not have grown.
    assert set(PLACEMENT_SCHEMA["properties"]["placements"]["items"]["required"]) == {
        "id", "design_id", "anchor", "rotation_rad", "size_mm", "mirror",
    }
