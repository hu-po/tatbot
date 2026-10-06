"""InkLang configuration guards for non-TypeScript consumers.

The TypeScript core (web/inkmap/src/core/inklang/) owns parsing, grounding, and
reverse description. These tests keep the language-neutral files honest for
every non-web consumer, so a web-only edit cannot silently break the simulator
side. Stdlib only; this module is not another InkLang implementation.
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
with open(CONFIG / "inklang-intent.schema.json") as f:
    INTENT_SCHEMA = json.load(f)
with open(CONFIG / "region-atlas.schema.json") as f:
    ATLAS_SCHEMA = json.load(f)
with open(CONFIG / "inklang-resolution.schema.json") as f:
    RESOLUTION_SCHEMA = json.load(f)
with open(CONFIG / "inklang-errors.json") as f:
    ERROR_VOCAB = json.load(f)
with open(CONFIG / "examples" / "inklang" / "corpus-v1.json") as f:
    CORPUS = json.load(f)

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
    # Every shipped atlas speaks the current lexicon and carries the grounding
    # data consumers would otherwise be tempted to recompute differently.
    bodies_dir = Path(__file__).resolve().parents[2] / "web" / "inkmap" / "public" / "bodies"
    atlases = sorted(bodies_dir.glob("*.regions.json"))
    assert [path.name for path in atlases] == ["mhr-soma-v1.regions.json"]
    for path in atlases:
        with open(path) as f:
            atlas = json.load(f)
        assert atlas["atlas_schema_version"] == 2, path.name
        assert atlas["inklang_version"] == SITES_LEX["inklang"], path.name
        assert len(atlas["body"]["rest_surface_sha256"]) == 64, path.name
        assert len(atlas["body"]["asset_sha256"]) == 64, path.name
        assert set(atlas["sites"]) == set(SITES_LEX["sites"]), f"{path.name}: sites differ from the lexicon"
        n_sites = len(atlas["sites"])
        assert all(v == -1 or 0 <= (v >> 2) < n_sites for v in atlas["faces"]), path.name
        assert atlas["regions"], path.name
        for key, region in atlas["regions"].items():
            assert region["site_id"] in SITES_LEX["sites"], f"{path.name}: {key}"
            anchor = region["default_anchor"]
            assert abs(sum(anchor["barycentric"]) - 1.0) <= 1e-6, f"{path.name}: {key}"
            code = atlas["faces"][anchor["face"]]
            assert atlas["sites"][code >> 2] == region["site_id"], f"{path.name}: {key}"


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


def test_placement_schema_is_v6_with_site_language():
    assert PLACEMENT_SCHEMA["properties"]["schema_version"]["const"] == 6
    placement = PLACEMENT_SCHEMA["properties"]["placements"]["items"]["properties"]
    assert "site" in placement and "language" in placement
    assert set(PLACEMENT_SCHEMA["properties"]["placements"]["items"]["required"]) == {
        "id", "design_id", "anchor", "rotation_rad", "size_mm", "mirror",
    }


def test_inklang_contract_versions_are_independent():
    assert INTENT_SCHEMA["properties"]["intent_schema_version"]["const"] == 1
    assert ATLAS_SCHEMA["properties"]["atlas_schema_version"]["const"] == 2
    assert RESOLUTION_SCHEMA["properties"]["resolution_schema_version"]["const"] == 2
    assert PLACEMENT_SCHEMA["properties"]["schema_version"]["const"] == 6
    assert "pose" not in PLACEMENT_SCHEMA["properties"]


def test_inklang_error_vocabulary_is_unique_and_complete():
    assert ERROR_VOCAB["schema_version"] == 1
    records = ERROR_VOCAB["errors"]
    codes = [record["code"] for record in records]
    assert len(codes) == len(set(codes))
    assert all(code.startswith("INKLANG_") for code in codes)
    assert {record["status"] for record in records} == {"needs_choice", "rejected"}
    required = {
        "INKLANG_UNKNOWN_SITE",
        "INKLANG_AMBIGUOUS_LATERALITY",
        "INKLANG_SURFACE_MISMATCH",
        "INKLANG_OFFSET_OUT_OF_BOUNDS",
        "INKLANG_SEMANTIC_MISMATCH",
    }
    assert required <= set(codes)


def test_normative_corpus_covers_every_leaf_and_grounding_axis_on_the_fixed_body():
    assert CORPUS["corpus_schema_version"] == 1
    cases = CORPUS["cases"]
    assert len(cases) >= 120
    ids = {case["id"] for case in cases}
    assert all(
        set(case["request"]) in ({"prompt"}, {"prompt", "policy", "seed"})
        for case in cases
    )
    assert all(
        case["request"].get("policy") == "seeded-v1" and isinstance(case["request"].get("seed"), int)
        for case in cases
        if set(case["request"]) != {"prompt"}
    )
    body = "mhr-soma-v1"
    for site_id in SITES_LEX["sites"]:
        assert any(value.startswith(f"{body}:leaf:") and value.endswith(f":{site_id}") for value in ids)
    for aspect in ("inner", "outer", "front", "back", "side", "top"):
        assert any(value.startswith(f"{body}:aspect:{aspect}:") for value in ids)
    for level in ("upper", "mid", "lower"):
        assert f"{body}:level:{level}" in ids
    for relation in ("above", "below", "behind", "in_front", "beside", "between"):
        assert f"{body}:relation:{relation}" in ids
