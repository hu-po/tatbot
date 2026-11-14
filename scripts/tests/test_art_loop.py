"""python/tatbot_art loop: gene scores from votes and the next round's prompts."""
from tatbot_art.inks import load_catalog
from tatbot_art.loop import PALETTES, TEMPLATES, gene_scores, palette_key, propose, votes_by_design


def _designs():
    def d(style, subject, palette):
        return {"genes": {"style": style, "subject": subject, "palette": palette, "variant": "base"}}
    return {"a": d("american-traditional", "a sea lion barking on a rock", "trad-primaries"),
            "b": d("american-traditional", "a San Francisco cable car", "trad-primaries"),
            "c": d("linocut", "a sea lion barking on a rock", "orange-black"),
            "e": d("geometric", "a robot arm holding a tattoo machine", "red-green")}


def test_votes_sum_across_viewers():
    assert votes_by_design({"u1": {"a": 1, "c": -1}, "u2": {"a": 1}}) == {"a": 2, "c": -1}


def test_gene_scores_shrink_toward_zero_and_skip_unvoted():
    scores = gene_scores(_designs(), {"u1": {"a": 1, "b": 1, "c": -1}})
    assert scores["style"]["american-traditional"] == {"score": 0.5, "sum": 2, "n": 2}
    assert scores["style"]["linocut"]["score"] < 0 and "geometric" not in scores["style"]
    assert scores["subject"]["a sea lion barking on a rock"]["sum"] == 0


def test_propose_makes_new_prompts_that_name_their_inks():
    designs = _designs()
    prompts = propose(designs, {"u1": {"a": 1, "b": 1, "c": -1, "e": -1}}, load_catalog(), round_no=2, n=8, seed=3)
    assert len(prompts) == 8 and len({p["id"] for p in prompts}) == 8
    seen = {(d["genes"]["style"], d["genes"]["subject"], d["genes"]["palette"]) for d in designs.values()}
    for p in prompts:
        assert (p["style"], p["subject"], p["palette"]) not in seen
        assert p["style"] in TEMPLATES and p["inks"][-1] == "true_black"
        assert p["inks"] == PALETTES.get(p["palette"], p["inks"]) and palette_key(p["inks"]) == p["palette"]
        assert "#" in p["text"] and p["subject"] in p["text"]
    exploit = propose(designs, {"u1": {"a": 1, "b": 1, "c": -1, "e": -1}}, load_catalog(), round_no=2, n=8,
                      explore=0.0, seed=3)
    assert all(p["palette"] in PALETTES for p in exploit)
    assert sum(p["style"] == "american-traditional" for p in exploit) >= sum(p["style"] == "linocut" for p in exploit)


def test_palette_key_names_a_known_load():
    assert palette_key(["bright_red", "true_black"]) == "red-black"
    assert palette_key(["silver", "true_black"]) == "silver+true_black"


def test_exploring_rounds_spread_over_styles_fills_and_free_palettes():
    from tatbot_art.acquire import SOLID_FILLS
    from tatbot_art.loop import FILLS, SEPARATION
    assert set(FILLS) <= set(SOLID_FILLS) and "maze" not in FILLS  # only fills that stay in their regions
    prompts = propose(_designs(), {"u1": {"a": 1}}, load_catalog(), round_no=2, n=15, explore=0.7, seed=5)
    styles = [p["style"] for p in prompts]
    assert len(set(styles)) >= 8 and max(styles.count(s) for s in styles) <= 2
    assert len({p["fill"] for p in prompts}) >= 4 and all(p["fill"] in FILLS for p in prompts)
    for p in prompts:
        assert p["separation"] == SEPARATION[p["style"]] and p["inks"][-1] == "true_black" and len(p["inks"]) <= 5
        assert "#000000 black" in p["text"] and "#3C3C3F" not in p["text"]
