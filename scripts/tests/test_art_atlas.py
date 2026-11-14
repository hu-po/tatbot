"""python/tatbot_art map, pyramid, lineage and asks: the browsable side of the voting loop."""
import json

import numpy as np
from PIL import Image
from tatbot_art.atlas import descriptor, graph
from tatbot_art.inks import load_catalog
from tatbot_art.loop import SEPARATION, TEMPLATES, ballots, parents_for, propose, votes_by_design
from tatbot_art.pyramid import on_disk, promotions, stars, study_docs, tiers


def _doc(style, subject, inks, fill="packed", round_no=1, drawn=False, **extra):
    from tatbot_art.loop import palette_key
    views = {"flow": "/_blob/f", "ours": "/_blob/o", **({"drawn": "/_blob/d"} if drawn else {})}
    return {"round": round_no, "style": style, "subject": subject, "fill": fill, "views": views,
            "prompt": TEMPLATES.get(style, "{subject}").format(subject=subject, palette="x"),
            "inks": [{"id": i} for i in inks],
            "genes": {"style": style, "subject": subject, "palette": palette_key(inks), "fill": fill,
                      "variant": "base"}, **extra}


RED = ["bright_red", "true_black"]
SKY = ["blue_sky", "golden_yellow", "true_black"]


def _wall():
    return {"t1": _doc("american-traditional", "a koi fish", RED, drawn=True),
            "t2": _doc("american-traditional", "a tiger head", RED),
            "s1": _doc("stained-glass", "a koi fish", SKY, drawn=True),
            "s2": _doc("stained-glass", "a luna moth", SKY),
            "l1": _doc("linocut", "an octopus", RED, drawn=True),
            "l2": _doc("linocut", "a fox curled asleep", SKY)}


def test_vote_documents_and_plain_ballots_read_the_same():
    plain = {"u": {"t1": 1, "s2": -1}}
    docs = {"u": {"v": {"t1": 1, "s2": -1}, "s": {"t1": 1}, "at": "x"}}
    assert ballots(docs) == ballots(plain) == plain
    assert votes_by_design(docs) == {"t1": 1, "s2": -1} and stars(docs) == {"t1"} and stars(plain) == set()


def test_tiers_run_sketch_drawn_study_shortlist():
    wall = _wall()
    wall["t1~contour"] = {**_doc("american-traditional", "a koi fish", RED, "contour", drawn=True), "study_of": "t1"}
    level = tiers(wall, {"u": {"v": {}, "s": {"s1": 1}}})
    assert level["t2"] == 0 and level["l1"] == 1 and level["t1"] == level["t1~contour"] == 2 and level["s1"] == 3


def test_promotion_draws_first_then_studies_round_robin_by_style_within_budget():
    wall = _wall()
    votes = {"u": {"t1": 2, "t2": 1, "s1": 1, "s2": 1, "l1": 1, "l2": -1}}
    fills = ["packed", "contour", "flow"]
    work = promotions(wall, votes, fills, budget=100)
    assert [(w["kind"], w["id"]) for w in work] == [("draw", "t2"), ("draw", "s2"), ("study", "t1"), ("study", "l1"),
                                                     ("study", "s1")]
    assert work[2]["fills"] == ["contour", "flow"]
    assert [w["id"] for w in promotions(wall, votes, fills, budget=4)] == ["t2", "s2", "t1"]
    wall["t1~flow"] = {**wall["t1"], "study_of": "t1"}
    assert "t1" not in [w["id"] for w in promotions(wall, votes, fills, budget=100)]


def test_study_documents_take_the_source_genes_and_their_own_fill_and_paths(tmp_path):
    d = tmp_path / "separations" / "t1~maze"
    d.mkdir(parents=True)
    (d / "study.json").write_text(json.dumps({"schema": "tatbot.art-study/1", "of": "t1", "fill": "maze"}))
    (d / "separation.json").write_text(json.dumps({"time": {"minutes": 40.0, "dips": 9}, "colours": [{"layers": [1]}],
                                                   "fidelity": {"vs_target": {"mean": 6.5}}}))
    base = {"t1": {**_doc("american-traditional", "a koi fish", RED, drawn=True), "minutes": 99.0}}
    sib = study_docs(base, tmp_path, {"t1~maze-drawn": "abc"})["t1~maze"]
    assert sib["study_of"] == "t1" and sib["parents"] == ["t1"] and sib["fill"] == sib["genes"]["fill"] == "maze"
    assert sib["views"] == {"flow": "/_blob/f", "ours": "/_blob/o", "drawn": "/_blob/abc"} and sib["minutes"] == 40.0
    assert base["t1"]["genes"]["fill"] == "packed"


def _image(path, colour, block):
    img = np.full((64, 64, 3), 255, np.uint8)
    img[block] = colour
    Image.fromarray(img).save(path)
    return path


def test_the_map_places_lookalikes_together_and_keeps_studies_out_of_the_neighbours(tmp_path):
    red_a = descriptor(_image(tmp_path / "a.jpg", (220, 40, 30), np.s_[10:50, 10:50]))
    red_b = descriptor(_image(tmp_path / "b.jpg", (230, 50, 40), np.s_[12:52, 12:52]))
    blue = descriptor(_image(tmp_path / "c.jpg", (30, 90, 220), np.s_[0:20, 40:64]))
    assert np.linalg.norm(red_a - red_b) < np.linalg.norm(red_a - blue)
    wall = _wall()
    wall["t1~maze"] = {**wall["t1"], "study_of": "t1"}
    looks = {k: (red_a if k[0] == "t" else red_b if k[0] == "l" else blue) for k in wall}
    doc = graph(wall, looks, SEPARATION, {"t1": 1}, "now")
    n = len(doc["ids"])
    assert n == 7 and len(doc["xy"]) == len(doc["cluster"]) == n and doc["clusters"]
    study = doc["ids"].index("t1~maze")
    assert all(study not in (a, b) and 0 <= a < b < n for a, b, _ in doc["near"])
    assert doc["cluster"][study] == doc["cluster"][doc["ids"].index("t1")]
    assert sum(c["size"] for c in doc["clusters"]) == n - 1
    assert json.loads(json.dumps(doc)) == doc


def test_parents_are_the_kept_designs_sharing_two_genes():
    wall = _wall()
    totals = {"t1": 2, "t2": 1, "s1": -1}
    assert parents_for("american-traditional", "a koi fish", "red-black", wall, totals) == ["t1", "t2"]
    assert parents_for("american-traditional", "an astronaut floating", "warm", wall, totals) == []


def test_open_asks_take_the_first_prompts_and_keep_their_anchor_genes():
    wall, cat = _wall(), load_catalog()
    asks = {"a1": {"kind": "near", "anchors": ["t1"], "n": 2, "at": "1", "steer": "more negative space"},
            "a2": {"kind": "branch", "anchors": ["s1"], "n": 1, "at": "2"},
            "a3": {"kind": "blend", "anchors": ["t1", "s1"], "n": 2, "at": "3"},
            "a4": {"kind": "cluster", "anchors": ["l1", "l2"], "n": 1, "at": "4"},
            "done": {"kind": "near", "anchors": ["t1"], "n": 1, "at": "0", "status": "done"}}
    prompts = propose(wall, {"u": {"t1": 1}}, cat, round_no=4, n=9, seed=1, asks=asks)
    assert len(prompts) == 9 and [p.get("ask") for p in prompts[:6]] == ["a1", "a1", "a2", "a3", "a3", "a4"]
    near, branch, blend = prompts[:2], prompts[2], prompts[3:5]
    assert all(p["style"] == "american-traditional" and p["inks"] == RED and p["subject"] != "a koi fish" for p in near)
    assert all(p["text"].endswith("more negative space") and p["parents"] == ["t1"] for p in near)
    assert branch["subject"] == "a koi fish" and branch["inks"] == SKY and branch["style"] != "stained-glass"
    assert {(p["style"], tuple(p["inks"])) for p in blend} == {("american-traditional", tuple(SKY)),
                                                               ("stained-glass", tuple(RED))}
    assert prompts[5]["style"] == "linocut" and prompts[5]["variant"] == "ask-cluster"
    assert prompts[0]["id"].startswith("r4-01-") and all(p["text"] and p["separation"] for p in prompts)


def test_an_ask_that_does_not_fit_the_round_waits():
    asks = {"big": {"kind": "near", "anchors": ["t1"], "n": 5, "at": "1"}}
    prompts = propose(_wall(), {}, load_catalog(), round_no=4, n=3, seed=1, asks=asks)
    assert len(prompts) == 3 and not any(p.get("ask") for p in prompts)


def test_promotion_counts_what_the_corpus_already_acquired(tmp_path):
    wall = _wall()
    (tmp_path / "separations" / "t2").mkdir(parents=True)
    (tmp_path / "separations" / "t2" / "drawn.png").write_bytes(b"")
    (tmp_path / "separations" / "t1~maze").mkdir()
    (tmp_path / "separations" / "t1~maze" / "study.json").write_text(json.dumps({"of": "t1", "fill": "maze"}))
    seen = on_disk(wall, tmp_path)
    assert "drawn" not in wall["t2"]["views"] and seen["t2"]["views"]["drawn"] and seen["t1~maze"]["study_of"] == "t1"
    work = promotions(seen, {"u": {"t1": 1, "t2": 1}}, ["packed", "maze"], budget=10)
    assert [(w["kind"], w["id"]) for w in work] == [("study", "t2")]


def test_a_starred_sketch_is_drawn_so_its_bundle_stops_waiting():
    votes = {"u": {"v": {"t2": 1}, "s": {"t2": 1}}}
    assert [(w["id"], w["kind"]) for w in promotions(_wall(), votes, ["packed", "contour"], budget=10)] == [("t2", "draw")]


def test_stencil_preview_keeps_the_recorded_size_offset_and_underlying_print():
    from tatbot_art.wall_previews import place_on_stencil

    sheet = Image.new("RGB", (160, 240), "white")
    sheet.putpixel((46, 50), (30, 30, 30))
    ink = Image.new("RGB", (40, 80), "white")
    ink.paste((200, 30, 40), (20, 0, 40, 80))

    preview = place_on_stencil(sheet, ink, (40, 60), (10, 20), (-5, 10))

    assert preview.size == sheet.size
    assert preview.getpixel((46, 50)) == (30, 30, 30)
    assert preview.getpixel((79, 40)) == preview.getpixel((79, 119)) == (200, 30, 40)
    for point in ((79, 39), (79, 120), (80, 40)):
        assert preview.getpixel(point) == (255, 255, 255)
    assert sheet.getpixel((79, 40)) == (255, 255, 255)


def test_wall_renditions_require_the_original_digest_and_keep_recipe_metadata(tmp_path):
    from tatbot_art.wall_previews import enrich

    base = {"image_sha256": "source", "canvas_mm": [16, 24], "layers": [{"ink": "true_black", "fill": "edges"}]}
    alt = {**base, "layers": [{"ink": "bright_red", "fill": "solid:contour"}],
           "tools": {"bright_red": "lutin-5rl-mast"}, "line_mm": {"bright_red": 800},
           "time": {"minutes": 10, "dips": 6}, "colours": [], "fidelity": {"vs_target": {"mean": 4.5}}}
    reports = {"t1": base, "t1~red-5rl": alt, "t1~unrelated": {**alt, "image_sha256": "another"}}
    for identifier, report in reports.items():
        path = tmp_path / "separations" / identifier / "separation.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
    designs = {"t1": _wall()["t1"]}

    enrich(designs, tmp_path, load_catalog(), {"t1~red-5rl-drawn": "one", "t1~unrelated-drawn": "two"})

    assert set(designs) == {"t1"}
    [preview] = designs["t1"]["renditions"]
    assert preview["id"] == "t1~red-5rl" and preview["views"]["drawn"] == "/_blob/one"
    assert preview["recipe"] == {"strokes": ["solid:contour"],
                                  "inks": [{"name": "Bright Red", "cartridge": "MAST Pro 1005RL"}]}
    assert preview["canvas_mm"] == [16, 24] and preview["minutes"] == 10


def test_wall_stencil_asset_cannot_be_reused_at_a_different_scale(tmp_path, monkeypatch):
    from tatbot_art import wall_previews

    separation = tmp_path / "design"
    separation.mkdir()
    Image.new("RGB", (40, 60), "black").save(separation / "target.png")
    out = tmp_path / "previews"
    report = {"canvas_mm": [10, 15], "page": {"stencil": "print", "at_mm": [0, 0], "fits": True}}
    monkeypatch.setattr(wall_previews, "read_print", lambda _path: (
        {"page_mm": [40, 60], "reference_id": "reference"}, Image.new("RGB", (160, 240), "white")))

    wall_previews.stencil_for(separation, report, {}, out)
    [asset] = out.glob("design-stencil-ours-*.png")
    assets = {asset.stem: "original-preview"}
    assert wall_previews.stencil_for(separation, report, assets, None)["url"] == "/_blob/original-preview"

    report["canvas_mm"] = [20, 30]
    assert wall_previews.stencil_for(separation, report, assets, None)["url"] is None
