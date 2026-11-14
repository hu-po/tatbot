"""python/tatbot_art bundle: a drawn separation as its acquired layers, inks, program and render."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tatbot_ink
import yaml
from tatbot_art import acquire, bundle
from tatbot_art.inks import load_catalog
from tatbot_contracts.artwork import freeze_artwork
from tatbot_contracts.paths import freeze_program

REPO = Path(__file__).resolve().parents[2]
ADAPTER = "dbv3-batik-paths/1"
CANVAS = (0.02, 0.024)


def _artwork(ink: str, rgb, paths, k: int) -> dict:
    elements = [{"id": f"p{j}", "kind": "path", "closed": False, "fill": False, "width_m": 0.0006, "deposition": 1,
                 "points_m": [list(map(float, p)) for p in pts]} for j, pts in enumerate(paths)]
    geometry = {"canvas_m": {"width": CANVAS[0], "height": CANVAS[1]}, "inks": [{"id": ink, "color_srgb": rgb}],
                "layers": [{"id": "layer-1", "ink_id": ink, "elements": elements}], "negative_space_masks": []}
    program = freeze_program(geometry, source_sha256=f"{k}" * 64, name=f"layer-{k}", adapter=ADAPTER)[0]
    return freeze_artwork(program, name=f"layer-{k}", source={
        "kind": "generated", "identifier": f"layer-{k}", "license": None, "attribution": "test",
        "generation": {"prompt": "a test", "model": None, "model_revision": None, "seed": None, "tracing": "test"}},
        conversion={"adapter": ADAPTER, "recipe_sha256": f"{k + 4}" * 64, "chord_error_m": 0.000005})


def _separation(root: Path, did="gx-1", *, drawn=True, line=None) -> Path:
    """A drawn separation as `tatbot art separate` leaves it: jobs, their acquisitions, inks.yaml, report."""
    sep = root / "separations" / did
    layers = [("bright_red", "solid", [[(0.004, 0.004), (0.016, 0.004), (0.016, 0.02)]], [0.98, 0.31, 0.2]),
              ("true_black", "solid", [[(0.002, 0.002), (0.018, 0.022)], [(0.018, 0.002), (0.002, 0.022)]],
               [0.24, 0.24, 0.25])]
    report_layers, pairs = [], []
    for k, (ink, fill, paths, rgb) in enumerate(layers):
        job = sep / "jobs" / f"gx-1-{k:02d}-{ink.replace('_', '-')}-{fill}"
        job.mkdir(parents=True)
        (job / "job.json").write_text(json.dumps({"job": k}) + "\n")
        acq = sep / "acq" / f"{job.name}-{hashlib.sha256((job / 'job.json').read_bytes()).hexdigest()[:8]}"
        acq.mkdir(parents=True)
        (acq / "artwork.json").write_text(json.dumps(_artwork(ink.replace("_", "-"), rgb, paths, k + 1)) + "\n")
        report_layers.append({"job": job.name, "ink": ink, "fill": fill, "angle_deg": 45.0, "area_mm2": 1.0})
        pairs.append((SimpleNamespace(ink_id=ink), acq / "artwork.json"))
    stale = sep / "acq" / "gx-1-00-bright-red-solid-00000000"   # a source's copy a fill study keeps beside its own
    stale.mkdir()
    (stale / "artwork.json").write_text("{}\n")
    line = line or {"true_black": 30, "bright_red": 10}
    acquire.inks_file(sep / "inks.yaml", pairs, line, "lutin-3rl-bugpin")
    report = {"schema": "tatbot.art-separation/1", "image": f"/corpus/flow/gx/{did.split('~')[0]}.jpg",
              "image_sha256": "c" * 64, "generation": {"prompt": "a test", "model": "test"}, "style": "geometric",
              "solid_fill": None, "canvas_mm": [CANVAS[0] * 1000, CANVAS[1] * 1000], "px_per_mm": 10.0,
              "inks": {"bright_red": {"source": "vendor-photo"}, "true_black": {"source": "vendor-photo"}},
              "weakest_source": "vendor-photo", "layers": report_layers, "refused": [],
              "line_mm": line}
    if drawn:
        report.update({"time": {"minutes": 1.0}, "fidelity": {"vs_target": {"mean": 5.0}}})
    (sep / "separation.json").write_text(json.dumps(report) + "\n")
    return sep


def _compile(artworks, inks, out, extra):
    """`tatbot ros compile` in process, with the arguments the bundle passes it."""
    out.mkdir(parents=True, exist_ok=True)
    stencil = extra[extra.index("--stencil") + 1] if "--stencil" in extra else None
    at_m = [float(v) / 1000 for a in extra if a.startswith("--at=") for v in a[5:].split(",")]
    program = tatbot_ink.compile(list(artworks), repo=REPO, inks_path=inks, stencil=stencil, at_m=at_m)
    tatbot_ink.write_program(program, out / "program.json")
    return program["stats"]


def _strokes(program):
    return [(op["resource_id"], op["points_m"]) for op in program["ops"] if op["op"] == "stroke"]


def test_layers_take_each_jobs_own_acquisition_in_drawing_order(tmp_path):
    sep = _separation(tmp_path)
    lays = bundle.layers(sep)
    assert [lay["layer"] for lay in lays] == ["00-bright-red-solid", "01-true-black-solid"]
    assert all("00000000" not in str(lay["path"]) for lay in lays)


def test_bundle_carries_each_layer_and_draws_what_the_separation_drew(tmp_path):
    sep = _separation(tmp_path)
    out = tmp_path / "out" / "gx-1"
    result = bundle.build(sep, out, load_catalog(), _compile)
    assert not (out / "design.json").exists()
    for lay, row in zip(bundle.layers(sep), result["layers"], strict=True):
        # the frozen records travel byte for byte, so inks.yaml still binds them
        assert (out / row["file"]).read_bytes() == lay["path"].read_bytes()
        assert row["artwork_sha256"] == json.loads(lay["path"].read_text())["content_sha256"]
    layered = tatbot_ink.compile([lay["path"] for lay in bundle.layers(sep)], repo=REPO, inks_path=sep / "inks.yaml")
    program = json.loads((out / "program" / "program.json").read_text())
    assert _strokes(program) == _strokes(layered)
    assert "set" not in result and [i["cap"] for i in result["inks"]] == ["inkcap_large_2", "inkcap_medium_1"]
    assert result["time"]["minutes"] > 0 and result["compile"]["strokes"] == len(_strokes(program))
    assert (out / "render.png").exists() and result["substrate"] == {"inks": "silicon_skin", "render": "silicone",
                                                                     "spread_mm": 0.0}
    for name, digest in result["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_position_and_ink_substrate_reach_the_program(tmp_path):
    sep = _separation(tmp_path)
    out = tmp_path / "out"
    centred = bundle.build(sep, tmp_path / "centred", load_catalog(), _compile)
    moved = bundle.build(sep, out, load_catalog(), _compile, at_mm=(10.0, -20.0), ink_substrate="human_skin")
    assert moved["placement"]["at_mm"] == [10.0, -20.0] and "rotation_deg" not in moved["placement"]
    program = json.loads((out / "program" / "program.json").read_text())
    base = json.loads((tmp_path / "centred" / "program" / "program.json").read_text())
    shift = {(round(x1 - x0, 9), round(y1 - y0, 9)) for (_, a), (_, b) in zip(_strokes(base), _strokes(program), strict=True)
             for (x0, y0), (x1, y1) in zip(a, b, strict=True)}
    assert shift == {(0.01, -0.02)} and centred["placement"]["at_mm"] == [0.0, 0.0]
    assert program["substrate"] == "human_skin" and "substrate: human_skin" in (out / "inks.yaml").read_text()


def test_inks_that_bind_other_artwork_are_refused(tmp_path):
    sep = _separation(tmp_path)
    (sep / "inks.yaml").write_text((sep / "inks.yaml").read_text().replace("artwork_sha256: ", "artwork_sha256: 0", 1))
    with pytest.raises(ValueError, match="binds other artwork"):
        bundle.build(sep, tmp_path / "out", load_catalog(), _compile)


def test_run_bundles_drawn_designs_and_lists_sketches_as_waiting(tmp_path):
    _separation(tmp_path, "gx-1")
    _separation(tmp_path, "gx-2", drawn=False)
    index = bundle.run(["gx-1", "gx-2", "gx-3"], tmp_path, tmp_path / "out", load_catalog(), _compile, log=lambda _: None)
    assert [b["id"] for b in index["bundles"]] == ["gx-1"]
    assert [w["id"] for w in index["waiting"]] == ["gx-2", "gx-3"]
    assert json.loads((tmp_path / "out" / "bundles.json").read_text()) == index


def test_one_load_gives_every_design_the_same_caps_by_combined_line(tmp_path):
    _separation(tmp_path, "gx-1", line={"true_black": 30, "bright_red": 10})   # its own draft: black in the large cap
    _separation(tmp_path, "gx-2", line={"true_black": 5, "bright_red": 40})    # its own draft: red in the large cap
    _separation(tmp_path, "gx-3", drawn=False)
    index = bundle.run(["gx-1", "gx-2", "gx-3"], tmp_path, tmp_path / "out", load_catalog(), _compile,
                       shared_load=True, log=lambda _: None)
    assert index["load"] == {"bright_red": "inkcap_large_2", "true_black": "inkcap_medium_1"}
    for did in ("gx-1", "gx-2"):
        out = tmp_path / "out" / did
        draft = yaml.safe_load((out / "inks.yaml").read_text())
        assert {r["ink_id"]: r["slot"] for r in draft["resources"]} == index["load"]
        assert {r["activation"]["slot"] for r in draft["resources"]} == {acquire.WATER_SLOT}
        program = json.loads((out / "program" / "program.json").read_text())
        mine = tatbot_ink.compile([lay["path"] for lay in bundle.layers(tmp_path / "separations" / did)], repo=REPO,
                                  inks_path=tmp_path / "separations" / did / "inks.yaml")
        assert [p for _, p in _strokes(program)] == [p for _, p in _strokes(mine)]   # same paths, other caps
        assert {i["ink"]: i["cap"] for i in json.loads((out / "bundle.json").read_text())["inks"]} == index["load"]
    assert [w["id"] for w in index["waiting"]] == ["gx-3"]


def test_poses_put_designs_at_their_own_places_on_one_page(tmp_path):
    _separation(tmp_path, "gx-1")
    _separation(tmp_path, "gx-2")
    index = bundle.run(["gx-1", "gx-2"], tmp_path, tmp_path / "out", load_catalog(), _compile,
                       poses={"gx-1": (0.0, 30.0)}, at_mm=(0.0, -30.0), log=lambda _: None)
    assert [b["at_mm"] for b in index["bundles"]] == [[0.0, 30.0], [0.0, -30.0]]
    ys = {}
    for did in ("gx-1", "gx-2"):
        program = json.loads((tmp_path / "out" / did / "program" / "program.json").read_text())
        ys[did] = [y for _, pts in _strokes(program) for _, y in pts]
    assert min(ys["gx-1"]) > max(ys["gx-2"])   # the 24 mm canvases, 60 mm apart, do not meet


def test_one_load_refuses_more_inks_than_caps(tmp_path):
    seps = [_separation(tmp_path, f"gx-{k}") for k in range(3)]
    for k, sep in enumerate(seps):   # three designs, two inks each, six inks in all
        text = (sep / "inks.yaml").read_text()
        (sep / "inks.yaml").write_text(text.replace("ink_id: bright_red", f"ink_id: ink_{k}a")
                                       .replace("ink_id: true_black", f"ink_id: ink_{k}b"))
    with pytest.raises(ValueError, match="more caps than"):
        bundle.one_load(seps)


def test_chosen_is_the_stars_and_with_kept_every_positive_total():
    votes = {"u1": {"v": {"a": 1, "b": -1, "c": 1}, "s": {"d": 1, "e": 0}}, "u2": {"v": {"c": -1}}}
    assert bundle.chosen(votes, kept=False) == ["d"]
    assert bundle.chosen(votes, kept=True) == ["a", "d"]
