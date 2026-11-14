"""python/tatbot_art separation: tone inverse, colour assignment, cleaning, layers, jobs and the plan."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from drawingbot.job import validate_job
from tatbot_art import acquire, pipeline
from tatbot_art.color import from_8bit
from tatbot_art.inks import InkLook, load_catalog
from tatbot_art.png import write_png
from tatbot_art.separate import Layer, assign, candidates, clean, layers_for, separate, widen
from tatbot_art.tone import load as load_tone

YELLOW, BLACK, RED = (253, 201, 73), (20, 20, 22), (250, 80, 50)


def _look(ink_id, rgb):
    return InkLook(ink_id, ink_id, tuple(from_8bit(rgb)), "nominal", False)


LOOKS = [_look("golden_yellow", YELLOW), _look("true_black", BLACK)]


def test_tone_inverse_runs_from_white_to_the_solid_grey():
    tone = load_tone()
    assert tone.grey(0.0) == 255 and tone.grey(1.0) == tone.solid_grey
    light, mid = tone.grey(0.35), tone.grey(0.7)
    assert 255 > light > mid > tone.solid_grey
    assert float(tone.coverage(mid)) == pytest.approx(0.7, abs=0.05)


def _flash(size=80):
    img = np.ones((size, size, 3))
    img[10:70, 10:40] = from_8bit(YELLOW)
    img[10:70, 45:70] = from_8bit(BLACK)
    img[38, 5:75] = from_8bit(BLACK)  # a one-pixel black line across both
    return img


def test_assign_snaps_flat_colours_to_the_load_and_paper():
    swatches = candidates(LOOKS, 0.6, mixes=False)
    labels = assign(_flash(), swatches)
    assert labels[0, 0] == -1
    assert [lay.ink_id for lay in swatches[labels[20, 20]].layers] == ["golden_yellow"]
    assert swatches[labels[20, 50]].layers[0].ink_id == "true_black"


def test_clean_keeps_a_thin_black_line_and_drops_a_thin_colour_sliver():
    swatches = candidates(LOOKS, 0.6, mixes=False)
    img = _flash()
    img[60, 5:9] = from_8bit(YELLOW)  # a yellow sliver one pixel wide, outside the block
    labels = assign(img, swatches)
    black = {k for k, sw in enumerate(swatches) if sw.layers[0].ink_id == "true_black" and sw.layers[0].coverage == 1}
    out = clean(labels, black, line_px=6.0, min_area_px=2)
    assert out[38, 41] in black  # the line between the blocks survives
    assert out[60, 6] == -1      # the sliver went to the paper around it


def test_widen_grows_a_black_line_to_one_drawn_line():
    swatches = candidates(LOOKS, 0.6, mixes=False)
    labels = widen(assign(_flash(), swatches), swatches, "true_black", line_px=6.0)
    column = labels[30:46, 41]
    assert (column == labels[38, 41]).sum() >= 5


def test_layers_trace_and_pack_the_darkest_ink_and_hatch_tints():
    swatches = candidates(LOOKS, 0.6, mixes=False)
    tint = next(k for k, sw in enumerate(swatches) if sw.layers[0].ink_id == "golden_yellow" and sw.layers[0].coverage < 0.5)
    labels = assign(_flash(), swatches)
    labels[10:30, 10:40] = tint
    cov = {i: np.zeros(labels.shape) for i in ("golden_yellow", "true_black")}
    for k, sw in enumerate(swatches):
        for lay in sw.layers:
            cov[lay.ink_id][labels == k] = lay.coverage
    fills = [(lay.ink_id, lay.fill) for lay in layers_for(cov, LOOKS, load_tone(), 6.0, True, 10.0)]
    assert fills == [("golden_yellow", "hatch"), ("golden_yellow", "solid"), ("true_black", "solid"), ("true_black", "edges")]


def test_job_is_a_valid_dbv3_job_with_the_ink_angle_and_no_255_channel(tmp_path):
    layer = Layer("golden_yellow", "hatch", np.full((40, 60), 120, np.uint8), 15.0, 10.0)
    look = _look("golden_yellow", (255, 201, 73))
    prov = acquire.provenance("t-00-golden-yellow-hatch", None, "x.png")
    d = acquire.write_job(tmp_path, "t-00-golden-yellow-hatch", layer, look, (6.0, 4.0), 0.6, 7, prov)
    job = validate_job(json.loads((d / "job.json").read_text()))
    assert job["pfm"] == "Streamlines Flow Field" and job["settings"]["Start Angle"] == 165
    assert max(job["drawing_set"]["pens"][0]["rgba"][:3]) == 254
    assert acquire.dbv3_angle(45.0) == 135.0 and acquire.dbv3_angle(-45.0) == 45.0


def test_inks_file_puts_water_in_a_large_cap_and_the_most_line_next(tmp_path):
    arts = []
    for ink in ("golden_yellow", "true_black"):
        art = tmp_path / f"{ink}.json"
        art.write_text(json.dumps({"content_sha256": ink[0] * 64}))
        arts.append((Layer(ink, "solid", None, 0.0, 0.0), art))
    text = acquire.inks_file(tmp_path / "inks.yaml", arts, {"golden_yellow": 100.0, "true_black": 900.0},
                             "lutin-3rl-bugpin").read_text()
    assert text.index("- id: true_black") < text.index("- id: golden_yellow")
    assert "slot: inkcap_large_2" in text.split("- id: golden_yellow")[0]
    assert text.count("slot: inkcap_large_1") == 2 and "pen_id: golden-yellow" in text


def test_plan_without_dbv3_writes_target_jobs_and_report(tmp_path):
    src = tmp_path / "flash.png"
    write_png(src, np.clip(np.rint(_flash(160) * 255), 0, 255).astype(np.uint8))
    sep, _looks, jobs, canvas, report = pipeline.plan(
        src, ["golden_yellow", "true_black"], load_catalog(), tmp_path / "out", width_mm=20.0, style="flash",
        px_per_mm=10.0, line_width_mm=0.6, mixes=True, seed=1, tone_path=None)
    assert (tmp_path / "out" / "target.png").exists() and canvas[0] == pytest.approx(20.0)
    assert {lay["fill"] for lay in report["layers"]} >= {"solid", "edges"}
    for job in jobs:
        validate_job(json.loads((job / "job.json").read_text()))
    assert report["inks"]["golden_yellow"]["source"] == "vendor-photo" and len(sep.layers) == len(jobs)


def test_separate_refuses_an_unknown_style():
    with pytest.raises(ValueError, match="unknown style"):
        separate(_flash(), LOOKS, style="cubist", tone=load_tone(), px_per_mm=10.0, line_width_mm=0.6)


@pytest.mark.skipif(not os.environ.get("TATBOT_DBV3_APP"), reason="set TATBOT_DBV3_APP to an activated DBV3 Premium")
def test_separate_end_to_end_with_dbv3(tmp_path):
    src = tmp_path / "flash.png"
    write_png(src, np.clip(np.rint(_flash(300) * 255), 0, 255).astype(np.uint8))
    sep, _looks, jobs, canvas, report = pipeline.plan(
        src, ["golden_yellow", "true_black"], load_catalog(), tmp_path / "out", width_mm=30.0, style="flash",
        px_per_mm=10.0, line_width_mm=0.6, mixes=False, seed=1, tone_path=None)
    report = pipeline.build(sep, load_catalog(), jobs, canvas, report, tmp_path / "out",
                            app=Path(os.environ["TATBOT_DBV3_APP"]), substrate="silicone")
    assert report["compile"]["strokes"] > 0 and report["fidelity"]["vs_target"]["mean"] < 15


def test_a_solid_fill_texture_swaps_the_pfm_and_its_source_grey(tmp_path):
    from tatbot_art.acquire import SOLID_FILLS
    from tatbot_art.pipeline import _texture
    layers = [Layer("true_black", "solid", np.where(np.eye(8, dtype=bool), 70, 255).astype(np.uint8), 45.0, 1.0),
              Layer("true_black", "edges", np.full((8, 8), 255, np.uint8), 45.0, 1.0)]
    _texture(layers, "contour")
    assert [lay.fill for lay in layers] == ["solid:contour", "edges"]
    assert set(np.unique(layers[0].source)) == {SOLID_FILLS["contour"][3], 255}
    d = acquire.write_job(tmp_path, "t-00-true-black-solid-contour", layers[0], _look("true_black", BLACK), (0.8, 0.8),
                          0.6, 1, acquire.provenance("t-00-true-black-solid-contour", None, "x.png"))
    job = validate_job(json.loads((d / "job.json").read_text()))
    assert job["pfm"] == "Streamlines Edge Field" and job["drawing_set"]["distribution_type"] == "Luminance weighted"


@pytest.mark.parametrize("fill", sorted(acquire.SOLID_FILLS))
def test_every_solid_fill_writes_a_valid_job_with_its_family_distribution(tmp_path, fill):
    layer = Layer("true_black", f"solid:{fill}", np.full((8, 8), 70, np.uint8), 45.0, 1.0)
    jid = f"t-00-true-black-solid-{fill}"
    d = acquire.write_job(tmp_path, jid, layer, _look("true_black", BLACK), (0.8, 0.8), 0.6, 1,
                          acquire.provenance(jid, None, "x.png"))
    job = validate_job(json.loads((d / "job.json").read_text()))
    pfm, distribution = acquire.SOLID_FILLS[fill][:2]
    assert job["pfm"] == pfm and job["drawing_set"]["distribution_type"] == distribution
    family = {"Sketch": "Even weighted", "Streamlines": "Luminance weighted"}.get(pfm.split()[0], "Single pen")
    assert distribution == family
