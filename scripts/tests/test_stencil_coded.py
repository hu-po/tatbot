"""Coded flower-of-life stencil: the code's window margin, and the coded tracker on bench scenes."""

import json

import cv2
import numpy as np
import pytest
import stencil_bench_scene as bench
import stencil_bench_score as scoring
import stencil_coded as coded
import stencil_coded_tracker as tracker
from cli_runner import REPO, tatbot

cv2.setNumThreads(1)


@pytest.fixture(scope="module")
def prints(tmp_path_factory):
    """Three prints of the default coded design, differing only in print ID."""
    root = tmp_path_factory.mktemp("coded")
    return [bench.load_artwork(coded.generate(f"test-{name}", root/name)) for name in "abc"]


@pytest.fixture(scope="module")
def cameras():
    return bench.wrist_models(REPO)+[bench._nominal_overhead()]


def located(artwork, cameras, prints, index=2, kind="positive", degradation="clean", **transfer):
    params = bench.sample_params("train", index, cameras, "wrist", degradation)
    params["kind"] = kind
    params["surface"]["radius_mm"] = None
    params["pose"].update(target_uv=[.5, .5], distance_mm=150., tilt_deg=10.)
    params["transfer"].update(transfer)
    camera = {c.name: c for c in cameras}[params["camera"]]
    scene = bench.Scene(params, artwork, camera, distractor=prints[2])
    decoder = tracker.CodedTracker()
    decoder.prepare(prints[:2])
    result = decoder.locate(scene.render(), camera.intrinsics())
    return result, scoring.score_scene(scene, result, 1.)


def page_edges(artwork):
    code = coded.load_code(artwork.reference.parent)
    return code, [tuple(e) for e in code["edges"]]


def test_the_artwork_decodes_its_own_code_under_every_symmetry(prints):
    code, edges = page_edges(prints[0])
    book = tracker.Codebook(prints[0], code)
    for sym in (0, 4, 7, 11):
        sigma, m, matrix = tracker.TRANSFORMS[sym]
        observed = [(*coded.map_edge(q, r, k, bit, sigma, m, matrix, (3, -5), book.reflect_flips), 1.)
                    for q, r, k, bit in edges]
        best = tracker.vote(observed, [book])[0]
        assert best["agree"] == len(edges) and best["disagree"] == 0
        assert best["log10_chance"] < -100
        # the winning hypothesis maps every observed edge back onto its page edge
        agree, disagree = book.agreement([e[:4] for e in observed], best["sym"], best["offset"])
        assert (agree, disagree) == (len(edges), 0)


def test_windows_decode_with_their_claimed_erasures(prints):
    """Every radius-3 window, read under a random symmetry with (margin - 1) edges erased, still
    names exactly one page hypothesis; the stored summary is what the code achieves."""
    code, edges = page_edges(prints[0])
    layout = coded.Layout(**code["geometry"], centre_mm=code["centre_mm"])
    bits = np.array([e[3] for e in edges])
    flips = code["style"]["bit"] == "side"
    sizes, margins = coded.window_margins(layout, bits, code["window_radius"], reflect_flips=flips)
    assert code["window"]["distance_min"] == margins.min() >= 4
    assert code["window"]["erasable_fraction_p50"] >= .25
    book = tracker.Codebook(prints[0], code)
    lookup = {e[:3]: e[3] for e in edges}
    rng = np.random.default_rng(0)
    slots = coded.window_slots(code["window_radius"])
    for centre_index in rng.choice(len(layout.nodes), 12, replace=False):
        cq, cr = layout.nodes[centre_index]
        window = [(cq+dq, cr+dr, k) for dq, dr, k in slots if (cq+dq, cr+dr, k) in lookup]
        keep = rng.permutation(len(window))[margins[centre_index]-1:]
        sigma, m, matrix = tracker.TRANSFORMS[int(rng.integers(12))]
        observed = []
        for i in keep:
            q, r, k = window[i]
            observed.append((*coded.map_edge(q, r, k, lookup[(q, r, k)], sigma, m, matrix, (7, 2), flips), 1.))
        hypotheses = tracker.vote(observed, [book])
        assert hypotheses[0]["agree"] == len(keep) and hypotheses[1]["agree"] < len(keep)
        agree, disagree = book.agreement([o[:4] for o in observed], hypotheses[0]["sym"], hypotheses[0]["offset"])
        assert (agree, disagree) == (len(keep), 0)


def test_a_clean_transfer_decodes_to_sub_millimetre_with_its_print_id(prints, cameras):
    result, record = located(prints[1], cameras, prints)
    assert result.status == "accepted", result.reason
    assert result.pattern_id == prints[1].pattern_id and record["success"]
    assert result.extra["print_id"] == coded.load_code(prints[1].reference.parent)["print_id"]
    assert record["p50_mm"] < .5 and record["model_p50_mm"] < 1.
    assert result.extra["mirrored"] is False


def test_a_mirrored_transfer_is_reported_mirrored(prints, cameras):
    result, record = located(prints[0], cameras, prints, mirrored=True)
    assert result.status == "accepted", result.reason
    assert result.extra["mirrored"] is True and record["success"]


def test_a_washed_transfer_still_decodes(prints, cameras):
    result, record = located(prints[0], cameras, prints, degradation="full", washoff=.3, speckle=.03,
                             color="violet", transmittance_rgb=list(bench.INKS["violet"]), density=1.,
                             skin_rgb=[243, 218, 186])
    assert result.status == "accepted", result.reason
    assert record["success"] and not record["false_accept"]


@pytest.mark.parametrize("kind", ["other_print", "blank"])
def test_other_prints_and_blank_skin_are_refused(prints, cameras, kind):
    result, record = located(prints[0], cameras, prints, kind=kind)
    assert result.status != "accepted", (result.pattern_id, result.reason)
    assert not record["false_accept"]


def test_generator_is_deterministic_and_subtle(prints, tmp_path):
    again = coded.generate("test-a", tmp_path/"again")
    assert json.loads((again/"coded.json").read_text()) == coded.load_code(prints[0].reference.parent)
    card = scoring.subtlety(prints[0])
    assert card["largest_solid_blob_mm2"] < 10 and card["black_fraction_frame"] < .3


def test_cli_plans_the_coded_bench():
    result = tatbot("vision", "stencil", "bench", "--dry-run", "--candidate", "coded-flower-of-life",
                    "--tracker", "coded", "--set", "knot_mm=1.4")
    assert result.returncode == 0, result.stderr
    assert "--candidate coded-flower-of-life" in result.stdout and "--tracker coded" in result.stdout


BEAD = {"spacing_mm": 7, "frame_mm": 14, "knot_mm": 2.6, "stroke_mm": .3, "halo_mm": 1.}


@pytest.mark.parametrize("style", [{**BEAD, "bit": "seed"}, {**BEAD, "bit": "teardrop"},
                                   {"spacing_mm": 6, "frame_mm": 12, "knot_mm": 2.2, "stroke_mm": .4, "ornament": "dots"}])
def test_every_bit_style_decodes_a_clean_mirrored_transfer(tmp_path, cameras, style):
    """Full-petal styles carry a bit a reflection does not flip; the decoder must still name the
    mirror and every junction."""
    styled = [bench.load_artwork(coded.generate(f"style-{n}", tmp_path/n, **style)) for n in "ab"]
    result, record = located(styled[0], cameras, styled+[styled[1]], mirrored=True)
    assert result.status == "accepted", result.reason
    assert record["success"] and result.extra["mirrored"] is True
