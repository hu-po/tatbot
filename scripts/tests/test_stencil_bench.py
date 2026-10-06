"""Stencil bench tier 0: truth geometry, deterministic banks, degradation control, scoring."""

import json

import cv2
import numpy as np
import pytest
import stencil_bench_scene as bench
import stencil_bench_score as scoring
from cli_runner import REPO, tatbot
from stencil_bench_trackers import ACCEPTED, Located, Oracle, SiftBaseline

cv2.setNumThreads(1)
ARTWORK = REPO/"docs/assets/stencil-frames"


@pytest.fixture(scope="module")
def artwork():
    return bench.load_artwork(ARTWORK/"tatbot-43")


@pytest.fixture(scope="module")
def cameras():
    return bench.wrist_models(REPO)+[bench._nominal_overhead()]


def scene_with(artwork, cameras, index=0, *, radius=None, degradation="full", role="wrist", **transfer):
    params = bench.sample_params("train", index, cameras, role, degradation)
    params["kind"] = "positive"
    params["surface"]["radius_mm"] = radius
    params["transfer"].update(transfer)
    camera = {c.name: c for c in cameras}[params["camera"]]
    return bench.Scene(params, artwork, camera, distractor=bench.load_artwork(ARTWORK/"tatbot-44"))


@pytest.mark.parametrize("radius", [None, 35.0])
def test_truth_projects_and_backprojects_onto_the_same_skin_point(artwork, cameras, radius):
    scene = scene_with(artwork, cameras, 2, radius=radius, wobble_mm=1.2, mirrored=True)
    uv, visible = scene.frame_cells()
    assert visible.mean() > .2
    s = scene.truth_skin_mm(uv[visible])
    pixels, seen = scene.project(s)
    assert seen.all()
    back = scene.backproject(pixels)
    assert np.nanmax(np.linalg.norm(back-s, axis=1)) < .02
    # The wobble moves the transferred geometry off the ideal page, by about its RMS.
    ideal = uv[visible]*np.array(artwork.page_mm)
    ideal[:, 0] = artwork.page_mm[0]-ideal[:, 0]
    shift = np.linalg.norm(s-ideal, axis=1)
    assert .4 < np.sqrt(np.mean(shift**2)) < 2.


def test_rendered_ink_lands_on_the_truth_pixels(artwork, cameras):
    scene = scene_with(artwork, cameras, 2, degradation="clean")
    gray = cv2.cvtColor(scene.render(), cv2.COLOR_BGR2GRAY).astype(float)
    h, w = artwork.ink.shape
    rows, cols = np.nonzero(artwork.ink)
    pick = np.random.default_rng(0).choice(len(rows), 4000, replace=False)
    ink_uv = np.c_[(cols[pick]+.5)/w, (rows[pick]+.5)/h]
    blank_uv = np.c_[np.random.default_rng(1).uniform(.3, .7, (4000, 2))]   # the clear centre
    values = []
    for uv in (ink_uv, blank_uv):
        pixels, visible = scene.truth_pixels(uv)
        pixels = np.rint(pixels[visible]).astype(int)
        values.append(gray[pixels[:, 1], pixels[:, 0]].mean())
    assert values[0] < values[1]-25


def test_banks_are_deterministic_and_disjoint(cameras):
    first = [bench.sample_params("train", i, cameras) for i in range(16)]
    again = [bench.sample_params("train", i, cameras) for i in range(16)]
    holdout = [bench.sample_params("holdout", i, cameras) for i in range(16)]
    assert json.dumps(first) == json.dumps(again)
    assert all(a["seed"] != b["seed"] and a["transfer"] != b["transfer"] for a, b in zip(first, holdout, strict=True))
    assert [p["kind"] for p in first].count("blank") == 2
    assert [p["kind"] for p in first].count("other_print") == 2


@pytest.mark.parametrize("target", [0.0, 0.3, 0.6])
def test_washoff_removes_the_requested_fraction_of_the_frame(artwork, cameras, target):
    scene = scene_with(artwork, cameras, 5, washoff=target, speckle=0.0)
    assert abs(scene.transfer.washed_fraction-target) < .08


def test_scorer_accepts_truth_and_flags_wrong_places_and_blanks(artwork, cameras):
    scene = scene_with(artwork, cameras, 2, radius=45.0)
    oracle = Oracle()
    oracle.truth = scene
    record = scoring.score_scene(scene, oracle.locate(None, {}), 1.)
    assert record["success"] and not record["false_accept"]
    assert record["p95_mm"] < .05 and record["localized_fraction"] > .9
    liar = Oracle(offset_mm=6.)
    liar.truth = scene
    record = scoring.score_scene(scene, liar.locate(None, {}), 1.)
    assert record["false_accept"] and not record["success"]
    blank = scene_with(artwork, cameras, 3)
    blank.params["kind"] = "blank"
    blank.transfer = None
    claim = Located(ACCEPTED, artwork.pattern_id, np.array([[.1, .1]]), np.array([[10., 10.]]))
    assert scoring.score_scene(blank, claim, 1.)["false_accept"]
    card = scoring.scorecard([record], {"tracker": "oracle"}, artwork)
    assert card["overall"]["false_accepts"] == 1
    assert abs(card["subtlety"]["black_fraction_page"]-artwork.meta["settings"]["black_fraction"]) < .01


def test_sift_baseline_locates_a_clean_transfer_to_sub_millimetre(artwork, cameras):
    tracker = SiftBaseline()
    tracker.prepare([artwork])
    scene = scene_with(artwork, cameras, 2, degradation="clean")
    located = tracker.locate(scene.render(), scene.camera.intrinsics())
    record = scoring.score_scene(scene, located, 1.)
    assert record["success"], record["reason"]
    assert record["p50_mm"] < 1 and record["model_p95_mm"] < 2


def test_a_calibrated_overhead_camera_is_posed_over_the_right_arms_pivot(tmp_path):
    """The robot-world record registers the URDF root; the pivot is in the right arm's base.
    A camera the bundle puts straight over the pivot must sit straight over it here too."""
    from ink_spec import base_from_root_matrix
    from stencil_coded_live import drawing_pads
    pivot_base = drawing_pads(REPO).get("right")
    if pivot_base is None:
        pytest.skip("no right-arm drawing pad in this checkout's workspace")
    pivot_root = np.linalg.inv(base_from_root_matrix(REPO, "right")) @ [*pivot_base, 1.]
    (tmp_path/"world.json").write_text(json.dumps({"world_from_base": np.eye(4).tolist()}))
    (tmp_path/"bundle.json").write_text(json.dumps({"cameras": {"camera1": {
        "intrinsics": {"width": 640, "height": 480, "fx": 500., "fy": 500., "cx": 320., "cy": 240.},
        "world_from_camera": {"rotation": np.diag([1., -1., -1.]).ravel().tolist(),
                              "translation_m": (pivot_root[:3]+[0., 0., .8]).tolist()}}}}))
    [camera] = bench.overhead_models(REPO, tmp_path/"bundle.json", tmp_path/"world.json")
    assert camera.basis == "calibration-bundle"
    np.testing.assert_allclose(camera.surface_from_camera_candidates[0][:3, 3], [0., 0., 800.], atol=1e-6)


def test_overhead_distortion_inverse_round_trips(artwork, cameras):
    camera = bench._nominal_overhead()
    camera.dist = (-.33, .075, 0., 0., 0.)
    params = bench.sample_params("train", 1, [camera], "overhead")
    params["kind"] = "positive"
    scene = bench.Scene(params, artwork, camera)
    uv, visible = scene.frame_cells()
    s = scene.truth_skin_mm(uv[visible])
    pixels, _ = scene.project(s)
    assert np.nanmax(np.linalg.norm(scene.backproject(pixels)-s, axis=1)) < .05


def test_cli_plans_the_bench_with_pinned_dependencies():
    result = tatbot("vision", "stencil", "bench", "--dry-run", "--seed", "1", "--set", "frame_mm=10")
    assert result.returncode == 0, result.stderr
    assert "scripts/vision/stencil_bench.py" in result.stdout
    assert "--set frame_mm=10" in result.stdout and "opencv-python-headless" in result.stdout
    refused = tatbot("vision", "stencil", "bench", "--dry-run", "--artwork", "x", "--marked")
    assert refused.returncode != 0


def test_bench_run_writes_a_scorecard_and_contact_sheets(tmp_path, monkeypatch):
    import stencil_bench
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path/"logs"))
    monkeypatch.setenv("TATBOT_RUN_CONSOLE", "shell")
    assert stencil_bench.main(["--artwork", str(ARTWORK/"tatbot-43"), "--scenes", "4", "--workers", "1",
                               "--camera", "wrist", "--degradation", "clean",
                               "--calibration", str(tmp_path/"absent.json")]) == 0
    run = next((tmp_path/"logs/stencil-bench").glob("*T*Z-*"))
    card = json.loads((run/"scorecard.json").read_text())
    assert card["overall"]["scenes"] == 4 and card["overall"]["false_accepts"] == 0
    assert card["overall"]["success_rate"] == 1.0
    assert set(card["curves"]) >= {"washoff", "radius_mm", "color", "camera"}
    assert (run/"worst.jpg").stat().st_size > 0 and len((run/"scenes.jsonl").read_text().splitlines()) == 4
