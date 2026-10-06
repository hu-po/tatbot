"""The registration's chain without the stack or a camera: sightings synthesised with the tags off their measured
seat and the joints off by known offsets give both back within their sigma where a rigid fit cannot; held out, the
model that made the data beats the rigid one, by run and by hold; the seated layout is the seat on the measured one,
in the form the layout exporter adopts; unreached, tagless and moving holds are not sightings."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tatbot_calib import chain, register  # noqa: E402

K = np.array([[322.2, 0.0, 323.4], [0.0, 321.7, 181.4], [0.0, 0.0, 1.0]])
DIST = np.array([-0.0527, 0.057, -0.0008, -0.0001, -0.0179])
TRUE_DQ = np.array([0.0, 0.020, -0.015, 0.025, 0.010, 0.0, 0.0])   # rad, joint_1..joint_4
TRUE_SEAT = (0.012, -0.030, 0.010, -0.006, 0.007, 0.002)             # rotvec (rad), translation (m)


def _chain():
    from tatbot_description import robot_description
    from tatbot_motion import Kinematics

    repo = register.Path(__file__).resolve().parents[3]
    register._lib(repo)
    from fiducials import load_inventory

    kin = Kinematics(robot_description(None, arms=("right",)), "right")
    return chain.Chain(kin, "right", load_inventory(repo / "config" / "fiducials.json").target("wrist"))


def _camera(rotvec=(2.6, 0.3, -0.2), at=(0.05, 0.10, 0.75)):
    camera = np.eye(4)
    camera[:3, :3] = cv2.Rodrigues(np.array(rotvec))[0]
    camera[:3, 3] = at
    return camera


def _run(c, name, camera, rng, holds=20, noise=0.3):
    """Sightings at random joints, projected where the tags stand on the true seat with the joints at q + dq (what
    the camera sees), detected with `noise` px; the run's adopted camera is the truth nudged, as a rigid fit leaves
    it."""
    seat = chain._pose(TRUE_SEAT)
    sightings = []
    for hold in range(holds):
        q = np.zeros(7)
        q[:6] = rng.uniform(-0.5, 0.5, 6)
        for tag in c.target.ids:
            s = {"hold": hold, "q": q, "tag": tag}
            pixels = register._project(camera, c.corners(s, TRUE_DQ, seat), K, DIST) + rng.normal(0.0, noise, (4, 2))
            sightings.append({**s, "pixels": pixels})
    start = camera.copy()
    start[:3, 3] += [0.004, -0.003, 0.002]
    return chain.Run(name, K, DIST, sightings, start)


def test_the_seat_and_the_offsets_come_back_within_their_sigma_where_a_rigid_fit_cannot():
    c, rng = _chain(), np.random.default_rng(3)
    run = _run(c, "a", _camera(), rng, holds=24)
    fitted = chain.fit(c, [run])
    for j in chain.FREE_JOINTS:
        name = c.kin.joint_names[j].split("/")[-1]
        assert abs(fitted["dq"][j] - TRUE_DQ[j]) < max(4 * fitted["sigma"][name], 1e-3), name
    seat, sd = chain._vector(fitted["seat"]), np.asarray(fitted["sigma"]["seat"])
    assert np.all(np.abs(seat - TRUE_SEAT) < np.maximum(4 * sd, [1e-3] * 3 + [5e-4] * 3))
    assert fitted["median_px"] < 0.5                                   # the detection noise, nothing left over
    assert chain.fit(c, [run], (), False)["median_px"] > 2 * fitted["median_px"]


def test_held_out_the_model_that_made_the_data_beats_the_rigid_one():
    c, rng = _chain(), np.random.default_rng(5)
    runs = [_run(c, "a", _camera(), rng, holds=10),
            _run(c, "b", _camera((2.5, 0.2, -0.3), (0.08, 0.05, 0.72)), rng, holds=10),
            _run(c, "c", _camera((2.7, 0.35, -0.1), (0.02, 0.12, 0.78)), rng, holds=10)]
    report, fits = chain.compare(c, runs)
    assert report["best"] == "seat+joints" and set(fits) == set(chain.MODELS)
    held = {name: row["held_out"] for name, row in report["models"].items()}
    assert held["rigid"]["by"] == "run" and set(held["rigid"]["each"]) == {"a", "b", "c"}
    assert held["seat+joints"]["median_px"] < 0.35 * held["rigid"]["median_px"]   # 0.30 with the seated layout
    row = report["models"]["seat+joints"]["fit"]
    assert set(row["dq_mrad"]) == {"joint_1", "joint_2", "joint_3", "joint_4"}
    assert row["seat"]["parent_frame"] == c.target.parent_frame


def test_one_run_is_held_out_by_hold():
    c, rng = _chain(), np.random.default_rng(11)
    run = _run(c, "a", _camera(), rng, holds=10)
    seated, rigid = chain.hold_out(c, [run], (), True), chain.hold_out(c, [run], (), False)
    assert seated["by"] == "hold" and seated["holds"] == 10
    assert seated["median_px"] < 0.6 * rigid["median_px"]


def test_the_seated_layout_is_the_seat_on_the_measured_one_in_the_exporters_form():
    c, rng = _chain(), np.random.default_rng(7)
    run = _run(c, "a", _camera(), rng, holds=12)
    fitted = chain.fit(c, [run])
    record = chain.layout_record(c, [run], fitted)
    assert record["link"] == c.target.parent_frame and record["mode"] == "corner_reprojection"
    for tag, measured in c.parent_from_tag.items():
        assert np.allclose(record["link_from_tag"][str(tag)], fitted["seat"] @ measured)
    assert record["observations"] == len(run.sightings)
    assert record["pose_observations_by_tag"] == {str(t): 12 for t in c.target.ids}
    # a 0.3 px corner at ~0.7 m and 322 px focal length is about 0.7 mm
    assert record["corner_px_median"] < 0.5 and 0.2 < record["residual_mm_median"] < 1.5


def test_unreached_tagless_and_moving_holds_are_not_sightings():
    corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
    rows = [{"hold": 0, "reached": True, "q": [0.0] * 7, "still_rad": 1e-4, "tags": {"2": corners}},
            {"hold": 1, "reached": False, "error": "not reached"},
            {"hold": 2, "reached": True, "q": [0.1] * 7, "still_rad": 1e-4, "tags": {}},
            {"hold": 3, "reached": True, "q": [0.2] * 7, "still_rad": 0.01, "tags": {"3": corners}}]
    assert [(s["hold"], s["tag"]) for s in chain.sightings_of(rows)] == [(0, 2)]
