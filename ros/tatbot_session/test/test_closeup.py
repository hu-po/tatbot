"""The close-up ink inspection's pure parts (tatbot_session.closeup): the gauge-anchored camera pose, and each
stroke's ink share on a page image, at the plan and at the shift that finds the most."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tatbot_session import closeup, gauge
from tatbot_session import inspect as ins


def test_the_anchored_pose_takes_its_tilt_from_the_paper_and_its_place_from_the_tip():
    true = np.eye(4)
    true[:3, :3] = Rotation.from_euler("xyz", [2.4, 0.3, -0.5]).as_matrix()
    true[:3, 3] = [0.30, 0.05, 0.16]
    page_up = np.array([0.0, 0.0, 1.0])
    contact = np.array([-0.018, 0.007, 0.16])            # the gauge's tip in the camera frame
    tcp = (true @ [*contact, 1.0])[:3]                   # where the arm puts that tip
    cad = true.copy()
    cad[:3, :3] = Rotation.from_rotvec([0.02, -0.015, 0.0]).as_matrix() @ true[:3, :3]
    cad[:3, 3] += [0.004, -0.003, 0.008]                 # the CAD mount: ~1.4 deg and ~9 mm off
    n_cam = true[:3, :3].T @ page_up                     # the paper's normal as the frame measures it
    frame = gauge.Frame(np.zeros((1, 3)), np.zeros(3), n_cam, 0.0)
    pose = closeup.anchored(cad, tcp, frame, {"tip_cam_m": contact.tolist(), "offset_m": 0.0}, page_up)
    assert np.allclose(pose[:3, :3] @ n_cam, page_up, atol=1e-9)        # the paper's normal is the page's
    assert np.allclose((pose @ [*contact, 1.0])[:3], tcp, atol=1e-9)    # the tip lands on the arm's tip
    # what remains is a turn about the page's normal, which the paper cannot show: a page point 20 mm from the
    # tip lands within a few tenths of a millimetre of where the true camera sees it
    point = tcp + [0.02, 0.0, -0.012]
    assert np.linalg.norm(np.linalg.inv(pose) @ [*point, 1.0] - np.linalg.inv(true) @ [*point, 1.0]) < 5e-4


@pytest.fixture
def page():
    extent, ppm = (-0.02, -0.02, 0.02, 0.02), 10.0
    gx, _ = ins.page_grid(extent, ppm)
    image = np.full((*gx.shape, 3), (225, 228, 232), np.uint8)          # paper, a soft shadow over half of it
    image[:, : gx.shape[1] // 2] = (175, 178, 182)
    return extent, ppm, image


def test_a_stroke_with_ink_on_it_counts_and_one_without_does_not(page):
    extent, ppm, image = page
    program = {"ops": [{"op": "stroke", "id": "a", "resource_id": "r", "points_m": [[-0.01, 0.005], [0.01, 0.005]]},
                       {"op": "stroke", "id": "b", "resource_id": "q", "points_m": [[-0.01, -0.008], [0.01, -0.008]]}]}
    ink = np.round(ins.to_image_px([[-0.01, 0.005], [0.01, 0.005]], extent, ppm)).astype(np.int32)
    cv2.polylines(image, [ink], False, (210, 150, 60), 5)                # a 0.5 mm sky-blue line along stroke a
    valid = np.ones(image.shape[:2], bool)
    result = closeup.coverage(program, closeup.ink_score(image, valid, ppm), valid, extent, ppm, search_m=0.002)
    shares = {s["id"]: s["ink"] for s in result["strokes"]}
    assert shares["a"] > 0.9 and shares["b"] < 0.05
    assert result["resources"]["r"]["ink"] > 0.9 and result["best_shift_m"] == [0.0, 0.0]


def test_ink_drawn_off_its_plan_reads_at_its_shift(page):
    extent, ppm, image = page
    program = {"ops": [{"op": "stroke", "id": "a", "resource_id": "r", "points_m": [[-0.01, 0.0], [0.01, 0.0]]}]}
    drawn = np.round(ins.to_image_px([[-0.01, 0.004], [0.01, 0.004]], extent, ppm)).astype(np.int32)
    cv2.polylines(image, [drawn], False, (40, 40, 40), 5)                 # drawn 4 mm off the plan
    valid = np.ones(image.shape[:2], bool)
    result = closeup.coverage(program, closeup.ink_score(image, valid, ppm), valid, extent, ppm, search_m=0.006)
    assert result["strokes"][0]["ink"] < 0.05
    assert result["best_shift_m"][1] == pytest.approx(0.004, abs=0.0011)
    assert result["at_best_shift"]["strokes"][0]["ink"] > 0.9


def test_a_hold_rectifies_on_the_paper_its_own_depth_measures():
    page = np.eye(4)
    page[:3, 3] = [0.29, 0.0, 0.003]                     # the draw's page, 2 mm under the paper now
    cam = np.eye(4)
    cam[:3, 3] = [0.30, 0.01, 0.17]
    frame = gauge.Frame(np.zeros((1, 3)), np.array([0.01, -0.01, -0.165]), np.array([0.0, 0.0, -1.0]), 0.0)
    here = closeup.on_paper(page, cam, frame)
    assert here[2, 3] == pytest.approx(0.005) and np.allclose(here[:2, 3], page[:2, 3])
    assert np.allclose(here[:3, :3], page[:3, :3])
