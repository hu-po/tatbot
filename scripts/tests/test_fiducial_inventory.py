"""Canonical inventory, detector, ambiguity, and generated-artifact contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vision"))

from ee_fiducial import WristLayout  # noqa: E402
from export_wrist_tags import quality_gate, record_from_solve  # noqa: E402
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from fiducials.detector import FiducialDetector  # noqa: E402
from tag_scan import detection_candidates, resolve_duplicates  # noqa: E402


def test_repository_inventory_is_complete_and_explicit():
    inventory = load_inventory()
    assert inventory.family == "apriltag_16h5"
    assert inventory.target("wrist").ids == (3, 6, 7, 8)
    assert inventory.target("wrist").edge_m == pytest.approx(0.056)
    assert inventory.target("board").ids == (3, 4, 5, 6, 7, 8, 9, 10, 11)
    assert inventory.target("board").edge_m == pytest.approx(0.044)
    assert inventory.target("board").grid == ((3, 4, 5), (6, 7, 8), (9, 10, 11))
    assert inventory.target("board").calibration_root_id == 10
    assert inventory.target("board").max_calibration_regression_mm == 1.0
    assert inventory.exclusive_ids("board") == (4, 5, 9, 10, 11)
    assert inventory.owners(8) == ("wrist", "board", "palette")
    assert inventory.owners(3) == ("wrist", "board")
    assert inventory.owners(6) == ("wrist", "board")
    assert inventory.owners(7) == ("wrist", "board")
    assert inventory.target("palette").ids == (8,)
    assert inventory.target("palette").edge_m == pytest.approx(0.041)
    assert inventory.spare_ids == ()
    assert inventory.target("wrist").minimum_calibration_poses_per_id == 3
    assert inventory.target("wrist").max_calibration_parent_distance_mm == 200.0
    assert inventory.target("wrist").parent_frame == "right/gripper_left"
    groups = {inventory.target(name).ambiguity_group for name in ("wrist", "board", "palette")}
    assert groups == {"phase_separated_calibration_ids"}


def test_undeclared_cross_target_duplicate_is_rejected(tmp_path):
    path = tmp_path / "fiducials.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "family": "apriltag_16h5",
        "targets": {
            "one": {"role": "one", "ids": [8], "edge_m": 0.056},
            "two": {"role": "two", "ids": [8], "edge_m": 0.041},
        },
    }))
    with pytest.raises(ValueError, match="ambiguity_group"):
        load_inventory(path)


def test_required_detector_profiles_are_fail_closed(tmp_path):
    record = json.loads((ROOT / "config" / "fiducials.json").read_text())
    del record["detector"]["live"]
    path = tmp_path / "missing-live.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="missing detector profiles.*live"):
        load_inventory(path)


def test_shared_detector_recovers_configured_tag_and_corner_contract():
    inventory = load_inventory()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
    marker = cv2.aruco.generateImageMarker(dictionary, inventory.target("wrist").ids[0], 240)
    image = np.full((320, 320), 255, np.uint8)
    image[40:280, 40:280] = marker
    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detection = FiducialDetector(inventory.known_ids).detect("synthetic", frame, 123)[0]
    assert detection.tag_id == inventory.target("wrist").ids[0]
    assert detection.timestamp_ns == 123
    assert detection.corners_px.shape == (4, 2)
    assert np.allclose(detection.corners_px, [[40, 40], [279, 40], [279, 279], [40, 279]], atol=2)
    assert np.allclose(
        tag_model_corners(2.0),
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
    )


def test_reused_ids_require_board_only_context():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float64)
    board8 = square + [100, 100]
    palette8 = square + [900, 600]
    board, palette = resolve_duplicates([(8, board8), (8, palette8)])
    assert board == {}
    assert palette is None

    siblings = [(4, square + [80, 100]), (5, square + [120, 100]),
                (9, square + [100, 80])]
    board, palette = resolve_duplicates([*siblings, (8, board8), (8, palette8)])
    assert np.array_equal(board[8], board8)
    assert np.array_equal(palette, palette8)

    board3 = square + [90, 100]
    wrist3 = square + [850, 550]
    board, palette = resolve_duplicates([*siblings, (3, board3), (3, wrist3)])
    assert np.array_equal(board[3], board3)
    assert palette is None

    candidates = detection_candidates([(8, board8), (8, palette8), (3, board3)])
    assert len(candidates[8]) == 2
    assert len(candidates[3]) == 1


def test_wrist_publish_requires_distinct_poses_for_every_id():
    wrist = load_inventory().target("wrist")
    solved = {
        "link_from_tag": {str(tag_id): np.eye(4).tolist() for tag_id in wrist.ids},
        "observations": 20,
        "pose_observations_by_tag": {str(tag_id): 3 for tag_id in wrist.ids},
        "corner_px_median": 1.0,
        "residual_mm_median": 1.0,
    }
    quality_gate(solved, wrist)
    solved["pose_observations_by_tag"]["7"] = 2
    with pytest.raises(ValueError, match="distinct arm poses per id"):
        quality_gate(solved, wrist)


def test_wrist_publish_requires_solve_in_configured_parent(tmp_path):
    inventory = load_inventory()
    wrist = inventory.target("wrist")
    solved = {
        "link": "right/realsense_link",
        "link_from_tag": {str(tag_id): np.eye(4).tolist() for tag_id in wrist.ids},
        "observations": 20,
        "pose_observations_by_tag": {str(tag_id): 3 for tag_id in wrist.ids},
        "corner_px_median": 1.0,
        "residual_mm_median": 1.0,
    }
    with pytest.raises(ValueError, match="must equal configured parent_frame"):
        record_from_solve(solved, tmp_path / "robot_world.json", inventory)

    solved["link"] = wrist.parent_frame
    record = record_from_solve(solved, tmp_path / "robot_world.json", inventory)
    assert record["parent_frame"] == "right/gripper_left"


def test_checked_in_layout_is_calibrated_and_matches_inventory(tmp_path):
    path = ROOT / "config" / "wrist_tags_measured.json"
    raw = json.loads(path.read_text())
    inventory = load_inventory()
    assert tuple(raw["target_ids"]) == inventory.target("wrist").ids
    layout = WristLayout.load(path)
    assert layout.parent_frame == "right/gripper_left"

    raw["calibration_status"] = "pending_recalibration"
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pending_recalibration"):
        WristLayout.load(pending)


def test_wrist_layout_schema_and_parent_are_fail_closed(tmp_path):
    source = ROOT / "config" / "wrist_tags_measured.json"
    record = json.loads(source.read_text())
    record["calibration_status"] = "calibrated"

    record["schema_version"] = 1
    path = tmp_path / "wrong-schema.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unsupported wrist layout schema"):
        WristLayout.load(path)

    record["schema_version"] = 2
    record["parent_frame"] = "right/realsense_link"
    path = tmp_path / "wrong-parent.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="parent must be right/gripper_left"):
        WristLayout.load(path)

    record["parent_frame"] = "right/gripper_left"
    record["target_ids"] = [3, 6, 7, 8, 8]
    path = tmp_path / "duplicate-id.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="wrist target ids must be"):
        WristLayout.load(path)
