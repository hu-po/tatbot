"""Canonical inventory, detector, ambiguity, and generated-artifact contracts."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from ee_fiducial import WristLayout  # noqa: E402
from export_wrist_tags import marker_asset_relpath, quality_gate, record_from_solve  # noqa: E402
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from fiducials.detector import DetectorConfig, FiducialDetector, tag_dictionary  # noqa: E402
from tag_scan import detection_candidates, resolve_duplicates  # noqa: E402


def test_repository_inventory_is_complete_and_explicit():
    inventory = load_inventory()
    assert inventory.family == "apriltag_36h11"
    assert inventory.target("board").family == "apriltag_16h5"
    assert inventory.target("wrist").family == "apriltag_36h11"
    assert inventory.target("palette").family == "apriltag_36h11"
    assert inventory.target("wrist").ids == (2, 3, 4)
    assert inventory.target("wrist_left").ids == (5, 30, 1)
    assert inventory.target("wrist").edge_m == pytest.approx(0.047)
    assert inventory.target("board").ids == (3, 4, 5, 6, 7, 8, 9, 10, 11)
    assert inventory.target("board").edge_m == pytest.approx(0.044)
    assert inventory.target("board").grid == ((3, 4, 5), (6, 7, 8), (9, 10, 11))
    assert inventory.target("board").calibration_root_id == 10
    assert inventory.target("board").max_calibration_regression_mm == 1.0
    assert inventory.exclusive_ids("board") == inventory.target("board").ids
    assert inventory.owners(8) == ("board",)
    assert inventory.owners(3, "apriltag_16h5") == ("board",)
    assert inventory.owners(6) == ("board",)
    assert inventory.owners(7) == ("board",)
    # the v11 palette's roof sticker: its black square triangulates to 46 mm
    assert inventory.target("palette").ids == (0,)
    assert inventory.target("palette").edge_m == pytest.approx(0.046)
    assert inventory.spare_ids == ()
    assert inventory.target("wrist").minimum_calibration_poses_per_id == 3
    assert inventory.target("wrist").max_calibration_parent_distance_mm == 225.0
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
    dictionary = tag_dictionary(inventory.target("wrist").family)
    marker = cv2.aruco.generateImageMarker(dictionary, inventory.target("wrist").ids[0], 240)
    image = np.full((320, 320), 255, np.uint8)
    image[40:280, 40:280] = marker
    frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detection = FiducialDetector.from_inventory(inventory).detect("synthetic", frame, 123)[0]
    assert detection.tag_id == inventory.target("wrist").ids[0]
    assert detection.timestamp_ns == 123
    assert detection.corners_px.shape == (4, 2)
    assert np.allclose(detection.corners_px, [[40, 40], [279, 40], [279, 279], [40, 279]], atol=2)
    assert np.allclose(
        tag_model_corners(2.0),
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
    )


def _marker_cells(family: str, tag_id: int) -> np.ndarray:
    """The tag's 8x8 cells (6x6 data inside a one-cell black border), 255 white."""
    draw = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker   # OpenCV < 4.7: drawMarker
    return draw(tag_dictionary(family), tag_id, 8)


def _area_sampled(cells: np.ndarray, image_from_cell: np.ndarray, size: int = 96, supersample: int = 8) -> np.ndarray:
    """The tag drawn through a homography (cell units -> pixels, pixel centres at integers) on a white margin, each
    pixel the mean over its area, then the optics' blur: the corners are known off the pixel grid."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    ys, xs = np.mgrid[0:size, 0:size].astype(float)
    cell_from_image = np.linalg.inv(image_from_cell)
    total = np.zeros((size, size))
    for dy in offsets:
        for dx in offsets:
            u, v, w = cell_from_image @ np.stack([xs.ravel() + dx, ys.ravel() + dy, np.ones(size * size)])
            u, v = (u / w).reshape(size, size), (v / w).reshape(size, size)
            inside = (u >= 0) & (u < 8) & (v >= 0) & (v < 8)
            white = cells[np.clip(v, 0, 7).astype(int), np.clip(u, 0, 7).astype(int)] > 127
            total += np.where(inside & ~white, 25.0, 235.0)
    return np.clip(np.round(cv2.GaussianBlur(total / supersample**2, (0, 0), 0.6)), 0, 255).astype(np.uint8)


def test_the_edge_fit_places_a_small_tags_corners_where_the_tag_is():
    """A 23 px tag turned and seen at a slant, as the overhead D555 sees the wrist and roof tags: the edge fit puts
    each corner within a quarter pixel of where the tag's black square ends, off the pixel grid; the refinement
    OpenCV's AprilTag path offers (+0.44 px off in x and y) is refused like any unknown one."""
    inventory = load_inventory()
    wrist = inventory.target("wrist")
    turn, side = np.radians(23.0), 23.0
    image_from_cell = np.array([[np.cos(turn), -np.sin(turn), 0.0], [np.sin(turn), np.cos(turn), 0.0],
                                [0.0, 0.0, 1.0]]) @ np.diag([side / 8.0, side / 8.0, 1.0])
    image_from_cell = np.array([[1.0, 0.0, 48.3], [0.0, 1.0, 47.6], [0.0, 0.0, 1.0]]) @ image_from_cell @ np.array(
        [[1.0, 0.0, -4.0], [0.0, 1.0, -4.0], [0.004, -0.006, 1.0]])            # centred, and a slant
    image = _area_sampled(_marker_cells(wrist.family, wrist.ids[1]), image_from_cell)
    truth = np.array([[0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 8.0, 1.0], [0.0, 8.0, 1.0]]) @ image_from_cell.T
    truth = truth[:, :2] / truth[:, 2:]
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges"),
                                               target="wrist")
    (detection,) = detector.detect("synthetic", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0)
    assert detection.tag_id == wrist.ids[1]
    assert np.abs(detection.corners_px - truth).max() < 0.25
    with pytest.raises(ValueError, match="unknown corner refinement"):
        FiducialDetector.from_inventory(inventory, DetectorConfig(refinement="apriltag"))


def test_the_edge_fit_finds_a_square_tags_edges_between_pixels():
    """A 24 px tag square to the pixel grid, its edges part-way into pixels (a downscaled print): the contour
    pass's whole-pixel corners move to the edges."""
    inventory = load_inventory()
    palette = inventory.target("palette")
    marker = _marker_cells(palette.family, palette.ids[0])
    big = np.full((1200, 1200), 255, np.uint8)
    big[400:800, 400:800] = cv2.resize(marker, (400, 400), interpolation=cv2.INTER_NEAREST)
    scale = 1.0 / 16.5
    small = cv2.resize(big, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = FiducialDetector.from_inventory(inventory, DetectorConfig(min_side_px=8.0, refinement="edges"),
                                               target="palette")
    (detection,) = detector.detect("synthetic", cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), 0)
    lo, hi = 400 * scale - 0.5, 800 * scale - 0.5      # the black square's outer edges, pixel centres at integers
    assert np.abs(detection.corners_px - [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]).max() < 0.1


def test_reused_ids_require_board_only_context(monkeypatch):
    import tag_scan
    monkeypatch.setattr(tag_scan, "PALETTE_IDS", {8})
    monkeypatch.setattr(tag_scan, "WRIST_IDS", {3, 6, 7})
    monkeypatch.setattr(tag_scan, "BOARD_UNIQUE_IDS", {4, 5, 9, 10, 11})
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
    solved["pose_observations_by_tag"][str(wrist.ids[-1])] = 2
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


def test_checked_in_layout_matches_inventory_and_enforces_status(tmp_path):
    path = ROOT / "config" / "wrist_tags_measured.json"
    raw = json.loads(path.read_text())
    inventory = load_inventory()
    assert tuple(raw["target_ids"]) == inventory.target("wrist").ids
    if raw["calibration_status"] == "calibrated":
        layout = WristLayout.load(path)
        assert layout.parent_frame == "right/gripper_left"
    else:
        with pytest.raises(ValueError, match="not calibrated"):
            WristLayout.load(path)

    raw["calibration_status"] = "pending_recalibration"
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pending_recalibration"):
        WristLayout.load(pending)


def test_foamcore_mount_envelope_accepts_offset_but_still_rejects_far_tag(tmp_path):
    inventory = load_inventory()
    wrist = inventory.target("wrist")
    solved = {
        "link": wrist.parent_frame,
        "link_from_tag": {str(tag_id): np.eye(4).tolist() for tag_id in wrist.ids},
        "observations": 189,
        "pose_observations_by_tag": {str(tag_id): 19 for tag_id in wrist.ids},
        "corner_px_median": 8.91,
        "residual_mm_median": 5.452,
    }
    solved["link_from_tag"][str(wrist.ids[1])][1][3] = 0.2073
    record_from_solve(solved, tmp_path / "robot_world.json", inventory)
    solved["link_from_tag"][str(wrist.ids[1])][1][3] = 0.225
    with pytest.raises(ValueError, match="implausibly far"):
        record_from_solve(solved, tmp_path / "robot_world.json", inventory)


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


def test_pending_layout_does_not_render_old_mount_tag_frames():
    from export_wrist_tags import render_urdf_block
    record = json.loads((ROOT / "config/wrist_tags_measured.json").read_text())
    record["calibration_status"] = "pending_recalibration"
    assert "<joint" not in render_urdf_block(record, "synthetic")
    record["calibration_status"] = "calibrated"
    record["tags"] = {str(i): {"ee_from_tag": np.eye(4).tolist()} for i in record["target_ids"]}
    block = render_urdf_block(record, "synthetic")
    assert 'right/wrist_tag8' not in block
    assert block.count('<joint') == 3
    assert block.count('<visual>') == 3
    assert 'meshes/tags/36h11_002_47mm/tag.glb' in block


@pytest.mark.parametrize("target_name", ["wrist", "wrist_left", "palette"])
def test_viewer_assets_encode_the_canonical_fiducial_ids(target_name):
    inventory = load_inventory()
    target = inventory.target(target_name)
    detector = FiducialDetector.from_inventory(inventory, target=target_name)
    for tag_id in target.ids:
        asset = ROOT / "urdf" / marker_asset_relpath(target.family, tag_id, target.edge_m)
        assert asset.is_file()
        blob = asset.read_bytes()
        magic, version, total = struct.unpack_from("<III", blob)
        assert (magic, version, total) == (0x46546C67, 2, len(blob))
        json_length, json_type = struct.unpack_from("<II", blob, 12)
        assert json_type == 0x4E4F534A
        gltf = json.loads(blob[20:20 + json_length])
        assert gltf["images"] == [{"bufferView": 3, "mimeType": "image/png"}]
        image = cv2.imread(str(asset.with_name("tag.png")))
        assert image is not None
        found = detector.detect("viewer-asset", image, 1)
        assert [d.tag_id for d in found] == [tag_id]
        # The quad is the detector's tag frame (TAG_CORNER_SIGNS: TL/TR/BR/BL
        # of the printed pattern at (-,+)/(+,+)/(+,-)/(-,-)), and glTF's
        # texture origin is the image's top-left, so the vertex at sign
        # (sx, sy) must carry uv ((sx+1)/2, (1-sy)/2): image top row at +y.
        # The wrong pairing renders every fiducial mirrored about its x axis.
        binary_offset = 20 + json_length + 8
        positions = struct.unpack_from("<12f", blob, binary_offset)
        uvs = struct.unpack_from("<8f", blob, binary_offset + 48)
        for vertex in range(4):
            sx, sy = np.sign(positions[3 * vertex]), np.sign(positions[3 * vertex + 1])
            assert (sx, sy) in {tuple(sign) for sign in np.sign(tag_model_corners(1.0)[:, :2])}
            assert uvs[2 * vertex:2 * vertex + 2] == pytest.approx([(sx + 1) / 2, (1 - sy) / 2])


def test_unmeasured_palette_does_not_render_old_sticker():
    import xml.etree.ElementTree as ET
    urdf = ET.parse(ROOT / "urdf/palette.urdf").getroot()
    assert urdf.find("link[@name='palette_tag']/visual") is None


def test_palette_viewer_asset_embeds_printed_mount_material():
    geometry = json.loads((ROOT / "config/palette_geometry.json").read_text())
    blob = (ROOT / geometry["viewer_mesh"]).read_bytes()
    json_length, json_type = struct.unpack_from("<II", blob, 12)
    assert json_type == 0x4E4F534A
    gltf = json.loads(blob[20:20 + json_length])
    assert gltf["materials"] == [{
        "doubleSided": True,
        "name": "Tatbot printed mount black",
        "pbrMetallicRoughness": {
            "baseColorFactor": pytest.approx([31 / 255, 31 / 255, 33 / 255, 1.0]),
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        },
    }]
    assert all(
        primitive["material"] == 0
        for mesh in gltf["meshes"]
        for primitive in mesh["primitives"]
    )


@pytest.mark.parametrize("family,tag_id", [("apriltag_16h5", 3), ("apriltag_36h11", 5)])
@pytest.mark.parametrize("turns", range(4))
def test_both_families_keep_corner_order_under_rotation(family, tag_id, turns):
    marker = cv2.aruco.generateImageMarker(tag_dictionary(family), tag_id, 240)
    frame = np.full((320, 320, 3), 255, np.uint8)
    frame[40:280, 40:280] = np.rot90(marker, turns)[:, :, None]
    found = FiducialDetector.from_inventory(load_inventory()).detect("synthetic", frame, 123)
    assert [d.tag_id for d in found] == [tag_id]
    expected = np.roll([[40, 40], [279, 40], [279, 279], [40, 279]], turns, axis=0)
    assert np.allclose(found[0].corners_px, expected, atol=2)


def test_mixed_scene_filters_wrong_family_and_separates_the_two_wrist_targets():
    frame = cv2.imread(str(ROOT / 'scripts/tests/fixtures/fiducials/mixed.png'))
    found = FiducialDetector.from_inventory(load_inventory()).detect('synthetic', frame, 1)
    assert sorted((d.family, d.tag_id) for d in found) == [("apriltag_16h5", 3), ("apriltag_36h11", 3), ("apriltag_36h11", 5)]
    board = FiducialDetector.from_inventory(load_inventory(), target='board').detect('synthetic', frame, 1)
    assert [d.tag_id for d in board] == [3]
    # Each physical arm's triplet is its own target: a pink sighting never
    # counts for blue and vice versa.
    pink = FiducialDetector.from_inventory(load_inventory(), target='wrist').detect('synthetic', frame, 1)
    blue = FiducialDetector.from_inventory(load_inventory(), target='wrist_left').detect('synthetic', frame, 1)
    assert [d.tag_id for d in pink] == [3]
    assert [d.tag_id for d in blue] == [5]


def test_reused_numeric_ids_are_retained_only_under_their_actual_family():
    from tag_scan import detect, family_groups
    inventory = load_inventory()
    frame = cv2.imread(str(ROOT / 'scripts/tests/fixtures/fiducials/mixed.png'))
    detector = FiducialDetector.from_inventory(inventory)
    detector.keep_best_per_id = True
    records = detector.detect('synthetic', frame, 1)
    assert len(records) == 3  # ID 3 in both families survives deduplication.
    families = family_groups(records)
    board = families[inventory.target('board').family]
    blue = families[inventory.target('wrist_left').family]
    assert board['corners']['3'][0][0] == pytest.approx(40, abs=2)
    assert blue['corners']['3'][0][0] == pytest.approx(360, abs=2)
    assert detect(frame) == []  # Every seen ID is cross-family ambiguous.


def test_palette_is_retained_when_wrist_tags_are_far_from_it():
    from types import SimpleNamespace

    from tag_scan import family_groups
    records = [SimpleNamespace(family='apriltag_36h11', tag_id=tag_id,
                               corners_px=np.array([[x, 0], [x + 50, 0], [x + 50, 50], [x, 50]], float))
               for tag_id, x in [(3, 0), (5, 100), (22, 1000)]]
    assert set(family_groups(records)['apriltag_36h11']['corners']) == {'3', '5', '22'}


@pytest.mark.parametrize('mutation,match', [
    ('missing', 'explicitly'), ('unsupported', 'supported'),
    ('overlap', 'ambiguity_group'), ('capacity', 'capacity')])
def test_mixed_inventory_fails_closed(tmp_path, mutation, match):
    record = json.loads((ROOT / 'config/fiducials.json').read_text())
    wrist = record['targets']['wrist']
    if mutation == 'missing':
        del wrist['family']
    elif mutation == 'unsupported':
        wrist['family'] = 'unknown'
    elif mutation == 'overlap':
        wrist['ids'][0] = 5
        del wrist['ambiguity_group']
    else:
        wrist['ids'][0] = 587
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        load_inventory(path)


def test_legacy_schema_keeps_single_family(tmp_path):
    record = json.loads((ROOT / 'config/fiducials.json').read_text())
    record['schema_version'] = 1
    record['family'] = 'apriltag_16h5'
    record['targets'] = {'board': record['targets']['board']}
    del record['targets']['board']['family']
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(record))
    assert load_inventory(path).target('board').family == 'apriltag_16h5'


@pytest.mark.parametrize('target_name', ['wrist', 'wrist_left', 'palette'])
def test_generated_sheet_encodes_inventory_and_metric_print_size(target_name):
    from PIL import Image
    inventory = load_inventory()
    target = inventory.target(target_name)
    path = ROOT / 'docs' / f'{target_name}-tags-{target.family.removeprefix("apriltag_")}.png'
    with Image.open(path) as sheet:
        assert sheet.info['dpi'] == pytest.approx((300, 300), abs=0.01)
    frame = cv2.imread(str(path))
    found = FiducialDetector.from_inventory(inventory, target=target_name).detect('print', frame, 0)
    assert sorted(d.tag_id for d in found) == sorted(target.ids)
    for detection in found:
        assert detection.side_px / 300 * 25.4 == pytest.approx(target.edge_m * 1000, abs=0.1)


def test_refresh_restamps_a_calibrated_layout_only_when_its_geometry_still_matches():
    """An inventory edit that leaves ids, edge and parent alone (a plausibility
    bound, a note) re-stamps the hash on refresh and keeps the provenance; one
    that changes the geometry the hash guards still refuses to relabel."""
    from export_wrist_tags import normalize_existing
    raw = json.loads((ROOT / 'config/wrist_tags_measured.json').read_text())
    raw['calibration_status'] = 'calibrated'
    raw['tags'] = {str(i): {'ee_from_tag': np.eye(4).tolist()} for i in raw['target_ids']}
    raw['inventory_hash'] = 'previous-inventory'
    raw['source_metrics'] = {'observations': 7}
    refreshed = normalize_existing(raw, load_inventory())
    assert refreshed['inventory_hash'] == load_inventory().inventory_hash
    assert refreshed['source_metrics'] == {'observations': 7}
    relabelled = dict(raw, parent_frame='right/ee_gripper_link')
    with pytest.raises(ValueError, match='different parent frame'):
        normalize_existing(relabelled, load_inventory())
    wrong_ids = dict(raw, target_ids=[40, 41, 46])
    with pytest.raises(ValueError, match='ids must be'):
        normalize_existing(wrong_ids, load_inventory())
