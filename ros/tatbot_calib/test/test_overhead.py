"""The overhead D555 through the wrist D405 (tatbot_calib.overhead): a known registration recovered from simulated
targets and depth through the D555's own optics, from a start turned like the cube's (2.5 deg, 6 mm); a bad tile read
left out; the wrist's views placing a tag and a tile's corners; the record and its installation."""
from __future__ import annotations

import json

import numpy as np
import pytest
from tatbot_calib import overhead, register

K = np.array([[645.7, 0.0, 647.3], [0.0, 644.8, 363.2], [0.0, 0.0, 1.0]])   # the D555's colour, 2026-10-18
DIST = np.array([-0.0527, 0.0570, -0.0008, -0.0001, -0.0179])
TRUE = np.array([[-0.0764, -0.8133, 0.5769, -0.0044], [-0.9701, -0.0731, -0.2314, 0.2606],
                 [0.2303, -0.5773, -0.7834, 0.6564], [0.0, 0.0, 0.0, 1.0]])   # the installed one, as the truth here
START = TRUE @ overhead.pose(np.r_[np.radians([1.5, -1.2, 1.5]), 0.004, -0.003, 0.003])
WRIST_K = np.array([[651.1, 0.0, 650.4], [0.0, 650.3, 358.1], [0.0, 0.0, 1.0]])
WRIST_DIST = np.zeros(5)
SQUARE = np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]])


def corners(centre, half):
    return np.asarray(centre, float) + half * SQUARE


def scene_and_clouds(world_from_base, rng):
    """The table (z 0.03), the palette's block at the station (to 0.08) and the holder's (to 0.05): the D555's points
    of it and the wrist's, by region."""
    def box(x0, x1, y0, y1, top, step=0.003):
        x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
        tops = np.c_[x.ravel(), y.ravel(), np.full(x.size, top)]
        return tops, np.tile([0.0, 0.0, 1.0], (len(tops), 1))

    parts = [box(0.15, 0.45, 0.0, 0.40, 0.03), box(0.26, 0.34, 0.26, 0.36, 0.08), box(0.25, 0.35, 0.05, 0.15, 0.05)]
    points = np.concatenate([p for p, _ in parts])
    normals = np.concatenate([n for _, n in parts])
    # the boxes' walls, so the depth sees across as well as up
    for x0, x1, y0, y1, top in ((0.26, 0.34, 0.26, 0.36, 0.08), (0.25, 0.35, 0.05, 0.15, 0.05)):
        z = np.linspace(0.031, top - 0.001, 12)
        for x in (x0, x1):
            y = np.linspace(y0, y1, 30)
            yy, zz = np.meshgrid(y, z)
            points = np.r_[points, np.c_[np.full(yy.size, x), yy.ravel(), zz.ravel()]]
            normals = np.r_[normals, np.tile([np.sign(x - (x0 + x1) / 2), 0.0, 0.0], (yy.size, 1))]
        for y in (y0, y1):
            x = np.linspace(x0, x1, 30)
            xx, zz = np.meshgrid(x, z)
            points = np.r_[points, np.c_[xx.ravel(), np.full(xx.size, y), zz.ravel()]]
            normals = np.r_[normals, np.tile([0.0, np.sign(y - (y0 + y1) / 2), 0.0], (xx.size, 1))]
    seen = points @ world_from_base[:3, :3].T + world_from_base[:3, 3]
    scene = overhead.Scene(seen, normals @ world_from_base[:3, :3].T, np.ones(len(seen), bool))
    near = {"station": np.linalg.norm(points[:, :2] - [0.30, 0.31], axis=1) < 0.08,
            "holder": np.linalg.norm(points[:, :2] - [0.30, 0.10], axis=1) < 0.08}
    clouds = [(region, points[rng.choice(np.flatnonzero(mask), 1500, replace=False)]
               + rng.normal(0.0, 0.0005, (1500, 3))) for region, mask in near.items()]
    return scene, clouds


def targets(world_from_base, rng, noise_px=0.3):
    """The roof tag at the station, 4 x 3 tile markers on the holder: their corners where the D555 sees them."""
    out = [overhead.Target("station", "roof tag", corners([0.30, 0.31, 0.081], 0.023), None)]
    for i, (x, y) in enumerate((x, y) for x in (0.27, 0.29, 0.31, 0.33) for y in (0.07, 0.10, 0.13)):
        out.append(overhead.Target("holder", f"tile {i}", corners([x, y, 0.051], 0.009), None, 0.8))
    for t in out:
        seen = t.base @ world_from_base[:3, :3].T + world_from_base[:3, 3]
        t.pixels = overhead.project(seen, K, DIST) + rng.normal(0.0, noise_px, (4, 2))
    return out


def turn_and_shift(a, b):
    return overhead.turn_deg(a, b), float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def test_a_registration_is_recovered_from_both_regions_tags_and_depth():
    rng = np.random.default_rng(0)
    scene, clouds = scene_and_clouds(TRUE, rng)
    fitted = overhead.fit(START, targets(TRUE, rng), clouds, scene, K, DIST)
    turn, shift = turn_and_shift(fitted, TRUE)
    assert turn < 0.05 and shift < 0.0005, (turn, shift)
    assert turn_and_shift(START, TRUE)[0] > 2.0


def test_a_bad_tile_read_is_left_out_and_each_region_is_predicted_by_the_other():
    rng = np.random.default_rng(1)
    scene, clouds = scene_and_clouds(TRUE, rng)
    ts = targets(TRUE, rng)
    ts[3].pixels = ts[3].pixels + [9.0, -6.0]
    fitted, report = overhead.solve(START, ts, clouds, scene, K, DIST)
    assert report["dropped"] == ["tile 2"]
    assert turn_and_shift(fitted, TRUE)[0] < 0.05
    rows = report["variants"]
    assert set(rows) == {"installed", "fitted", "without the station", "without the holder", "tags alone", "depth alone"}
    assert rows["fitted"]["station"]["tags_m"] < 0.0005 and rows["fitted"]["holder"]["tags_m"] < 0.0005
    assert all(rows["installed"][r]["tags_m"] > 3 * rows["fitted"][r]["tags_m"] for r in ("station", "holder"))
    assert abs(rows["fitted"]["holder"]["depth_median_m"]) < 0.0005


def test_a_holder_no_d555_read_places_is_fitted_by_the_wrists_depth_alone():
    """The forearm tile's 12 mm markers read 0 of 3 in the D555 (2026-10-22): the roof tag and both regions' depth
    still recover the registration the cube's start is 2.5 deg off."""
    rng = np.random.default_rng(2)
    scene, clouds = scene_and_clouds(TRUE, rng)
    roof = [t for t in targets(TRUE, rng) if t.region == "station"]
    assert overhead.unmeasured(roof, clouds[:1] + [clouds[1]] * 3) is None
    fitted, report = overhead.solve(START, roof, clouds, scene, K, DIST)
    turn, shift = turn_and_shift(fitted, TRUE)
    assert turn < 0.05 and shift < 0.0005 and report["holder_by"] == "depth alone", (turn, shift)


def test_stencilds_page_places_the_wrists_tile_corners_where_the_d555_sees_them():
    """The forearm tile's corners in the base (the wrist's) against stencild's page in the world: under the true
    registration each lands where the D555 sees it; the page counts as one rigid observation, never left out."""
    from types import SimpleNamespace

    tile = SimpleNamespace(corners={i: np.array([[x - 0.006, y + 0.006], [x + 0.006, y + 0.006], [x + 0.006, y - 0.006],
                                                  [x - 0.006, y - 0.006]])
                                    for i, (x, y) in enumerate((x, y) for x in (-0.02, 0.0, 0.02) for y in (-0.03, 0.03))})
    base_from_page = overhead.pose(np.r_[0.0, 0.0, 0.4, 0.30, 0.10, 0.05])
    corners = {i: (base_from_page @ np.c_[c, np.zeros(4), np.ones(4)].T).T[:, :3] for i, c in tile.corners.items()}
    page = overhead.page_targets(tile, corners, TRUE @ base_from_page, 0.002, K, DIST)
    assert [t.name for t in page] == [f"page {i}" for i in range(6)]
    assert max(overhead.target_miss_m(TRUE, t, K, DIST) for t in page) < 1e-6
    assert min(overhead.target_miss_m(START, t, K, DIST) for t in page) > 0.002
    rng = np.random.default_rng(3)
    scene, clouds = scene_and_clouds(TRUE, rng)
    roof = [t for t in targets(TRUE, rng) if t.region == "station"]
    off = overhead.page_targets(tile, corners, TRUE @ overhead.pose(np.r_[0, 0, 0, 0.003, 0, 0]) @ base_from_page, 0.002,
                                K, DIST)   # stencild 3 mm off: past OUTLIER_M, yet kept whole
    fitted, report = overhead.solve(START, roof + off, clouds, scene, K, DIST)
    assert report["dropped"] == [] and report["holder_by"] == "page"
    assert turn_and_shift(fitted, TRUE)[0] < 0.2
    off[4].base = off[4].base + [0.008, 0.0, 0.0175]   # a wrist read off the print (2026-10-23's marker 23)
    fitted, report = overhead.solve(START, roof + off, clouds, scene, K, DIST)
    assert report["dropped"] == ["page 4"] and turn_and_shift(fitted, TRUE)[0] < 0.2


@pytest.mark.parametrize("names, holder_views, why", [
    (["roof tag", "tile 0"], 0, None),
    (["roof tag"], 3, None),
    (["roof tag"], 2, "the holder has no tile marker"),
    (["tile 0"], 9, "no roof tag"),
])
def test_a_region_is_measured_by_its_shared_targets_or_the_holder_by_its_depth(names, holder_views, why):
    ts = [overhead.Target("holder" if n.startswith("tile") else "station", n, np.zeros((4, 3)), np.zeros((4, 2)))
          for n in names]
    clouds = [("station", np.zeros((10, 3)))] + [("holder", np.zeros((10, 3)))] * holder_views
    said = overhead.unmeasured(ts, clouds)
    assert said is None if why is None else why in said


@pytest.mark.parametrize("frame", ["world", "right/base_link"])
def test_the_bridges_page_is_placed_in_the_base(frame):
    from types import SimpleNamespace

    from tatbot_description.transforms import matrix_quat

    page_pose = overhead.pose(np.r_[0.0, 0.0, 0.4, 0.3, 0.1, 0.05])
    q = matrix_quat(page_pose)
    msg = SimpleNamespace(pose=SimpleNamespace(header=SimpleNamespace(frame_id=frame), pose=SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=page_pose[0, 3], y=page_pose[1, 3], z=page_pose[2, 3]),
                             orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))))   # Page's nesting
    want = page_pose if frame.endswith("base_link") else np.linalg.inv(TRUE) @ page_pose
    assert np.allclose(overhead.page_in_base(msg, TRUE), want, atol=1e-9)


def look_down(at, height, aside):
    """base <- camera over `at`, `aside` off it, looking straight down (optical z down, x along the base's x)."""
    out = np.eye(4)
    out[:3, :3] = np.diag([1.0, -1.0, -1.0])
    out[:3, 3] = np.asarray(at, float) + [aside[0], aside[1], height]
    return out


def test_the_wrists_views_place_the_roof_tag_and_the_tiles_corners():
    rng = np.random.default_rng(2)
    model = 0.023 * SQUARE
    tag = np.eye(4)
    tag[:3, 3] = [0.30, 0.31, 0.081]
    views = []
    for aside in ((0.0, 0.0), (0.04, 0.0), (0.0, -0.04), (-0.03, 0.03)):
        b_c = look_down(tag[:3, 3], 0.22, aside)
        cam = (np.linalg.inv(b_c) @ tag @ np.c_[model, np.ones(4)].T).T[:, :3]
        views.append((b_c, WRIST_K, WRIST_DIST, overhead.project(cam, WRIST_K, WRIST_DIST) + rng.normal(0, 0.3, (4, 2))))
    prior = tag @ overhead.pose([0.02, -0.01, 0.03, 0.004, -0.003, 0.005])
    placed, per_view = overhead.tag_in_base(views, model, prior)
    assert np.linalg.norm(placed[:3, 3] - tag[:3, 3]) < 3e-4 and np.all(per_view < 1.0)

    truth = {i: corners([0.27 + 0.02 * (i % 4), 0.07 + 0.03 * (i // 4), 0.051 + 0.002 * (i % 3)], 0.009)
             for i in range(12)}
    tiles = []
    for aside in ((0.0, 0.0), (0.03, 0.03), (-0.03, 0.03), (0.03, -0.03), (-0.03, -0.03)):
        b_c = look_down([0.30, 0.10, 0.05], 0.25, aside)
        reads, depths = {}, {}
        for i, c in truth.items():
            cam = (np.linalg.inv(b_c)[:3, :3] @ c.T).T + np.linalg.inv(b_c)[:3, 3]
            reads[i] = overhead.project(cam, WRIST_K, WRIST_DIST) + rng.normal(0.0, 0.4, (4, 2))
            depths[i] = cam[:, 2] + rng.normal(0.0, 0.0005, 4)
        tiles.append((b_c, WRIST_K, WRIST_DIST, reads, depths))
    reads0 = tiles[0][3]
    reads0[5] = reads0[5] + 40.0   # a wrong read
    start = {i: c + rng.normal(0.0, 0.004, (4, 3)) for i, c in truth.items()}
    placed, rms, out = overhead.tile_in_base(tiles, start)
    assert out == 1 and rms < 1.0
    assert max(np.max(np.linalg.norm(placed[i] - truth[i], axis=1)) for i in truth) < 6e-4


def test_the_record_keeps_the_installed_cameras_and_adoption_installs_it_with_the_golden(tmp_path, monkeypatch):
    installed = {"schema": "tatbot.arm-registration/1", "arm": "right", "calibration_id": "c" * 64,
                 "camera_bundle_sha256": "b" * 64, "camera_bundle_path": "/bundle.json",
                 "root_from_arm_base": np.eye(4).tolist(), "world_from_arm_base": START.tolist(),
                 "world_from_root": START.tolist()}
    rng = np.random.default_rng(3)
    scene, clouds = scene_and_clouds(TRUE, rng)
    fitted, report = overhead.solve(START, targets(TRUE, rng), clouds, scene, K, DIST)
    candidate = overhead.record(installed, fitted, report, "20261022T160000Z-test")
    assert candidate["calibration_id"] == installed["calibration_id"] and candidate["rig"]["method"] == "wrist"
    np.testing.assert_allclose(candidate["world_from_root"], fitted, atol=1e-12)
    assert set(candidate["hold_out"]) == {"station", "holder"}
    monkeypatch.setattr(register, "VISION", tmp_path / "vision")
    monkeypatch.setattr(register, "STACK_CALIB", tmp_path / "calib")
    (tmp_path / "vision").mkdir()
    (tmp_path / "vision" / "arm-registration-right-current.json").write_text(json.dumps(installed))
    golden = {"world_from_base": START.tolist(), "link_from_tag": {"2": [[1.0]]}, "mode": "d555_wrist_tags"}
    (tmp_path / "vision" / "robot-world-current.json").write_text(json.dumps(golden))
    run = tmp_path / "run"
    run.mkdir()
    (run / "candidate.json").write_text(json.dumps(candidate))
    done = overhead.adopt(run, "right")
    assert len(done) == 3
    for directory in ("vision", "calib"):
        now = json.loads((tmp_path / directory / "arm-registration-right-current.json").read_text())
        assert now["rig"]["method"] == "wrist"
    assert json.loads((run / "replaced" / "vision" / "arm-registration-right-current.json").read_text()) == installed
    world = json.loads((tmp_path / "vision" / "robot-world-current.json").read_text())
    np.testing.assert_allclose(world["world_from_base"], fitted, atol=1e-12)
    assert world["link_from_tag"] == golden["link_from_tag"] and (run / "replaced" / "robot-world-current.json").is_file()


def test_the_tiles_frame_is_the_pages_whatever_the_target_frame():
    pytest.importorskip("cv2")
    page = np.eye(4)
    page[:3, 3] = [0.30, 0.10, 0.05]
    page[:3, :3] = overhead.pose([0.0, 0.0, 0.4, 0, 0, 0])[:3, :3]
    target = page @ np.diag([1.0, -1.0, -1.0, 1.0])

    class Tile:
        corners = {i: np.array([[x - 0.009, y + 0.009], [x + 0.009, y + 0.009], [x + 0.009, y - 0.009],
                                [x - 0.009, y - 0.009]]) for i, (x, y) in enumerate(((-0.03, -0.04), (0.03, -0.04),
                                                                                    (0.0, 0.05)))}

    from fiducials import stencil

    world = TRUE
    depth = np.zeros((720, 1280), np.float32)
    reads = {}
    for i in Tile.corners:
        layout = stencil.target_corners(Tile, i)
        seen = (world @ target @ np.c_[layout, np.ones(4)].T).T[:, :3]
        reads[i] = overhead.project(seen, K, DIST)
        for (u, v), z in zip(np.round(reads[i]).astype(int), seen[:, 2], strict=True):
            depth[v - 2:v + 3, u - 2:u + 3] = z
    placed = overhead.tile_frame(Tile, reads, depth, K, DIST, world)
    assert overhead.turn_deg(placed, page) < 0.3 and np.linalg.norm(placed[:3, 3] - page[:3, 3]) < 0.001


def test_the_holder_views_start_from_rest_and_a_landed_arm_that_never_wakes_stops_the_run(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class Rig:
        safety = SimpleNamespace(landed=True)

        def spin(self, seconds):
            calls.append("spin")

    rig = Rig()
    monkeypatch.setattr(overhead.program, "retreat", lambda r, kin: calls.append("land"))
    capture = overhead.Capture.__new__(overhead.Capture)
    capture.cal, capture.kin = SimpleNamespace(rig=rig, arm="right"), None

    def wake(arm):
        calls.append(f"wake {arm}")
        rig.safety = SimpleNamespace(landed=False)

    capture.from_rest(wake)
    assert calls[:3] == ["land", "wake right", "spin"]
    rig.safety = SimpleNamespace(landed=True)
    with pytest.raises(RuntimeError, match="did not wake"):
        capture.from_rest(lambda arm: None)
