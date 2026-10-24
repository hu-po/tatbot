"""fiducials.stencil: a fiducial-tile print read in a wrist page image and placed in an overhead RGB-D view."""

import dataclasses
import itertools

import cv2
import numpy as np
import pytest
import stencil_tile
from fiducials import stencil
from fiducials.detector import DetectorConfig, tag_dictionary

INK = np.float32([110, 30, 95])     # a violet transfer, BGR
SKIN = np.float32([150, 170, 215])
ARM = np.uint8([60, 60, 60])


def sheet(layout):
    """The tile's ink at stencil_tile.PPM px/mm (1 = ink), bled back 0.4 mm as a transfer does, and its Tile."""
    img, record = stencil_tile.build(layout, 0)
    r = round(stencil_tile.SHRINK_MM * stencil_tile.PPM)
    ink = cv2.dilate((255 - img).astype(np.float32) / 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return ink, record, stencil.tile(record)


def sample(ink, record, page_xy):
    """The ink at page-frame points (metres, y up), bilinear."""
    (ox, oy), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    ppm = stencil_tile.PPM
    sx = (page_xy[..., 0] * 1000 + tw / 2 + ox) * ppm - 0.5
    sy = (th / 2 - page_xy[..., 1] * 1000 + oy) * ppm - 0.5
    return cv2.remap(ink, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def on_skin(ink_alpha, seed=0):
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(ink_alpha, (0, 0), 0.8)[..., None] * 0.75
    out = SKIN * (1 - a) + INK * a + rng.normal(0, 3, (*ink_alpha.shape, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def page_image(layout, tx, ty, turn_deg, ppm=5.0):
    """A rectified page image (inspect.page_grid) of a print moved p = R q + t from the believed page."""
    ink, record, t = sheet(layout)
    extent = (-t.size_m[0] / 2 - 0.01, -t.size_m[1] / 2 - 0.01, t.size_m[0] / 2 + 0.01, t.size_m[1] / 2 + 0.01)
    step = 1e-3 / ppm
    xs = extent[0] + (np.arange(round((extent[2] - extent[0]) / step)) + 0.5) * step
    ys = extent[3] - (np.arange(round((extent[3] - extent[1]) / step)) + 0.5) * step
    q = np.stack(np.meshgrid(xs, ys), axis=-1)
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    p = np.stack([c * q[..., 0] - s * q[..., 1] + tx, s * q[..., 0] + c * q[..., 1] + ty], axis=-1)
    return on_skin(sample(ink, record, p)), t, extent


@pytest.mark.parametrize("layout", sorted(stencil_tile.LAYOUTS))
@pytest.mark.parametrize("move", [(0.0, 0.0, 0.0), (0.004, -0.003, 5.0), (-0.007, 0.002, -9.0)])
def test_a_moved_print_reads_back_its_move(layout, move):
    image, t, extent = page_image(layout, *move)
    fit = stencil.fit(image, t, extent, 5.0)
    assert fit["accepted"] and fit["markers"] >= len(t.ids) - 3
    assert abs(fit["tx_m"] - move[0]) < 1e-4 and abs(fit["ty_m"] - move[1]) < 1e-4
    assert abs(np.degrees(fit["theta_rad"]) - move[2]) < 0.05
    assert fit["margin"] == 1.0 and fit["sides"] == ["markers"]


def test_a_misread_marker_is_left_out():
    _, _, t = sheet("forearm")
    ids = sorted(t.ids)
    seen = {i: t.corners[i] + [0.003, 0.0] for i in ids[:5]}
    seen[ids[5]] = t.corners[ids[6]]   # a code read where another marker is
    fit = stencil.fit_points(seen, t)
    assert fit["ids"] == ids[:5] and abs(fit["tx_m"] + 0.003) < 1e-9 and fit["accepted"]


def test_one_marker_is_accepted_only_near_its_printed_place():
    """A lone marker 3 mm from where the page puts it is taken; one a pitch away, where a misread would stand, not."""
    _, _, t = sheet("forearm")
    i = min(t.ids)
    near = stencil.fit_points({i: t.corners[i] + [0.003, 0.0]}, t)
    assert near["markers"] == 1 and near["accepted"] and abs(near["tx_m"] + 0.003) < 1e-9
    far = stencil.fit_points({i: t.corners[i] + [0.0165, 0.0]}, t)
    assert not far["accepted"] and far["margin"] == 0.0


def overhead_view(layout, camera_from_target, fx=645.0, size=(1280, 720), look=None, *, bend=(0.0, 0.0),
                  slide=(0.0, 0.0), hide=None, noise_m=0.0):
    """An aligned RGB-D view of the print on its plane: (image, depth_m, rays) as the observer's rgbd_frame. With
    `look` (LOOKS), the ink as the overhead D555's soft colour shows it at its measured levels; else on_skin's.
    `bend` (kx, ky): the skin z = kx x^2 + ky y^2 in the target frame (x u, y v down, z into the print) instead,
    the print laid on it along x and y and slid `slide` (m) over it; `hide(local_xy)`: the skin the arm hides (no
    depth, the arm's colour); `noise_m`: the depth's per-pixel noise."""
    if look is None:
        ink, record, t = sheet(layout)
    else:
        img, record = stencil_tile.build(layout, 0)
        ink, t = (255 - img).astype(np.float32) / 255, stencil.tile(record)
        if LOOKS[look]["bled"]:
            r = round(stencil_tile.SHRINK_MM * stencil_tile.PPM)
            ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        # the optics' blur and each pixel's area: the ink blurred by 0.8 px at the print's distance
        ink = cv2.GaussianBlur(ink, (0, 0), 0.8 * stencil_tile.PPM * camera_from_target[2, 3] / fx * 1000)
    w, h = size
    k = np.array([[fx, 0, w / 2 - 0.5], [0, fx, h / 2 - 0.5], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(u - k[0, 2]) / fx, (v - k[1, 2]) / fx, np.ones_like(u, float)], axis=-1)
    n, o = camera_from_target[:3, 2], camera_from_target[:3, 3]
    if bend == (0.0, 0.0):
        depth = (n @ o) / (rays @ n)
    else:   # the nearer crossing of each ray with the skin, in the target frame
        (kx, ky), a, b = bend, -(camera_from_target[:3, :3].T @ o), rays @ camera_from_target[:3, :3]
        qa = kx * b[..., 0] ** 2 + ky * b[..., 1] ** 2
        qb = 2 * (kx * a[0] * b[..., 0] + ky * a[1] * b[..., 1]) - b[..., 2]
        qc = kx * a[0] ** 2 + ky * a[1] ** 2 - a[2]
        with np.errstate(invalid="ignore"):
            depth = 2 * qc / (np.sqrt(qb * qb - 4 * qa * qc) - qb)
    local = (rays * depth[..., None] - o) @ camera_from_target[:3, :3]   # target frame: x u, y v (down)
    xy = local[..., :2] - np.asarray(slide)
    alpha = sample(ink, record, np.stack([xy[..., 0], -xy[..., 1]], axis=-1))
    if look is None:
        image = on_skin(alpha)
    else:
        ground, colour = (np.float32(LOOKS[look][key]) for key in ("ground", "ink"))
        image = ground - alpha[..., None] * (ground - colour) + np.random.default_rng(0).normal(0, 2.5, (h, w, 3))
        image = np.clip(image, 0, 255).astype(np.uint8)
    depth = depth + np.random.default_rng(1).normal(0, noise_m, depth.shape) if noise_m else depth
    if hide is not None:
        hidden = hide(local[..., :2])
        image[hidden], depth[hidden] = ARM, 0.0
    return image, depth, rays, t


# The forearm tile as the overhead D555 showed it on 2026-10-19 (BGR; green, the channel read, darkest tenth of a
# marker against the ground between markers): the paper print, its black printed 0.4 mm thin and not bled back, at
# 188 on 219; the violet skin transfer, bled back, at 127 on 208. Rendered, 185 on 218 and 118 on 206.
LOOKS = {"paper": {"ground": (208, 219, 223), "ink": (165, 172, 178), "bled": False},
         "transfer": {"ground": (196, 208, 219), "ink": (140, 105, 130), "bled": True}}


def forearm_pose(turn_deg, z=0.75):
    """camera <- target of the forearm tile as the overhead sees it: tilted 37.5 deg, left of and above the
    image centre, turned turn_deg in its plane."""
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([np.radians(37.5), 0, 0]))[0] @ cv2.Rodrigues(np.array([0, 0, np.radians(turn_deg)]))[0]
    pose[:3, 3] = [-0.045, -0.056, z]
    return pose


def test_the_overhead_places_the_silicone_sheet_by_its_markers():
    turn = np.radians(20)
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([0.15, -0.1, turn]))[0]
    pose[:3, 3] = [0.03, -0.02, 0.75]
    image, depth, rays, t = overhead_view("silicone", pose)
    found = stencil.rgbd_pose(image, depth, rays, t, **stencil.overhead_reading(t))
    assert found["found"] and set(t.anchors) <= set(found["ids"])
    err = np.linalg.inv(pose) @ found["camera_from_target"]
    assert np.linalg.norm(err[:3, 3]) < 0.0015
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(err[:3, :3])[0])) < 0.5


def test_the_silicone_tile_keeps_the_plain_overhead_reading():
    """Its 18 mm markers are read as before: no upsampling, OpenCV's perimeter floor and threshold, skin CLAHE."""
    t = sheet("silicone")[2]
    reading = stencil.overhead_reading(t)
    assert not stencil.small_markers(t) and reading["contrast"] == stencil.SKIN_CONTRAST == (2.0, 8)
    assert reading["det"].config == DetectorConfig(adaptive_window_max=53, min_side_px=6.0, refinement="edges",
                                                   cell_margin=0.25, skin=True)
    reading["det"].detect("tile", np.full((720, 1280), 200, np.uint8), 0)
    assert reading["det"].params.minMarkerPerimeterRate == pytest.approx(0.03)
    assert reading["det"].params.adaptiveThreshConstant == pytest.approx(7.0)
    forearm = stencil.overhead_reading(sheet("forearm")[2])
    assert forearm["det"].config.scale == 2.0 and forearm["contrast"] == stencil.SMALL_CONTRAST


@pytest.mark.parametrize("look", sorted(LOOKS))
@pytest.mark.parametrize("turn", [0, 45, 90, 180, 270])
def test_the_overhead_places_the_forearm_tile_the_plain_reading_misses(look, turn):
    """The forearm tile's 12 mm markers at ~9 px a side and 1.5 px a cell, their perimeter under OpenCV's floor
    (0.03 of 1280 px): the plain reading places it in no view, the small-marker reading in every one."""
    pose = forearm_pose(turn)
    image, depth, rays, t = overhead_view("forearm", pose, look=look)
    assert stencil.small_markers(t)
    assert not stencil.rgbd_pose(image, depth, rays, t, stencil.detector(t, min_side_px=6.0))["found"]
    found = stencil.rgbd_pose(image, depth, rays, t, **stencil.overhead_reading(t))
    assert found["found"] and found["markers"] >= 5
    err = np.linalg.inv(pose) @ found["camera_from_target"]
    assert np.linalg.norm(err[:3, 3]) < 0.0005
    assert np.degrees(np.linalg.norm(cv2.Rodrigues(err[:3, :3])[0])) < 0.5


def foreign_tag(image, tag_id, at, side_px=60):
    """A 16h5 tag of side_px drawn on a white margin into the image, its top-left at `at` (an ArUco sheet or a
    wrist-tag face that decodes as one of the tile's codes)."""
    margin = side_px // 2
    tag = cv2.aruco.generateImageMarker(tag_dictionary("apriltag_16h5"), tag_id, side_px)
    x, y = at
    image[y - margin:y + side_px + margin, x - margin:x + side_px + margin] = 255
    image[y:y + side_px, x:x + side_px] = tag[..., None]
    return image


def test_find_reads_in_black_only_the_codes_the_violet_view_missed(monkeypatch):
    """find's second pass (skin_gray, a print in black) fits the edges of the codes the violet view missed alone:
    fitting every code again cost 15 ms of a frame's 67 on the silicone tile's 19 markers (2026-10-24), each frame read
    25 times a gauge hold by two cameras. With none missed it is not run; what it finds is as before."""
    image, t, _ = page_image("silicone", 0.0, 0.0, 0.0)
    det = stencil.detector(t)
    before = {k: v.copy() for k, v in stencil.find(image, t, det).items()}
    violet = {d.tag_id for d in det.detect("tile", stencil.violet_gray(image), 0)}
    asked, detect = [], det.detect
    monkeypatch.setattr(det, "detect", lambda *a, only=None, **k: asked.append(only) or detect(*a, only=only, **k))
    again = stencil.find(image, t, det)
    assert asked == [None] + ([frozenset(t.ids) - violet] if violet < set(t.ids) else [])
    assert again.keys() == before.keys() and all(np.array_equal(again[k], before[k]) for k in before)
    monkeypatch.setattr(det, "detect", lambda *a, only=None, **k: asked.append(only) or [
        d for d in detect(*a, only=only, **k) if d.tag_id != next(iter(before))])
    asked.clear()
    stencil.find(image, t, det)
    assert next(iter(before)) in asked[1]   # a code the violet view missed is read in black


def test_a_large_false_read_of_a_code_does_not_hide_the_prints_own():
    """A 60 px false read of one of the print's codes beside it, in black: find reads the violet print first, so it
    keeps the print's own read, and the overhead's reads (find_all, the false one too) are held to the printed size,
    so the print's own read of that code still places it."""
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([np.radians(37.5), 0.0, 0.0]))[0]
    pose[:3, 3] = [-0.045, -0.056, 0.5]
    image, depth, rays, t = overhead_view("forearm", pose)
    det = stencil.detector(t, min_side_px=6.0)
    clean = stencil.rgbd_pose(image, depth, rays, t, det)
    code = clean["ids"][0]
    image = foreign_tag(image, code, (1100, 560))   # clear of the CLAHE tiles about the print
    assert np.ptp(stencil.find(image, t, det)[code][:, 0]) < 40     # the print's own read, not the black false one
    assert max(np.ptp(d.corners_px[:, 0]) for d in stencil.find_all(image, t, det) if d.tag_id == code) > 40
    found = stencil.rgbd_pose(image, depth, rays, t, det)
    assert found["found"] and found["ids"] == clean["ids"]
    assert np.allclose(found["camera_from_target"], clean["camera_from_target"], atol=1e-9)


def test_no_marker_is_a_reason():
    image = np.full((360, 640, 3), 180, np.uint8)
    found = stencil.rgbd_pose(image, np.full((360, 640), 0.7), np.ones((360, 640, 3)), sheet("forearm")[2])
    assert found == {"found": False, "reason": "no marker of the print read"}
    image = foreign_tag(image, min(sheet("forearm")[2].ids), (300, 150))   # a 60 px tag 0.7 m away: 65 mm
    rays = np.stack([*np.meshgrid((np.arange(640) - 319.5) / 645.0, (np.arange(360) - 179.5) / 645.0),
                     np.ones((360, 640))], axis=-1)
    found = stencil.rgbd_pose(image, np.full((360, 640), 0.7), rays, sheet("forearm")[2])
    assert found == {"found": False, "reason": "no marker of the print read: 1 reads of its codes off the print "
                                                 "(not its printed size on the depth)"}


def test_a_foreign_read_far_from_the_print_does_not_move_it(monkeypatch):
    """Three reads of the tile's codes elsewhere in the scene, at its printed size, agree with nothing on the print."""
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 0.75]
    image, depth, rays, t = overhead_view("silicone", pose)
    real = stencil.find_all

    def with_strays(img, tile, det=None, roi_xyxy=None, **kw):
        found = real(img, tile, det, roi_xyxy, **kw)
        for k, d in enumerate(found[:3]):   # the same code read across the table
            found[k] = dataclasses.replace(d, corners_px=d.corners_px + [-300 - 100 * k, 150])
        return found
    monkeypatch.setattr(stencil, "find_all", with_strays)
    out = stencil.rgbd_pose(image, depth, rays, t, **stencil.overhead_reading(t))
    assert out["found"] and np.linalg.norm(out["camera_from_target"][:3, 3] - pose[:3, 3]) < 0.0015


# Skins the overhead reads a tile on, z = kx x^2 + ky y^2 in the target frame: the silicone mount as its gauge holds
# fit it (2026-10-19, r ~168 mm along v) and a forearm (r 40 mm around u).
MOUNT = (0.0, 2.98)
FOREARM = (1 / (2 * 0.040), 0.0)


def skin_pose(z):
    """camera <- target of a print on a skin, 30 deg off the optical axis and turned 10 deg."""
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.array([np.radians(30.0), 0.0, np.radians(10.0)]))[0]
    pose[:3, 3] = [0.0, 0.0, z]
    return pose


def on_skin_view(bend, z, layout="forearm", **view):
    """(the tile placed in an overhead view of it on a skin, that view's depth, rays, tile)."""
    image, depth, rays, t = overhead_view(layout, skin_pose(z), bend=bend, noise_m=0.0007, **view)
    found = stencil.rgbd_pose(image, depth, rays, t, **stencil.overhead_reading(t))
    assert found["found"]
    return found, depth, rays, t


def corners(found, key="points_camera"):
    """id -> its four published corners (ids[k] names rows 4k:4k+4)."""
    rows = np.asarray(found[key])
    return dict(zip(found["ids"], rows.reshape(-1, 4, rows.shape[1]), strict=True))


@pytest.mark.parametrize("bend, z, hidden", [(MOUNT, 0.70, "lower"), (FOREARM, 0.45, "flank")])
def test_the_arm_over_part_of_a_curved_print_moves_none_of_the_rest(bend, z, hidden):
    """Each corner sits at its own depth, so the arm over half the print moves no other marker's points: the depth
    alone moves them under 0.2 mm, and with the detector's corners shifting by the arm's edge 0.1 mm median, under
    1 mm (one plane over the sheet's visible part moved them 0.9 and 2.2 mm median, 2.1 and 4.0 mm at most)."""
    hide = {"lower": lambda xy: xy[..., 1] > 0.0, "flank": lambda xy: xy[..., 0] > 0.0}[hidden]
    full, _, rays, _ = on_skin_view(bend, z)
    part, depth, _, _ = on_skin_view(bend, z, hide=hide)
    again = {i: stencil.corner_depth(depth, px) for i, px in corners(full, "pixels").items()}
    moved = [np.linalg.norm(stencil._ray(rays, corners(full, "pixels")[i]) * d[:, None] - corners(full)[i], axis=1)
             for i, d in again.items() if d is not None]
    assert len(moved) >= 2 and np.max(moved) < 0.0002
    common = sorted(set(full["ids"]) & set(part["ids"]))
    shift = np.concatenate([np.linalg.norm(corners(part)[i] - corners(full)[i], axis=1) for i in common])
    assert len(common) >= 2 and np.median(shift) < 0.0001 and shift.max() < 0.001


@pytest.mark.parametrize("bend, slide", [(MOUNT, (0.003, 0.0)), (FOREARM, (0.0, 0.003))])
def test_a_print_slid_on_curved_skin_reads_its_slide_at_the_same_markers(bend, slide):
    """3 mm along the skin where it does not bend reads 3 mm."""
    before = on_skin_view(bend, 0.70)[0]
    after = on_skin_view(bend, 0.70, slide=slide)[0]
    common = sorted(set(before["ids"]) & set(after["ids"]))
    moved = [np.linalg.norm(corners(after)[i].mean(axis=0) - corners(before)[i].mean(axis=0)) for i in common]
    assert len(common) >= 4 and abs(np.median(moved) - 0.003) < 0.0001


@pytest.mark.parametrize("layout, bend, z, least", [("forearm", FOREARM, 0.45, 12), ("forearm", FOREARM, 0.60, 7),
                                                    ("silicone", MOUNT, 0.70, 15)])
def test_a_curved_skin_keeps_every_marker_read_at_its_size(layout, bend, z, least):
    """On a 40 mm forearm the flat layout fitted to its markers misses the outer ones by mm (at 0.6 m one plane and
    its 4 mm cut kept 6), and over the silicone mount by up to 14 mm: agreeing by their distances, within the curve
    each layout allows, every read the size check keeps is kept."""
    found, depth, rays, t = on_skin_view(bend, z, layout)
    reading = stencil.overhead_reading(t)
    sized = set()
    for d in stencil.find_all(*overhead_view(layout, skin_pose(z), bend=bend, noise_m=0.0007)[:1], t,
                              reading["det"], contrast=reading["contrast"]):
        lift = stencil.corner_depth(depth, d.corners_px)
        if lift is not None and stencil._printed_size(stencil._ray(rays, d.corners_px) * lift[:, None],
                                                      t.sides_m[d.tag_id]):
            sized.add(d.tag_id)
    assert found["ids"] == sorted(sized) and len(sized) >= least


def wrapped(t, radius_m, pose, along=0):
    """id -> (4, 3) camera points of the tile's printed corners laid unstretched on skin curved to radius_m, its
    arcs along u (along=0: a forearm round v) or v (1: the silicone mount) keeping their printed lengths; flat with
    no radius."""
    out = {}
    for i in t.ids:
        local = stencil.target_corners(t, i)
        if radius_m:
            turn = local[:, along] / radius_m
            local[:, along], local[:, 2] = radius_m * np.sin(turn), radius_m * (1 - np.cos(turn))
        out[i] = local @ pose[:3, :3].T + pose[:3, 3]
    return out


@pytest.mark.parametrize("radius_m", [0.040, 0.035])
def test_markers_on_a_forearm_agree_by_their_distances(radius_m):
    """Every marker of a print on a 35-40 mm forearm, 0.3 mm noise on each corner, agrees with every other: the flat
    layout's rigid fit leaves its outer corners over 2 mm off."""
    t = sheet("forearm")[2]
    rng = np.random.default_rng(0)
    points = {(i, 0): c + rng.normal(0, 0.0003, c.shape) for i, c in wrapped(t, radius_m, skin_pose(0.5)).items()}
    keys = sorted(points)
    flat = stencil.kabsch(np.concatenate([stencil.target_corners(t, k[0]) for k in keys]),
                          np.concatenate([points[k] for k in keys]))
    off = max(np.abs(stencil.target_corners(t, k[0]) @ flat[:3, :3].T + flat[:3, 3] - points[k]).max() for k in keys)
    assert off > 0.002
    assert stencil._consensus(t, points, 0.004) == keys


def test_a_code_read_off_its_printed_place_agrees_with_no_one():
    """On the forearm: a code read where its neighbour is printed (that neighbour's own read lost), and a second
    read of another code 30 mm off the print, are left out; every other marker is kept."""
    t = sheet("forearm")[2]
    points = {(i, 0): c for i, c in wrapped(t, 0.040, skin_pose(0.5)).items()}
    ids = sorted(t.ids)
    misread = ids[0]
    centre = t.corners[misread].mean(axis=0)
    neighbour = min((i for i in ids if i != misread), key=lambda i: np.linalg.norm(t.corners[i].mean(axis=0) - centre))
    points[(misread, 0)] = points.pop((neighbour, 0))
    points[(ids[5], 1)] = points[(ids[5], 0)] + [0.03, 0.0, 0.0]
    keep = stencil._consensus(t, points, 0.004)
    assert keep == [(i, 0) for i in ids if i not in (misread, neighbour)]


def test_each_layout_allows_the_curve_of_the_skin_it_goes_on():
    """The forearm tile a forearm's (35 mm); the silicone tile a practice skin's over the mount (r ~167 mm), never a
    forearm's; a layout the table does not name, the tightest."""
    assert sheet("forearm")[2].curve_radius_m == 0.035
    assert sheet("silicone")[2].curve_radius_m == 0.10
    assert stencil.tile({**sheet("forearm")[1], "layout": "unnamed"}).curve_radius_m == 0.035


@pytest.mark.parametrize("layout, radius_m, along", [("forearm", None, 0), ("forearm", 0.040, 0),
                                                     ("silicone", None, 1), ("silicone", 0.167, 1)])
def test_a_code_read_again_off_its_place_beside_two_true_reads_takes_neither_place(layout, radius_m, along):
    """Two true reads (the silicone tile's anchors: the overhead's usual view of it with the arm over the sheet) and a
    third code read again 10-57 mm off its printed place in the print's plane, or 10 mm off that plane, flat or on
    the skin the tile goes on: both true reads are always kept, and a read 20 mm or more off its place never is.
    Ties among such three were broken by the keys' order and the silicone tile allowed a forearm's curve: 242 of
    these 2904 views on the mount dropped a true anchor for the stray, the pose up to 22 mm off (2026-10-19 review)."""
    t = sheet(layout)[2]
    ids = sorted(t.anchors) if layout == "silicone" else sorted(t.ids)
    pose = skin_pose(0.67)
    truth = wrapped(t, radius_m, pose, along)
    rng = np.random.default_rng(0)
    cases = [(pair, other) for pair in itertools.combinations(ids, 2) for other in ids if other not in pair]
    cases = [cases[k] for k in rng.choice(len(cases), min(12, len(cases)), replace=False)]
    grid = np.arange(-0.04, 0.0401, 0.01)
    for (pair, other), dx, dy, dn in itertools.product(cases, grid, grid, (-0.01, 0.0, 0.01)):
        if np.hypot(dx, dy) < 0.005 and dn == 0.0:
            continue
        points = {(i, 0): truth[i] + rng.normal(0, 0.0003, (4, 3)) for i in pair}
        points[(other, 1)] = truth[other] + pose[:3, :3] @ [dx, dy, dn] + rng.normal(0, 0.0003, (4, 3))
        keep = stencil._consensus(t, points, 0.004)
        assert {(i, 0) for i in pair} <= set(keep), (pair, other, dx, dy, dn)
        assert (other, 1) not in keep or np.hypot(dx, dy) < 0.015, (pair, other, dx, dy, dn)


def test_two_true_reads_and_a_stray_keep_the_true_pair_whatever_the_codes_order():
    """The three tie on every count but their distances' fit: the true pair wins under either order of the keys."""
    t = sheet("silicone")[2]
    a, b, c = sorted(t.anchors)[:3]
    pose = skin_pose(0.67)
    truth = wrapped(t, None, pose)
    for stray in ((a, 1), (c, 0), (c, 9)):   # sorting first, between and last
        points = {(a, 0): truth[a], (b, 0): truth[b], stray: truth[stray[0]] + pose[:3, :3] @ [0.02, -0.04, 0.0]}
        if stray[0] == a:
            points = {(c, 0): truth[c], (b, 0): truth[b], stray: points[stray]}
        true = sorted(k for k in points if k != stray)
        assert stencil._consensus(t, points, 0.004) == true


def turned_by(a, b) -> float:
    """The angle between two directions, degrees."""
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


@pytest.mark.parametrize("layout, radius_m, along, side", [("forearm", 0.045, 0, 1), ("forearm", 0.060, 0, -1),
                                                           ("forearm", 0.060, 0, 0), ("silicone", 0.167, 1, 1)])
def test_a_print_on_curved_skin_is_placed_by_its_centre_on_the_skin(layout, radius_m, along, side):
    """The print laid unstretched on a forearm (round v) or the silicone mount (round u), its centre on the crest,
    the markers of one flank alone in view (`side`; 0: all of them), 0.3 mm on each corner: the layout laid round a
    cylinder through them puts the centre on the skin within 0.5 mm and its z along the skin's normal there within
    1.5 deg, where on the forearm their chord plane (the flat rigid fit) tilts 4 deg or more toward them; the turn
    stays the flat fit's."""
    t = sheet(layout)[2]
    pose = skin_pose(0.6)
    rng = np.random.default_rng(3)
    across = 1 - along   # the print's axis the skin bends along
    ids = sorted(i for i in t.ids if side * t.corners[i][:, along].mean() >= 0.0)
    points = np.concatenate([wrapped(t, radius_m, pose, along)[i] + rng.normal(0, 0.0003, (4, 3)) for i in ids])
    flat = stencil.kabsch(np.concatenate([stencil.target_corners(t, i) for i in ids]), points)
    placed, radius, rms, _ = stencil.bent(t, ids, points, flat)
    assert np.linalg.norm(placed[:3, 3] - pose[:3, 3]) < 0.0005
    assert turned_by(placed[:3, 2], pose[:3, 2]) < 1.5 and turned_by(placed[:3, across], pose[:3, across]) < 1.5
    assert abs(radius / radius_m - 1) < 0.25 and rms < 0.0007   # the noise: 0.3 mm a coordinate, 0.52 in 3-D
    if side and layout == "forearm":
        assert turned_by(flat[:3, 2], pose[:3, 2]) > 4.0


def test_a_print_on_flat_skin_or_markers_in_a_row_keep_the_flat_fit():
    """Flat skin fits no bend worth having; three markers in one row (or two) cannot show one: the flat fit stands."""
    t = sheet("forearm")[2]
    pose = skin_pose(0.6)
    truth = wrapped(t, None, pose)
    ids = sorted(t.ids)
    points = np.concatenate([truth[i] for i in ids])
    flat = stencil.kabsch(np.concatenate([stencil.target_corners(t, i) for i in ids]), points)
    placed, radius, _, _ = stencil.bent(t, ids, points, flat)
    assert np.linalg.norm(placed[:3, 3] - pose[:3, 3]) < 1e-5 and turned_by(placed[:3, 2], pose[:3, 2]) < 0.01
    assert radius > 10.0
    row = sorted(t.ids, key=lambda i: (round(t.corners[i][:, 1].mean(), 3), t.corners[i][:, 0].mean()))[:3]
    assert len({round(t.corners[i][:, 1].mean(), 3) for i in row}) == 1
    curved = wrapped(t, 0.045, pose)
    for ids in (row, row[:2]):
        points = np.concatenate([curved[i] for i in ids])
        flat = stencil.kabsch(np.concatenate([stencil.target_corners(t, i) for i in ids]), points)
        placed, radius, _, _ = stencil.bent(t, ids, points, flat)
        assert radius == np.inf and np.array_equal(placed, flat)


def test_the_overhead_places_a_forearm_print_by_the_skin_under_its_centre():
    """The overhead's view of a 40 mm forearm, the arm over one flank (two of the print's three columns in view):
    the pose's z is the skin's normal at the print's centre within 2 deg (the markers' chord plane tilted 11.6), its
    origin on the skin there within 0.7 mm. One column alone in view keeps the chord (test above)."""
    pose = skin_pose(0.45)
    found, depth, rays, t = on_skin_view(FOREARM, 0.45, hide=lambda xy: xy[..., 0] > 0.012)
    flat = stencil.kabsch(np.concatenate([stencil.target_corners(t, i) for i in found["ids"]]), found["points_camera"])
    assert found["radius_m"] < 0.06 and turned_by(flat[:3, 2], pose[:3, 2]) > 5.0
    assert turned_by(found["camera_from_target"][:3, 2], pose[:3, 2]) < 2.0
    assert np.linalg.norm(found["camera_from_target"][:3, 3] - pose[:3, 3]) < 0.0007
