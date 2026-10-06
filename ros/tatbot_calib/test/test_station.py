"""The station fix: a fresh roof-tag solve into an arm's frame, the refusal of anything older than its run, and
the comparison that catches a palette moved between two fixes (probe-calibration plan, principle 8)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from tatbot_calib import station
from tatbot_description import repo_root

PALETTE = repo_root(None) / "urdf" / "palette.urdf"
T0 = station.parse_utc("2026-09-28T16:00:00Z")


def rigid(xyz, yaw_deg=0.0):
    out = np.eye(4)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    out[:2, :2] = [[c, -s], [s, c]]
    out[:3, 3] = xyz
    return out


def fix_at(xyz, yaw_deg=0.0, source="overhead_tag", utc="2026-09-28T16:00:00Z"):
    base_from_palette = rigid(xyz, yaw_deg)
    ball = (base_from_palette @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3]
    return station.StationFix("left", base_from_palette, ball, utc, source)


def test_the_ball_hangs_where_the_palette_urdf_puts_it():
    assert np.allclose(station.ball_in_palette(PALETTE), [0.0, 0.0, 0.0556])


@pytest.mark.parametrize('mutation', ['parent', 'moving', 'duplicate', 'missing'])
def test_station_obstacles_refuse_an_unsupported_neighbour_joint(tmp_path, mutation):
    import copy
    import xml.etree.ElementTree as ET

    tree = ET.parse(PALETTE)
    joint = next(j for j in tree.getroot().iter('joint') if j.find('child').get('link') == 'inkcap_large_2')
    if mutation == 'parent':
        joint.find('parent').set('link', 'palette_camera')
    elif mutation == 'moving':
        joint.set('type', 'revolute')
    elif mutation == 'duplicate':
        tree.getroot().append(copy.deepcopy(joint))
    else:
        tree.getroot().remove(joint)
    changed = tmp_path/'palette.urdf'
    tree.write(changed)
    with pytest.raises(ValueError, match='exactly one|fixed directly'):
        station.inkcap_rims(changed, repo_root(None)/'config/palette.yaml')


@pytest.mark.parametrize('field,value', [('diameter_m', -.001), ('height_m', -.001),
                                        ('diameter_m', float('nan')), ('height_m', float('inf'))])
def test_station_obstacles_refuse_unusable_cap_dimensions(tmp_path, field, value):
    import yaml

    palette = yaml.safe_load((repo_root(None)/'config/palette.yaml').read_text())
    palette['sizes']['large'][field] = value
    changed = tmp_path/'palette.yaml'
    changed.write_text(yaml.safe_dump(palette))
    with pytest.raises(ValueError, match='finite positive metres'):
        station.inkcap_rims(PALETTE, changed)


@pytest.mark.parametrize('present', [True, None, False])
def test_tilted_cap_proxy_covers_its_sides_and_retains_an_absent_support(tmp_path, present):
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace

    import yaml
    from tatbot_description.transforms import rpy_matrix

    tree = ET.parse(PALETTE)
    slot = 'inkcap_medium_1'
    joint = next(j for j in tree.getroot().iter('joint') if j.find('child').get('link') == slot)
    joint.find('origin').set('rpy', '.3 -.2 .5')
    changed = tmp_path/'palette.urdf'
    tree.write(changed)
    config = repo_root(None)/'config/palette.yaml'
    sizes = yaml.safe_load(config.read_text())
    size = sizes['sizes'][sizes['slots'][slot]['size']]
    pose = rpy_matrix([.2, .3, .01], [-.2, .1, .4])
    support = rpy_matrix([float(v) for v in joint.find('origin').get('xyz').split()], [.3, -.2, .5])
    load = {name: SimpleNamespace(cap_present=present) for name in sizes['slots']}
    rims = station.inkcap_rims(changed, config, load=load, base_from_palette=pose)
    _, top, radius, post = next(p for p in station.parts(pose, rims, changed) if p[0] == slot)
    assert post
    height = 0. if present is False else size['height_m']
    angles = np.linspace(0, 2*np.pi, 181)
    for z in np.linspace(0., height, 11):
        cylinder = np.array([size['diameter_m']/2*np.cos(angles), size['diameter_m']/2*np.sin(angles),
                             np.full_like(angles, z), np.ones_like(angles)])
        samples = (pose @ support @ cylinder)[:3].T
        assert np.max(np.linalg.norm(samples[:, :2]-top[:2], axis=1)) <= radius+1e-12
        assert np.max(samples[:, 2]) <= top[2]+1e-12
    if present is False:
        assert top[2] < next((pose @ np.append(p, 1.))[2] for n, p, _ in station.inkcap_rims(
            changed, config, base_from_palette=pose) if n == slot)


# The D555 over the demo table: 640x360 colour, the palette 0.66 m away (2026-09-29).
K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]])
CAMERA_FROM_BASE = np.array([[-0.7073, -0.7048, 0.0546, 0.2343], [-0.6846, 0.6636, -0.3017, 0.0007],
                             [0.1764, -0.2508, -0.9518, 0.7901], [0.0, 0.0, 0.0, 1.0]])
MODEL = 0.023 * np.array([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]])   # 46 mm tag


def _orthonormal(m):
    u, _, vt = np.linalg.svd(m[:3, :3])
    out = m.copy()
    out[:3, :3] = u @ vt
    return out


def seen(base_from_palette, noise_px=0.0, seed=0):
    """The roof tag's corners as the D555 sees the palette at base_from_palette."""
    import cv2

    tag = base_from_palette @ station.palette_from_tag(PALETTE) @ np.c_[MODEL, np.ones(4)].T
    camera = (_orthonormal(CAMERA_FROM_BASE) @ tag)[:3].T
    pixels, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), K, np.zeros(5))
    return pixels.reshape(4, 2) + np.random.default_rng(seed).normal(0.0, noise_px, (4, 2))


def test_the_tag_hangs_where_the_palette_urdf_puts_it():
    tag = station.palette_from_tag(PALETTE)
    assert np.allclose(tag[:3, 3], [0.068, -0.026, 0.0082])
    # the pattern's +x 135 degrees from the palette's +x, as the D555 found it on the seat (2026-09-30)
    assert np.allclose(tag[:3, :3] @ [1.0, 0.0, 0.0], [-2**-0.5, 2**-0.5, 0.0], atol=1e-9)


def test_a_small_tag_seen_by_the_d555_places_the_level_palette_and_its_ball():
    """23 px of tag: the free pose's tilt is ambiguous, the level pose puts the ball within a millimetre."""
    pytest.importorskip("cv2")
    truth = rigid([0.0952, 0.3116, 0.0202], yaw_deg=67.3)
    pixels = seen(truth, noise_px=0.05)
    assert 20.0 < np.mean(np.linalg.norm(pixels - np.roll(pixels, 1, axis=0), axis=1)) < 26.0
    got, rms = station.level_pose(pixels, K, np.zeros(5), _orthonormal(CAMERA_FROM_BASE), MODEL,
                                  station.palette_from_tag(PALETTE))
    ball = lambda pose: (pose @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3]   # noqa: E731
    assert np.linalg.norm(ball(got) - ball(truth)) < 0.0015 and rms < 0.2
    assert abs(math.degrees(math.atan2(got[1, 0], got[0, 0])) - 67.3) < 0.5 and got[2, 2] == pytest.approx(1.0)


def test_the_depth_places_the_tag_where_the_light_moves_its_corners():
    """2026-09-30, the room dimmed: the 23 px tag imaged 1.5% larger, and its corners put the palette 10.8 mm
    higher, while the D555's depth held it within 0.1 mm. Placed by the depth, the ball stands where it is; its
    yaw stays the corners'."""
    pytest.importorskip("cv2")
    truth = rigid([0.0952, 0.3116, 0.0202], yaw_deg=67.3)
    pixels = seen(truth)
    centre = pixels.mean(axis=0)
    dim = centre + 1.015 * (pixels - centre)          # the edges' bias in the dim room
    camera_from_base = _orthonormal(CAMERA_FROM_BASE)
    palette_tag = station.palette_from_tag(PALETTE)
    corners, _ = station.level_pose(dim, K, np.zeros(5), camera_from_base, MODEL, palette_tag)
    ball = lambda pose: (pose @ np.append(station.ball_in_palette(PALETTE), 1.0))[:3]   # noqa: E731
    assert np.linalg.norm(ball(corners) - ball(truth)) > 0.006
    tag = camera_from_base @ truth @ palette_tag      # the palette's roof, its plane in the depth
    ys, xs = np.mgrid[0:360, 0:640]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    depth = (tag[:3, 2] @ tag[:3, 3]) / (rays @ tag[:3, 2])
    depth = depth + np.random.default_rng(0).normal(0.0, 0.001, depth.shape)
    found = station.tag_centre_by_depth(dim, depth.astype(np.float32), K, np.zeros(5))
    assert found is not None and np.linalg.norm(found - tag[:3, 3]) < 0.0005
    placed = station.placed_at(corners, palette_tag, (np.linalg.inv(camera_from_base) @ np.append(found, 1.0))[:3])
    assert np.linalg.norm(ball(placed) - ball(truth)) < 0.001 and np.allclose(placed[:3, :3], corners[:3, :3])
    assert station.tag_centre_by_depth(dim, np.zeros((360, 640), np.float32), K, np.zeros(5)) is None


def test_a_tag_not_standing_level_is_refused():
    pytest.importorskip("cv2")
    tilted = rigid([0.0952, 0.3116, 0.0202], yaw_deg=67.3)
    c, s = math.cos(math.radians(25.0)), math.sin(math.radians(25.0))
    tilted[:3, :3] = tilted[:3, :3] @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    with pytest.raises(ValueError, match="tilts"):
        station.level_pose(seen(tilted), K, np.zeros(5), _orthonormal(CAMERA_FROM_BASE), MODEL,
                           station.palette_from_tag(PALETTE))


def test_shots_average_into_one_fix_and_a_disagreeing_one_refuses_it():
    shots = [rigid([0.0952 + dx, 0.3116, 0.0202], yaw_deg=67.3 + dyaw) for dx, dyaw in
             ((0.0, 0.0), (0.0003, 0.05), (-0.0003, -0.05))]
    fix = station.from_poses("right", shots, "2026-09-29T16:00:00Z", PALETTE, "overhead_tag", {"camera": "d555"})
    assert np.allclose(fix.ball, [0.0952, 0.3116, 0.0202 + 0.0556], atol=1e-6)
    assert fix.detail["shots"] == 3 and fix.detail["shot_spread_mm"] == pytest.approx(0.3, abs=1e-3)
    assert fix.source == "overhead_tag" and fix.detail["camera"] == "d555"
    with pytest.raises(ValueError, match="disagree"):
        station.from_poses("right", [*shots, rigid([0.1002, 0.3116, 0.0202], yaw_deg=67.3)], "2026-09-29T16:00:00Z",
                           PALETTE, "overhead_tag")
    with pytest.raises(ValueError, match="no shot"):
        station.from_poses("right", [], "2026-09-29T16:00:00Z", PALETTE, "overhead_tag")


def test_a_fix_from_before_the_run_or_too_old_is_refused():
    fix = fix_at([0.21, -0.27, 0.064])
    assert station.require_fresh(fix, T0 - 60.0, now=T0 + 30.0) == pytest.approx(30.0)
    with pytest.raises(station.StaleStationError, match="before this run started"):
        station.require_fresh(fix, T0 + 60.0, now=T0 + 90.0)   # the palette can have moved since
    with pytest.raises(station.StaleStationError, match="ago"):
        station.require_fresh(fix, T0 - 60.0, now=T0 + 700.0)
    with pytest.raises(station.StaleStationError, match="ahead"):
        station.require_fresh(fix, T0 - 60.0, now=T0 - 30.0)
    # a few seconds of skew between the camera node's stamp and this node's clock is not a refusal
    assert station.require_fresh(fix, T0 + 3.0, now=T0 - 2.0) == 0.0


def test_a_moved_station_is_told_from_the_repeatability_of_its_source():
    before = fix_at([0.21, -0.27, 0.064])
    assert not station.station_moved(before, fix_at([0.2105, -0.27, 0.064]))["moved"]   # 0.5 mm: the D555's noise
    moved = station.station_moved(before, fix_at([0.2115, -0.27, 0.064]))
    assert moved["moved"] and moved["shift_m"] == pytest.approx(0.0015)
    assert moved["tolerance_m"] == station.MOVED_M["overhead_tag"]
    assert station.station_moved(before, fix_at([0.21, -0.27, 0.064], yaw_deg=1.0))["moved"]   # turned on the spot
    with pytest.raises(ValueError, match="two arms"):
        station.station_moved(before, station.StationFix("right", before.base_from_palette, before.ball,
                                                          before.measured_utc, before.source))


def test_keep_outs_follow_the_loaded_palette_asset(tmp_path):
    """A replaced camera or switch must not leave old hardcoded collision centres."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(PALETTE)
    camera = next(j for j in tree.getroot().iter('joint')
                  if j.find('child').get('link') == 'palette_camera')
    camera.find('origin').set('xyz', '.2 .03 .06')
    changed = tmp_path / 'palette.urdf'
    tree.write(changed)
    pose = rigid([.1, .2, .3], 90)
    camera_zone = next(part for part in station.parts(pose, palette_urdf=changed)
                       if part[0] == 'palette_camera')
    assert camera_zone[1] == pytest.approx([.07, .4, .36])
    tree.getroot().remove(camera)
    tree.write(changed)
    with pytest.raises(ValueError, match='palette_camera'):
        station.parts(pose, palette_urdf=changed)


def test_hardware_station_cannot_use_an_unconfirmed_tag_frame(tmp_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(PALETTE)
    tree.getroot().set('tag_pose_status', 'nominal')
    pending = tmp_path / 'palette.urdf'
    tree.write(pending)
    assert station.palette_from_tag(pending).shape == (4, 4)  # synthetic geometry remains usable
    with pytest.raises(ValueError, match='orientation is unconfirmed'):
        station.palette_from_tag(pending, require_confirmed=True)
    tree.getroot().set('tag_pose_status', 'confirmed')
    tree.write(pending)
    assert station.palette_from_tag(pending, require_confirmed=True).shape == (4, 4)


def test_estop_keep_out_covers_the_bottom_of_the_installed_housing():
    _, centre, radius, _ = next(part for part in station.parts(np.eye(4)) if part[0] == "palette_estop")
    # v11 panel Z32, housing radius 22, bottom Z-20.4 (mm).
    for xy in ((0.096, 0.040), (0.052, 0.040), (0.074, 0.062), (0.074, 0.018)):
        assert np.linalg.norm(np.array([*xy, -0.0204]) - centre) <= radius
    # The actuator reaches Z42 with radius 16.
    assert np.linalg.norm(np.array([0.090, 0.040, 0.042]) - centre) <= radius
