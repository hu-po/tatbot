"""The wrist gauge's fit (tatbot_session.gauge): one pen tip fixed in the camera's frame and frames whose paper
planes turn by the tool's +-2 deg, as the trips of 2026-10-03 did; the contacts fit to zero, the trips in the air
read their own heights."""
from __future__ import annotations

import numpy as np
import pytest
from tatbot_session import gauge


def _turned(n, deg):
    axis = np.cross(n, [1.0, 0.0, 0.0])
    axis /= np.linalg.norm(axis)
    a = np.radians(deg)
    return n * np.cos(a) + np.cross(axis, n) * np.sin(a) + axis * (axis @ n) * (1 - np.cos(a))


def test_the_fit_puts_the_contacts_at_zero_and_reads_the_air_trips():
    rng = np.random.default_rng(0)
    n0 = np.array([-0.727, 0.243, -0.642])
    n0 /= np.linalg.norm(n0)
    end = np.array([-0.018, 0.007, 0.160])        # the cone's end in the camera frame
    pen = end + np.outer(rng.uniform(0.0, 0.03, 4000), n0) + rng.normal(0.0, 0.0015, (4000, 3)) * [1, 1, 0]
    truth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003, 0.006, 0.010, 0.016]
    frames = []
    for k, h in enumerate(truth):
        n = _turned(n0, 2.0 if k % 2 else -2.0)
        lowest = pen[np.argmin(pen @ n0)]
        c = lowest - (h + 0.0025) * n + rng.normal(0.0, 0.00005, 3)   # the cone's end 2.5 mm over a contact
        frames.append(gauge.Frame(pen + rng.normal(0.0, 0.0002, pen.shape), c, n, 0.0004))
    cal, heights = gauge.fit(frames)
    assert cal["contacts"] == 6
    np.testing.assert_allclose(heights, truth, atol=4e-4)
    assert gauge.height(frames[8], cal) == pytest.approx(0.010, abs=4e-4)


def test_a_fitted_gauge_finds_a_pen_of_any_colour_on_its_axis():
    h, w, f = 480, 640, 400.0
    v, u = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 0.200)                          # the paper, square to the camera
    pen = (np.abs(u - 320) <= 20) & (v >= 240) & (v <= 400)
    z[pen] = 0.12 + (v[pen] - 240) * 0.0002             # a pink pen leaning toward the paper
    colour = np.full((h, w, 3), 230, np.uint8)
    colour[pen] = (140, 65, 235)
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 320.0, "ppy": 240.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]}}
    raw = np.round(z / 0.0001).astype(np.uint16)
    pts = np.stack([(u[pen] - 320) / f * z[pen], (v[pen] - 240) / f * z[pen], z[pen]], axis=1)
    m, a = gauge._axis(pts)
    a = a if a[2] > 0 else -a
    cal = {"cone_end_cam_m": (m + np.percentile((pts - m) @ a, 99.5) * a).tolist(), "axis_cam": a.tolist()}
    assert gauge.measure(raw, colour, meta) is None     # not teal: the colour finds no pen
    frame = gauge.measure(raw, colour, meta, cal)
    assert frame is not None and len(frame.pen) >= 100
    np.testing.assert_allclose(frame.n, [0.0, 0.0, -1.0], atol=1e-3)
    assert frame.c[2] == pytest.approx(0.200, abs=2e-4)
    assert np.max(gauge._along(frame.pen, cal)[1]) < gauge.AXIS_M


def test_pink_silicone_is_a_surface_beside_a_print_standing_off_it():
    h, w, f = 480, 640, 400.0
    v, u = np.mgrid[0:h, 0:w]
    z = np.full((h, w), 0.200)
    pen = (np.abs(u - 320) <= 20) & (v >= 240) & (v <= 400)
    z[pen] = 0.12 + (v[pen] - 240) * 0.0002
    colour = np.full((h, w, 3), (166, 181, 232), np.uint8)   # practice skin's peach (BGR), S ~73
    colour[pen] = (200, 160, 30)                             # the teal cartridge
    stripe = (u < 100) & ~pen
    z[stripe] = 0.190                                        # a violet print 10 mm proud of the skin...
    colour[stripe] = (200, 40, 120)                          # ...too saturated to be surface
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 320.0, "ppy": 240.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]}}
    frame = gauge.measure(np.round(z / 0.0001).astype(np.uint16), colour, meta)
    assert frame is not None and frame.c[2] == pytest.approx(0.200, abs=2e-4)


def test_only_what_the_depth_puts_on_the_paper_is_paper():
    h, w, f = 120, 160, 100.0
    z = np.full((h, w), 0.200)
    z[40:60, 70:90] = 0.150                              # a black pen 50 mm over the paper
    z[80:90, 20:30] = 0.0                                # and a hole in the depth
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 80.0, "ppy": 60.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 80.0], [0.0, f, 60.0], [0.0, 0.0, 1.0]]}}
    paper = gauge.Frame(np.zeros((1, 3)), np.array([0.0, 0.0, 0.200]), np.array([0.0, 0.0, -1.0]), 0.0)
    seen = gauge.on_paper(np.round(z / 0.0001).astype(np.uint16), meta, paper, (h, w, 3), close_px=3, trim_px=3)
    assert seen[10, 10] and seen[50, 110] and not seen[50, 80] and not seen[40, 70] and not seen[85, 25]


def _forearm_frame(radius_m: float | None, mat_m: float | None = None):
    """A depth frame of skin 150 mm from the camera, flat or a forearm of `radius_m` (its axis along the image's v,
    its crest at u = 320), and a pen leaning in from above whose tip stands 12 mm over the crest; with `mat_m`, a
    mat that far under the crest past the skin's edges (a raised holder)."""
    h, w, f = 480, 640, 400.0
    v, u = np.mgrid[0:h, 0:w]
    dx = (u - 320.0) / f
    z = np.full((h, w), 0.150)
    if radius_m is not None:
        for _ in range(8):   # the ray (dx z, dy z, z) on the cylinder z = z0 + R - sqrt(R^2 - x^2)
            z = 0.150 + radius_m - np.sqrt(np.maximum(radius_m ** 2 - (dx * z) ** 2, 0.0))
        z[np.abs(dx * z) > 0.95 * radius_m] = 0.0 if mat_m is None else 0.150 + mat_m
    tip, back = np.array([0.0, 0.0, 0.138]), np.array([0.0, -0.03, 0.095])
    for s in np.linspace(0.0, 1.0, 400):   # the pen: discs of 2 mm along its axis, nearest wins
        p = tip + s * (back - tip)
        pu, pv, rad = 320.0 + f * p[0] / p[2], 240.0 + f * p[1] / p[2], f * 0.002 / p[2]
        disc = (u - pu) ** 2 + (v - pv) ** 2 <= rad ** 2
        z[disc] = np.where((z[disc] == 0) | (z[disc] > p[2]), p[2], z[disc])
    colour = np.full((h, w, 3), 210, np.uint8)
    meta = {"depth_units_m": 0.0001, "intrinsics": {"fx": f, "fy": f, "ppx": 320.0, "ppy": 240.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]},
            "color_intrinsics": {"k": [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]}}
    axis = (tip - back) / np.linalg.norm(tip - back)
    cal = {"cone_end_cam_m": (tip - 0.001 * axis).tolist(), "axis_cam": axis.tolist(), "tip_cam_m": tip.tolist(),
           "offset_m": 0.0}
    return np.round(z / 0.0001).astype(np.uint16), colour, meta, cal


def test_on_a_forearm_the_curved_gauge_reads_the_skin_under_the_tip_where_a_plane_misses_it():
    """One plane through the gauge's skin on a 40 mm-radius forearm sits under the skin at the tip and puts the pen
    too high over it, so a ride would go too deep; the quadric over the same skin reads the 12 mm there."""
    raw, colour, meta, cal = _forearm_frame(0.040)
    plane = gauge.measure(raw, colour, meta, cal)
    local = gauge.measure(raw, colour, meta, cal, curved=True)
    assert plane is not None and local is not None and len(local.patch) > gauge.CURVED_MIN_POINTS
    assert gauge.height(plane, cal) - 0.012 > 0.002                      # the plane: 2.5 mm too far (a ride too deep)
    # a quadric over +-40 mm of a 40 mm radius misses its quartic: 0.2 mm short, the pen that much shallower
    assert -0.00025 < gauge.height(local, cal) - 0.012 < 0.0
    np.testing.assert_allclose(local.n, [0.0, 0.0, -1.0], atol=0.01)     # the crest's normal, toward the camera


def test_on_a_flat_page_the_curved_gauge_reads_what_the_plane_does():
    raw, colour, meta, cal = _forearm_frame(None)
    plane = gauge.measure(raw, colour, meta, cal)
    local = gauge.measure(raw, colour, meta, cal, curved=True)
    assert plane.patch is None and gauge.height(plane, cal) == pytest.approx(0.012, abs=1e-4)
    assert gauge.height(local, cal) == pytest.approx(gauge.height(plane, cal), abs=5e-5)


def test_on_a_raised_holder_the_curved_gauge_reads_the_skin_not_the_mat_beside_it():
    """A skin pinned over a holder of 50 mm radius whose crest stands 50 mm over the mat (2026-10-08): the mat fills
    the image box past the skin's edges, and window glare left half the skin's pixels unread: then most of the box
    was mat, and a fit over all of it read the mat (62 mm here). Only the surface within SUPPORT_M of the tip is read."""
    raw, colour, meta, cal = _forearm_frame(0.050, mat_m=0.050)
    skin = (raw >= 1490) & (raw < 1900)                  # nearer than the mat, past the pen
    raw[skin & (np.random.default_rng(0).random(raw.shape) < 0.5)] = 0
    local = gauge.measure(raw, colour, meta, cal, curved=True)
    assert local is not None and gauge.height(local, cal) == pytest.approx(0.012, abs=2.5e-4)
    np.testing.assert_allclose(local.n, [0.0, 0.0, -1.0], atol=0.01)


def test_a_small_top_square_to_the_pen_is_read_along_it_and_a_wall_along_the_pen_is_not():
    """The probe body's top under the upright pen, 3.5 mm inside its edge (2026-10-22): from the sides where the pen
    hides the top, most of what lies near its axis is the body's wall, which runs along the pen; a plane through it
    read the top up to 13 mm low. The top is the densest band along the axis; a wall-filled support reads nothing."""
    rng = np.random.default_rng(1)
    axis = np.array([0.3, -0.2, 0.93])
    axis /= np.linalg.norm(axis)
    tip = np.array([-0.011, 0.006, 0.147])
    cal = {"tip_cam_m": tip.tolist(), "axis_cam": axis.tolist(), "offset_m": 0.0}
    side = np.cross(axis, [1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    disc = rng.uniform(-0.006, 0.006, (3000, 2))
    disc = disc[np.hypot(*disc.T) < 0.006]
    top = tip + 0.013 * axis + disc[:, :1] * side + disc[:, 1:] * other + rng.normal(0.0, 0.0003, (len(disc), 3))
    wall = (tip + 0.0035 * side + np.outer(rng.uniform(0.013, 0.040, 1500), axis)
            + np.outer(rng.uniform(-0.004, 0.004, 1500), other))
    c, n, spread = gauge._square(np.vstack([top, wall]), cal)
    assert gauge.height(gauge.Frame(top, c, n, spread), cal) == pytest.approx(0.013, abs=3e-4)
    np.testing.assert_allclose(n, -axis)
    assert gauge._square(np.vstack([top[:300], wall]), cal) is None


def test_pooled_depth_is_each_pixels_median_where_most_frames_read_it():
    frames = [np.array([[1000, 0, 2000]], np.uint16), np.array([[1002, 0, 0]], np.uint16),
              np.array([[1100, 1500, 0]], np.uint16), np.array([[1001, 0, 0]], np.uint16),
              np.array([[999, 1500, 2000]], np.uint16)]
    np.testing.assert_array_equal(gauge.pool(frames), [[1001, 0, 0]])   # a pixel read in 2 of 5 is a hole
    np.testing.assert_array_equal(gauge.pool(frames, need=0.4), [[1001, 1500, 2000]])


def test_a_pooled_reading_keeps_every_colour_frame_for_the_prints_markers():
    """A curved hold's frames read in pooled fives: each reading the first colour of its five and their pooled depth,
    and every colour frame kept beside them, which the print's markers are read in (inspect.hold_tip `every`)."""
    colours = [np.full((1, 3, 3), k, np.uint8) for k in range(10)]
    depths = [np.array([[1000 + k, 0, 2000]], np.uint16) for k in range(10)]
    shots = gauge.WristGauge._pooled((colours, depths, {"m": 1}), 5)
    assert [int(c[0, 0, 0]) for c in shots[0]] == [0, 5] and len(shots[1]) == 2 and shots[2] == {"m": 1}
    assert [int(c[0, 0, 0]) for c in shots[3]] == list(range(10))
    flat = (colours[:5], depths[:5], {})
    assert gauge.WristGauge._pooled(flat, 1) is flat                       # unpooled: the frames as taken


# Monk Skin Tone scale swatches 1-10 (sRGB), the scale's published hex values.
MONK = ["f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96", "a07e56", "825c43", "604134", "3a312a", "292420"]


def test_the_surface_colour_takes_human_skin_and_a_violet_transfer_on_it():
    """The gauge's surface is not silicone's alone: Monk tones 1-6 in neutral and warm (3000 K-ish) light, and a
    half-strength violet stencil transfer on any of tones 1-9, read as surface; the teal pen body and tone 10 (too dark
    for the value floor) do not."""
    import cv2

    def hsv(rgb):
        return cv2.cvtColor(np.uint8([[np.clip(rgb, 0, 255)[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]

    violet = np.array([110.0, 60.0, 140.0])
    for i, swatch in enumerate(MONK, 1):
        rgb = np.array([int(swatch[j:j + 2], 16) for j in (0, 2, 4)], float)
        for gain in ((1.0, 1.0, 1.0), (1.08, 1.0, 0.85)):
            lit = rgb * gain
            if i <= 6:
                assert gauge.surface_colour(hsv(lit)), (i, gain)
            if i <= 9:
                assert gauge.surface_colour(hsv(0.5 * lit + 0.5 * violet)), (i, gain)
    assert not gauge.surface_colour(hsv(np.array([int(MONK[9][j:j + 2], 16) for j in (0, 2, 4)], float)))
    assert not gauge.surface_colour(hsv(np.array([20.0, 170.0, 160.0])))   # the ballpoint's teal body
