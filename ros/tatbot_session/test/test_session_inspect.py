"""The wrist-camera inspect geometry: aiming, projection and page rectification."""
import numpy as np
import pytest
from tatbot_session import geometry as g
from tatbot_session import inspect as ins

K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])
PAGE = g.rpy_matrix([0.33, 0.06, 0.044], [0.02, -0.01, -1.5])


def test_look_down_is_a_rotation_looking_down_the_page_normal():
    cam = ins.look_down(PAGE, (0.01, -0.02), 0.09)
    assert np.allclose(cam[:3, :3].T @ cam[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(cam[:3, :3]), 1.0)
    assert np.allclose(cam[:3, 2], -PAGE[:3, 2]) and np.allclose(cam[:3, 0], PAGE[:3, 1])
    under = ins.page_to_pixel(np.array([[0.01, -0.02]]), PAGE, cam, K, np.zeros(5))
    assert np.allclose(under, [[640.0, 360.0]])
    # 10 mm along page y moves 640 * 10 / 90 px along image x
    along = ins.page_to_pixel(np.array([[0.01, -0.01]]), PAGE, cam, K, np.zeros(5))
    assert np.allclose(along - under, [[640.0 * 0.01 / 0.09, 0.0]], atol=1e-6)


def test_tcp_target_puts_the_camera_where_asked():
    tcp_from_cam = g.rpy_matrix([0.01, 0.0, -0.08], [0.0, 0.35, 0.0])
    cam = ins.look_down(PAGE, (0.0, 0.0), 0.09)
    assert np.allclose(ins.tcp_target(cam, tcp_from_cam) @ tcp_from_cam, cam)


def test_rectify_maps_a_camera_image_back_onto_the_page():
    pytest.importorskip("cv2")
    cam = ins.look_down(PAGE, (0.0, 0.0), 0.09)
    # a camera image of a page with one bright dot at page (5, 20) mm
    image = np.zeros((720, 1280), np.uint8)
    u, v = ins.page_to_pixel(np.array([[0.005, 0.020]]), PAGE, cam, K, np.zeros(5))[0]
    image[int(round(v)) - 2:int(round(v)) + 3, int(round(u)) - 2:int(round(u)) + 3] = 255
    extent = (-0.041, -0.066, 0.041, 0.066)
    page = ins.rectify(image, PAGE, cam, K, np.zeros(5), extent, 10.0)
    assert page.shape == (1320, 820)
    row, col = np.unravel_index(np.argmax(page), page.shape)
    x, y = ins.to_image_px([[0.005, 0.020]], extent, 10.0)[0]
    assert abs(col - x) <= 3 and abs(row - y) <= 3


def test_look_at_tilts_toward_the_azimuth_and_keeps_looking_at_the_point():
    cam = ins.look_at(PAGE, (0.0, 0.01), tilt_rad=np.radians(45), azimuth_rad=np.radians(90), distance_m=0.15)
    assert np.allclose(cam[:3, :3].T @ cam[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(cam[:3, :3]), 1.0)
    target = (PAGE @ [0.0, 0.01, 0.0, 1.0])[:3]
    assert np.allclose(cam[:3, 3] + 0.15 * cam[:3, 2], target)
    assert np.isclose(-cam[:3, 2] @ PAGE[:3, 2], np.cos(np.radians(45)))
    assert np.allclose(ins.page_to_pixel(np.array([[0.0, 0.01]]), PAGE, cam, K, np.zeros(5)), [[640.0, 360.0]])


def _page(extent, ppm, clear, *, shift=(0.0, 0.0), theta=0.0):
    """A synthetic page image: white paper, a dark 14 mm border band around the clear centre, moved by
    shift and turned by theta (the print's true pose in the believed page frame)."""
    gx, gy = ins.page_grid(extent, ppm)
    c, s = np.cos(-theta), np.sin(-theta)
    x, y = c * (gx - shift[0]) - s * (gy - shift[1]), s * (gx - shift[0]) + c * (gy - shift[1])
    hx, hy = clear[0] / 2, clear[1] / 2
    band = (np.abs(x) < hx + 0.014) & (np.abs(y) < hy + 0.014) & ~((np.abs(x) < hx) & (np.abs(y) < hy))
    img = np.full(gx.shape, 220, np.uint8)
    img[band & ((np.round(x * 2000) + np.round(y * 2000)) % 3 != 0)] = 30
    return img


def test_border_edges_find_a_shifted_turned_print():
    pytest.importorskip("cv2")
    extent, ppm, clear = (-0.060, -0.085, 0.060, 0.085), 5.0, (0.062, 0.112)
    img = _page(extent, ppm, clear, shift=(0.004, 0.009), theta=np.radians(1.5))
    sol = ins.solve_in_plane(ins.border_edges(img, extent, ppm, clear), clear)
    assert sol is not None and len(sol["sides"]) == 4
    # the correction maps believed coordinates onto the print: t = -R shift
    assert abs(sol["tx_m"] + 0.004) < 0.0006 and abs(sol["ty_m"] + 0.009) < 0.0006
    assert abs(sol["theta_rad"] + np.radians(1.5)) < np.radians(0.4)


CLEAR = (0.062, 0.112)
LATTICE_M, KNOT_M, LINE_M = 0.0065, 0.0011, 0.0006   # a coded print: 6.5 mm hex lattice, 2.2 mm knots
PHASE_M = (0.000875, 0.0)   # a junction's offset from the page centre: the default coded design's phase
CODED_INNER = {"left": -0.030525, "right": 0.032275, "bottom": -0.055192, "top": 0.055192}   # its innermost ink


def _coded_lattice():
    """Knot centres and lattice lines (page frame, m) of a coded-like border: the junctions of a hex
    lattice in the 14 mm frame at least 0.2 mm outside the clear centre, as the generator places them,
    so the knots on the innermost ones reach past its edge by up to their radius: 0.5 mm in at the
    left, 1.3 mm short at the right, 0.8 mm in at the bottom and top (CODED_INNER)."""
    q, r = (v.ravel() for v in np.meshgrid(np.arange(-20, 21), np.arange(-20, 21)))
    pts = np.c_[PHASE_M[0] + LATTICE_M * (q + r / 2), PHASE_M[1] + LATTICE_M * r * np.sqrt(3) / 2]

    def in_band(p):
        inner = (np.abs(p[..., 0]) < CLEAR[0] / 2 + 0.0002) & (np.abs(p[..., 1]) < CLEAR[1] / 2 + 0.0002)
        return ~inner & (np.abs(p[..., 0]) <= CLEAR[0] / 2 + 0.0138) & (np.abs(p[..., 1]) <= CLEAR[1] / 2 + 0.0138)

    knots = pts[in_band(pts)]
    pairs = [(a, b) for i, a in enumerate(knots) for b in knots[i + 1:] if np.hypot(*(a - b)) < LATTICE_M * 1.01]
    lines = [(a, b) for a, b in pairs if in_band(a + (b - a) * np.linspace(0, 1, 9)[:, None]).all()]
    return knots, lines


def _coded_page(extent, ppm, *, shift=(0.0, 0.0), theta=0.0, scale=1.0, ink=30, mat=None):
    """A synthetic page image of the coded-like border (anti-aliased), the print moved by shift and turned
    by theta as in _page, and imaged `scale` times its size. With `mat` (a grey), the 100 x 150 mm page
    lies on a mat of that grey."""
    import cv2

    img = np.full(ins.page_grid(extent, ppm)[0].shape, 220 if mat is None else mat, np.uint8)
    c, s = scale * np.cos(theta), scale * np.sin(theta)

    def px(p):   # print frame -> image pixels, 4 fractional bits
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        believed = np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]] + shift
        return np.round(ins.to_image_px(believed, extent, ppm) * 16).astype(np.int32)

    if mat is not None:
        cv2.fillPoly(img, [px([[-0.05, -0.075], [0.05, -0.075], [0.05, 0.075], [-0.05, 0.075]])], 220, cv2.LINE_AA,
                     shift=4)
    knots, lines = _coded_lattice()
    for a, b in lines:
        (p0, p1) = px([a, b])
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), ink, int(round(LINE_M * 1000 * ppm)),
                 cv2.LINE_AA, shift=4)
    for k in px(knots):
        cv2.circle(img, tuple(int(v) for v in k), int(round(scale * KNOT_M * 1000 * ppm * 16)), ink, -1, cv2.LINE_AA,
                   shift=4)
    return img


def _truth(shift, theta, scale=1.0):
    """The correction _coded_page's print needs: believed b = scale R(theta) p + shift, so p = S R(-theta) b
    + t with t = -R(-theta) shift / scale."""
    c, s = np.cos(theta), np.sin(theta)
    return -np.array([c * shift[0] + s * shift[1], -s * shift[0] + c * shift[1]]) / scale


def test_border_fit_takes_each_side_about_the_prints_own_inner_edge():
    """A coded print's knots reach past the clear centre's edge by a different amount on each side.
    Fitted against +-clear_m / 2 the print comes out ~0.9 mm off in x with its x scale 2 % off, on every
    print; against its own inner edges (settings.json border_inner_mm) it is right."""
    pytest.importorskip("cv2")
    extent, ppm = ins.PAGE_EXTENT, 5.0
    for shift, theta in (((0.0, 0.0), 0.0), ((0.004, 0.009), np.radians(1.5)), ((-0.003, 0.002), np.radians(-0.8))):
        img = _coded_page(extent, ppm, shift=shift, theta=theta)
        truth = _truth(shift, theta)
        edges = ins.border_edges(img, extent, ppm, CLEAR, inner_m=CODED_INNER)
        sol = ins.solve_in_plane(edges, CLEAR, inner_m=CODED_INNER)
        assert sol is not None and len(sol["sides"]) == 4
        assert abs(sol["tx_m"] - truth[0]) < 0.0003 and abs(sol["ty_m"] - truth[1]) < 0.0003
        assert abs(sol["theta_rad"] + theta) < np.radians(0.2)
        legacy = ins.solve_in_plane(ins.border_edges(img, extent, ppm, CLEAR), CLEAR)
        assert legacy is not None and abs(legacy["tx_m"] - truth[0]) > 0.0006 and abs(legacy["scale"][0] - 1) > 0.015


def test_artwork_fit_places_a_print_whose_edges_do_not_fit():
    """A transfer with two sides solid and two wiped off: the edge fit refuses it, the artwork's frame
    still places it within half its 2 mm step and turns it within its sigma; a page with no print on it
    matches nothing."""
    pytest.importorskip("cv2")
    extent, ppm, size = ins.PAGE_EXTENT, 5.0, ins.PAGE_SIZE
    artwork = _coded_page((-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2), 10.0)
    for shift, theta in (((-0.005, 0.006), np.radians(1.0)), ((0.004, -0.003), np.radians(-2.5))):
        img = _coded_page(extent, ppm, shift=shift, theta=theta)
        gx, gy = ins.page_grid(extent, ppm)
        img[(gx < shift[0] - CLEAR[0] / 4) | (gy < shift[1] - CLEAR[1] / 4)] = 220   # left and bottom: blank skin
        truth = _truth(shift, theta)
        assert ins.solve_in_plane(ins.border_edges(img, extent, ppm, CLEAR, inner_m=CODED_INNER), CLEAR,
                                  inner_m=CODED_INNER) is None
        sol = ins.artwork_fit(img, artwork, extent, ppm, size, CLEAR)
        assert sol is not None and sol["sides"] == ["artwork"]
        assert abs(sol["tx_m"] - truth[0]) < 0.001 and abs(sol["ty_m"] - truth[1]) < 0.001
        assert abs(sol["theta_rad"] + theta) < sol["sigma"][2]
    assert ins.artwork_fit(np.full_like(img, 220), artwork, extent, ppm, size, CLEAR) is None


def test_hold_tip_reads_the_tip_on_the_print_whatever_page_seeds_it():
    """A gauge hold's wrist frame, 13 mm over a coded print seen 20 deg off its normal: the tip's place on the print
    comes from the camera's own geometry (the gauge's surface, the print's artwork, the gauge's tip point), the
    same within half a millimetre from a believed page 4 mm and 1 deg off it or from the print itself."""
    import cv2
    from tatbot_session import gauge

    size = ins.PAGE_SIZE
    artwork = _coded_page((-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2), 10.0)
    sheet_extent, sheet_ppm = (-0.09, -0.12, 0.09, 0.12), 10.0
    sheet = _coded_page(sheet_extent, sheet_ppm)
    cam_from_print = np.linalg.inv(ins.look_at(np.eye(4), (0.005, 0.0), tilt_rad=np.radians(20),
                                               azimuth_rad=np.radians(200), distance_m=0.13))
    v, u = np.mgrid[0:720, 0:1280].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    print_from_cam = np.linalg.inv(cam_from_print)
    o, d = print_from_cam[:3, 3], rays @ print_from_cam[:3, :3].T
    hit = o[:2] + (-o[2] / d[..., 2])[..., None] * d[..., :2]   # each pixel's point on the print (m)
    px = ins.to_image_px(hit.reshape(-1, 2), sheet_extent, sheet_ppm).reshape(720, 1280, 2).astype(np.float32) - 0.5
    image = cv2.remap(sheet, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderValue=220)
    surface = gauge.Frame(np.zeros((0, 3)), cam_from_print[:3, 3], cam_from_print[:3, 2], 0.0)
    meta = {"color_intrinsics": {"k": K.tolist(), "coeffs": [0.0] * 5}}
    tip = np.array([0.004, -0.007, 0.013])
    tip_cam = (cam_from_print @ [*tip, 1.0])[:3]
    page = {"size_m": size, "clear_m": CLEAR}
    for off in (ins.correction_matrix(0.0, 0.0, 0.0), ins.correction_matrix(0.003, -0.004, np.radians(1.0))):
        found = ins.hold_tip([np.dstack([image] * 3)], [surface], meta, tip_cam, cam_from_print @ off, page, artwork)
        assert found is not None and np.allclose(found["tip_m"], tip, atol=0.0005)
    assert ins.hold_tip([np.dstack([image] * 3)], [None], meta, tip_cam, cam_from_print, page, artwork) is None


def test_holds_shift_is_the_median_of_where_the_tip_stood_less_where_it_was_sent():
    """Believed + shift = print; a hold whose fit jumped a band is dropped; fewer than three agreeing give none."""
    sent = [(-0.0175, -0.02), (0.0175, -0.02), (0.0175, 0.02), (0.0, 0.0), (-0.0175, 0.02)]
    off = [(-0.0016, -0.0035), (-0.0014, -0.0037), (-0.0015, -0.0036), (-0.0015, -0.0036), (-0.0015, 0.020)]
    holds = [{"xy": xy, "tip_m": [xy[0] + dx, xy[1] + dy, 0.012]} for xy, (dx, dy) in zip(sent, off, strict=True)]
    found = ins.holds_shift(holds)
    assert np.allclose(found["shift_m"], [-0.0015, -0.0036]) and (found["holds"], found["dropped"]) == (4, 1)
    assert ins.holds_shift(holds[:2]) is None
    assert ins.holds_shift(holds[:2] + holds[4:]) is None   # two agree, the third is a band off


def test_border_scan_stops_short_of_the_page_edge():
    """A border too faint to find, on a page imaged a few % small on a dark mat: scanning out to search_m
    the mat beyond the page's edge passed for the border on three sides (a fit 15 mm off); bounded by
    the page's margin and frame the scan finds nothing and the fit refuses. A border it can see on the
    same mat still fits."""
    pytest.importorskip("cv2")
    extent, ppm = ins.PAGE_EXTENT, 5.0

    def fit(img, **kw):
        return ins.solve_in_plane(ins.border_edges(img, extent, ppm, CLEAR, inner_m=CODED_INNER, **kw), CLEAR,
                                  inner_m=CODED_INNER)

    assert fit(_coded_page(extent, ppm, scale=0.96, ink=180, mat=60)) is None
    scale, shift, theta = 0.97, (0.003, 0.0), 0.02
    truth = _truth(shift, theta, scale)
    faint = _coded_page(extent, ppm, shift=shift, theta=theta, scale=scale, ink=180, mat=60)
    assert fit(faint) is None
    unbounded = fit(faint, size_m=None)
    assert unbounded is not None and abs(unbounded["tx_m"] - truth[0]) > 0.010   # what the bound stops
    sol = fit(_coded_page(extent, ppm, shift=shift, theta=theta, scale=scale, mat=60))
    assert sol is not None and abs(sol["tx_m"] - truth[0]) < 0.0003 and abs(sol["ty_m"] - truth[1]) < 0.0003


def test_solve_in_plane_scales_about_the_aim_point():
    """A view aimed off the page centre images the print a few % large about the point it aims at: the fit
    returns the print's own offset when its scale is about that point, and (1 - s) c more about the page
    centre."""
    aim, k, t = np.array([-0.015, 0.037]), 1.03, np.array([0.0012, -0.0008])   # image scale k, the print's offset t
    edges = []
    for side, (axis, _) in ins.SIDES.items():
        for a in np.linspace(-0.02, 0.02, 6):
            q = np.zeros(2)
            q[axis], q[1 - axis] = CODED_INNER[side], a    # on the print's inner edge
            p = aim + k * (q - t - aim)                    # where the view shows it
            edges.append({"side": side, "axis": axis, "edge_m": float(p[axis]), "along_m": float(p[1 - axis])})
    sol = ins.solve_in_plane(edges, CLEAR, inner_m=CODED_INNER, centre_m=aim)
    assert sol is not None  # solve_in_plane returns a fit dict for valid edges
    assert np.allclose([sol["tx_m"], sol["ty_m"], sol["theta_rad"]], [t[0], t[1], 0.0], atol=1e-9)
    assert np.allclose(sol["scale"], 1 / k)
    off = ins.solve_in_plane(edges, CLEAR, inner_m=CODED_INNER)
    assert off is not None  # solve_in_plane returns a fit dict for valid edges
    assert np.allclose([off["tx_m"], off["ty_m"]], t + (1 - 1 / k) * aim, atol=1e-9)


def test_a_held_scale_lets_a_stray_far_side_point_be_rejected():
    """A locate view saw the top border as one point at the pen cone, 7.8 mm inside the print's edge; with
    the y scale free the fit matched it exactly and moved the page ~4 mm in y (2026-09-28)."""
    t = np.array([0.0006, -0.0125])
    edges = []
    for side in ("left", "right", "bottom"):
        axis = ins.SIDES[side][0]
        for a in np.linspace(-0.02, 0.02, 7):
            q = np.zeros(2)
            q[axis], q[1 - axis] = CODED_INNER[side] - t[axis], a
            edges.append({"side": side, "axis": axis, "edge_m": float(q[axis]), "along_m": float(q[1 - axis])})
    edges.append({"side": "top", "axis": 1, "edge_m": 0.0474, "along_m": 0.0})
    free = ins.solve_in_plane(edges, CLEAR, inner_m=CODED_INNER)
    held = ins.solve_in_plane(edges, CLEAR, inner_m=CODED_INNER, free_scale=(True, False))
    assert free is not None and held is not None  # solve_in_plane returns a fit dict for valid edges
    assert abs(free["ty_m"] - t[1]) > 0.002 and free["scale"][1] != 1.0
    assert np.allclose([held["tx_m"], held["ty_m"]], t, atol=1e-9) and held["scale"][1] == 1.0
    assert held["points"] == len(edges) - 1     # the stray point is rejected, not fitted


def test_inner_edges_default_to_the_clear_centre_and_refuse_a_misread_key():
    assert ins.inner_edges(CLEAR) == {"left": -0.031, "right": 0.031, "bottom": -0.056, "top": 0.056}
    assert ins.inner_edges(CLEAR, CODED_INNER) == CODED_INNER
    for bad in ([-0.0305, 0.0322, -0.0551, 0.0552],                                # a list: which side is which
                {"left": 19.473, "right": 82.211, "bottom": 130.133, "top": 19.812},   # settings.json mm, unconverted
                {"left": -0.0305, "right": 0.0322, "bottom": 0.0551, "top": -0.0552},  # y down
                {"left": -0.0305, "right": 0.0322, "top": 0.0552}):
        with pytest.raises(ValueError):
            ins.inner_edges(CLEAR, bad)


def test_fuse_weights_by_inverse_variance():
    f = ins.fuse([((0.0, 0.0, 0.0), (0.004, 0.004, 0.02)), ((0.002, -0.004, 0.01), (0.002, 0.002, 0.02))])
    assert abs(f["tx_m"] - 0.0016) < 1e-9 and abs(f["ty_m"] + 0.0032) < 1e-9 and abs(f["theta_rad"] - 0.005) < 1e-9
    assert f["sigma"][0] < 0.002


def test_ink_alignment_finds_the_shift_and_coverage():
    pytest.importorskip("cv2")
    import cv2

    extent, ppm, clear = (-0.041, -0.066, 0.041, 0.066), 10.0, (0.062, 0.112)
    program = {"ops": [{"op": "stroke", "points_m": [[-0.01, 0.0], [0.01, 0.0], [0.01, 0.015]]}]}
    img = np.full(ins.page_grid(extent, ppm)[0].shape, 220, np.uint8)
    drawn = ins.to_image_px(np.array([[-0.01, 0.0], [0.01, 0.0]]) + [0.003, -0.002], extent, ppm)   # half drawn
    cv2.polylines(img, [np.round(drawn).astype(np.int32)], False, 30, 5)
    fit = ins.ink_alignment(img, program, extent, ppm, clear)
    assert fit is not None and abs(fit["dx_m"] - 0.003) < 0.0006 and abs(fit["dy_m"] + 0.002) < 0.0006
    assert 0.45 < fit["coverage"] < 0.7


def test_page_height_from_depth_finds_the_plane_under_clutter():
    rng = np.random.default_rng(3)
    page = g.rpy_matrix([0.30, 0.02, 0.040], [0.02, -0.01, 0.3])
    xy = rng.uniform([-0.05, -0.075], [0.05, 0.075], size=(4000, 2))
    paper = np.c_[xy, np.full(len(xy), 0.0012) + rng.normal(0, 0.0002, len(xy))]   # 1.2 mm over the prior
    cone = np.c_[rng.uniform(-0.02, 0.0, (300, 2)), rng.uniform(0.004, 0.012, 300)]   # the pen cone
    local = np.vstack([paper, cone])
    base = (page @ np.c_[local, np.ones(len(local))].T)[:3].T
    fit = ins.page_height_from_depth(base, page)
    assert fit is not None and abs(fit["offset_m"] - 0.0012) < 0.0002 and fit["rms_m"] < 0.0005


def test_depth_observation_distinguishes_a_missing_patch_from_a_wrong_height_prior():
    rng = np.random.default_rng(7)
    xy = rng.uniform([-.04, -.06], [.04, .06], (1000, 2))
    pts = np.c_[xy, np.full(len(xy), .035)]
    observation = {}
    assert ins.page_height_from_depth(pts, np.eye(4), diagnostics=observation) is None
    assert observation["reason"] == "depth_outside_prior_band"
    assert observation["page_xy_points"] == 1000 and observation["prior_band_points"] == 0
    assert np.allclose(observation["page_xy_height_quantiles_m"], [.035] * 3)
    # The saved observation exposes a plane excluded by the prior; the default fit stays unchanged.
    fit = ins.page_height_from_depth(pts, np.eye(4), band_m=.04)
    assert fit is not None and fit["offset_m"] == pytest.approx(.035)
    pts[:, 0] += .20
    assert ins.page_height_from_depth(pts, np.eye(4), diagnostics=observation) is None
    assert observation["reason"] == "insufficient_page_xy_depth"
    assert observation["page_xy_points"] == 0 and observation["page_xy_height_quantiles_m"] is None


def test_depth_observation_reports_unsupported_plane_and_empty_depth():
    rng = np.random.default_rng(9)
    xy = rng.uniform([-.01, -.01], [.01, .01], (1000, 2))
    observation = {}
    # A 45 degree plane has plenty of in-band points but disagrees with the expected page normal.
    assert ins.page_height_from_depth(np.c_[xy, xy[:, 0]], np.eye(4), diagnostics=observation) is None
    assert observation["reason"] == "no_supported_plane" and observation["prior_band_points"] == 1000
    assert ins.page_height_from_depth(np.empty((0, 3)), np.eye(4), diagnostics=observation) is None
    assert observation["reason"] == "insufficient_page_xy_depth" and observation["points"] == 0


def test_fuse_heights_weights_and_warns_without_dropping():
    f = ins.fuse_heights([("overhead", 0.0, 0.002), ("wrist depth", -0.003, 0.0015), ("touch 0", -0.0035, 0.002),
                          ("touch 1", -0.009, 0.002)], warn_m=0.002)
    assert -0.006 < f["offset_m"] < -0.002 and len(f["sources"]) == 4
    assert any(w.startswith("touch 1") for w in f["warn"])


def test_ink_alignment_reports_no_ink_when_the_plan_cannot_reach_it():
    pytest.importorskip("cv2")
    import cv2

    extent, ppm, clear = (-0.041, -0.066, 0.041, 0.066), 10.0, (0.062, 0.112)
    program = {"ops": [{"op": "stroke", "points_m": [[-0.01, 0.0], [0.01, 0.0]]}]}
    img = np.full(ins.page_grid(extent, ppm)[0].shape, 220, np.uint8)
    cv2.circle(img, tuple(np.round(ins.to_image_px([[0.02, 0.04]], extent, ppm)[0]).astype(int)), 6, 30, -1)   # a touch dot
    fit = ins.ink_alignment(img, program, extent, ppm, clear)
    assert fit is None or fit["found"] is False


def test_ink_alignment_ignores_a_neighbouring_drawing():
    """Two small drawings side by side: each is scored against its own ink only."""
    pytest.importorskip("cv2")
    import cv2

    extent, ppm, clear = (-0.041, -0.066, 0.041, 0.066), 10.0, (0.062, 0.112)
    plan = np.array([[-0.025, 0.030], [-0.005, 0.030], [-0.005, 0.045]])   # the left slot
    program = {"ops": [{"op": "stroke", "points_m": plan.tolist()}]}
    img = np.full(ins.page_grid(extent, ppm)[0].shape, 220, np.uint8)
    for shift in ([0.002, 0.001], [0.030, 0.0]):   # this drawing, drawn 2 mm off; its neighbour 30 mm right
        cv2.polylines(img, [np.round(ins.to_image_px(plan + shift, extent, ppm)).astype(np.int32)], False, 30, 5)
    fit = ins.ink_alignment(img, program, extent, ppm, clear)
    assert fit is not None and fit["found"]
    assert abs(fit["dx_m"] - 0.002) < 0.0006 and abs(fit["dy_m"] - 0.001) < 0.0006
    assert fit["coverage"] > 0.9 and fit["p95_gap_m"] < 0.001


def test_aim_skips_a_view_that_leaves_a_joint_at_its_limit():
    """The cheapest view put joint_1 on its limit, where the next plan refused to start; aim takes the next."""
    from types import SimpleNamespace

    up = np.eye(4)
    up[2, 3] = 0.1   # every solution holds the pen 100 mm over the page

    kin = SimpleNamespace(arm="right", lower=np.array([-3.0, 0.0, 0.0, -3.0, -3.0, -3.0, 0.0]),
                          upper=np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.04]),
                          fk=lambda q: up, frame=lambda q, f: up)
    at_limit = np.array([0.0, 0.003, 1.0, 0.0, 0.0, 0.0, 0.0])   # joint_1 3 mrad over its lower limit
    clear = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    calls = []

    def solve_ik(kin, target, seed):
        calls.append(1)
        return at_limit.copy() if len(calls) == 1 else clear.copy()

    loose = {"reference": np.zeros(7), "max_pen_tilt_deg": 180.0, "cube_up_m": -1.0}
    got = ins.aim(kin, solve_ik, np.eye(4), (0.0, 0.0), np.zeros(7), frame="cam", clearance_m=0.015,
                  tilts_deg=(20,), distances_m=(0.18,), azimuths=[0.0, 1.0], rolls_deg=range(0, 1), **loose)
    assert got is not None and np.allclose(got[0], clear)
    none = ins.aim(kin, lambda *_: at_limit.copy(), np.eye(4), (0.0, 0.0), np.zeros(7), frame="cam",
                   clearance_m=0.015, tilts_deg=(20,), distances_m=(0.18,), azimuths=[0.0], rolls_deg=range(0, 1), **loose)
    assert none is None


def test_aim_keeps_the_wrist_cube_on_its_side_and_clear_of_the_table():
    """The cheapest post-draw view rolled the wrist 90 deg: the pen stayed 20 mm up but the fiducial cube's tag
    face came 33 mm over the table, 140-180 mm past the camera toward the palette (2026-09-30)."""
    from types import SimpleNamespace

    def pose(z, y=0.0):
        out = np.eye(4)
        out[1, 3], out[2, 3] = y, z
        return out

    # three solutions, cheapest first: the cube low over the table, the cube across the side line, clear
    cube = {1.0: pose(0.033), 2.0: pose(0.150, y=0.200), 3.0: pose(0.150)}

    def frame(q, name):
        if name == "right/wrist_tag4":
            return cube.get(float(q[0]), pose(0.150))
        if name == "right/missing_frame":
            raise ValueError("URDF lacks frame")
        return pose(0.100 if name == "right/tool_mount" else 0.150)

    kin = SimpleNamespace(arm="right", lower=np.full(7, -9.0), upper=np.full(7, 9.0), fk=lambda q: pose(0.020),
                          frame=frame)
    solutions = iter(np.array([k, 0, 0, 0, 0, 0, 0], float) for k in (1.0, 2.0, 3.0))
    got = ins.aim(kin, lambda *_: next(solutions), np.eye(4), (0.0, 0.0), np.zeros(7), frame="cam",
                  clearance_m=0.015, tilts_deg=(30,), distances_m=(0.18,), azimuths=[0.0], rolls_deg=range(0, 181, 90),
                  max_y=0.100, reference=np.zeros(7), max_pen_tilt_deg=180.0)
    assert got is not None and got[0][0] == 3.0
    assert ins.frames_of(kin, np.zeros(7), ("{arm}/wrist_tag4", "{arm}/missing_frame")) == ["right/wrist_tag4"]


def test_aim_keeps_the_pen_down_the_wrist_near_pen_down_and_the_cube_up():
    """Scan views stay upright (operator, 2026-09-30): the pen mostly down, no big wrist turns, the cube up."""
    from types import SimpleNamespace

    def tilted(deg, z=0.150):
        """A pose `deg` off pen-down (tcp z into the page) at height z; the camera rides the tcp here."""
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        out = np.eye(4)
        out[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        out[2, 3] = z
        return out

    def frame(q, name):
        if name == "right/tool_mount":
            return tilted(0.0, 0.100)
        if "wrist_tag" in name and float(q[0]) == 3.0:
            return tilted(0.0, 0.110)             # the cube only 10 mm over the mount: turned over
        return tilted(0.0, 0.150)

    # solutions in cost order: the pen 60 deg off down, a wrist joint turned 1.2 rad, the cube low, upright
    kin = SimpleNamespace(arm="right", lower=np.full(7, -9.0), upper=np.full(7, 9.0),
                          fk=lambda q: tilted(60.0 if float(q[0]) == 1.0 else 0.0), frame=frame)
    solutions = iter([np.array([1.0, 0, 0, 0, 0, 0, 0]), np.array([2.0, 0, 0, 1.2, 0, 0, 0]),
                      np.array([3.0, 0, 0, 0, 0, 0, 0]), np.array([4.0, 0, 0, 0.2, 0, 0, 0])])
    got = ins.aim(kin, lambda *_: next(solutions), np.eye(4), (0.0, 0.0), np.zeros(7), frame="cam",
                  clearance_m=0.015, tilts_deg=(20,), distances_m=(0.18,), azimuths=[0.0, 1.0, 2.0, 3.0],
                  rolls_deg=range(0, 1), reference=np.zeros(7), max_pen_tilt_deg=45.0)
    assert got is not None and got[0][0] == 4.0


def test_view_targets_drop_a_tilted_pen_before_any_ik():
    page = np.eye(4)
    down = np.array([0.0, 0.0, -1.0])
    targets = list(ins.view_targets(page, (0.0, 0.0), np.eye(4), down, tilts_deg=(0, 45, 80), distances_m=(0.2,),
                                    azimuths=[0.0], rolls_deg=(0,), max_pen_tilt_deg=30.0))
    assert [t[0]["tilt_deg"] for t in targets] == [0.0]


def test_path_clear_refuses_a_way_that_passes_through_the_arm_itself():
    """The inspect moves bypass the session's guard; the view and the joint way to it are checked here."""
    def self_gap(q):  # the arm meets itself around joint_1 = 0.5 rad
        return abs(float(q[0]) - 0.5) - 0.1

    start = np.zeros(7)
    assert ins.path_clear(self_gap, start, np.array([0.3, 0, 0, 0, 0, 0, 0]))
    assert not ins.path_clear(self_gap, start, np.array([0.39, 0, 0, 0, 0, 0, 0]))  # 1 mm clear: under the margin
    assert not ins.path_clear(self_gap, start, np.array([1.0, 0, 0, 0, 0, 0, 0]))   # passes through it
    assert not ins.path_clear(self_gap, start, np.array([0.5, 0, 0, 0, 0, 0, 0]))   # ends inside it
    inside = np.array([0.45, 0, 0, 0, 0, 0, 0])
    assert ins.path_clear(self_gap, inside, np.array([0.0, 0, 0, 0, 0, 0, 0]))      # an arm inside backs out


def test_analyse_drops_views_far_from_the_expected_shift_and_pairs_each_with_its_border():
    """Two of three views matched ink 13 mm off (a misregistered view, or another drawing): the median
    alone reports their shift; the gate around the rig's known shift keeps the one good view, and its
    placement is its own shift plus its own border correction."""
    pytest.importorskip("cv2")
    import cv2

    extent, ppm, clear = (-0.060, -0.085, 0.060, 0.085), 5.0, (0.062, 0.112)
    plan = np.array([[-0.025, 0.030], [-0.005, 0.030], [-0.005, 0.045]])
    program = {"ops": [{"op": "stroke", "points_m": plan.tolist()}]}

    def view(border_shift, ink_shift):
        img = _page(extent, ppm, clear, shift=border_shift)
        px = ins.to_image_px(plan + np.asarray(border_shift) + ink_shift, extent, ppm)
        cv2.polylines(img, [np.round(px).astype(np.int32)], False, 30, 3)
        return img

    views = [view((0.0, 0.002), (-0.0065, -0.0005)), view((0.0, 0.0), (0.0065, 0.0)), view((0.0, 0.0), (0.0065, 0.0))]
    plain = ins.analyse(views, program, clear, extent, ppm)
    assert plain["ink"]["dx_m"] > 0.004   # the two bad views carry the median
    gated = ins.analyse(views, program, clear, extent, ppm, ink_prior_m=[-0.0065, -0.0005], ink_gate_m=0.003)
    assert gated["ink"]["views"] == 1 and gated["ink"]["used"] == [0] and gated["ink"]["gated"]
    # view 0 sees the print 2 mm up and the ink 2 mm up with it: on the print it is off by the ink shift alone
    assert abs(gated["placement"]["dx_m"] + 0.0065) < 0.001 and abs(gated["placement"]["dy_m"] + 0.0005) < 0.001
    assert "1 of 3 views" in ins.summary(gated)


def test_border_scan_starts_where_the_drawing_ends():
    """A drawing in the clear centre: scanned from search_m inside each edge, the first dark run after clean
    paper is ink (a resume's drawn wave, 2026-10-05: x scale 1.21-1.26, the print 10-13 mm off). Scanned
    from the program's drawing's own edge, the fit finds the print."""
    import cv2

    extent, ppm = ins.PAGE_EXTENT, 5.0
    shift, theta = (-0.002, 0.001), np.radians(0.5)
    img = _coded_page(extent, ppm, shift=shift, theta=theta)
    c, s = np.cos(theta), np.sin(theta)
    band = np.array([[0.018, -0.035], [0.023, -0.035], [0.023, 0.035], [0.018, 0.035]])   # filled ink, print frame
    cv2.fillPoly(img, [np.round(ins.to_image_px(band @ [[c, s], [-s, c]] + shift, extent, ppm)).astype(np.int32)], 40)
    program = {"ops": [{"op": "tool_change"}, {"op": "stroke", "points_m": [[-0.022, -0.040], [0.023, 0.040]]}]}
    truth = _truth(shift, theta)

    def fit(**kw):
        return ins.solve_in_plane(ins.border_edges(img, extent, ppm, CLEAR, inner_m=CODED_INNER, **kw), CLEAR,
                                  inner_m=CODED_INNER)

    fooled = fit()
    assert fooled is not None and abs(fooled["tx_m"] - truth[0]) > 0.003 and fooled["scale"][0] > 1.1
    assert ins.drawing_extent(program) == (-0.022, -0.040, 0.023, 0.040)
    sol = fit(drawn_m=ins.drawing_extent(program))
    assert sol is not None and abs(sol["tx_m"] - truth[0]) < 0.0003 and abs(sol["ty_m"] - truth[1]) < 0.0003
    assert abs(sol["scale"][0] - 1) < 0.01
