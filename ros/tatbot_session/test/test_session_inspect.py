"""The wrist-camera inspect geometry: aiming, projection and page rectification."""
import numpy as np
import pytest
from tatbot_session import geometry as g
from tatbot_session import inspect as ins

K = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])
PAGE = g.rpy_matrix([0.33, 0.06, 0.044], [0.02, -0.01, -1.5])


def look_down(base_from_page, xy, height_m):
    """ins.look_at straight down the page normal from height_m."""
    return ins.look_at(base_from_page, xy, tilt_rad=0.0, azimuth_rad=0.0, distance_m=height_m)


def test_look_down_is_a_rotation_looking_down_the_page_normal():
    cam = look_down(PAGE, (0.01, -0.02), 0.09)
    assert np.allclose(cam[:3, :3].T @ cam[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(cam[:3, :3]), 1.0)
    assert np.allclose(cam[:3, 2], -PAGE[:3, 2]) and np.allclose(cam[:3, 0], PAGE[:3, 1])
    under = ins.page_to_pixel(np.array([[0.01, -0.02]]), PAGE, cam, K, np.zeros(5))
    assert np.allclose(under, [[640.0, 360.0]])
    # 10 mm along page y moves 640 * 10 / 90 px along image x
    along = ins.page_to_pixel(np.array([[0.01, -0.01]]), PAGE, cam, K, np.zeros(5))
    assert np.allclose(along - under, [[640.0 * 0.01 / 0.09, 0.0]], atol=1e-6)


def test_tcp_target_puts_the_camera_where_asked():
    tcp_from_cam = g.rpy_matrix([0.01, 0.0, -0.08], [0.0, 0.35, 0.0])
    cam = look_down(PAGE, (0.0, 0.0), 0.09)
    assert np.allclose(ins.tcp_target(cam, tcp_from_cam) @ tcp_from_cam, cam)


def test_rectify_maps_a_camera_image_back_onto_the_page():
    pytest.importorskip("cv2")
    cam = look_down(PAGE, (0.0, 0.0), 0.09)
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


def _forearm(crest_m, at_y_m, radius_m=0.03):
    """A forearm along page x as the overhead's depth reads it (page frame, 2 mm apart): its crest `crest_m` over
    the page at y `at_y_m`, the table 70 mm under the page, and the arm's own wrist 100 mm over it."""
    x, y = np.meshgrid(np.arange(-0.12, 0.12, 0.002), np.arange(-0.15, 0.15, 0.002))
    off = np.abs(y - at_y_m)
    z = np.where(off < radius_m, crest_m - (radius_m - np.sqrt(np.clip(radius_m ** 2 - off ** 2, 0.0, None))), -0.07)
    wrist = np.c_[np.full((60, 2), 0.01), np.full(60, 0.10)]
    return np.r_[np.c_[x.ravel(), y.ravel(), z.ravel()], wrist]


def test_a_wrist_search_aims_over_the_skin_the_overhead_depth_reads_each_view_over_its_own_highest_skin():
    """2026-10-21: the overhead read no marker of an inked forearm print and two locate views read none either; the
    set-up refused, its views and holds on the forearm's flank where the stale page stood. The search is centred on
    the skin the overhead's depth reads near the last page now (not the table, nor the arm): that centre first, then
    the points the print's extent from it with skin under them, each raised lift_m over its own highest skin (no
    surface fitted), never below the page."""
    half = (0.0325, 0.0415)   # a 45 x 63 mm print and its 10 mm round it
    aims = ins.skin_aims(_forearm(0.004, 0.05), half, lift_m=0.008)   # risen 4 mm and moved 50 mm along page y
    assert np.allclose(aims[0]["xy"], [0.0, 0.05], atol=0.002)       # centred on the skin now, not the last page
    assert aims[0]["rise_m"] == pytest.approx(0.012) and aims[0]["points"] >= ins.SEARCH_MIN_POINTS   # its crest, 8 mm up
    assert len(aims) == 9 and min(a["xy"][1] for a in aims) > 0.0 and all(0 < a["rise_m"] <= 0.0121 for a in aims)
    sunk = ins.skin_aims(_forearm(-0.04, 0.0), half, lift_m=0.008)      # a limb lowered: views stand on the page
    assert len(sunk) == 9 and np.allclose(sunk[0]["xy"], 0.0, atol=0.002) and all(a["rise_m"] == 0.0 for a in sunk)
    assert ins.skin_aims(np.zeros((0, 3)), half, lift_m=0.008) == []
    assert ins.skin_aims(_forearm(0.2, 0.0), half, lift_m=0.008) == []  # nothing within the band: no search


def test_a_wrist_search_with_no_skin_by_the_last_page_aims_over_the_skin_nearest_it_on_the_mat():
    """2026-10-21: a fresh run seeded from the last run's page; the forearm lay elsewhere, the overhead's depth read
    no skin within 50 mm of that page's plane about it, and no search ran. The mat about it is read for what stands
    15-120 mm over it (not the mat, nor the arm's wrist 170 mm up): the skin nearest the last page centres the
    search, each aim over its own highest skin; a bare mat aims nothing."""
    half = (0.0325, 0.0415)
    moved = _forearm(-0.02, 0.13)             # 130 mm along page y and its crest 20 mm under the last page
    assert ins.skin_aims(moved, half, lift_m=0.008) == []
    wide = ins.wide_aims(moved, half, lift_m=0.008)
    assert wide and all(a["wide"] for a in wide) and 0.095 < wide[0]["xy"][1] < 0.14 and abs(wide[0]["xy"][0]) < 0.01
    assert wide[0]["rise_m"] == pytest.approx(-0.02 + 0.008, abs=0.003)   # over its crest, under the last page
    assert all(a["xy"][1] > 0.05 for a in wide)                           # none over the arm's wrist by the page
    mat = _forearm(-0.07, 0.09)[:-60]
    mat[:, 2] = -0.07
    assert ins.wide_aims(mat, half, lift_m=0.008) == []


def test_look_at_tilts_toward_the_azimuth_and_keeps_looking_at_the_point():
    cam = ins.look_at(PAGE, (0.0, 0.01), tilt_rad=np.radians(45), azimuth_rad=np.radians(90), distance_m=0.15)
    assert np.allclose(cam[:3, :3].T @ cam[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(cam[:3, :3]), 1.0)
    target = (PAGE @ [0.0, 0.01, 0.0, 1.0])[:3]
    assert np.allclose(cam[:3, 3] + 0.15 * cam[:3, 2], target)
    assert np.isclose(-cam[:3, 2] @ PAGE[:3, 2], np.cos(np.radians(45)))
    assert np.allclose(ins.page_to_pixel(np.array([[0.0, 0.01]]), PAGE, cam, K, np.zeros(5)), [[640.0, 360.0]])


def _tile_page(extent, ppm, *, shift=(0.0, 0.0)):
    """A rectified page image of the forearm fiducial tile, its markers dark on pale skin, the print moved by `shift`
    in the believed page frame (scripts/lib/stencil_tile.py, seed 0)."""
    import sys

    import cv2
    pytest.importorskip("PIL")
    sys.path.insert(0, str(ins.repo_root(None) / "scripts/vision"))
    import stencil_tile
    from fiducials import stencil

    ink, record = stencil_tile.build("forearm", 0)
    (ox, oy), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    gx, gy = ins.page_grid(extent, ppm)
    sx = (((gx - shift[0]) * 1000 + tw / 2 + ox) * stencil_tile.PPM - 0.5).astype(np.float32)
    sy = ((th / 2 - (gy - shift[1]) * 1000 + oy) * stencil_tile.PPM - 0.5).astype(np.float32)
    img = cv2.remap(ink, sx, sy, cv2.INTER_AREA, borderValue=255)
    return (90 + img.astype(np.float32) * (210 - 90) / 255).astype(np.uint8), stencil.tile(record)


CLEAR = (0.062, 0.112)


def test_hold_tip_reads_a_fiducial_tile_by_its_markers():
    """The same hold over a fiducial-tile print (scripts/lib/stencil_tile.py): its markers, each at its printed place,
    put the tip on the print within a quarter millimetre from a believed page 6 mm and 2 deg off, and say how many
    agreed; read in `every` frame the hold took when the gauge's own show none of them (a faint transfer decodes a
    different few markers each frame, 2026-10-24)."""
    import sys

    import cv2
    pytest.importorskip("PIL")
    sys.path.insert(0, str(ins.repo_root(None) / "scripts/vision"))
    import stencil_tile
    from fiducials import stencil
    from tatbot_session import gauge

    ink, record = stencil_tile.build("forearm", 0)
    tile = stencil.tile(record)
    (ox, oy), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    cam_from_print = np.linalg.inv(ins.look_at(np.eye(4), (0.005, 0.0), tilt_rad=np.radians(20),
                                               azimuth_rad=np.radians(200), distance_m=0.13))
    v, u = np.mgrid[0:720, 0:1280].astype(float)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    print_from_cam = np.linalg.inv(cam_from_print)
    o, d = print_from_cam[:3, 3], rays @ print_from_cam[:3, :3].T
    hit = (o[:2] + (-o[2] / d[..., 2])[..., None] * d[..., :2]) * 1000   # each pixel's point on the print (mm)
    sx = ((hit[..., 0] + tw / 2 + ox) * stencil_tile.PPM - 0.5).astype(np.float32)
    sy = ((th / 2 - hit[..., 1] + oy) * stencil_tile.PPM - 0.5).astype(np.float32)
    image = cv2.remap(ink, sx, sy, cv2.INTER_AREA, borderValue=255)
    image = (90 + image.astype(np.float32) * (200 - 90) / 255).astype(np.uint8)   # violet on skin, in green
    surface = gauge.Frame(np.zeros((0, 3)), cam_from_print[:3, 3], cam_from_print[:3, 2], 0.0)
    meta = {"color_intrinsics": {"k": K.tolist(), "coeffs": [0.0] * 5}}
    tip = np.array([0.004, -0.007, 0.013])
    tip_cam = (cam_from_print @ [*tip, 1.0])[:3]
    page = {"size_m": list(tile.size_m), "clear_m": list(tile.size_m)}
    for off in (ins.correction_matrix(0.0, 0.0, 0.0), ins.correction_matrix(0.006, -0.004, np.radians(2.0))):
        found = ins.hold_tip([np.dstack([image] * 3)], [surface], meta, tip_cam, cam_from_print @ off, page, tile)
        assert found is not None and np.allclose(found["tip_m"], tip, atol=0.00025)
        assert found["accepted"] and found["markers"] >= 10 and found["margin"] == 1.0
    blank = np.dstack([np.full_like(image, 200)] * 3)                       # skin alone: no marker in it
    assert ins.hold_tip([blank], [surface], meta, tip_cam, cam_from_print, page, tile) is None
    found = ins.hold_tip([blank], [surface], meta, tip_cam, cam_from_print, page, tile,
                         every=[blank, np.dstack([image] * 3)])
    assert found is not None and found["accepted"] and np.allclose(found["tip_m"], tip, atol=0.00025)


def test_holds_shift_is_the_median_of_where_the_tip_stood_less_where_it_was_sent():
    """Believed + shift = print; a hold whose fit jumped a band is dropped; fewer than three agreeing give none."""
    sent = [(-0.0175, -0.02), (0.0175, -0.02), (0.0175, 0.02), (0.0, 0.0), (-0.0175, 0.02)]
    off = [(-0.0016, -0.0035), (-0.0014, -0.0037), (-0.0015, -0.0036), (-0.0015, -0.0036), (-0.0015, 0.020)]
    holds = [{"xy": xy, "tip_m": [xy[0] + dx, xy[1] + dy, 0.012]} for xy, (dx, dy) in zip(sent, off, strict=True)]
    found = ins.holds_shift(holds)
    assert np.allclose(found["shift_m"], [-0.0015, -0.0036]) and (found["holds"], found["dropped"]) == (4, 1)
    assert ins.holds_shift(holds[:2]) is None
    assert ins.holds_shift(holds[:2] + holds[4:]) is None   # two agree, the third is a band off


def test_holds_shift_says_its_farthest_reading_and_the_places_two_markers_read():
    """off_m: the farthest from the shift of the holds kept and of every one by two markers or more, dropped or not;
    places: the kept ones' by two markers or more, a hold made again (nearer than span_m) one place."""
    sent = [(-0.0175, -0.02), (-0.0172, -0.02), (0.0175, 0.02), (0.0, 0.0), (-0.0175, 0.02)]
    off = [(0.001, 0.0), (0.0012, 0.0), (0.0018, 0.0), (0.001, 0.0), (0.012, 0.0)]
    holds = [{"xy": xy, "tip_m": [xy[0] + dx, xy[1] + dy, 0.01], "markers": m}
             for xy, (dx, dy), m in zip(sent, off, (3, 2, 1, 1, 1), strict=True)]
    found = ins.holds_shift(holds)
    assert found["dropped"] == 1 and found["off_m"] == pytest.approx(0.0007) and found["places"] == 1
    holds[4]["markers"] = 4                                  # a dropped reading by four markers counts
    assert ins.holds_shift(holds)["off_m"] == pytest.approx(0.0109)
    holds[2]["markers"] = 2
    assert ins.holds_shift(holds)["places"] == 2


def test_holds_shift_leaves_out_the_holds_the_track_would_not_take():
    """A hold whose markers were not accepted (margin 0) does not vote: four such holds agreeing on a misread no
    longer outvote three accepted ones, and the accepted ones alone are too few when only two remain."""
    sent = [(-0.022, 0.022), (-0.037, 0.039), (-0.006, 0.039), (-0.037, 0.004), (-0.006, 0.004), (-0.020, 0.004),
            (-0.022, 0.039)]
    off = [(0.001, -0.006)] * 3 + [(0.013, -0.006)] * 4
    margins = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    holds = [{"xy": xy, "tip_m": [xy[0] + dx, xy[1] + dy, 0.010], "margin": m}
             for xy, (dx, dy), m in zip(sent, off, margins, strict=True)]
    assert np.allclose(ins.holds_shift([{k: h[k] for k in ("xy", "tip_m")} for h in holds])["shift_m"], [0.013, -0.006])
    found = ins.holds_shift(holds)
    assert np.allclose(found["shift_m"], [0.001, -0.006]) and found["holds"] == 3
    assert ins.holds_shift(holds[1:]) is None


def test_holds_shift_on_a_curved_page_takes_the_holds_own_turn():
    """On a curved page the page turns by the median of each hold's own marker fit's turn (2026-10-19: 0.1-0.4 deg
    apart over a forearm's holds, the locate's views 0.4-0.8 deg apart) and shifts by each tip less where that turn
    puts where it was sent; a flat page's turn stays the locate's, and a wild one is not taken (the places' own turn
    still is)."""
    sent = [(-0.0175, -0.02), (0.0175, -0.02), (0.0175, 0.02), (0.0, 0.0), (-0.0175, 0.02)]
    turn, shift = np.radians(0.5), np.array([0.001, -0.002])
    rot = ins.correction_matrix(0.0, 0.0, turn)[:2, :2]
    holds = [{"xy": xy, "tip_m": [*(rot @ xy + shift), 0.01], "turn_rad": turn + np.radians(d)}
             for xy, d in zip(sent, (0.1, -0.1, 0.0, 0.2, -0.2), strict=True)]
    found = ins.holds_shift(holds, turned=True)
    assert np.isclose(found["turn_rad"], turn) and np.allclose(found["shift_m"], shift, atol=1e-9)
    assert ins.holds_shift(holds)["turn_rad"] == 0.0
    wild = [{**h, "turn_rad": np.radians(10.0)} for h in holds]
    assert np.isclose(ins.holds_shift(wild, turned=True)["turn_rad"], turn)
    assert ins.holds_shift(wild, turned=True)["fit_turn_rad"] == pytest.approx(turn)


HOLDS = np.array([(-0.0135, -0.0127), (0.0128, -0.0127), (0.0128, 0.0114), (-0.0135, 0.0114), (-0.0003, -0.0006)])


def _turned_holds(deg, sent=HOLDS, centre=(0.004, -0.002), slide=(0.0012, -0.0008)):
    """Holds of a print turned deg about `centre` and slid from the page believed: where each tip stands on it."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    tip = (sent - centre) @ np.array([[c, -s], [s, c]]).T + centre + slide
    return [{"xy": xy.tolist(), "tip_m": [*t, 0.010], "margin": 1.0} for xy, t in zip(sent, tip, strict=True)]


def test_holds_take_out_a_turned_pages_turn_with_its_shift():
    """2026-10-19, a forearm print: the five holds put the page 2-3 deg off the locate's (a spot read again later
    agreed to 0.04-0.23 mm), and a shift alone left 0.9 mm at the corner holds. A page truly turned 3 deg about a
    point off the holds' centre: the move carries every hold's believed point onto where its tip stood."""
    holds = _turned_holds(3.0)
    found = ins.holds_shift(holds, turned=True)
    assert np.isclose(np.degrees(found["turn_rad"]), 3.0) and found["holds"] == 5
    assert ins.holds_shift(holds)["turn_rad"] == 0.0                                # a flat page: the locate's
    move = ins.correction_matrix(*found["move"])
    for h in holds:
        assert np.allclose((move @ [*h["xy"], 0.0, 1.0])[:2], h["tip_m"][:2], atol=1e-9)
    shift_only = np.array([h["tip_m"][:2] for h in holds]) - HOLDS - np.median(
        [np.subtract(h["tip_m"][:2], h["xy"]) for h in holds], axis=0)
    assert np.linalg.norm(shift_only, axis=1).max() > 0.0009                       # what the shift alone left
    assert found["spread_m"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_a_turn_is_taken_only_from_enough_holds_spread_wide_and_within_the_locates_reach():
    """Four holds (one misread turns the page and nothing shows it), four places with one hold made again (recorded
    twice, 2026-10-19), five within 2 mm of each other, or a turn past 5 deg (a misread: the locate holds the yaw to
    1-1.5 deg) keep the median shift alone."""
    again = [*_turned_holds(3.0)[:4], {**_turned_holds(3.0)[0], "xy": (HOLDS[0] + [0.0006, -0.0004]).tolist()}]
    for holds in (_turned_holds(3.0)[:4], again, _turned_holds(3.0, sent=HOLDS * 0.1), _turned_holds(6.0)):
        found = ins.holds_shift(holds, turned=True)
        r = np.array([np.subtract(h["tip_m"][:2], h["xy"]) for h in holds])
        assert found["turn_rad"] == 0.0 and abs(found["fit_turn_rad"]) > 0.04
        assert np.allclose(found["shift_m"], np.median(r, axis=0)) and np.allclose(found["move"][:2], found["shift_m"])


def test_inner_edges_are_the_clear_centres():
    assert g.inner_edges(CLEAR) == {"left": -0.031, "right": 0.031, "bottom": -0.056, "top": 0.056}


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


def test_aim_rejects_each_bad_posture_and_takes_the_next_clear_view():
    """Joint margin, arm reach, pen tilt, wrist turn, cube height and side clearance each reject a cheaper view.
    The same chooser returns None when that is the only view; removing the arm cap restores the cheaper choice."""
    from types import SimpleNamespace

    def pose(deg=0.0, z=0.150, y=0.0):
        out = ins.rpy_matrix([0.0, y, z], [np.pi + np.radians(deg), 0.0, 0.0])
        return out

    clear = np.array([0.1, 0, 0, 0.6, 0, 0, 0.0])
    bad = np.array([0.7, 0, 0, 0, 0, 0, 0.0])
    for reason in ("joint limit", "arm reach", "pen tilt", "wrist turn", "cube below mount", "cube near table", "side"):
        rejected = bad.copy()
        if reason == "joint limit":
            rejected[1] = -8.997   # only 3 mrad from its limit
        if reason == "wrist turn":
            rejected[3] = 1.2

        def frame(q, name, reason=reason, rejected=rejected):
            if "missing" in name:
                raise ValueError("URDF lacks frame")
            if name == "right/tool_mount":
                return pose(z=0.100)
            if name == "right/wrist_tag4" and np.array_equal(q, rejected):
                return pose(z={"cube below mount": 0.110, "cube near table": 0.033}.get(reason, 0.150),
                            y=0.200 if reason == "side" else 0.0)
            return pose()

        kin = SimpleNamespace(arm="right", lower=np.full(7, -9.0), upper=np.full(7, 9.0), frame=frame,
                              fk=lambda q, reason=reason, rejected=rejected: pose(60.0 if reason == "pen tilt" and np.array_equal(q, rejected) else 0.0))
        options = {"frame": "cam", "clearance_m": 0.015, "reference": np.zeros(7), "max_y": 0.100,
                   "max_arm_rad": 0.6 if reason == "arm reach" else None, "max_pen_tilt_deg": 45.0,
                   "cube_up_m": -1.0 if reason == "cube near table" else ins.CUBE_UP_M,
                   "tilts_deg": (20,), "distances_m": (0.18,), "rolls_deg": (0,)}
        choices = iter([rejected, clear])
        got = ins.aim(kin, lambda *_, choices=choices: next(choices).copy(), np.eye(4), (0, 0), np.zeros(7),
                      azimuths=(0.0, 1.0), **options)
        assert got is not None and np.array_equal(got[0], clear), reason
        assert ins.aim(kin, lambda *_, rejected=rejected: rejected.copy(), np.eye(4), (0, 0), np.zeros(7),
                       azimuths=(0.0,), **options) is None, reason
        if reason == "arm reach":
            choices = iter([rejected, clear])
            got = ins.aim(kin, lambda *_, choices=choices: next(choices).copy(), np.eye(4), (0, 0), np.zeros(7),
                          azimuths=(0.0, 1.0), **{**options, "max_arm_rad": None})
            assert np.array_equal(got[0], rejected)
    assert ins.frames_of(kin, np.zeros(7), ("{arm}/wrist_tag4", "{arm}/missing_frame")) == ["right/wrist_tag4"]


def test_the_lower_wrist_camera_is_a_wrist_body_kept_on_the_arms_side_and_over_the_page():
    """The lower wrist D405 (2026-10-23) is one of the wrist's bodies (WRIST_BODY): a view whose lower camera crosses
    the side line, or stands under BODY_CLEARANCE_M over the page, is refused as one of the upper's would be."""
    from tatbot_session import inspect as ins

    at = {"right/realsense_lower_link": [0.3, -0.1, 0.2]}

    class Kin:
        arm, lower, upper = "right", np.full(7, -3.0), np.full(7, 3.0)

        def frame(self, q, name):
            pose = np.eye(4)
            pose[:3, 3] = at.get(name, [0.3, -0.1, 0.2])
            return pose

        def fk(self, q):
            return self.frame(q, "right/tcp")

    kin = Kin()
    body = ins.frames_of(kin, np.zeros(7), ins.WRIST_BODY)
    assert "right/realsense_lower_link" in body and "right/realsense_link" in body

    def clear():
        return ins.view_clear(kin, np.zeros(7), np.eye(4), margin_rad=0.0, clearance_m=0.015, body=body,
                              body_clearance_m=ins.BODY_CLEARANCE_M, side=body, max_y=0.0)
    assert clear()
    at["right/realsense_lower_link"] = [0.3, 0.02, 0.2]      # past the side line
    assert not clear()
    at["right/realsense_lower_link"] = [0.3, -0.1, 0.05]     # under the bodies' clearance
    assert not clear()


def test_a_top_down_view_is_aimed_about_the_page_levelled_with_the_side_line_farther(monkeypatch):
    """2026-10-21: views 35-40 deg off saw a forearm print's markers squashed; the owner wants them straight down.
    top_down aims about the page levelled (its centre and turn kept) on TOP_DOWN's grid, the side line farther."""
    calls, tilted = [], ins.rpy_matrix([0.3, 0.0, 0.1], [0.2, 0.0, 0.5])
    monkeypatch.setattr(ins, "aim", lambda kin, solve, page, xy, q, **k: calls.append((page, k)))
    ins.top_down(None, None, tilted, (0.0, 0.0), np.zeros(7), max_y=0.1, frame="cam")
    (page, k), = calls
    assert np.allclose(page[:3, 2], [0, 0, 1.0]) and np.allclose(page[:3, 3], tilted[:3, 3])
    assert abs(page[:3, 0] @ tilted[:3, 0]) > np.cos(0.21) and k["max_y"] == pytest.approx(0.1 + ins.TOP_DOWN_SIDE_M)
    assert {key: k[key] for key in ins.TOP_DOWN} == ins.TOP_DOWN and ins.TOP_DOWN["tilts_deg"] == (0, 5, 10)


def test_an_over_centre_view_is_the_ready_pose_raised_square_to_the_page():
    """2026-10-24: the locate's first look stands the pen square to the page `height_m` over its centre at the ready
    move's heading, solved from the ready move's seed with the carriage where it is (inspect.over_centre); it keeps the
    joint margins, the pen's and the wrist bodies' clearance over the page and a way off the arm itself, and no side
    line (the ready pose under it has none); the markers it sees by `see`, None without."""
    from types import SimpleNamespace

    page = ins.rpy_matrix([0.30, -0.05, 0.10], [0.12, 0.0, 0.4])
    heading = ins.rpy_matrix([0.0, 0.0, 0.0], [np.pi, 0.0, 0.7])
    solved, at = [], {}

    def frame(q, name):   # the wrist's bodies 0.2 m over the page and 0.5 m past any side line
        pose = np.eye(4)
        pose[:3, 3] = at.get(name, page[:3, 3] + 0.2 * page[:3, 2] + [0.0, 0.5, 0.0])
        return pose
    kin = SimpleNamespace(arm="right", lower=np.full(7, -3.0), upper=np.full(7, 3.0), frame=frame,
                          fk=lambda q: solved[-1][0])

    def solve(kin, target, seed):
        solved.append((target, seed.copy()))
        return np.full(7, 0.1)
    here = np.r_[np.full(6, 0.5), 0.02]
    q, view = ins.over_centre(kin, solve, page, (0.0, 0.0), here, heading, frame="cam", clearance_m=0.015,
                              height_m=0.050, seed=ins.REST)
    target, seed = solved[-1]
    assert np.allclose(target[:3, 3], page[:3, 3] + 0.050 * page[:3, 2]) and np.allclose(target[:3, 2], -page[:3, 2])
    assert target[:3, 0] @ heading[:3, 0] > 0.99 and np.allclose(seed, [*ins.REST[:6], 0.02])
    assert np.allclose(q, 0.1) and view == {"over_m": 0.050, "seen": None}
    _, view = ins.over_centre(kin, solve, page, (0.0, 0.0), here, heading, frame="cam", clearance_m=0.015,
                              height_m=0.100, see=lambda base_from_camera: 4)
    assert view == {"over_m": 0.100, "seen": 4}
    assert ins.over_centre(kin, solve, page, (0.0, 0.0), here, heading, frame="cam", clearance_m=0.015,
                           height_m=0.050, self_gap=lambda q: -0.01) is None              # its way crosses the arm
    at["right/realsense_lower_link"] = page[:3, 3] + 0.03 * page[:3, 2]
    assert ins.over_centre(kin, solve, page, (0.0, 0.0), here, heading, frame="cam", clearance_m=0.015,
                           height_m=0.050) is None                                         # a body near the page

    def fails(*a):
        raise ValueError("no solution")
    assert ins.over_centre(kin, fails, page, (0.0, 0.0), here, heading, frame="cam", clearance_m=0.015,
                           height_m=0.050) is None


def test_a_view_is_chosen_by_the_markers_it_sees_before_its_joint_cost():
    """2026-10-22: the locate's cheapest views put the print behind the tool (75% of the markers in view read, 92% of
    those the tool left clear). With `see` the views seeing the most come first, and none seeing fewer is solved once
    one of them is reachable; without it the cheapest is taken as before."""
    from types import SimpleNamespace

    down = ins.rpy_matrix([0.0, 0.0, 0.15], [np.pi, 0.0, 0.0])
    kin = SimpleNamespace(arm="right", lower=np.full(7, -9.0), upper=np.full(7, 9.0), frame=lambda q, name: down,
                          fk=lambda q: down)
    options = {"frame": "cam", "clearance_m": 0.015, "reference": np.zeros(7), "max_arm_rad": None,
               "tilts_deg": (20,), "distances_m": (0.18,), "rolls_deg": (0,), "azimuths": (0.0, np.pi / 2),
               "cube_up_m": -1.0}
    cheap, dear, solved = np.zeros(7), np.array([0.3, 0, 0, 0, 0, 0, 0.0]), []

    def solve(kin, target, seed):   # the view toward +y (azimuth 90 deg) costs more
        solved.append(target[:3, 3].copy())
        return (dear if target[1, 3] > 0.01 else cheap).copy()

    assert np.array_equal(ins.aim(kin, solve, np.eye(4), (0, 0), np.zeros(7), **options)[0], cheap)
    solved.clear()
    q, view = ins.aim(kin, solve, np.eye(4), (0, 0), np.zeros(7), see=lambda cam: 3 if cam[1, 3] > 0.01 else 1,
                      **options)
    assert np.array_equal(q, dear) and view["seen"] == 3 and len(solved) == 1
    q, view = ins.aim(kin, solve, np.eye(4), (0, 0), np.zeros(7), around=((0.0, 0.0), (0.0, 0.05)), **options)
    assert view["aside_m"] == [0.0, 0.0] and len({tuple(t.round(4)) for t in solved}) == 4


def test_the_markers_a_view_sees_are_in_frame_past_the_tool_and_big_enough():
    """markers_seen counts a marker when its four corners land in the frame, none on the tool's footprint, its sides
    MARKER_PX long on average; top_down_distances goes as high as the print's smallest marker keeps that size."""
    from types import SimpleNamespace

    k = np.array([[651.0, 0.0, 640.0], [0.0, 651.0, 360.0], [0.0, 0.0, 1.0]])
    square = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * 0.009   # 18 mm, TL/TR/BR/BL in the page frame
    tile = SimpleNamespace(ids=[0, 1, 2], corners={0: square + [-0.03, 0], 1: square + [0.03, 0], 2: square + [0.5, 0]},
                           sides_m={0: 0.018, 1: 0.018, 2: 0.018})
    camera = ins.rpy_matrix([0.0, 0.0, 0.25], [np.pi, 0.0, 0.0])   # looking straight down from 0.25 m
    clear, left = np.zeros((720, 1280), bool), np.zeros((720, 1280), bool)
    left[:, :640] = True
    assert ins.markers_seen(np.eye(4), camera, tile, k, clear) == 2      # the third is out of the frame
    assert ins.markers_seen(np.eye(4), camera, tile, k, left) == 1       # the tool hides one
    assert ins.markers_seen(np.eye(4), camera, tile, k, clear, min_px=60.0) == 0   # 47 px a side
    assert ins.top_down_distances(tile, 651.0) == (ins.TOP_DOWN_M, ins.TOP_DOWN_MAX_M)
    tile.sides_m = {0: 0.012}
    assert ins.top_down_distances(tile, 651.0) == (ins.TOP_DOWN_M,)


def test_the_tools_footprint_is_its_datasheet_profile_moved_onto_the_measured_tip():
    """The tool rides the camera, so it covers the same pixels in every view: its profile of revolution about the
    mount's z, projected, moved so its tip lies on the gauge's measured tip (2026-10-22: the CAD mount put the tip
    11 mm off it, and the footprint then told which of 477 markers were read in 87% against 91%)."""
    from types import SimpleNamespace

    mount = np.eye(4)
    mount[:3, 3] = [0.0, 0.03, 0.05]   # 30 mm below the camera's axis, 50 mm out, along it
    kin = SimpleNamespace(arm="right", frame=lambda q, name: mount if name.endswith("tool_mount") else np.eye(4))
    k = np.array([[651.0, 0.0, 640.0], [0.0, 651.0, 360.0], [0.0, 0.0, 1.0]])
    profile = [(0.0, 0.01), (0.05, 0.01)]

    def at(x, y, z):
        return int(360 + 651 * y / z), int(640 + 651 * x / z)
    cad = ins.tool_footprint(kin, np.zeros(7), k, profile, "cam")
    moved = ins.tool_footprint(kin, np.zeros(7), k, profile, "cam", tip_cam=[0.02, 0.03, 0.1])
    assert cad[at(0.0, 0.03, 0.075)] and not cad[20, 1260] and 0.01 < cad.mean() < 0.5
    assert moved[at(0.02, 0.03, 0.075)] and not moved[at(-0.012, 0.03, 0.075)]


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


def test_analyse_drops_views_far_from_the_expected_shift_and_pairs_each_with_its_print_fit():
    """Two of three views matched ink 13 mm off (a misregistered view, or another drawing): the median
    alone reports their shift; the gate around the rig's known shift keeps the one good view, and its
    placement is its own shift plus its own print correction (the tile's markers)."""
    pytest.importorskip("cv2")
    import cv2

    extent, ppm, clear = (-0.060, -0.085, 0.060, 0.085), 5.0, (0.110, 0.150)
    plan = np.array([[-0.025, 0.050], [-0.005, 0.050], [-0.005, 0.065]])   # above the 45 x 63 mm tile's markers
    program = {"ops": [{"op": "stroke", "points_m": plan.tolist()}]}
    tile = None

    def view(border_shift, ink_shift):
        nonlocal tile
        img, tile = _tile_page(extent, ppm, shift=border_shift)
        px = ins.to_image_px(plan + np.asarray(border_shift) + ink_shift, extent, ppm)
        cv2.polylines(img, [np.round(px).astype(np.int32)], False, 30, 3)
        return img

    views = [view((0.0, 0.002), (-0.0065, -0.0005)), view((0.0, 0.0), (0.0065, 0.0)), view((0.0, 0.0), (0.0065, 0.0))]
    plain = ins.analyse(views, program, clear, extent, ppm, tile=tile)
    assert plain["ink"]["dx_m"] > 0.004   # the two bad views carry the median
    gated = ins.analyse(views, program, clear, extent, ppm, tile=tile, ink_prior_m=[-0.0065, -0.0005], ink_gate_m=0.003)
    assert gated["ink"]["views"] == 1 and gated["ink"]["used"] == [0] and gated["ink"]["gated"]
    # view 0 sees the print 2 mm up and the ink 2 mm up with it: on the print it is off by the ink shift alone
    assert abs(gated["placement"]["dx_m"] + 0.0065) < 0.001 and abs(gated["placement"]["dy_m"] + 0.0005) < 0.001
    assert "1 of 3 views" in ins.summary(gated)


