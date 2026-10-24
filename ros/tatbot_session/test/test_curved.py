"""tatbot_session.curved: the wrist camera's turn (fit_turn, hold_patch) and the check holds' places."""
import numpy as np
import pytest
from tatbot_session import curved, geometry


def _rot(axis, deg):
    return curved.rotation(np.asarray(axis, float) * np.radians(deg))


def test_hold_patch_turns_the_skin_about_the_tip_in_the_cameras_frame():
    tip = np.array([0.0, 0.0, 0.15])
    patch = np.array([[0.02, 0.0, 0.15], [0.0, 0.02, 0.15]])

    class Frame:
        pass
    frame = Frame()
    frame.patch = patch
    base_from_camera, tcp = np.eye(4), np.eye(4)
    tcp[:3, 3] = tip
    plain = curved.hold_patch(frame, {"tip_cam_m": tip.tolist()}, base_from_camera, tcp)
    turned = curved.hold_patch(frame, {"tip_cam_m": tip.tolist()}, base_from_camera, tcp, turn=[0.0, 0.05, 0.0])
    assert np.allclose(plain, patch)
    assert np.allclose(turned - tip, (patch - tip) @ curved.rotation([0.0, 0.05, 0.0]).T)


def test_fit_turn_finds_the_cameras_turn_from_several_setups():
    """Holds over a 40 mm forearm from two setups whose camera stands at other headings; every patch turned by the
    same 40 mrad in the camera's frame. Its part square to the page normal comes back; about the normal, 0."""
    rng = np.random.default_rng(1)
    truth = np.array([0.03, -0.02, 0.025])
    runs = []
    for heading in (0.0, 25.0):
        page = np.eye(4)
        holds = []
        for x, y in ((0.0, 0.0), (-0.012, -0.012), (0.012, -0.012), (0.012, 0.012), (-0.012, 0.012)):
            surface = np.array([x, y, np.sqrt(0.04 ** 2 - x ** 2) - 0.04])
            base_from_camera = np.eye(4)   # camera looking down -z at 0.15 m over the tip, turned by heading
            base_from_camera[:3, :3] = _rot([0, 0, 1], heading) @ _rot([1, 0, 0], 180.0)
            tip_base = surface + [0.0, 0.0, 0.010]
            tip_cam = np.array([0.0, 0.0, 0.15])
            base_from_camera[:3, 3] = tip_base - base_from_camera[:3, :3] @ tip_cam
            xy = rng.uniform(-0.025, 0.025, (800, 2)) + [x, y]
            skin = np.c_[xy, np.sqrt(0.04 ** 2 - np.clip(xy[:, 0], -0.039, 0.039) ** 2) - 0.04]
            cam = (skin - base_from_camera[:3, 3]) @ base_from_camera[:3, :3]
            seen = (cam - tip_cam) @ curved.rotation(-truth).T + tip_cam   # what a camera turned by truth reports
            tcp = np.eye(4)
            tcp[:3, 3] = tip_base
            holds.append({"patch_cam": seen + rng.normal(0, 0.0002, seen.shape), "tip_cam": tip_cam,
                          "base_from_camera": base_from_camera, "tcp": tcp, "point": surface})
        runs.append({"page": page, "holds": holds})
    found = curved.fit_turn(runs)
    normal_cam = np.array([0.0, 0.0, -1.0])            # the page normal in these cameras' frames
    square = truth - (truth @ normal_cam) * normal_cam
    assert np.allclose(found, square, atol=0.004)


def test_check_points_are_the_corner_edges_midpoints_whatever_the_holds_order():
    holds = [[0.0, 0.0], [-0.01, -0.02], [0.01, -0.02], [0.01, 0.02], [-0.01, 0.02]]
    mids = curved.check_points(holds)
    assert sorted(map(tuple, np.round(mids, 4))) == sorted([(0.0, -0.02), (0.01, 0.0), (0.0, 0.02), (-0.01, 0.0)])
    assert len(curved.check_points(holds[1:4])) == 3


def test_a_large_drawing_gets_a_grid_of_holds_within_reach_of_every_point():
    """10-13: a 75 mm filler stood 37 mm from the corners and centre; reach is 20 mm."""
    program = {"ops": [{"op": "stroke", "points_m": [[-0.0375, -0.0375], [0.0375, 0.0375]]},
                       {"op": "stroke", "points_m": [[-0.0375, 0.0375], [0.0375, -0.0375], [0.0, 0.0375]]}]}
    holds = curved.layout(program, [0.090, 0.134], reach_m=0.020)
    assert np.allclose(holds[0], [0.0, 0.0]) and len(holds) == 17
    assert curved.coverage(program, holds, 0.020) <= 0.020


def test_layout_holds_the_centre_first():
    program = {"ops": [{"op": "stroke", "points_m": [[-0.01, -0.015], [0.01, 0.015]]}]}
    holds = curved.layout(program, [0.043, 0.088])
    assert np.allclose(holds[0], [0.0, 0.0]) and len(holds) == 5
    assert pytest.approx(np.abs(holds[1:]).max(axis=0).tolist()) == [0.01, 0.015]


def _forearm(radius=0.04):
    from tatbot_motion.surface import HeightField, PageSurface, fit_height

    xy = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 31), np.linspace(-0.04, 0.04, 21)), -1).reshape(-1, 2)
    pts = np.c_[xy, np.sqrt(radius ** 2 - xy[:, 0] ** 2) - radius]
    return PageSurface(np.eye(4), fit_height(pts, degree=4)), PageSurface(np.eye(4), HeightField())


def test_page_to_pixel_puts_print_points_on_the_skin_and_none_where_it_faces_away():
    """A camera 0.15 m over the crest looking down: a flat surface projects as the plane does; on the forearm a
    print point is the lifted skin point's pixel, and past the flank (the skin facing away) there is none."""
    from tatbot_session import inspect as ins

    forearm, flat = _forearm()
    k, dist = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]), np.zeros(5)
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera[:3, 3] = [0.0, 0.0, 0.15]
    pts = np.array([[0.0, 0.0], [0.012, 0.01], [-0.02, -0.03]])
    assert np.allclose(ins.page_to_pixel(pts, np.eye(4), camera, k, dist, flat),
                       ins.page_to_pixel(pts, np.eye(4), camera, k, dist))
    on_skin = ins.page_to_pixel(pts, np.eye(4), camera, k, dist, forearm)
    lifted = forearm.lift(pts)[0]
    cam = (np.linalg.inv(camera) @ np.c_[lifted, np.ones(3)].T)[:3].T
    assert np.allclose(on_skin, np.c_[600 * cam[:, 0] / cam[:, 2] + 320, 600 * cam[:, 1] / cam[:, 2] + 240])
    side = forearm.chart.to_chart([[0.037, 0.0]])   # the far flank, ~68 deg round: off to the side, ~73 deg + away
    assert not np.isfinite(ins.page_to_pixel(side + [0.012, 0.0], np.eye(4), camera, k, dist, forearm)).all()


def test_foot_gives_the_print_point_and_height_a_point_was_lifted_from():
    forearm, _ = _forearm()
    uv = np.array([[0.011, -0.007]])
    point, _ = forearm.lift(uv, offset_m=0.006)
    where, gap, normal = forearm.foot(point[0])
    assert np.allclose(where, uv[0], atol=1e-4) and gap == pytest.approx(0.006, abs=5e-5)   # the chart's 1 mm grid
    assert np.allclose(normal, forearm.lift(uv)[1][0], atol=1e-3)


# --- the rigid follow of a curved page (curved.RigidWatch, rigid_fit, ambiguity) ---------------------------------


def _moved(surface, slide=(0.0, 0.0), turn_deg=0.0):
    """The skin moved rigidly in its own page plane (a forearm slid along the rest, turned about the page normal)."""
    m = np.eye(3)
    c, s = np.cos(np.radians(turn_deg)), np.sin(np.radians(turn_deg))
    m[:2, :2], m[:2, 2] = [[c, -s], [s, c]], slide
    return surface.moved(surface.base @ curved.in_plane(m)), m


def _reading(surface, skin, xy):
    """The watch's reading with the pen 10 mm along the page's normal over plane point xy: the print point under it
    on the skin as it is (act) and as the holds left it (exp), in uv and on the plane."""
    tcp = surface.lift(surface.chart.to_chart([xy]), offset_m=0.010)[0][0]
    act_uv, exp_uv = skin.foot(tcp)[0], surface.foot(tcp)[0]
    act_xy, exp_xy = surface.chart.to_plane([act_uv, exp_uv])
    return act_xy, exp_xy, act_uv, exp_uv


def test_a_reading_within_follow_is_the_stencil_staying_and_one_past_it_waits_for_a_second():
    forearm, _ = _forearm(0.035)
    watch = curved.RigidWatch()
    still, _ = _moved(forearm, (0.0005, -0.0008))                        # the readings' scatter: under follow_m
    assert watch.add(*_reading(forearm, still, [0.0, 0.0])) == "added" and not watch.confirmed
    slid, m = _moved(forearm, (0.0, 0.003))                             # 3 mm along the arm
    assert watch.add(*_reading(forearm, slid, [0.004, -0.010])) == "held"
    assert watch.add(*_reading(forearm, slid, [0.005, -0.009])) == "held"   # the same place again: no second view
    assert watch.add(*_reading(forearm, still, [0.0, 0.010])) == "added" and not watch.held   # it stayed after all
    assert watch.add(*_reading(forearm, slid, [0.004, -0.010])) == "held"
    assert watch.add(*_reading(forearm, slid, [-0.006, 0.008])) == "moved" and watch.confirmed
    plane, on_print = watch.step()
    assert np.allclose(plane[:2, 2], [0.0, 0.003], atol=1.5e-4) and np.allclose(plane[:2, :2], np.eye(2), atol=0.02)
    assert np.allclose(on_print[:2, 2], [0.0, 0.003], atol=1.5e-4)


def test_two_readings_that_disagree_on_one_move_confirm_nothing():
    """A misread past follow_m and a real move elsewhere: no one rigid move takes both."""
    forearm, _ = _forearm(0.035)
    watch = curved.RigidWatch()
    left, _ = _moved(forearm, (-0.003, 0.0))
    right, _ = _moved(forearm, (0.0, 0.004))
    assert watch.add(*_reading(forearm, left, [0.0, -0.010])) == "held"
    assert watch.add(*_reading(forearm, right, [0.0, 0.010])) == "held" and watch.step()[0].tolist() == np.eye(3).tolist()


def test_a_turned_stencil_is_fitted_with_its_turn_once_the_readings_spread():
    src = np.array([[-0.012, -0.012], [0.012, -0.010], [0.0, 0.014]])
    c, s = np.cos(np.radians(3.0)), np.sin(np.radians(3.0))
    dst = src @ np.array([[c, -s], [s, c]]).T + [0.001, 0.002]
    m = curved.rigid_fit(src, dst, spread_m=0.010)
    assert np.allclose(curved.moved_by(m, src), dst, atol=1e-12)
    near = curved.rigid_fit(src[:2] * 0.1, dst[:2] * 0.1, spread_m=0.010)   # 2.4 mm apart: a slide alone
    assert np.allclose(near[:2, :2], np.eye(2))


def test_the_rigid_follow_puts_every_print_point_on_the_moved_skin_and_stretches_no_stroke():
    """The skin slid 2.5 mm along the arm and turned 2 deg on its rest: two readings confirm it, and the page surface
    moved by the watch's step lifts a 24 mm stroke round the curve onto the moved skin, every point at its print mm,
    its length on the skin unchanged (the per-stroke field's correction, 2026-10-08, stretched one 5 mm)."""
    forearm, _ = _forearm(0.035)
    skin, truth = _moved(forearm, (0.0006, 0.0025), turn_deg=2.0)
    watch = curved.RigidWatch()
    for xy in ([-0.010, -0.012], [0.010, 0.012]):
        watch.add(*_reading(forearm, skin, xy))
    assert watch.confirmed
    plane, _ = watch.step()
    followed = forearm.moved(forearm.base @ curved.in_plane(plane))
    stroke = np.c_[np.linspace(-0.012, 0.012, 49), np.full(49, 0.004)]   # across the curve, in print mm
    drawn, want = followed.lift(stroke)[0], skin.lift(stroke)[0]
    assert np.max(np.linalg.norm(drawn - want, axis=1)) < 2e-4
    length = lambda p: float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))   # noqa: E731
    assert length(drawn) == pytest.approx(length(forearm.lift(stroke)[0]), rel=1e-9)
    turn = np.degrees(np.arctan2(plane[1, 0], plane[0, 0]))
    assert turn == pytest.approx(2.0, abs=0.2) and np.allclose(plane[:2, 2], truth[:2, 2], atol=3e-4)


def test_a_move_along_the_arm_is_unambiguous_and_one_across_a_strong_curve_is_not():
    """A forearm turned about its own axis moves the print round the curve and keeps its shape; a slid one carries
    the shape. The print reads the two alike: along the axis (or on a flat page) they agree, across a 30 mm radius
    a 1.5 mm move puts a 30 mm design's flanks a millimetre apart."""
    forearm, flat = _forearm(0.030)
    uv = np.stack(np.meshgrid(np.linspace(-0.015, 0.015, 7), np.linspace(-0.015, 0.015, 7)), -1).reshape(-1, 2)
    along, across = np.eye(3), np.eye(3)
    along[1, 2], across[0, 2] = 0.003, 0.0015
    assert curved.ambiguity(forearm, along, along, uv)[0] < 1e-4
    assert curved.ambiguity(flat, across, across, uv)[0] < 1e-9
    apart, tilt = curved.ambiguity(forearm, across, across, uv)
    assert apart > 0.0005 and tilt > 0.02


def test_rigid_defaults_and_overrides():
    cfg = curved.config({"page": {"surface": {"rigid": {"follow_m": 0.003}}}})
    assert cfg["rigid"]["follow_m"] == 0.003 and cfg["rigid"]["max_m"] == curved.RIGID["max_m"]


# --- a human forearm's curve under a small design (the field, its holds, the check gate) ---------------------------


def _arm(radius, taper=0.0, tilt_deg=0.0, axis_deg=0.0):
    """Skin height over the page: a cylinder of `radius` about an axis along page y turned axis_deg, its radius
    tapering by `taper` per metre along it, the axis tilting tilt_deg along the arm."""
    c, s, t = np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg)), np.tan(np.radians(tilt_deg))

    def h(xy):
        xy = np.atleast_2d(xy)
        across, along = c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]
        r = radius * (1.0 + taper * along)
        return np.sqrt(np.maximum(r ** 2 - across ** 2, 0.0)) - radius + t * along
    return h


def _hold(h, xy, rng, bias=0.0):
    """A hold's skin as the wrist camera sees it (curved.fit's input): within 40 mm of the tip, 8 mm about its foot
    left out, on a 40 x 60 mm print, facing a camera 60 mm along the arm and 120 mm up within 60 deg, 0.2 mm noise."""
    tip = np.r_[xy, h(xy)[0] + 0.010]
    g = rng.uniform(-0.04, 0.04, (5000, 2)) + xy
    g = g[(np.abs(g[:, 0]) <= 0.018) & (np.abs(g[:, 1]) <= 0.028)]
    p = np.c_[g, h(g)]
    e = 1e-5
    n = np.c_[-(h(g + [e, 0]) - h(g - [e, 0])) / (2 * e), -(h(g + [0, e]) - h(g - [0, e])) / (2 * e), np.ones(len(g))]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    view = tip + [0.0, -0.06, 0.12] - p
    facing = np.sum(n * view / np.linalg.norm(view, axis=1, keepdims=True), axis=1) > np.cos(np.radians(60.0))
    keep = (np.linalg.norm(p - tip, axis=1) < 0.04) & (np.linalg.norm(g - xy, axis=1) > 0.008) & facing
    p = p[keep][:1500]
    p[:, 2] += rng.normal(0.0, 0.0002, len(p)) + bias
    return {"xy": xy, "point": np.r_[xy, h(xy)[0] + bias], "patch": p, "tip": tip}


@pytest.mark.parametrize("radius, taper, tilt_deg, axis_deg, within", [
    (0.060, 0.0, 0.0, 0.0, 0.00035), (0.045, 3.0, 10.0, 0.0, 0.00035), (0.030, 3.0, 15.0, 20.0, 0.00035),
    (0.030, 0.0, 0.0, 90.0, 0.0006)])   # the camera looking across the arm: the far flank grazing (turn the arm)
def test_five_holds_carry_a_forearms_curve_under_a_30_mm_design(radius, taper, tilt_deg, axis_deg, within):
    """A human forearm (30-60 mm radius, tapering 3%/cm, its axis tilted along the arm, turned on its rest) under a
    30 x 30 mm design on a 40 x 60 mm print: the drawing's corners and centre (curved.layout) cover it within the
    reach, the degree-4 field from their skin lies within 0.35 mm of it with each hold 0.2 mm off at random (0.6 with
    the camera looking across the arm), and check holds at the edges' midpoints read it well inside the 0.9 mm gate
    (stack.yaml max_loo_m). Noise-free the field is within 0.1 mm in every case: the holds are not the limit."""
    h, rng = _arm(radius, taper, tilt_deg, axis_deg), np.random.default_rng(7)
    program = {"ops": [{"op": "stroke", "id": "s", "points_m": [[-0.015, -0.015], [0.015, -0.015], [0.015, 0.015],
                                                                 [-0.015, 0.015]]}]}
    holds = curved.layout(program, [0.034, 0.054])
    assert len(holds) == 5 and curved.coverage(program, holds, 0.020) < 0.020
    reads = [_hold(h, xy, rng, bias=rng.normal(0.0, 0.0002)) for xy in holds]
    field, report = curved.fit(np.eye(4), reads, {"degree": 4}, bounds=(0.018, 0.028))
    grid = np.stack(np.meshgrid(np.linspace(-0.015, 0.015, 31), np.linspace(-0.015, 0.015, 31)), -1).reshape(-1, 2)
    assert np.max(np.abs(field(grid) - h(grid))) < within
    for xy in curved.check_points(holds):
        check = _hold(h, xy, rng)
        assert abs(curved.ring_miss(np.eye(4), field, check["patch"], check["point"])) < 0.0006


def test_the_relief_is_reported_along_the_strokes_not_over_the_forearms_flanks_the_holds_see():
    """A 14 x 13 mm heart on a thin forearm (25 mm radius, 3%/cm taper, crest along the drawing): its five holds read
    the skin within ~1 mm of one another, while their skin reaches the arm's flanks, where the field falls 10 mm and
    more. The tiny heart's setups logged 18-24 mm "of relief over the drawing" from that skin's extent where the field
    spanned 0.8 mm along their strokes (2026-10-21). The relief is now the field's along the strokes; the field there
    stays on the skin and within the gauge points' span but for the arm's own curve past the corner holds."""
    h, rng = _arm(0.025, 3.0), np.random.default_rng(3)
    heart = np.array([[-0.007, 0.002], [-0.0035, 0.0065], [0.0, 0.003], [0.0035, 0.0065], [0.007, 0.002],
                      [0.0, -0.0065], [-0.007, 0.002]])
    t = np.linspace(0.0, 1.0, 20)[:, None]
    points = np.concatenate([a + t * (b - a) for a, b in zip(heart[:-1], heart[1:], strict=True)])
    program = {"ops": [{"op": "stroke", "id": "s", "points_m": points.tolist()}]}
    holds = curved.layout(program, [0.034, 0.054])
    reads = [_hold(h, xy, rng, bias=rng.normal(0.0, 0.0002)) for xy in holds]
    field, report = curved.fit(np.eye(4), reads, {"degree": 4}, bounds=(0.018, 0.028), drawing=points)
    gauged = [r["point"][2] for r in reads]
    assert report["relief_extent_m"] > 0.008 > 0.0015 > report["relief_m"]
    assert report["relief_m"] == pytest.approx(np.ptp(field(points)))
    assert report["gauged_m"] == pytest.approx([min(gauged), max(gauged)], abs=1e-5)   # the fitted camera turn
    assert np.max(np.abs(field(points) - h(points))) < 0.00035
    assert report["drawing_m"][0] > min(gauged) - 0.0005 and report["drawing_m"][1] < max(gauged) + 0.0005
    _, alone = curved.fit(np.eye(4), reads, {"degree": 4}, bounds=(0.018, 0.028))
    assert alone["relief_m"] == alone["relief_extent_m"] == pytest.approx(report["relief_extent_m"])
    assert alone["drawing_m"] is None


def test_a_small_print_keeps_its_holds_its_own_inset_inside_the_clear_centre():
    """A 30 x 30 mm drawing on the 45 x 63 mm forearm print (clear centre 32 x 50 mm): 5 mm inside that centre held
    its corner holds 4 mm inside the drawing's corners; the print's own 2.5 mm (page.inset_m) holds them 1.5 mm in."""
    program = {"ops": [{"op": "stroke", "points_m": [[-0.015, -0.015], [0.015, 0.015]]},
                       {"op": "stroke", "points_m": [[-0.015, 0.015], [0.015, -0.015]]}]}
    stock = curved.layout(program, [0.032, 0.050])
    small = curved.layout(program, [0.032, 0.050], inset_m=0.0025)
    assert pytest.approx(np.abs(stock[1:]).max(axis=0).tolist()) == [0.011, 0.015]
    assert pytest.approx(np.abs(small[1:]).max(axis=0).tolist()) == [0.0135, 0.015]
    assert curved.coverage(program, small, 0.020) < curved.coverage(program, stock, 0.020)


def test_a_checks_skin_is_turned_by_the_camera_tilt_the_fit_took_out_of_the_holds():
    """The fit takes the wrist camera's turn out of the holds' skin, about each tip, in the page's frame (camera_tilt);
    a check's skin, seen through the same camera, carries the same turn and is read turned alike (check_miss,
    untilt). Unturned, a live forearm's check stood 0.16 mm higher, and the run refused at 1.01 mm (2026-10-20). Here
    the skin one side of the tip, turned 12 mrad: 0.17 mm off unturned, on the field turned."""
    from tatbot_motion.surface import fit_height

    page = geometry.translation([0.30, 0.10, 0.05])
    page[:3, :3] = _rot([0.0, 0.0, 1.0], 30.0) @ _rot([1.0, 0.0, 0.0], 8.0)
    grid = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 31), np.linspace(-0.03, 0.03, 31)), -1).reshape(-1, 2)
    field = fit_height(np.c_[grid, np.zeros(len(grid))], degree=2)
    rng, tip, w = np.random.default_rng(0), np.array([0.0, 0.0, 0.010]), np.array([0.0, 0.012])
    side = np.c_[rng.uniform(0.008, 0.020, 400), rng.uniform(-0.005, 0.005, 400), np.zeros(400)]

    def base(local):
        return np.c_[local, np.ones(len(local))] @ page[:3, :].T
    seen, tip_b = base(curved._turned(side, tip, w)), base([tip])[0]
    raw, point = curved.check_miss(page, field, seen, base([[0.0, 0.0, 0.0]])[0])
    turned, _ = curved.check_miss(page, field, seen, base([[0.0, 0.0, 0.0]])[0], tip_b, -w)
    assert raw == pytest.approx(-0.012 * 0.014, abs=3e-5) and abs(turned) < 1e-6 and abs(point) < 1e-9
    assert np.allclose(curved.untilt(page, seen, tip_b, -w), base(side), atol=1e-9)
    assert np.array_equal(curved.untilt(page, seen, None, -w), seen) and np.array_equal(curved.untilt(page, seen, tip_b, None), seen)


# --- a stroke rides the skin its hover read along its own path (skin_along, ride_profile, Ridden) --------------------


def _tilted(surface, uv, skin_of, rng):
    """Skin points (base) over print points uv standing skin_of(uv) (m) off `surface` along its normals, 0.1 mm noise."""
    points, normals = surface.lift(uv)
    return points + (skin_of(uv) + rng.normal(0.0, 0.0001, len(uv)))[:, None] * normals


def _heart_surface():
    """A 25 mm forearm's field under a page turned and raised off the base."""
    from tatbot_motion.surface import PageSurface, fit_height

    h = _arm(0.025, 3.0)
    g = np.stack(np.meshgrid(np.linspace(-0.02, 0.02, 41), np.linspace(-0.02, 0.02, 41)), -1).reshape(-1, 2)
    page = geometry.translation([0.33, 0.02, 0.07])
    page[:3, :3] = _rot([0.0, 0.0, 1.0], 30.0) @ _rot([1.0, 0.0, 0.0], 5.0)
    return PageSurface(page, fit_height(np.c_[g, h(g)], degree=4))


@pytest.mark.parametrize("tilt_m", [0.0015, 0.0025])
def test_a_stroke_rides_its_skin_rising_freely_and_going_under_its_plan_at_most_a_millimetre(tilt_m):
    """The tiny heart's skin sank on one lobe against the other under one field (2026-10-21). A 14 mm stroke whose
    skin tilts from -tilt to +tilt along it: the ride follows its rise, goes at most profile_lower_m (1 mm) under the
    plan where it sank. Under profile_setup_m (4 mm) it is ridden, not a move: a 2.35 mm read set the snake's page up
    again at 2 mm and its re-fit tilted the pen into joint_3's limit (2026-10-23)."""
    rigid, rng, surface = curved.RIGID, np.random.default_rng(5), _heart_surface()
    path = np.c_[np.linspace(-0.007, 0.007, 29), np.full(29, 0.002)]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

    def skin_of(uv):
        return tilt_m * np.clip(uv[:, 0] / 0.007, -1.0, 1.0)
    uv = np.c_[rng.uniform(-0.010, 0.010, 4000), rng.uniform(-0.002, 0.006, 4000)]
    read = curved.skin_along(surface, _tilted(surface, uv, skin_of, rng), path, rigid["profile_near_m"])
    assert np.isfinite(read).all() and np.allclose(read, skin_of(path), atol=0.0003)
    part, ride = curved.ride_profile(read, arc, 0.0, 0.0, rigid)
    assert np.nanmax(np.abs(part)) < rigid["profile_setup_m"]
    assert ride.min() == pytest.approx(-rigid["profile_lower_m"], abs=1e-9)
    assert ride[-1] == pytest.approx(tilt_m, abs=0.0004)                      # the rise followed
    assert np.all(np.abs(np.diff(ride)) <= rigid["profile_slope"] * np.diff(arc) + 1e-9)
    assert np.all(ride >= np.maximum(part, -rigid["profile_lower_m"]) - 1e-9)   # never under the skin, clamped


def test_where_the_hover_read_no_skin_the_stroke_rides_as_before_and_never_lower():
    """Half a stroke inside the hover's blind 8 mm: its skin read under the field on the half seen goes under the plan
    there, at most a millimetre; the unread half rides the ring's ride (the fallback, never under the plan), and the
    change between them is raised to profile_slope, so no step goes into the skin. No skin read at all: no profile."""
    rigid, rng, surface = curved.RIGID, np.random.default_rng(6), _heart_surface()
    path = np.c_[np.linspace(-0.007, 0.007, 29), np.zeros(29)]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    uv = np.c_[rng.uniform(-0.010, -0.001, 3000), rng.uniform(-0.003, 0.003, 3000)]
    read = curved.skin_along(surface, _tilted(surface, uv, lambda p: np.full(len(p), -0.0008), rng), path,
                             rigid["profile_near_m"])
    seen = np.isfinite(read)
    assert seen[:8].all() and not seen[-10:].any()
    part, ride = curved.ride_profile(read, arc, 0.0003, 0.0, rigid)
    assert ride[:4] == pytest.approx(np.full(4, -0.0008), abs=0.0002) and np.all(ride[~seen] >= 0.0003 - 1e-12)
    assert np.all(np.abs(np.diff(ride)) <= rigid["profile_slope"] * np.diff(arc) + 1e-9)
    assert not np.isfinite(curved.skin_along(surface, np.zeros((0, 3)), path, 0.002)).any()


def test_a_ridden_surface_lifts_its_strokes_points_by_the_nearest_path_points_ride_and_nothing_else():
    """curved.Ridden: a stroke planned on it lifts each print point by its nearest path point's ride along the normal,
    on top of the pen's height; the surface's other answers (its pose, a point's height over it) are the field's."""
    surface = _heart_surface()
    path, extra = np.array([[0.0, 0.0], [0.005, 0.0], [0.010, 0.0]]), np.array([0.0, 0.0005, 0.001])
    ridden = curved.Ridden(surface, path, extra)
    uv = np.array([[0.0001, 0.0], [0.0049, 0.0002], [0.0102, 0.0]])
    (p0, n0), (p1, n1) = surface.lift(uv, 0.002), ridden.lift(uv, 0.002)
    assert np.allclose(n0, n1) and np.allclose(np.sum((p1 - p0) * n0, axis=1), extra)
    assert not ridden.flat and np.array_equal(ridden.base, surface.base)
    from tatbot_motion.surface import PageSurface
    assert not curved.Ridden(PageSurface(np.eye(4)), path, extra).flat and curved.Ridden(PageSurface(np.eye(4)), path,
                                                                                          0 * extra).flat
    assert ridden.height_over(p0[0])[0] == pytest.approx(surface.height_over(p0[0])[0])
    moved, plain = ridden.moved(np.eye(4)).lift(uv), surface.moved(np.eye(4)).lift(uv)
    assert np.allclose(np.sum((moved[0] - plain[0]) * plain[1], axis=1), extra)


def test_the_turned_checks_lift_runs_along_the_drawing_between_them_held_beyond_and_never_deeper():
    """On a skin round a bottle the hovers turned as the strokes read it 1.4-1.8 mm over the holds' field at one end of
    the drawing and 0.0 at the other (2026-10-24): two turned checks' readings run linearly between them along the
    line through them, held at either end; one runs to 0 at its mirror through the centre; a reading under the field
    lifts nothing."""
    at, read = [[0.0, 0.021], [0.0, -0.021]], [0.0015, 0.0]
    uv = np.array([[0.0, 0.030], [0.004, 0.021], [0.0, 0.0], [0.0, -0.021], [0.0, -0.040]])
    assert curved.turned_lift(at, read, uv) == pytest.approx([0.0015, 0.0015, 0.00075, 0.0, 0.0])
    assert curved.turned_lift(at[:1], read[:1], [[0.0, 0.021], [0.0, 0.0], [0.0, -0.021]]) == pytest.approx(
        [0.0015, 0.00075, 0.0])
    assert curved.turned_lift(at, [-0.001, 0.001], [[0.0, 0.021], [0.0, -0.021]]) == pytest.approx([0.0, 0.001])
    assert curved.turned_lift([], [], [[0.0, 0.0]]) == pytest.approx([0.0])


def test_a_turned_surface_lifts_the_strokes_by_the_turned_checks_and_answers_the_rest_as_the_field():
    """curved.Turned: what the planners draw on lifts each print point by the turned checks' lift along the normal,
    under a stroke's own ride (Ridden over it adds); its pose and a point's height over it are the field's."""
    surface = _heart_surface()
    turned = curved.Turned(surface, [[0.0, 0.01], [0.0, -0.01]], [0.001, 0.0])
    uv = np.array([[0.0, 0.01], [0.002, 0.0], [0.0, -0.01]])
    (p0, n0), (p1, n1) = surface.lift(uv, 0.002), turned.lift(uv, 0.002)
    assert np.allclose(n0, n1) and np.allclose(np.sum((p1 - p0) * n0, axis=1), [0.001, 0.0005, 0.0])
    assert turned.height_over(p0[0])[0] == pytest.approx(surface.height_over(p0[0])[0])
    ridden = curved.Ridden(turned, uv, np.array([0.0002, 0.0, 0.0]))
    assert np.allclose(np.sum((ridden.lift(uv, 0.002)[0] - p0) * n0, axis=1), [0.0012, 0.0005, 0.0])
    moved, plain = turned.moved(np.eye(4)).lift(uv), surface.moved(np.eye(4)).lift(uv)
    assert np.allclose(np.sum((moved[0] - plain[0]) * plain[1], axis=1), [0.001, 0.0005, 0.0])
    from tatbot_motion.surface import PageSurface
    assert curved.Turned(PageSurface(np.eye(4)), [[0.0, 0.0]], [-0.001]).flat
    assert not curved.Turned(PageSurface(np.eye(4)), [[0.0, 0.0]], [0.001]).flat


def test_a_strokes_ride_takes_out_what_the_pen_rides_by_already_at_each_point():
    """ride_profile's rise: the surface check's one number, or the turned checks' lift at each path point; the skin's
    part is the reading less it there."""
    rigid = curved.RIGID
    arc = np.linspace(0.0, 0.014, 15)
    read = np.full(15, 0.0015)
    one, _ = curved.ride_profile(read, arc, 0.0, 0.0005, rigid)
    each, ride = curved.ride_profile(read, arc, 0.0, np.linspace(0.0, 0.0015, 15), rigid)
    assert one == pytest.approx(np.full(15, 0.001))
    assert each[0] == pytest.approx(0.0015, abs=1e-6) and each[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(each) < 0) and np.all(ride >= -rigid["profile_lower_m"] - 1e-12)
