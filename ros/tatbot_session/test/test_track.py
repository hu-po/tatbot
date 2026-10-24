"""track.Tracker: the pen tip on the print, read in the wrist's frames and logged into the filter (ros/README.md 4.4,
stencil tracking)."""
import json

import numpy as np
import pytest
from tatbot_session.page_filter import PageFilter
from tatbot_session.track import Tracker, consistently_off

GAUGE = [[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02], [0.0, 0.0]]


def test_a_fit_the_runs_page_cannot_find_is_tried_from_the_overheads(monkeypatch):
    """A sheet slid 50 mm is past the wrist fit's reach from the run's page; the overhead saw where it went."""
    from tatbot_session import track

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, markers=None):
        found = cam_from_page[1, 3] > 0.01
        return {"tip_m": [0.0, 0.0, 0.011], "cam_from_print": np.eye(4).tolist(), "markers": 9 if found else 1,
                "ids": list(range(9 if found else 1)), "score": 1.0, "margin": 1.0 if found else 0.1,
                "rms_m": 0.0008, "accepted": found}
    monkeypatch.setattr(track, "_TILE", object())
    monkeypatch.setattr(track.gauge, "measure", lambda *a, **k: object())
    monkeypatch.setattr(track.ins, "hold_tip", hold_tip)
    moved = np.eye(4)
    moved[1, 3] = 0.05
    job = {"bgr": None, "depth": None, "meta": {}, "q": None, "used": np.eye(4), "tip_cam": None, "cal": {},
           "page": {}, "base_from_camera": np.eye(4)}
    assert not track._measure({**job, "overhead": None})["accepted"]
    out = track._measure({**job, "overhead": moved})
    assert out["accepted"] and out["prior"] == "overhead" and out["inliers"] == 9


def test_the_overheads_prior_is_the_runs_page_moved_in_its_own_plane_as_the_sheet_moved():
    """The wrist fit's retry prior: the run's page moved by the sheet's slide since setup that it has not followed
    (ArmExecutor.sheet_slide, in-plane: by the print's markers when kept), never the overhead pose's tilt or height."""
    from types import SimpleNamespace

    tracker = Tracker.__new__(Tracker)
    used, slide = np.eye(4), np.eye(4)
    used[:3, 3] = [0.3, 0.0, 0.002]
    slide[:3, 3] = [0.0, 0.047, 0.0]
    tracker.ex = SimpleNamespace(sheet_slide=lambda: slide, slid=np.eye(4))
    np.testing.assert_allclose(tracker._overhead(used)[:3, 3], [0.3, 0.047, 0.002])
    tracker.ex.slid = slide                                              # the page followed it already
    assert tracker._overhead(used) is None
    tracker.ex.slid, slide[:3, 3] = np.eye(4), [0.0, 0.001, 0.0]         # under 2 mm: the same prior
    assert tracker._overhead(used) is None
    tracker.ex.sheet_slide = lambda: None                                # no reading since setup
    assert tracker._overhead(used) is None


def test_a_reading_is_the_print_markers_on_the_gauges_surface(monkeypatch, tmp_path):
    """The tip on the print as the gauge holds read it (inspect.hold_tip), accepted as hold_tip accepts it; no tile,
    no gauge surface or no marker seen is not found, with the reason in the log."""
    from types import SimpleNamespace

    from tatbot_session import track

    surface, seen = object(), {}

    def hold_tip(colors, frames, meta, tip_cam, cam_from_page, page, tile, markers=None):
        seen.update(frames=frames, cam_from_page=cam_from_page)
        return None if meta.get("blank") else {"tip_m": [0.001, 0.002, 0.012], "cam_from_print": np.eye(4).tolist(),
                                                "markers": meta["markers"], "ids": list(range(meta["markers"])),
                                                "score": 1.0, "margin": 1.0, "rms_m": 0.0009,
                                                "accepted": meta["markers"] >= 2}
    monkeypatch.setattr(track.gauge, "measure", lambda depth, bgr, meta, cal: surface if depth else None)
    monkeypatch.setattr(track.ins, "hold_tip", hold_tip)
    used = np.eye(4)
    used[0, 3] = 0.3
    job = {"bgr": None, "depth": 1, "meta": {"markers": 7}, "q": None, "used": used, "tip_cam": None, "cal": {},
           "page": {"size_m": [0.111, 0.1575], "clear_m": [0.111, 0.1575]}, "base_from_camera": np.eye(4),
           "overhead": None}
    monkeypatch.setattr(track, "_TILE", None)
    assert track._measure(job)["error"] == "no fiducial-tile print installed for the page"
    monkeypatch.setattr(track, "_TILE", object())
    out = track._measure(job)
    assert out["fit"] == "markers" and out["accepted"] and out["inliers"] == 7 and out["tip_m"][1] == 0.002
    assert seen["frames"] == [surface] and np.allclose(seen["cam_from_page"], used)   # the camera is base here
    assert not track._measure({**job, "meta": {"markers": 1}})["accepted"]
    out = track._measure({**job, "meta": {"blank": True}})
    assert not out.get("fit") and out["error"] == "no marker of the print seen"
    monkeypatch.setattr(track, "print_read", lambda job, markers=None: {"print_none": "no marker of the print read"})
    out = track._measure({**job, "depth": 0})                            # no surface by the tip: not a missing print
    assert not out.get("fit") and out["surface"] is False
    assert out["error"] == "gauge read no surface; no marker of the print read"

    events = []
    tracker = Tracker.__new__(Tracker)
    tracker.field, tracker._fed, tracker.ends, tracker.rigid = PageFilter(), 0.0, [], None
    tracker.ex = SimpleNamespace(run_dir=tmp_path, event=lambda kind, text, op_id="": events.append(text),
                                 _timed_moves=lambda since: [])
    row = {**track._measure(job), "tip_m": [0.002, 0.001, 0.011]}
    assert tracker._log("s0001", "s0001", np.array([0.002, 0.0]), np.zeros(2), 1.0, row) == "added"
    assert "7 markers, rms 0.90 mm" in events[0] and "+0.00 +1.00 mm from the plan" in events[0]
    assert np.allclose(tracker.field.samples[0][2], [0.002, 0.001])


def _wrist_frame(base_from_print):
    """A wrist frame of the forearm fiducial-tile print (scripts/lib/stencil_tile.py, seed 0) with no pen in view:
    its colour, raw depth and their metadata (the D405's colour and depth one view), the camera's pose; the Tile."""
    import cv2

    pytest.importorskip("PIL")
    import stencil_tile
    from fiducials import stencil
    from tatbot_session import inspect as ins

    ink, record = stencil_tile.build("forearm", 0)
    (ox, oy), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    print_from_cam = ins.look_at(np.eye(4), (0.005, 0.0), tilt_rad=np.radians(20), azimuth_rad=np.radians(200),
                                 distance_m=0.13)
    v, u = np.mgrid[0:720, 0:1280].astype(float)
    rays = np.stack([(u - 640.0) / 640.0, (v - 360.0) / 640.0, np.ones_like(u)], axis=-1)
    o, d = print_from_cam[:3, 3], rays @ print_from_cam[:3, :3].T
    z = -o[2] / d[..., 2]                                                 # each pixel's depth to the print
    hit = (o[:2] + z[..., None] * d[..., :2]) * 1000
    sx = ((hit[..., 0] + tw / 2 + ox) * stencil_tile.PPM - 0.5).astype(np.float32)
    sy = ((th / 2 - hit[..., 1] + oy) * stencil_tile.PPM - 0.5).astype(np.float32)
    grey = (90 + cv2.remap(ink, sx, sy, cv2.INTER_AREA, borderValue=255).astype(np.float32) * 110 / 255).astype(np.uint8)
    meta = {"depth_units_m": 1e-4, "intrinsics": {"fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0] * 3},
            "color_intrinsics": {"k": [[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]], "coeffs": [0.0] * 5}}
    return {"bgr": np.dstack([grey] * 3), "depth": np.round(z / 1e-4).astype(np.uint16), "meta": meta,
            "base_from_camera": base_from_print @ print_from_cam}, stencil.tile(record)


def test_a_frame_whose_gauge_read_no_surface_still_reads_the_print_on_its_own_depth(monkeypatch):
    """Hovering 6 mm over a forearm print (2026-10-19) the gauge read no pen or no surface by the tip in 43 of 61
    unread readings, each logged as no marker of the print seen. The print's markers lifted by the frame's own depth
    (print_read) still place it on the page in use, here 3 mm, -2 mm and 2 deg from it, and the camera's FK pose 1 cm
    off along the page's normal (the hovers' read) moves nothing."""
    from tatbot_session import gauge, track
    from tatbot_session import inspect as ins

    used = ins.rpy_matrix([0.32, 0.05, 0.03], [0.02, -0.01, -1.5])
    moved = ins.correction_matrix(0.003, -0.002, np.radians(2.0))
    frame, tile = _wrist_frame(used @ moved)
    monkeypatch.setattr(track, "_TILE", tile)
    job = {**frame, "used": used, "q": None, "tip_cam": np.zeros(3), "cal": {}, "overhead": np.eye(4)}
    assert gauge.measure(job["depth"], job["bgr"], job["meta"], {}) is None
    out = track._measure(job)
    assert out["surface"] is False and not out.get("fit") and out["print_markers"] >= 8
    err = ins.correction_matrix(*out["print_fit"]) @ moved                # the reading undoes the move
    assert np.linalg.norm(err[:2, 3]) < 3e-4 and abs(np.degrees(np.arctan2(err[1, 0], err[0, 0]))) < 0.3
    at = np.asarray(out["print_at"])
    assert np.allclose(out["print_off_m"], (moved @ [*at, 0.0, 1.0])[:2] - at, atol=3e-4)
    assert out["error"].startswith("gauge read no surface; print ") and out["error"].endswith(
        f"mm off the page by {out['print_markers']} markers")
    lifted = used @ ins.rpy_matrix([0.0, 0.0, 0.01], [0.0, 0.0, 0.0]) @ np.linalg.inv(used) @ job["base_from_camera"]
    assert np.allclose(track.print_read({**job, "base_from_camera": lifted})["print_fit"], out["print_fit"], atol=1e-9)


def test_a_readings_markers_go_to_the_wrist_pane_at_its_grabs_stamp_and_stay_out_of_track_jsonl(monkeypatch, tmp_path):
    """Every marker a reading decodes in its frame, kept by its fit or not, is handed to the camera's tap (the viewer's
    wrist pane) as the worker returns it, at the stamp the tap published that frame at, not a later grab's; the row
    logged to track.jsonl does not carry them."""
    import concurrent.futures
    from types import SimpleNamespace

    from tatbot_session import inspect as ins
    from tatbot_session import track

    used = ins.rpy_matrix([0.32, 0.05, 0.03], [0.02, -0.01, -1.5])
    frame, tile = _wrist_frame(used)
    monkeypatch.setattr(track, "_TILE", tile)
    out = track._measure({**frame, "used": used, "q": None, "tip_cam": np.zeros(3), "cal": {}, "overhead": None})
    assert out["surface"] is False and set(out["print_sol"]["ids"]) <= set(out["shown"]) and len(out["shown"]) >= 8
    assert all(np.shape(c) == (4, 2) for c in out["shown"].values())

    shown, futures = [], []
    cam = SimpleNamespace(stamp_ns=None, raw_depth=[frame["depth"]], depth_metadata=frame["meta"],
                          tap=SimpleNamespace(markers=shown.append, log=pytest.fail))

    def grab_both(n, fresh=False):
        cam.stamp_ns = 1_800_000_000 + len(futures)
        return [frame["bgr"]], []
    cam.grab_both = grab_both
    tracker = Tracker.__new__(Tracker)
    tracker.cam, tracker.pending, tracker.hover_frames, tracker.tip_cam, tracker.cal = cam, [], 1, np.zeros(3), {}
    tracker.pool = SimpleNamespace(submit=lambda fn, job: futures.append(concurrent.futures.Future()) or futures[-1])
    tracker._overhead, tracker._camera = (lambda used: None), (lambda q: np.eye(4))
    kin = SimpleNamespace(fk=lambda q: np.eye(4))
    tracker.ex = SimpleNamespace(used=lambda: np.eye(4), stack={"page": {"size_m": [0.1, 0.1], "clear_m": [0.1, 0.1]}},
                                 _curved=lambda: False, surface=lambda: None, kin=kin, skin_tilt=lambda: None,
                                 run_dir=tmp_path, _followed=lambda: np.eye(3), _timed_moves=lambda since: [],
                                 clock=SimpleNamespace(now=lambda: 0.0),
                                 io=SimpleNamespace(q=np.zeros(7)), event=lambda kind, text, op_id="": None)
    tracker._grab("s0001-hover", "s0001", np.zeros(2))
    tracker._grab("s0001", "s0001", np.zeros(2))                         # the next grab before the first is read
    futures[0].set_result(out)
    futures[1].set_result({"fit": None, "error": "no marker of the print seen", "shown": {}})
    (hover,) = shown                                                       # the second read no marker
    assert (hover["frame_ns"], hover["reading"], hover["size"]) == (1_800_000_000, "s0001-hover", [1280, 720])
    assert sorted(m["id"] for m in hover["markers"] if m["fit"]) == out["print_sol"]["ids"]
    tracker.field, tracker._fed, tracker.ends, tracker.rigid = PageFilter(), 0.0, [], None
    tracker.print_held, tracker.printed = [], []
    tracker.drain(wait=True)
    rows = [json.loads(line) for line in (tmp_path / "track.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and not any("shown" in row for row in rows) and rows[0]["print_markers"] >= 8


def _lifted(tile, base_from_print, ids, radius_m=None, noise_m=0.0002, seed=0):
    """Markers `ids` of `tile` laid on its print (round a cylinder about print v of radius_m, unstretched) as a view
    looking down on them lifts their corners by depth: id -> (4, 3) camera metres, and base_from_camera."""
    rng = np.random.default_rng(seed)
    base_from_camera = base_from_print @ np.array([[1.0, 0, 0, 0.004], [0, -1.0, 0, 0.002], [0, 0, -1.0, 0.16],
                                                   [0, 0, 0, 1.0]])
    out = {}
    for i in ids:
        u, v = tile.corners[i].T
        x, z = (u, 0 * u) if radius_m is None else (radius_m * np.sin(u / radius_m), radius_m * (np.cos(u / radius_m) - 1))
        base = (base_from_print @ np.c_[x, v, z, np.ones(4)].T).T
        cam = (np.linalg.inv(base_from_camera) @ base.T).T[:, :3]
        out[i] = cam + rng.normal(0.0, noise_m, cam.shape)
    return out, base_from_camera


def test_a_print_round_a_forearm_is_judged_about_the_skins_curve_not_a_plane():
    """About the flat layout, in its plane, a forearm's own curve reads as error: over 57 locate views of forearm prints
    (2026-10-20/21) 1.08 mm median where the layout laid round the skin read 0.88 across the corners' rays, and two
    views at the 1.6 mm cap read 1.13 and 1.53. The whole forearm print round a slim 26 mm wrist (as tight as its layout
    bends), its corners 0.3 mm out: about a plane past the cap, about the skin's curve within it, and accepted."""
    pytest.importorskip("PIL")
    import stencil_tile
    from fiducials import stencil
    from tatbot_session import track

    tile = stencil.tile(stencil_tile.build("forearm", 0)[1])
    ids = sorted(tile.corners)
    seen, base_from_camera = _lifted(tile, np.eye(4), ids, radius_m=0.026, noise_m=0.0003)
    points = np.concatenate([seen[i] for i in ids])
    target = np.concatenate([stencil.target_corners(tile, i) for i in ids])
    flat = stencil.kabsch(target, points)
    off = (np.c_[target, np.ones(len(target))] @ flat.T)[:, :3] - points
    off -= np.outer(off @ flat[:3, 2], flat[:3, 2])
    assert np.sqrt(np.mean(np.sum(off ** 2, axis=1))) > track.LIFTED_RMS_M
    fit = track.lifted_fit(seen, tile, base_from_camera, np.eye(4))
    assert fit["accepted"] and fit["markers"] == len(ids) and fit["rms_m"] < 0.0006


def _placement_error(fit, moved):
    from tatbot_session import inspect as ins

    err = ins.correction_matrix(fit["tx_m"], fit["ty_m"], fit["theta_rad"]) @ moved
    return float(np.linalg.norm(err[:2, 3])), abs(float(np.degrees(np.arctan2(err[1, 0], err[0, 0]))))


def test_the_print_is_placed_by_its_markers_lifted_corners_whatever_the_tilt_of_the_page_it_is_put_on():
    """2026-10-21: a fresh run's blind search stood on the last run's page, the forearm since at another place and
    angle. Carried onto that page's plane, a print tilted 25 deg off it reads its markers shrunk and their spacing
    short by the tilt's cosine; the corners' 3-D fit (track.lifted_fit) places it at its slide and turn in that plane
    within 0.3 mm and 0.3 deg whatever the tilt, on a 40 mm forearm's curve too. A misread code (a marker read as
    another id) keeps no printed distance to the rest and is left out; a lone marker a pitch from where the page puts
    it is no placement (lone_ok)."""
    pytest.importorskip("PIL")
    import stencil_tile
    from fiducials import stencil
    from tatbot_session import inspect as ins
    from tatbot_session import track

    tile = stencil.tile(stencil_tile.build("forearm", 0)[1])
    used = ins.rpy_matrix([0.32, 0.05, 0.03], [0.02, -0.01, -1.5])
    moved = ins.correction_matrix(0.003, -0.002, np.radians(2.0))
    ids = sorted(tile.corners, key=lambda i: np.linalg.norm(tile.corners[i].mean(axis=0) - [0.008, 0.016]))[:4]
    #  ^ a 2 x 2 piece off the print's centre, as a view sees part of it
    tilted = ins.rpy_matrix([0.008, 0.004, 0.0], [np.radians(25.0), np.radians(8.0), 0.0])     # the skin, not the page
    for radius in (None, 0.040):
        seen, base_from_camera = _lifted(tile, used @ np.linalg.inv(moved) @ tilted, ids, radius)
        fit = track.lifted_fit(seen, tile, base_from_camera, used)
        assert fit["accepted"] and fit["ids"] == sorted(ids) and fit["rms_m"] <= track.LIFTED_RMS_M
        slide, turn = _placement_error(fit, np.linalg.inv(moved) @ ins.rpy_matrix(
            [*(tilted[:2, 3]), 0.0], [0.0, 0.0, float(np.arctan2(tilted[1, 0], tilted[0, 0]))]))
        assert slide < 3e-4 and turn < 0.3 and fit["tilt_deg"] > 20.0
    # as carried onto the page's plane before (each corner's x, y there): the markers shrink, the print is misplaced
    planar = {i: (np.linalg.inv(used) @ base_from_camera @ np.c_[c, np.ones(4)].T)[:2].T for i, c in seen.items()}
    old = stencil.fit_points(planar, tile)
    assert old is None or not old["accepted"] or _placement_error(old, np.linalg.inv(moved) @ ins.rpy_matrix(
        [*(tilted[:2, 3]), 0.0], [0.0, 0.0, float(np.arctan2(tilted[1, 0], tilted[0, 0]))]))[0] > 0.001
    wrong = next(i for i in tile.corners if i not in ids)   # a marker read as another code
    misread = {**{i: c for i, c in seen.items() if i != ids[0]}, wrong: seen[ids[0]]}
    fit = track.lifted_fit(misread, tile, base_from_camera, used)
    assert fit["accepted"] and wrong not in fit["ids"] and fit["markers"] == 3
    far = ins.correction_matrix(0.02, 0.0, 0.0)               # one marker, 20 mm (over a pitch) from the page's
    lone, cam = _lifted(tile, used @ np.linalg.inv(far), ids[:1])
    assert not track.lifted_fit(lone, tile, cam, used)["accepted"]


def test_a_transfer_stretched_with_the_skin_is_placed_and_a_misread_or_a_false_code_still_is_not():
    """2026-10-21, a faded transfer on a forearm turned since it was laid: its true 15 and 23 lifted 10.4-13.9 mm a
    side (printed 12) and their corners -2.6..+1.8 mm off their printed distances, past the 15% size and 2.5 mm
    checks, and no view placed the print. A print stretched 12% along u and pressed 8% across it, a 2 x 2 piece, is
    placed within 0.5 mm and 1 deg of where it lies. A code read a pitch from where it is printed (a misread) and one
    read off the print (a false code at its printed size) are each left out, and alone place nothing; the reads of a
    code at two places are kept apart (track._read_places), the median of each."""
    import dataclasses

    pytest.importorskip("PIL")
    import stencil_tile
    from fiducials import stencil
    from tatbot_session import inspect as ins
    from tatbot_session import track

    tile = stencil.tile(stencil_tile.build("forearm", 0)[1])
    used = ins.rpy_matrix([0.32, 0.05, 0.03], [0.02, -0.01, -1.5])
    moved = ins.correction_matrix(0.002, 0.010, np.radians(2.0))
    ids = sorted(tile.corners, key=lambda i: np.linalg.norm(tile.corners[i].mean(axis=0) - [0.008, 0.016]))[:4]
    centre = np.mean([tile.corners[i].mean(axis=0) for i in ids], axis=0)
    skin = dataclasses.replace(tile, corners={i: (c - centre) * [1.12, 0.92] + centre for i, c in tile.corners.items()})
    seen, cam = _lifted(skin, used @ np.linalg.inv(moved), ids, 0.040)
    fit = track.lifted_fit(seen, tile, cam, used)
    assert fit["accepted"] and fit["ids"] == sorted(ids)
    slide, turn = _placement_error(fit, np.linalg.inv(moved))
    assert slide < 5e-4 and turn < 1.0
    near = sorted(tile.corners, key=lambda i: np.linalg.norm(tile.corners[i].mean(axis=0) - tile.corners[ids[0]].mean(axis=0)))
    other = next(i for i in near if i not in ids)   # the code of the nearest marker not in view, read at ids[0]'s place
    misread = {(other, 0): seen[ids[0]], **{(i, 0): seen[i] for i in ids[1:]}}
    fit = track.lifted_fit(misread, tile, cam, used)
    assert fit["accepted"] and other not in fit["ids"] and "off the others' printed distances" in fit["why"]
    false = seen[ids[1]] + [0.09, 0.05, 0.02]        # a code at its printed size 100 mm off, on the mat
    fit = track.lifted_fit({**{(i, 0): seen[i] for i in ids[:2]}, (ids[2], 1): false}, tile, cam, used)
    assert fit["accepted"] and fit["ids"] == sorted(ids[:2])
    assert not track.lifted_fit({(ids[2], 1): false}, tile, cam, used)["accepted"]
    square = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]])
    places = track._read_places([(23, square + 640), (23, square + 641), (23, square + [1074, 584]), (12, square)])
    assert sorted(places) == [(12, 0), (23, 0), (23, 1)] and np.allclose(places[(23, 0)], square + 640.5)


def test_a_curved_pages_locate_view_reads_the_print_by_each_corners_depth_not_the_overheads_plane():
    """A curved page's locate view (ArmExecutor._view_read) lifts each marker corner by its own depth: a prior 9 mm
    over the print, as the overhead's plane stood over a forearm's skin (2026-10-19), still reads the print's 3 mm,
    -2 mm and 2 deg; rectified onto that prior's plane, as a flat page's view is, the same view reads it mm off."""
    from types import SimpleNamespace

    from tatbot_session import inspect as ins
    from tatbot_session.executor import ArmExecutor

    used = ins.rpy_matrix([0.32, 0.05, 0.03], [0.02, -0.01, -1.5])
    moved = ins.correction_matrix(0.003, -0.002, np.radians(2.0))
    frame, tile = _wrist_frame(used @ moved)
    prior = used @ ins.rpy_matrix([0.0, 0.0, 0.009], [0.0, 0.0, 0.0])
    shown = []   # the viewer's wrist pane: the view's markers at the frame the tap published (wrist_tap)
    cam = SimpleNamespace(raw_depth=[frame["depth"]] * 3, depth_metadata=frame["meta"], stamp_ns=1_800_000_000,
                          k=np.asarray(frame["meta"]["color_intrinsics"]["k"]), dist=np.zeros(5),
                          tap=SimpleNamespace(markers=shown.append, log=pytest.fail))
    args = (cam, [frame["bgr"]] * 3, frame["base_from_camera"], prior, tile, (ins.page_extent(tile.size_m), 5.0))
    _, sol, _ = ArmExecutor._view_read(*args, True)
    err = ins.correction_matrix(sol["tx_m"], sol["ty_m"], sol["theta_rad"]) @ moved
    assert np.linalg.norm(err[:2, 3]) < 3e-4 and abs(np.degrees(np.arctan2(err[1, 0], err[0, 0]))) < 0.3
    assert sol["markers"] >= 8 and sol["sides"] == ["markers"] and len(sol["sigma"]) == 3   # as combine_views takes
    (view,) = shown
    assert (view["frame_ns"], view["size"], view["reading"], view["accepted"]) == (1_800_000_000, [1280, 720], "locate",
                                                                                   True)
    assert sorted(m["id"] for m in view["markers"] if m["fit"]) == sol["ids"] and len(view["markers"]) >= sol["markers"]
    assert all(0 <= u < 1280 and 0 <= v < 720 for m in view["markers"] for u, v in m["corners_px"])
    _, flat, _ = ArmExecutor._view_read(*args, False)
    assert flat is None or np.linalg.norm([flat["tx_m"] - sol["tx_m"], flat["ty_m"] - sol["ty_m"]]) > 0.002


def test_a_gauge_miss_is_held_to_where_the_holds_read_the_print(tmp_path):
    """The camera's FK pose puts the print ~5 mm off the page at the holds and the hovers alike, so a hover's gauge
    miss is judged by where the holds read it there (their median), with its own op's readings only: a sheet slid
    15 mm reads 15 mm."""
    from types import SimpleNamespace

    from tatbot_session import track

    tracker = Tracker.__new__(Tracker)
    tracker.field, tracker._fed, tracker.ends, tracker.rigid = PageFilter(), 0.0, [], None
    tracker.print_held, tracker.printed, events = [], [], []
    tracker.ex = SimpleNamespace(run_dir=tmp_path, event=lambda kind, text, op_id="": events.append(text),
                                 _timed_moves=lambda since: [])

    def hold(fit):
        return {"fit": "markers", "tip_m": [0.0, 0.0, 0.011], "cam_from_print": np.eye(4).tolist(), "inliers": 6,
                "score": 1.0, "margin": 1.0, "rms_m": 0.0008, "accepted": True, "print_fit": fit, "print_at": [0, 0]}

    def miss(fit, at=(0.004, -0.004)):
        return {"fit": None, "surface": False, "error": "gauge read no surface; print ...", "print_fit": fit,
                "print_at": list(at)}
    for k, fit in enumerate(([-0.0055, -0.0045, 0.002], [-0.0056, -0.0042, 0.0], [-0.0054, -0.0054, 0.001])):
        tracker._log(f"gauge-{k}", "", None, np.zeros(2), 0.0, hold(fit))     # the setup's holds (end None)
    assert tracker.print_moved("s0001") == []
    still = [-0.0060, -0.0040, 0.01]
    slid = [still[0] - np.cos(still[2]) * 0.015, still[1] - np.sin(still[2]) * 0.015, still[2]]   # 15 mm along x
    tracker._log("s0001-hover", "s0001", np.zeros(2), np.zeros(2), 1.0, miss(still))
    tracker._log("s0001-hover2", "s0001", np.zeros(2), np.zeros(2), 2.0, miss(slid))
    near, far = tracker.print_moved("s0001")
    assert near < 0.001 and abs(far - 0.015) < 0.001
    assert np.isclose(far, track.print_offset(tracker.print_held, slid, [0.004, -0.004]))
    tracker._log("s0002-hover", "s0002", np.zeros(2), np.zeros(2), 3.0, miss([-0.0055, -0.0045, 0.001]))
    assert tracker.print_moved("s0001") == [] and len(tracker.print_moved("s0002")) == 1
    assert events[3] == "track s0001-hover: gauge read no surface; print ..."
    assert len(tracker.field.samples) == 3                               # a miss adds nothing to the field


def test_two_strokes_ending_off_the_plan_one_way_are_consistently_off():
    """The incident's s0157 and s0158 ended 2.3 and 2.0 mm from the plan, one way; scattered or small ends are not."""
    s0157, s0158 = np.array([-0.00100, -0.00209]), np.array([-0.00093, -0.00172])
    assert consistently_off([s0157], 0.0015, 2) is None                      # one stroke is not a pattern
    assert np.allclose(consistently_off([np.zeros(2), s0157, s0158], 0.0015, 2), (s0157 + s0158) / 2)
    assert consistently_off([s0157, [0.0004, -0.0011]], 0.0015, 2) is None   # one under the limit
    assert consistently_off([[0.0018, 0.0], [-0.0018, 0.0]], 0.0015, 2) is None   # each past it, opposite ways
    assert consistently_off([s0157, s0158], 0.0015, 0) is None


def test_only_stroke_ends_count_toward_the_off_plan_pause(tmp_path):
    """A hover's reading carries its travel's older correction and a held one may be a misread: neither counts."""
    from types import SimpleNamespace

    tracker = Tracker.__new__(Tracker)
    tracker.field, tracker._fed, tracker.ends, tracker.pending = PageFilter(), 0.0, [], []
    tracker.off_m, tracker.off_count = 0.0015, 2
    tracker.ex = SimpleNamespace(run_dir=tmp_path, event=lambda kind, text, op_id="": None, _timed_moves=lambda since: [])

    def read(tip):
        return {"fit": "markers", "tip_m": [*tip, 0.011], "cam_from_print": np.eye(4).tolist(), "inliers": 9,
                "score": 1.0, "margin": 1.0, "rms_m": 0.0008, "accepted": True}
    plan, tcp = np.zeros(2), np.zeros(2)
    assert tracker._log("s0001-hover", "s0001", plan, tcp, 1.0, read([0.0, 0.0020])) == "added"
    assert tracker._log("s0001", "s0001", plan, tcp, 2.0, read([0.0, 0.0021])) == "added"
    assert tracker.off_plan() is None                                         # the hover did not count
    assert tracker._log("s0002", "s0002", plan, tcp, 3.0, read([0.0, -0.0025])) == "held"
    assert tracker.off_plan() is None                                         # nor did a held reading
    assert tracker._log("s0003", "s0003", plan, tcp, 4.0, read([0.0002, 0.0019])) == "added"
    mean, ends = tracker.off_plan()
    assert [op for op, _ in ends] == ["s0001", "s0003"] and np.allclose(mean, [0.0001, 0.002])


def test_on_a_curved_page_the_tracker_logs_and_moves_no_stroke(tmp_path):
    """2026-10-08, a skin pinned over a holder: the tracker's correction carried a 20 mm stroke 5 mm past the gauge
    holds onto skin turned 18 deg, and the needle went through the skin into the holder. A curved page's tracker
    corrects only with page.surface.track_correct."""
    from types import SimpleNamespace

    def tracker(mode, curved, **surface):
        stack = {"page": {"surface": surface}}
        ex = SimpleNamespace(run_dir=tmp_path, stack=stack, _curved=lambda: curved, overhead_at=0.0)
        return Tracker(ex, None, None, {"tip_cam_m": [0.0, 0.0, 0.15]}, {"mode": mode})

    assert tracker("correct", False).correcting and tracker("correct", False).rigid is None
    assert not tracker("correct", True).correcting
    assert tracker("correct", True, track_correct=True).correcting
    assert not tracker("log", False).correcting
    # it watches the stencil instead: the executor moves the whole surface with it (curved.RigidWatch)
    assert tracker("correct", True).rigid is not None and tracker("correct", True).hover_frames == 5
    assert tracker("log", True).rigid is None and tracker("correct", True, track_correct=True).rigid is None
    assert tracker("correct", True, rigid={"enabled": False}).rigid is None


def test_a_curved_pages_readings_are_judged_against_the_holds_and_stroke_ends_lose_their_residual(tmp_path):
    """The gauge holds read the print 1 mm off the TCP here (the arm's and the reading's own error): later readings
    that agree with them say the stencil stays; two 2.5 mm past them, 4 mm apart, say it moved, and their stroke
    end counts toward the off-plan pause without the holds' 1 mm (the filter's estimate of it: within 0.05 mm, its prior
    pulling it in a little)."""
    from types import SimpleNamespace

    from tatbot_motion.surface import PageSurface
    from tatbot_session import curved

    tracker = Tracker.__new__(Tracker)
    tracker.field, tracker._fed, tracker.ends, tracker.pending = PageFilter(), 0.0, [], []
    tracker.off_m, tracker.off_count, tracker.rigid = 0.0015, 2, curved.RigidWatch()
    flat = PageSurface(np.eye(4))
    tracker.ex = SimpleNamespace(run_dir=tmp_path, event=lambda kind, text, op_id="": None,
                                 surface=lambda: flat, _timed_moves=lambda since: [])

    def read(tip):
        return {"fit": "markers", "tip_m": [*tip, 0.011], "cam_from_print": np.eye(4).tolist(), "inliers": 9,
                "score": 1.0, "margin": 1.0, "rms_m": 0.0008, "accepted": True, "skin_base": [[0.0, 0.0, 0.0]] * 60}
    for k, u in enumerate(GAUGE):                                        # the setup's holds (end None: the seed)
        tracker._log(f"gauge-{k}", "", None, np.asarray(u), 0.0, read(np.asarray(u) + [0.001, 0.0]))
    assert len(tracker.field.samples) == 5 and tracker.skin.shape == (60, 3)
    tcp = np.array([0.005, 0.0])
    assert tracker._log("s0001-hover", "s0001", tcp, tcp, 1.0, read(tcp + [0.0013, 0.0004])) == "added"
    assert len(tracker.field.samples) == 5                               # the holds' field stays as it was
    assert tracker._log("s0001", "s0001", tcp, tcp, 2.0, read(tcp + [0.0035, 0.0])) == "held"
    far = np.array([-0.005, 0.0])
    assert tracker._log("s0002-hover", "s0002", far, far, 3.0, read(far + [0.0035, 0.0])) == "moved"
    plane, _ = tracker.rigid.step()
    assert np.allclose(plane[:2, 2], [-0.0025, 0.0], atol=5e-5)         # the print 2.5 mm +x: the page follows it
    rows = [json.loads(line) for line in (tmp_path / "track.jsonl").read_text().splitlines()]
    assert np.allclose(rows[-1]["moved_m"], [0.0025, 0.0], atol=5e-5) and "skin_base" not in rows[-1]
    assert tracker._log("s0002", "s0002", far, far, 4.0, read(far + [0.0035, 0.0])) == "moved"
    assert tracker.off_plan() is None                                    # one stroke end: not yet a pattern
    up = np.array([0.0, 0.008])
    assert tracker._log("s0003", "s0003", up, up, 5.0, read(up + [0.0035, 0.0])) == "moved"
    mean, ends = tracker.off_plan()
    assert [op for op, _ in ends] == ["s0002", "s0003"] and np.allclose(mean, [0.0025, 0.0], atol=5e-5)   # the holds' out


def test_the_tracker_takes_the_setups_camera_and_logs_its_readings_in_the_order_taken(tmp_path, monkeypatch):
    """The tracker opened its own wrist D405 (~2 s) after the setup's had closed, and read the holds' 9-11 frames one at
    a time in one worker (6 s, 2026-10-20). It takes the setup's camera, open already, once the holds' frames are
    written, and reads in three workers; whichever finishes first, the readings are logged in the order taken."""
    from types import SimpleNamespace

    from tatbot_session import gauge, track

    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"tip_cam_m": [0.0, 0.0, 0.15]}))
    monkeypatch.setattr(gauge, "calibration_path", lambda arm: cal)
    pools, done, cam = [], [], SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(track, "ProcessPoolExecutor", lambda n, **kw: pools.append(n) or SimpleNamespace())

    def opened(serial):
        raise AssertionError("a second camera opened")
    ex = SimpleNamespace(stack={"page": {"track": {"mode": "log"}, "pattern_id": "p", "locate": {}}}, arm="right",
                         sensing=SimpleNamespace(wrist="real"), run_dir=tmp_path, repo=tmp_path, _curved=lambda: False,
                         _settle=lambda: done.append("settled"), take_camera=lambda: cam, wrist_camera=opened,
                         event=lambda kind, text, op_id="": None, overhead_at=0.0)
    tracker = Tracker.start(ex)
    assert tracker.cam is cam and done == ["settled"] and pools == [3]
    logged = []
    monkeypatch.setattr(tracker, "_log", lambda name, *a, **k: logged.append(name) or name)

    def future(ready):
        return SimpleNamespace(done=lambda: ready[0], result=lambda: {})
    first, second = [False], [True]
    tracker.pending = [("a", "", None, None, 0.0, None, 0, future(first)),   # (..., follow, camera, future)
                      ("b", "", None, None, 0.0, None, 0, future(second))]
    assert tracker.drain() is None and logged == []                         # b is done, a is not: neither yet
    first[0] = True
    assert tracker.drain() == "b" and logged == ["a", "b"] and tracker.pending == []
