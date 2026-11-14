"""stencild samples (tatbot.target-pose/1) as the bridge reads them, and the pose helpers."""
import json
import math

import numpy as np
import pytest
from tatbot_bridge import page, stack, watch

PATTERN = "stencil-" + "a" * 64


def envelope(source="measured", pose=None, pattern=PATTERN, **extra):
    payload = {"target_id": pattern, "target_frame": "world", "source": source,
               "world_from_target": (np.eye(4).tolist() if pose is None else pose) if source == "measured" else None,
               "calibration_id": "bundle", "translation_sigma_m": 0.001 if source == "measured" else None,
               "rotation_sigma_rad": 0.01 if source == "measured" else None, "tag_ids": [],
               "support": {"physical_instance_id": "print-7", "physical_instance_identity_verified": True,
                           "motion_authority": False}}
    payload.update(extra)
    return json.dumps({"schema": page.SCHEMA, "producer": {"role": "track"}, "seq": 5,
                       "stamp": {"mono_ns": 0, "wall_ns": 1_790_000_000_123_456_789, "basis": "x"},
                       "payload": payload}).encode()


def test_measured_sample():
    s = page.parse(envelope(), PATTERN)
    # Guaranteed non-None because envelope() constructs a valid tatbot.target-pose/1 payload for PATTERN
    assert s is not None
    assert s["source"] == "measured" and np.allclose(s["world_from_target"], np.eye(4))
    assert s["stamp_ns"] == 1_790_000_000_123_456_789 and s["seq"] == 5
    assert (s["print_id"], s["identity_verified"], s["calibration_id"]) == ("print-7", True, "bundle")
    assert s["translation_sigma_m"] == 0.001


def test_lost_sample_has_no_pose():
    s = page.parse(envelope("lost"), PATTERN)
    # Guaranteed non-None because envelope("lost") produces a valid lost sample for PATTERN
    assert s is not None
    assert s["source"] == "lost" and s["world_from_target"] is None and math.isnan(s["translation_sigma_m"])


def test_other_prints_and_malformed_samples_are_ignored():
    assert page.parse(envelope(pattern="stencil-" + "b" * 64), PATTERN) is None
    assert page.parse(b"not json", PATTERN) is None
    assert page.parse(json.dumps({"schema": "tatbot.other/1"}), PATTERN) is None
    assert page.parse(envelope(pose=(2 * np.eye(4)).tolist()), PATTERN) is None   # not rigid
    assert page.parse(envelope(source="guessed"), PATTERN) is None
    s = page.parse(envelope(), None)
    # Guaranteed non-None because passing pattern_id=None matches any target_id in a valid envelope
    assert s is not None
    assert s["pattern_id"] == PATTERN


def pose_at(x, y=0.0):
    pose = np.eye(4)
    pose[:2, 3] = [x, y]
    return pose


def test_a_samples_copies_are_samples_of_their_own():
    other = {"world_from_target": pose_at(0.3).tolist(), "translation_sigma_m": 0.003, "rotation_sigma_rad": 0.02,
             "support": {"markers": {"read": 7}, "anchor_points_m": [[1, 2, 3]]}}
    bad = {**other, "world_from_target": (2 * np.eye(4)).tolist()}
    s = page.parse(envelope(support={"markers": {"read": 9}, "motion_authority": False, "bundle_id": "b",
                                     "copies": [other, bad]}), PATTERN)
    first, second = page.copies(s)
    assert np.allclose(first["world_from_target"], np.eye(4)) and "copies" not in first["support"]
    assert np.allclose(second["world_from_target"], pose_at(0.3)) and second["translation_sigma_m"] == 0.003
    assert second["support"]["markers"] == {"read": 7} and second["support"]["anchor_points_m"] == [[1, 2, 3]]
    assert second["support"]["bundle_id"] == "b" and "copies" not in second["support"]   # stencild's own, on each
    assert second["stamp_ns"] == first["stamp_ns"] and page.others_m(first, [first, second]) == [0.3]


def test_a_sample_without_copies_is_one_sheet():
    s = page.parse(envelope(), PATTERN)
    assert [sheet["support"] for sheet in page.copies(s)] == [s["support"]]


COPIES = {"same_m": 0.05}
PAGE, SPARE, FAR = 0.3, 0.5, 0.95     # three places along x, the arm's base at the origin


def sheet(x, read=20, t=0.0):
    return {"world_from_target": pose_at(x), "support": {"markers": {"read": read}}, "stamp_ns": int(t * 1e9)}


def chooser(place=None, chosen=False):
    return page.PageChooser(COPIES, place=None if place is None else [place, 0.0, 0.0], chosen=chosen)


def x_of(chosen):
    return None if chosen is None else round(float(chosen["world_from_target"][0, 3]), 3)


def test_the_page_stays_the_page_whichever_sheet_reads_more():
    c = chooser(PAGE)
    for t, (page_read, spare_read) in enumerate([(20, 21), (21, 20), (3, 22), (2, 22)]):
        turn = sorted([sheet(PAGE, page_read, t), sheet(SPARE, spare_read, t)], key=lambda s: -s["support"]["markers"]["read"])
        assert x_of(c.choose(turn)) == PAGE


def test_with_no_page_yet_one_sheet_in_view_is_it_and_of_several_none_until_one_is_chosen():
    assert x_of(chooser().choose([sheet(FAR)])) == FAR
    c = chooser()
    assert c.choose([sheet(FAR, 22), sheet(SPARE, 21), sheet(PAGE, 20)]) is None and c.place is None
    assert c.why_none() == "none chosen"
    assert c.choose([sheet(SPARE)]) is None   # the others hidden a moment: still none chosen


def test_a_slow_slide_carries_the_page_any_distance():
    c = chooser(PAGE)
    for t, x in enumerate(np.arange(PAGE, 0.6, 0.04)):
        assert x_of(c.choose([sheet(float(x), t=t)])) == round(float(x), 3)


def test_on_a_rig_with_one_sheet_the_moved_page_is_followed_at_once():
    c = chooser(PAGE)
    assert x_of(c.choose([sheet(0.1, t=1)])) == 0.1
    assert x_of(c.choose([sheet(0.1, t=2)])) == 0.1


def test_a_table_that_needed_a_choice_has_other_mounts_so_the_page_never_jumps():
    """Restarted from a chosen sheet with the arm over it: the one sheet in view is another mount's."""
    c = chooser(PAGE, chosen=True)
    assert c.choose([sheet(SPARE)]) is None
    c = chooser(PAGE)
    c.choose_at([PAGE, 0.0], np.eye(4))
    assert x_of(c.choose([sheet(PAGE)])) == PAGE and c.choose([sheet(SPARE)]) is None


def test_once_two_sheets_were_in_view_the_page_never_jumps_to_another():
    """The arm rests over the page and hides it, for as long as it likes: the other mount's sheet never stands in for
    it, and the page is the page again when it comes back."""
    c = chooser(PAGE)
    assert x_of(c.choose([sheet(PAGE, t=0), sheet(SPARE, t=0)])) == PAGE
    for t in range(1, 2000, 100):
        assert c.choose([sheet(SPARE, t=t)]) is None
    assert "where the page stood" in c.why_none()
    assert x_of(c.choose([sheet(SPARE, t=2000), sheet(PAGE, t=2000)])) == PAGE


def test_a_page_moved_too_far_at_once_is_lost_until_chosen_again():
    c = chooser(PAGE)
    c.choose([sheet(PAGE), sheet(FAR)])
    assert c.choose([sheet(0.1), sheet(FAR)]) is None   # moved 200 mm while hidden
    c.choose_at([0.11, 0.0], np.eye(4))
    assert x_of(c.choose([sheet(0.1), sheet(FAR)])) == 0.1
    assert c.wanted is None and x_of(c.choose([sheet(0.12), sheet(FAR)])) == 0.12   # followed from there


def test_a_chosen_sheet_is_found_in_the_arms_base_and_until_then_there_is_no_page():
    c = chooser(PAGE)
    c.choose([sheet(PAGE), sheet(SPARE)])
    base_from_world = np.linalg.inv(pose_at(0.2, 0.1))   # the arm's base stands at (0.2, 0.1) in the world
    c.choose_at([0.3, -0.1], base_from_world)            # the spare, in the base
    assert c.choose([sheet(PAGE)]) is None and "the chosen (+300, -100) mm" in c.why_none()
    assert x_of(c.choose([sheet(PAGE), sheet(SPARE)])) == SPARE
    assert x_of(c.choose([sheet(PAGE), sheet(SPARE)])) == SPARE   # and it stays the page


def test_page_at_names_an_arm_and_a_point_in_mm():
    arm, xy = page.parse_at(" xarm, 400, -100 ")
    assert arm == "xarm" and np.allclose(xy, [0.4, -0.1])
    for bad in ("xarm,400", ",1,2", "xarm,nan,0", "xarm,a,b"):
        with pytest.raises(ValueError):
            page.parse_at(bad)


def test_the_last_touched_or_chosen_page_is_where_to_look_first(tmp_path):
    touched = tmp_path / "page-touched-right.json"
    touched.write_text(json.dumps({"camera": pose_at(0.3).tolist()}))
    place = page.page_place(touched, pose_at(0.0, 0.2))
    assert np.allclose(place, [0.3, 0.2, 0.0])
    assert page.page_place(touched, None) is None
    assert page.page_place(tmp_path / "none.json", np.eye(4)) is None
    touched.write_text("{}")
    assert page.page_place(touched, np.eye(4)) is None


def test_a_stacks_chooser_starts_where_the_newer_of_the_touches_and_the_choice_put_the_page(tmp_path, monkeypatch):
    import os

    touched, chosen = tmp_path / "page-touched-right.json", tmp_path / "page-chosen-right.json"
    touched.write_text(json.dumps({"camera": pose_at(0.3).tolist()}))
    monkeypatch.setattr(stack, "log_root", lambda: tmp_path)
    c = stack.page_chooser({"page": {"copies": COPIES}}, {"left": None, "right": pose_at(0.0, 0.2)})
    assert np.allclose(c.place, [0.3, 0.2, 0.0])
    assert c.alone   # touches alone say nothing of other mounts
    chosen.write_text(json.dumps({"camera": pose_at(0.5).tolist()}))
    os.utime(touched, (1, 1))
    c = stack.page_chooser({"page": {"copies": COPIES}}, {"right": pose_at(0.0, 0.2)})
    assert np.allclose(c.place, [0.5, 0.2, 0.0]) and stack.chosen_path("right") == chosen and not c.alone
    assert stack.page_chooser({"page": {"copies": COPIES}}, {"left": pose_at(0.0)}).place is None


def test_the_watch_says_where_each_sheet_stands_and_which_is_the_page():
    page_sheet, spare = sheet(PAGE, 19), sheet(FAR, 20)
    assert watch.sheet_rows([spare, page_sheet], page_sheet, np.eye(4)) == [
        {"markers": 20, "page": False, "arm_mm": 950, "base_xy_mm": [950.0, 0.0]},
        {"markers": 19, "page": True, "arm_mm": 300, "base_xy_mm": [300.0, 0.0]}]
    assert watch.sheet_rows([spare], None, None) == [{"markers": 20, "page": False}]


def test_covariance_and_quaternion():
    cov = page.covariance(0.002, 0.01)
    assert np.allclose(np.diag(np.reshape(cov, (6, 6))), [4e-6] * 3 + [1e-4] * 3)
    assert page.covariance(math.nan, 0.0)[0] == -1.0
    rng = np.random.default_rng(3)
    randoms = []
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        randoms.append(q * np.sign(np.linalg.det(q)))
    # 180 degree turns (w = 0) about axes with one, two and three non-zero components
    flips = [2 * np.outer(a, a) - np.eye(3) for a in
             (np.array(v, float) / np.linalg.norm(v) for v in ([1, 0, 0], [0, 0, 1], [1, -1, 0], [0, 1, 1], [1, -2, 3]))]
    for q in randoms + flips + [np.diag([1.0, -1.0, -1.0]), np.eye(3)]:
        x, y, z, w = page.quaternion(q)
        back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(back, q, atol=1e-9)


def test_fixed_pose_from_xyz_rpy():
    m = page.from_xyz_rpy([0.371, -0.231, 0.0334], [0.0, 0.0, math.pi / 2])
    assert np.allclose(m[:3, 3], [0.371, -0.231, 0.0334])
    assert np.allclose(m[:3, 0], [0, 1, 0]) and np.allclose(m[:3, 2], [0, 0, 1])
    assert np.allclose(page.rpy_deg(page.from_xyz_rpy([0, 0, 0], np.radians([10, -20, 30]))[:3, :3]), [10, -20, 30])
