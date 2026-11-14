"""The pen's end in a wrist camera's own frames (tatbot_calib.seen): found where it holds still over backgrounds that
move, and the fitted mount pinned to it. 2026-11-11: the hand-eye's fit put the tcp 1.7 mm off the pen its frames
showed, and a 3RL's ink landed 4.5-5 mm off its markers."""
from __future__ import annotations

import numpy as np
import pytest
from tatbot_calib import seen, wristcam

cv2 = pytest.importorskip("cv2")

K = np.array([[651.08, 0.0, 650.44], [0.0, 650.32, 358.05], [0.0, 0.0, 1.0]])
DIST = np.array([-0.0527, 0.0588, 0.0, 0.0004, -0.0192])
END = np.array([678.0, 358.0])   # the pen's end, as the xArm's session camera sees a 3RL's
CONE = np.array([640.0, 372.0])


def _frames(count: int, clear: bool = False) -> list[np.ndarray]:
    """A pen (dark, or a clear tube a shade off its background) ending at END over a background that changes."""
    rng, frames = np.random.default_rng(0), []
    back = (END - CONE) / np.linalg.norm(END - CONE)
    side = np.array([-back[1], back[0]])
    pen = np.array([END + 1.5 * side, END - 1.5 * side, CONE - 60 * back - 22 * side, CONE - 60 * back + 22 * side])
    for _ in range(count):
        frame = np.full((720, 1280, 3), int(rng.integers(150, 220)), np.uint8)
        for _ in range(12):   # what the arm's move carries across the view: the palette's caps, a tag, a mat's edge
            x, y, w, h = rng.integers(0, 1280), rng.integers(0, 720), rng.integers(20, 200), rng.integers(20, 200)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (int(rng.integers(40, 240)),) * 3, -1)
        shade = (float(frame[0, 0, 0]) * 0.8,) * 3 if clear else (25, 25, 25)
        cv2.fillConvexPoly(frame, np.round(pen).astype(np.int32), shade, lineType=cv2.LINE_AA)
        frames.append(frame)
    return frames


@pytest.mark.parametrize("clear", [False, True])
def test_the_pens_end_is_found_where_the_fit_is_off(clear):
    """The fit's tcp 14 px (3.4 mm) off the pen: the end is read from the frames within two pixels, half a millimetre."""
    frames = _frames(24, clear)
    end = seen.tip(frames, END + [9.0, -11.0], CONE + [9.0, -11.0])
    assert end is not None and np.linalg.norm(end - END) < 2.0


def test_too_few_frames_read_no_end():
    assert seen.tip(_frames(6), END, CONE) is None


def test_pinned_images_the_tcp_at_the_pixel_and_keeps_its_depth():
    mount = wristcam._pose(np.array([[0.69, 0.23, -0.68], [0.0, 0.95, 0.32], [0.72, -0.22, 0.66]]),
                           [0.103, -0.052, -0.035])
    mount[:3, :3] = np.linalg.qr(mount[:3, :3])[0]
    tip_mount = np.array([0.0026, -0.0029, 0.0733])
    before = np.linalg.inv(mount)[:3, :3] @ tip_mount + np.linalg.inv(mount)[:3, 3]
    pixel = seen.image([before], K, DIST)[0] + [-7.0, 6.0]
    moved = seen.pinned(mount, tip_mount, pixel, K, DIST)
    after = np.linalg.inv(moved)[:3, :3] @ tip_mount + np.linalg.inv(moved)[:3, 3]
    assert np.allclose(seen.image([after], K, DIST)[0], pixel, atol=1e-3)
    assert after[2] == pytest.approx(before[2])
    assert np.allclose(moved[:3, :3], mount[:3, :3])


def _shots(count: int, end_m: float = 0.157) -> list:
    """Depth frames of the pen over a background 70 mm on: its near side rises 0.25 mm a pixel back from its end."""
    meta = {"depth_units_m": 1e-4, "width": 1280, "height": 720,
            "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "ppx": K[0, 2], "ppy": K[1, 2]},
            "color_intrinsics": {"k": K.tolist()},
            "color_from_depth": {"rotation": np.eye(3).ravel().tolist(), "translation_m": [0.0, 0.0, 0.0]}}
    back = (CONE - END) / np.linalg.norm(CONE - END)
    v, u = np.mgrid[0:720, 0:1280]
    along = (u - END[0]) * back[0] + (v - END[1]) * back[1]
    beside = np.abs(-(u - END[0]) * back[1] + (v - END[1]) * back[0])
    pen = (along >= 0) & (beside <= 1.5 + along * 0.3)
    depth = np.where(pen, end_m - 0.00025 * along, 0.227)
    return [(np.round(depth / 1e-4).astype(np.uint16), meta)] * count


def test_the_pens_depth_is_read_at_its_end_and_not_off_the_background():
    """The pen's near side carried to its end, 157.0 mm, with the fit's tip 2.2 mm nearer; a pen pointed 40 deg off
    reads the background beside it, which is no pen's depth."""
    assert seen.depth(_shots(12), END, CONE, 0.1548) == pytest.approx(0.157, abs=3e-4)
    turned = END + np.array([[0.77, -0.64], [0.64, 0.77]]) @ (CONE - END)
    assert seen.depth(_shots(12), END, turned, 0.1548) is None
    assert seen.depth(_shots(4), END, CONE, 0.1548) is None


def test_pinned_takes_the_depth_it_is_given():
    mount = np.eye(4)
    mount[:3, 3] = [0.01, 0.0, -0.08]
    tip_mount = np.array([0.0, 0.0, 0.0733])
    moved = seen.pinned(mount, tip_mount, [660.0, 350.0], K, DIST, 0.157)
    after = np.linalg.inv(moved)[:3, :3] @ tip_mount + np.linalg.inv(moved)[:3, 3]
    assert after[2] == pytest.approx(0.157) and np.allclose(seen.image([after], K, DIST)[0], [660.0, 350.0], atol=1e-3)


def test_pin_moves_the_fits_mount_and_keeps_it_beside():
    """wristcam.pin: the result's mount images the tcp at the pen's end; frames that show no pen keep the fit's."""
    tip_mount = np.array([0.0, 0.0, 0.0733])
    mount = np.eye(4)
    mount[:3, 3] = [0.0, 0.0, -0.08]   # the camera behind the mount, looking along the pen
    record = wristcam.gauge_record(mount, tip_mount, "", "")
    at, cone = seen.image([record["tip_cam_m"], record["cone_end_cam_m"]], K, DIST)
    result = {"mount_from_camera": mount.tolist()}
    line = wristcam.pin(result, _frames(4), [], tip_mount, K, DIST)
    assert "not made out" in line and np.allclose(result["mount_from_camera"], mount)
    assert np.linalg.norm(at - cone) < 1.0   # the pen points at the camera here: no direction, no end, the fit kept
    assert "not made out" in wristcam.pin(result, _frames(24), [], tip_mount, K, DIST)
