"""The page frame (ros/README.md section 3) from stencild's target frame, pinned on the print's u/v."""
import numpy as np
from tatbot_bridge.page import TARGET_FROM_PAGE, base_from_page, world_from_page

PAGE_MM = np.array([100.0, 150.0])


def target_point(u, v):
    """stencil_pose.py object points: ((uv - 0.5) * page_mm, 0) in metres; u right, v down the image."""
    x, y = (np.array([u, v]) - 0.5) * PAGE_MM / 1000
    return np.array([x, y, 0.0, 1.0])


def test_page_axes_from_target():
    page = world_from_page(np.eye(4))
    assert np.allclose(page[:3, 0], [1, 0, 0])   # x along u
    assert np.allclose(page[:3, 1], [0, -1, 0])  # y toward the top of the print (-v)
    assert np.allclose(page[:3, 2], [0, 0, -1])  # z out of the paper (target z is into it)
    assert np.isclose(np.linalg.det(TARGET_FROM_PAGE[:3, :3]), 1.0)


def test_uv_corners_in_the_page_frame():
    """u right -> +x, v down -> -y: the print's top-right corner (u=1, v=0) is (+50, +75) mm."""
    target_from_page = TARGET_FROM_PAGE
    page_from_target = np.linalg.inv(target_from_page)
    for (u, v), xy_mm in {(0.5, 0.5): (0, 0), (1, 0): (50, 75), (0, 0): (-50, 75),
                          (0, 1): (-50, -75), (1, 1): (50, -75), (0.75, 0.25): (25, 37.5)}.items():
        p = page_from_target @ target_point(u, v)
        assert np.allclose(p[:3] * 1000, [*xy_mm, 0.0]), (u, v, p)


def test_page_z_points_at_a_camera_looking_down():
    """A camera above the page sees the print upright: its optical z (into the paper) is target z, so the
    page z is toward the camera."""
    world_from_target = np.eye(4)
    world_from_target[:3, :3] = np.diag([1.0, -1.0, -1.0])  # target z = -world z: paper faces world +z
    world_from_target[:3, 3] = [0.1, 0.2, 0.0]
    wfp = world_from_page(world_from_target)
    assert np.allclose(wfp[:3, 2], [0, 0, 1])                 # page z up, out of the paper
    assert np.allclose(wfp[:3, 0], [1, 0, 0])
    assert np.allclose(wfp[:3, 3], [0.1, 0.2, 0.0])           # origin unchanged: the page centre


def rigid(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = q, rng.normal(size=3)
    return m


def test_registration_crossing_matches_the_session_chain():
    """base_from_page = inv(world_from_arm_base) @ world_from_target @ TARGET_FROM_PAGE, the frozen session's
    `arm_base = inv(world_from_arm_base) @ world_from_target` with the page frame's axes."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        wfb, wft = rigid(rng), rigid(rng)
        bfp = base_from_page(wfb, wft)
        session = np.linalg.inv(wfb) @ wft
        assert np.allclose(bfp, session @ TARGET_FROM_PAGE)
        # A print point lands at the same place in the base either way.
        p_page = np.array([0.02, 0.03, 0.0, 1.0])
        p_target = TARGET_FROM_PAGE @ p_page
        assert np.allclose(bfp @ p_page, np.linalg.inv(wfb) @ (wft @ p_target))
        assert np.isclose(np.linalg.det(bfp[:3, :3]), 1.0)
