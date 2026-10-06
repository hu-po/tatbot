"""A page placed by its artwork on the table plane (scripts/vision/stencil_plane_match.py), in a synthetic view
like the demo stack's D555: a dotted ring printed on a larger white sheet on a dark mat, 0.8 m from a tilted
camera at 640x360, with the table's depth. The match returns the page's centre and turn, its top away from the
arm; the sheet alone, without its ring, is no page; a drawing in the clear centre does not hide the page."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
import stencil_plane_match as match  # noqa: E402

PAGE_M = (0.100, 0.150)
K = np.array([[322.3, 0.0, 323.4], [0.0, 321.8, 181.4], [0.0, 0.0, 1.0]])
SIZE = (640, 360)
CLEAR_UV = (0.19, 19 / 150, 0.81, 131 / 150)   # ring_reference's clear centre, as a reference's tracking.json gives it


def ring_reference(px_per_mm=6):
    """A white page with a dotted band from 5 to 19 mm inside its edges, like a printed stencil frame."""
    width, height = int(100 * px_per_mm), int(150 * px_per_mm)
    image = np.full((height, width), 255, np.uint8)
    for x_mm in np.arange(7.0, 94.0, 3.2):
        for y_mm in np.arange(7.0, 144.0, 3.2):
            inside = 19.0 < x_mm < 81.0 and 19.0 < y_mm < 131.0
            if not inside:
                cv2.circle(image, (int(x_mm * px_per_mm), int(y_mm * px_per_mm)), int(1.1 * px_per_mm), 0, -1)
    return image


def drawing(reference, px_per_mm=6):
    """The reference with a drawing over its clear centre to 5 mm of its edges: 1 mm wavy lines 3 mm apart."""
    image = reference.copy()
    for i, y_mm in enumerate(np.arange(24.0, 126.0, 3.0)):
        x_mm = np.linspace(24.0, 76.0, 60)
        points = np.c_[x_mm, y_mm + 2.0 * np.sin(x_mm / 5.0 + i)] * px_per_mm
        cv2.polylines(image, [points.astype(np.int32)], False, 60, px_per_mm, cv2.LINE_AA)
    return image


def top_view(printed, seed=0):
    """The plane image find_page correlates (RECTIFIED_M per pixel): `printed` on its 216 x 279 mm sheet, turned
    20 deg, on a dark mat, blurred and noisy."""
    rng = np.random.default_rng(seed)
    page = match.reference_template(printed, PAGE_M)
    sheet = np.full((140, 108), 225.0, np.float32)
    top, left = (140 - page.shape[0]) // 2, (108 - page.shape[1]) // 2
    sheet[top:top + page.shape[0], left:left + page.shape[1]] = 70.0 + page * (155.0 / 255.0)
    matrix = cv2.getRotationMatrix2D((54.0, 70.0), 20.0, 1.0)
    matrix[:, 2] += (71.0, 55.0)
    view = np.full((250, 250), 70.0, np.float32)
    on = cv2.warpAffine(np.ones_like(sheet), matrix, (250, 250), flags=cv2.INTER_NEAREST) > 0
    view[on] = cv2.warpAffine(sheet, matrix, (250, 250), flags=cv2.INTER_LINEAR)[on]
    return cv2.GaussianBlur(view, (0, 0), 0.6) + rng.normal(0.0, 2.0, view.shape).astype(np.float32)


def rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def scene(page_root, reference, *, with_ring=True, seed=3):
    """(image BGR, depth m, rays, root_from_camera) of a table at z = 0.003 in root (the arm base) under a camera
    0.8 m up, tilted like the installed D555; the page printed on a 216 x 279 mm sheet on a dark mat."""
    rng = np.random.default_rng(seed)
    root_from_camera = np.eye(4)
    root_from_camera[:3, :3] = rotation("z", -35.0) @ rotation("x", 162.0)   # looking down, tilted
    root_from_camera[:3, 3] = [0.05, 0.25, 0.80]
    y, x = np.mgrid[:SIZE[1], :SIZE[0]]
    rays = np.stack(((x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x, float)), axis=-1)
    direction = rays @ root_from_camera[:3, :3].T
    origin = root_from_camera[:3, 3]
    scale = (0.003 - origin[2]) / direction[..., 2]
    points = origin + scale[..., None] * direction
    depth = scale                                           # rays have z = 1: the camera Z is the scale
    local = (points - page_root[:3, 3]) @ page_root[:3, :3]  # page target frame: x along u, y along v
    u, v = local[..., 0] / PAGE_M[0] + 0.5, local[..., 1] / PAGE_M[1] + 0.5
    image = np.full(depth.shape, 70.0)                      # the mat
    sheet = (np.abs(local[..., 0]) < 0.108) & (np.abs(local[..., 1]) < 0.1395)
    image[sheet] = 225.0
    on_page = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    if with_ring:
        ref_h, ref_w = reference.shape
        sample = reference[np.clip((v[on_page] * ref_h).astype(int), 0, ref_h - 1),
                           np.clip((u[on_page] * ref_w).astype(int), 0, ref_w - 1)].astype(float)
        image[on_page] = 225.0 * sample / 255.0 + 70.0 * (1 - sample / 255.0)
    image = cv2.GaussianBlur(image, (0, 0), 0.8) + rng.normal(0.0, 2.0, image.shape)
    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return bgr, depth + rng.normal(0.0, 0.001, depth.shape), rays, root_from_camera


def page_pose(centre, yaw_deg):
    pose = np.eye(4)
    pose[:3, :3] = rotation("z", yaw_deg) @ np.diag([1.0, -1.0, -1.0])      # z into the paper (down)
    pose[:3, 3] = centre
    return pose


def pad(centre, half=0.25):
    return np.asarray(centre) + half * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0.]])


def test_the_ring_on_its_sheet_gives_the_page_centre_and_turn_top_away_from_the_arm():
    reference = ring_reference()
    truth = page_pose([0.2755, 0.0297, 0.003], -88.0)
    image, depth, rays, root_from_camera = scene(truth, reference)
    found = match.find_page(image, depth, rays, K, np.zeros(5), root_from_camera, reference, PAGE_M,
                            pad([0.30, 0.0, 0.003]), np.eye(4))
    assert found["found"], found
    pose = found["pose"]
    assert np.linalg.norm(pose[:3, 3] - truth[:3, 3]) < 0.003
    assert math.degrees(math.acos(np.clip(pose[:3, 0] @ truth[:3, 0], -1, 1))) < 1.5
    assert pose[:3, 2] @ np.array([0.0, 0.0, -1.0]) > 0.99                  # into the paper
    # the print's bottom (+v) faces the arm at the origin, its top away
    assert pose[:3, 1] @ (np.zeros(3) - pose[:3, 3]) > 0
    assert found["score"] >= match.MIN_SCORE and found["margin"] >= match.MIN_MARGIN


def test_continuity_keeps_the_last_turn_over_the_arm_side_rule():
    pose = page_pose([0.3, 0.0, 0.003], 0.0)        # u along base +x, v toward -y: bottom away from the arm
    arm = np.eye(4)
    assert match.half_turned(pose, arm, None) == (pose[:3, 1] @ (arm[:3, 3] - pose[:3, 3]) < 0)
    assert match.half_turned(pose, arm, -pose[:3, 0]) is True     # the last pose pointed the other way
    assert match.half_turned(pose, arm, pose[:3, 0]) is False


@pytest.mark.parametrize("clear_uv", [None, CLEAR_UV])
def test_a_sheet_without_its_ring_is_no_page(clear_uv):
    reference = ring_reference()
    image, depth, rays, root_from_camera = scene(page_pose([0.2755, 0.0297, 0.003], -88.0), reference,
                                                 with_ring=False)
    found = match.find_page(image, depth, rays, K, np.zeros(5), root_from_camera, reference, PAGE_M,
                            pad([0.30, 0.0, 0.003]), np.eye(4), clear_uv=clear_uv)
    assert not found["found"] and "no clear page match" in found["reason"]


def test_a_drawing_in_the_clear_centre_does_not_hide_the_page():
    """Ink in the clear centre is image variance the blank template cannot match: a four-ink drawing held the
    uncovered page under MIN_SCORE for good (2026-10-05). Matched on its frame alone the drawn page scores as
    the undrawn one does, at the same place and turn."""
    reference = ring_reference()
    template = match.reference_template(reference, PAGE_M)
    clean = match.match_page(top_view(reference), template, clear_uv=CLEAR_UV)
    drawn = top_view(drawing(reference))
    whole, framed = match.match_page(drawn, template), match.match_page(drawn, template, clear_uv=CLEAR_UV)
    assert whole[0] < match.MIN_SCORE
    assert framed[0] >= match.MIN_SCORE + 0.1 and framed[1] >= match.MIN_MARGIN
    assert np.hypot(*np.subtract(framed[2], clean[2])) < 0.5 and framed[3] == clean[3]
