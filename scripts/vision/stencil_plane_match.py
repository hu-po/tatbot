"""A stencil page found by its artwork on the table plane, in a fixed RGB-D view too coarse to read its print.

The demo stack's one fixed camera is the D555 at 640x360 from about 0.8 m: 0.43 px per millimetre, where a coded
print's 2.2 mm knots are a pixel and its bits unreadable, so no print is decoded or feature-matched there. The
printed frame itself still shows as a grey ring on the sheet. Inside an arm's drawing pad this fits the table
plane to the aligned depth, resamples the colour image onto that plane at RECTIFIED_M per pixel (a top view as the
camera sees it), and correlates the reference artwork, blurred to the camera's resolution, over position and turn.
Only the printed frame is correlated, never the clear centre: that is where the arm draws, and ink there is image
variance the blank template cannot match. A peak whose correlation passes MIN_SCORE, clear of any other peak by
MIN_MARGIN, is the page: its centre and turn
on the plane give the print's pose in its target frame (x along u, y along v down the print, z into the paper).
The ring is nearly symmetric under a half turn, so the turn nearest a prior (the page's last pose, else the arm's
own forward axis) is chosen. The pose is a prior for the drawing arm, which measures the page again with its wrist
camera and touches before it draws; this never reads which print it is.
"""

from __future__ import annotations

import cv2
import numpy as np

RECTIFIED_M = 0.002             # the plane image's pixel: about the camera's own footprint at the table
PLANE_INLIER_M = 0.004          # a table point lies this close to the fitted plane
MIN_SCORE = 0.70                # normalized correlation of the band-passed ring with the band-passed sheet
MIN_MARGIN = 0.15               # the next peak more than a page-width away scores at least this much lower
# With the arm over the page, false peaks a quarter turn round and 60-120 mm away scored 0.47-0.60 (margins
# 0.03-0.28) and were published as measured at MIN_SCORE 0.55; the uncovered page scored 0.77-0.87 (margins
# 0.37-0.50) (2026-09-30). A drawing run no longer adopts a jump (ros/README.md, page motion); this keeps them
# off the bus. The margin does not separate them as cleanly: the synthetic ring's true peak clears by 0.28.
# Those scores correlated the whole page. A four-ink drawing in the clear centre then held the uncovered page at
# 0.61-0.69, lost for good; with the clear centre left out it scored 0.79-0.82, its pose within 1 mm of the
# undrawn page's, and a dark arm painted over it still scored at most 0.61 (2026-10-05).
# Both images are band-passed (a difference of Gaussians of these sigmas, in plane pixels) before correlating:
# the printed ring's texture passes, a sheet's edge on a dark mat does not dominate. Unfiltered, that edge
# out-scored the ring by 0.58 to 0.51 on the installed view; band-passed, the ring scored 0.85 (2026-09-29).
BAND_SIGMAS_PX = (1.0, 4.0)
TURN_STEP_DEG = 4.0
TRANSLATION_SIGMA_M = 0.004     # floors: the match's own scatter is smaller than the depth and
ROTATION_SIGMA_RAD = 0.026      # registration bias it cannot see (docs/vision.md)
DEPTH_RANGE_M = (0.3, 1.5)


def table_plane(points: np.ndarray, rng: np.random.Generator, iterations: int = 150):
    """(unit normal, offset) of the dominant plane n.p = d among `points` (N, 3): RANSAC on at most 4000 of
    them, then a least-squares refit on its inliers; None with too few points."""
    if len(points) < 50:
        return None
    sample_from = points[rng.choice(len(points), min(len(points), 4000), replace=False)]
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = sample_from[rng.integers(0, len(sample_from), 3)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        count = int(np.count_nonzero(np.abs(sample_from @ normal - normal @ a) < PLANE_INLIER_M))
        if count > best_count:
            best, best_count = (normal, float(normal @ a)), count
    if best is None:
        return None
    inliers = points[np.abs(points @ best[0] - best[1]) < PLANE_INLIER_M]
    centre = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][2]
    return normal, float(normal @ centre)


def _inside(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """Points inside a convex polygon (either winding), in the plane."""
    signs = []
    for a, b in zip(polygon_xy, np.roll(polygon_xy, -1, axis=0), strict=True):
        edge, rel = b - a, points_xy - a
        signs.append(edge[0] * rel[:, 1] - edge[1] * rel[:, 0])
    signs = np.stack(signs, axis=1)
    return np.all(signs >= 0, axis=1) | np.all(signs <= 0, axis=1)


def plane_axes(normal_toward_camera: np.ndarray, forward: np.ndarray):
    """In-plane axes (e1 along `forward` projected, e2 = e1 x n) for a top view as the camera sees it: image x
    along e1, image y along e2, and e1 x e2 = -n, into the table."""
    n = normal_toward_camera / np.linalg.norm(normal_toward_camera)
    e1 = forward - (forward @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(e1, n)


def rectify(image_gray: np.ndarray, k: np.ndarray, dist: np.ndarray, origin, e1, e2, half_m: float,
            step_m: float = RECTIFIED_M) -> np.ndarray:
    """The image resampled onto the plane (camera frame: origin + s e1 + t e2, s and t within half_m), one
    pixel per step_m; points behind the camera or outside the image read 0."""
    s = np.arange(-half_m, half_m, step_m)
    grid_s, grid_t = np.meshgrid(s, s)
    points = origin + grid_s[..., None] * e1 + grid_t[..., None] * e2
    ahead = points[..., 2] > 0.05
    pixels, _ = cv2.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, dist)
    maps = pixels.reshape(points.shape[:2] + (2,)).astype(np.float32)
    maps[~ahead] = -1
    return cv2.remap(image_gray, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def reference_template(reference_gray: np.ndarray, page_m, step_m: float = RECTIFIED_M,
                       blur_px: float = 0.7) -> np.ndarray:
    """The reference artwork (u right, v down) at step_m per pixel, blurred like the camera sees it."""
    size = (max(3, round(page_m[0] / step_m)), max(3, round(page_m[1] / step_m)))
    small = cv2.resize(reference_gray.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), blur_px)


def frame_mask(template: np.ndarray, clear_uv=None) -> np.ndarray:
    """1 over the template's printed frame, 0 over its clear centre `clear_uv` (u0, v0, u1, v1, the reference's
    `clear_center_uv`); all 1 without one."""
    keep = np.ones_like(template, dtype=np.float32)
    if clear_uv is not None:
        height, width = template.shape
        u0, v0, u1, v1 = clear_uv
        keep[round(v0 * height):round(v1 * height), round(u0 * width):round(u1 * width)] = 0.0
    return keep


def _turned(template: np.ndarray, degrees: float, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The template turned by `degrees` (image x toward image y) about its centre on a canvas that holds it,
    and `keep` (frame_mask) turned with it: only the page's frame is correlated, never the canvas corners."""
    height, width = template.shape
    side = int(np.ceil(np.hypot(width, height))) | 1
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -degrees, 1.0)
    matrix[:, 2] += ((side - width) / 2.0, (side - height) / 2.0)
    turned = cv2.warpAffine(template, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0.0)
    mask = cv2.warpAffine(keep, matrix, (side, side), flags=cv2.INTER_NEAREST, borderValue=0.0)
    return turned, mask


def _correlate(image: np.ndarray, template: np.ndarray, degrees: float, keep: np.ndarray) -> np.ndarray:
    turned, mask = _turned(template, degrees, keep)
    result = cv2.matchTemplate(image, turned, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)


def band_pass(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    return cv2.GaussianBlur(image, (0, 0), BAND_SIGMAS_PX[0]) - cv2.GaussianBlur(image, (0, 0), BAND_SIGMAS_PX[1])


def match_page(plane_image: np.ndarray, template: np.ndarray, *, clear_uv=None, step_deg: float = TURN_STEP_DEG):
    """(score, margin, centre (col, row) in plane pixels, turn degrees in [0, 180)) of the best correlation of
    the band-passed, turned template with the band-passed plane image over the page's frame (frame_mask), or
    None when the image is smaller than the template."""
    keep = frame_mask(template, clear_uv)
    image, template = band_pass(plane_image), band_pass(template)
    side = int(np.ceil(np.hypot(*template.shape))) | 1
    if side >= min(image.shape):
        return None
    best = None
    scores = {}
    for degrees in np.arange(0.0, 180.0, step_deg):
        result = _correlate(image, template, degrees, keep)
        scores[degrees] = result
        _, peak, _, location = cv2.minMaxLoc(result)
        if best is None or peak > best[0]:
            best = (peak, degrees, location)
    peak, degrees, location = best
    shape = (side, side)
    for fine in np.arange(degrees - step_deg, degrees + step_deg + 1e-9, 1.0):
        _, value, _, where = cv2.minMaxLoc(_correlate(image, template, fine, keep))
        if value > peak:
            peak, degrees, location = value, fine, where
    exclusion = max(template.shape)
    runner_up = -1.0
    for result in scores.values():
        masked = result.copy()
        y0, x0 = max(0, location[1] - exclusion), max(0, location[0] - exclusion)
        masked[y0:location[1] + exclusion + 1, x0:location[0] + exclusion + 1] = -1.0
        runner_up = max(runner_up, float(masked.max()))
    result = _correlate(image, template, degrees, keep)
    x, y = location
    dx = dy = 0.0
    if 0 < x < result.shape[1] - 1:
        left, mid, right = result[y, x - 1], result[y, x], result[y, x + 1]
        dx = 0.5 * (left - right) / max(left - 2 * mid + right, -1e9) if left - 2 * mid + right < 0 else 0.0
    if 0 < y < result.shape[0] - 1:
        up, mid, down = result[y - 1, x], result[y, x], result[y + 1, x]
        dy = 0.5 * (up - down) / max(up - 2 * mid + down, -1e9) if up - 2 * mid + down < 0 else 0.0
    centre = (x + dx + (shape[1] - 1) / 2.0, y + dy + (shape[0] - 1) / 2.0)
    return float(peak), float(peak - runner_up), centre, float(degrees % 180.0)


def find_page(image_bgr: np.ndarray, depth_m: np.ndarray, rays: np.ndarray, k: np.ndarray, dist: np.ndarray,
              root_from_camera: np.ndarray, reference_gray: np.ndarray, page_m, area_root: np.ndarray,
              arm_root: np.ndarray, previous_x_root: np.ndarray | None = None, rng=None, clear_uv=None) -> dict:
    """The page inside one pad area of the arm whose base pose in root is `arm_root`: {found, pose (root,
    target frame), score, margin, reason, ...}, matched on the frame around the reference's clear centre
    `clear_uv` (match_page)."""
    rng = rng or np.random.default_rng(0)
    area_root = np.asarray(area_root, float)
    camera_from_root = np.linalg.inv(root_from_camera)
    valid = np.isfinite(depth_m) & (depth_m > DEPTH_RANGE_M[0]) & (depth_m < DEPTH_RANGE_M[1])
    points = (rays * np.where(valid, depth_m, 0.0)[..., None])[valid]
    root_points = points @ root_from_camera[:3, :3].T + root_from_camera[:3, 3]
    in_area = _inside(root_points[:, :2], area_root[:, :2])
    plane = table_plane(points[in_area], rng)
    if plane is None:
        return {"found": False, "reason": "no table plane measured in the pad area"}
    normal, offset = plane
    if normal[2] > 0:                       # toward the camera: the camera looks along +z
        normal, offset = -normal, -offset
    centre_root = area_root.mean(axis=0)
    centre = camera_from_root[:3, :3] @ centre_root + camera_from_root[:3, 3]
    origin = centre - (centre @ normal - offset) * normal
    e1, e2 = plane_axes(normal, camera_from_root[:3, :3] @ np.asarray(arm_root, float)[:3, 0])
    half = 0.5 * float(max(np.linalg.norm(area_root[1] - area_root[0]), np.linalg.norm(area_root[2] - area_root[1])))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    plane_image = rectify(gray, k, dist, origin, e1, e2, half)
    template = reference_template(reference_gray, page_m)
    match = match_page(plane_image, template, clear_uv=clear_uv)
    if match is None:
        return {"found": False, "reason": "the pad area is smaller than the page"}
    score, margin, (col, row), turn = match
    record = {"score": score, "margin": margin, "plane_normal_camera": normal.tolist()}
    if score < MIN_SCORE or margin < MIN_MARGIN:
        return {"found": False, "reason": f"no clear page match (score {score:.2f}, margin {margin:.2f})", **record}
    s, t = -half + col * RECTIFIED_M, -half + row * RECTIFIED_M
    theta = np.radians(turn)
    x_cam = np.cos(theta) * e1 + np.sin(theta) * e2
    y_cam = -np.sin(theta) * e1 + np.cos(theta) * e2
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.column_stack((x_cam, y_cam, np.cross(x_cam, y_cam)))
    camera_pose[:3, 3] = origin + s * e1 + t * e2
    pose = root_from_camera @ camera_pose
    if half_turned(pose, np.asarray(arm_root, float), previous_x_root):
        pose[:3, :2] *= -1.0
    return {"found": True, "pose": pose, **record}


def half_turned(pose: np.ndarray, arm_root: np.ndarray, previous_x_root=None) -> bool:
    """Whether the page's pose needs a half turn about its normal: the ring cannot tell. Continuity with the
    page's last pose where that is decisive; else the print's top (-v, target -y) faces away from the drawing
    arm, its bottom toward it, the way the arm draws it."""
    if previous_x_root is not None:
        agreement = float(pose[:3, 0] @ np.asarray(previous_x_root, float))
        if abs(agreement) > 0.5:
            return agreement < 0
    toward_arm = arm_root[:3, 3] - pose[:3, 3]
    return float(pose[:3, 1] @ toward_arm) < 0


def page_corners(pose: np.ndarray, page_m, uv) -> np.ndarray:
    """Root-frame points of a page pose at reference UVs (0..1 across, 0..1 down)."""
    uv = np.asarray(uv, float)
    local = np.column_stack(((uv[:, 0] - 0.5) * page_m[0], (uv[:, 1] - 0.5) * page_m[1], np.zeros(len(uv))))
    return local @ pose[:3, :3].T + pose[:3, 3]
