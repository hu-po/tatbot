"""The pen's end where a wrist camera sees it, and the camera's mount pinned to it (ros/README.md section 7).

The pen and the wrist camera ride the one tool mount, so the pen's end stands at one pixel in every frame that camera
takes, whatever the arm does. The hand-eye's fit (wristcam.fit_mount) solves the camera's mount and the palette tag's
pose together, and what the tag is free to take the mount gives up: over six runs of three days the fit put the tcp
at pixels 3.4 mm apart on one camera while the pen's end in their own frames stood within a pixel, 0.25 mm
(2026-11-11: the session camera's fit 1.7 mm off its pen, and a 3RL's ink 4.5-5 mm off its markers).

`tip` finds that pixel in a run's frames: an edge of the pen holds still while the background's move with the arm,
so the median over the frames of each frame's gradient keeps the pen's outline alone, and the pen's end is the
outline's far end along the pen from its cone. `depth` reads how far that end stands from the camera in the same
camera's depth frames: the pen's near side a few pixels back from its end, carried to the end along the pen. The
tracker and the gauge read the skin from that depth stream, so the tip belongs at the range the stream itself gives
the pen: set to the arm's heights over the probe's top instead (calib gauge) it stood 2.2 mm nearer, the needles
rode about 1.5 mm deep and the ink 2.3 mm along the view from its markers (2026-11-11). `pinned` moves the fitted
mount until the tcp stands at that pixel and depth; the fit's turn stands.
"""
from __future__ import annotations

import numpy as np

WINDOW_PX = 60        # the outline is read this far about the fit's tip: the fit stood 4 mm, 17 px, off at worst
EDGE_SHARE = 0.3      # an outline pixel's median gradient over the window's strongest (its 98th percentile): 0.15 let
                      # a clear tube's background join it, 0.25-0.4 agreed within a pixel on every run
MIN_FRAMES = 12       # fewer and a background edge survives the median
END_PX = 1.5          # the outline's pixels this near its far end are the end
BACK_PX = (3, 20)     # the pen's depth is read this far back from its end: nearer, a pixel is half background
NEAR_M = 0.012        # the pen's depth this near the fit's tip's: the fit stood 2-4 mm off, the background 30+
BESIDE_PX = 2.5       # ... within this of the pen's line in the image
PAST_PX = 2.0         # ... which stands this far past the pen's own: the outline's dilation and the gradient's blur


def static_edges(frames) -> np.ndarray:
    """Each pixel's gradient magnitude, its median over the frames (BGR, one camera): what holds still."""
    import cv2

    mags = []
    for frame in frames:
        grey = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        mags.append(np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1)))
    return np.median(np.stack(mags), axis=0)


def tip(frames, tip_px, cone_px) -> np.ndarray | None:
    """The pen's end (x, y) in the frames' pixels, None when they are too few or show no outline by the cone.
    `tip_px`, `cone_px`: where the fit images the tcp and the pen's cone end, which say where to look and which way
    the pen points."""
    import cv2

    if len(frames) < MIN_FRAMES:
        return None
    tip_px, cone_px = np.asarray(tip_px, float), np.asarray(cone_px, float)
    along = tip_px - cone_px
    if np.linalg.norm(along) < 1.0:
        return None
    along /= np.linalg.norm(along)
    edges = static_edges(frames)
    x0, y0 = np.maximum((tip_px - WINDOW_PX).astype(int), 0)
    window = edges[y0:int(tip_px[1]) + WINDOW_PX, x0:int(tip_px[0]) + WINDOW_PX]
    if window.size == 0:
        return None
    outline = (window > EDGE_SHARE * np.quantile(window, 0.98)).astype(np.uint8)
    outline = cv2.dilate(outline, np.ones((3, 3), np.uint8))   # a thin end's two sides join
    _, labels = cv2.connectedComponents(outline)
    ys, xs = np.nonzero(outline)
    if len(xs) == 0:
        return None
    points = np.c_[xs + x0, ys + y0].astype(float)
    nearest = int(np.argmin(np.linalg.norm(points - cone_px, axis=1)))
    pen = points[labels[ys, xs] == labels[ys[nearest], xs[nearest]]]
    reach = (pen - cone_px) @ along
    return pen[reach >= reach.max() - END_PX].mean(axis=0) - PAST_PX * along


def depth(shots, end_px, cone_px, about_m: float) -> float | None:
    """The pen's end's depth (m, along the camera's axis) in `shots`, each a depth frame and its metadata (raw, as
    gauge._points reads them): each frame's line through the pen's depth BACK_PX back from its end, taken at the end,
    and the frames' median. Only depth within NEAR_M of `about_m` (the fit's) is the pen's: past its edge a pixel reads
    the background, centimetres on. None when fewer than MIN_FRAMES show the pen."""
    from tatbot_session import gauge

    end_px, cone_px = np.asarray(end_px, float), np.asarray(cone_px, float)
    back = (cone_px - end_px) / np.linalg.norm(cone_px - end_px)
    ends = []
    for raw, meta in shots:
        points, u, v = gauge._points(raw, meta, (int(meta["height"]), int(meta["width"])))
        off = np.c_[u, v] - end_px
        along, beside = off @ back, np.abs(off @ np.array([-back[1], back[0]]))
        on_pen = ((along >= BACK_PX[0]) & (along <= BACK_PX[1]) & (beside <= BESIDE_PX)
                  & (np.abs(points[:, 2] - about_m) <= NEAR_M))
        steps = np.round(along[on_pen]).astype(int)
        at = sorted(set(steps.tolist()))
        if len(at) < 6:
            continue
        near = [np.median(points[on_pen][steps == a, 2]) for a in at]
        ends.append(float(np.polyval(np.polyfit(at, near, 1), 0.0)))
    return float(np.median(ends)) if len(ends) >= MIN_FRAMES else None


def image(points_cam, k, dist) -> np.ndarray:
    """Camera-frame points' pixels through the colour stream's intrinsics."""
    import cv2

    pts = np.asarray(points_cam, float).reshape(-1, 1, 3)
    return cv2.projectPoints(pts, np.zeros(3), np.zeros(3), np.asarray(k, float), np.asarray(dist, float))[0].reshape(-1, 2)


def pinned(mount_from_camera: np.ndarray, tip_mount, pixel, k, dist, depth_m: float | None = None) -> np.ndarray:
    """`mount_from_camera` moved so the tcp images at `pixel`, at `depth_m` along the camera's axis (the fit's
    when None)."""
    import cv2

    camera_from_mount = np.linalg.inv(np.asarray(mount_from_camera, float))
    tcp = camera_from_mount[:3, :3] @ np.asarray(tip_mount, float) + camera_from_mount[:3, 3]
    ray = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), np.asarray(k, float),
                              np.asarray(dist, float)).reshape(2)
    z = tcp[2] if depth_m is None else float(depth_m)
    camera_from_mount[:3, 3] += np.array([ray[0] * z, ray[1] * z, z]) - tcp
    return np.linalg.inv(camera_from_mount)
