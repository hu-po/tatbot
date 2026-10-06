"""The cameras a policy looks through, and how a pinhole render becomes each one's stream.

MuJoCo renders square-pixel pinhole images with a centred principal point.
A real stream has its own focal lengths, principal point and lens model, so
the generator renders a larger pinhole frame and remaps it onto the real
pixel grid: the blue arm's wrist D405 (inverse Brown-Conrady, as librealsense
reports it) and the third-person scene camera over the rig (a wide-angle PoE
camera, OpenCV Brown-Conrady, non-square pixels on its sub stream).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

# Factory calibration of the left wrist D405 colour stream at 640x480, as the
# camera reported it in a 2026-09-19 capture's frame metadata (librealsense
# RS2_DISTORTION_INVERSE_BROWN_CONRADY; coefficients k1 k2 p1 p2 k3).
LEFT_WRIST_D405 = {
    "width": 640,
    "height": 480,
    "fx": 392.476806640625,
    "fy": 391.9150390625,
    "cx": 315.5647277832031,
    "cy": 238.0267791748047,
    "distortion": (-0.05427977070212364, 0.05887533351778984, -0.00028628899599425495,
                   0.000988920801319182, -0.019327452406287193),
}

# The scene camera, calibrated on its 2960x1668 main stream (a 44 mm board session, 1.37 px rms; OpenCV
# Brown-Conrady k1 k2 p1 p2 k3). Its 704x480 sub stream is the same full frame squeezed to that size
# (matching features between the two streams gives scales 0.2379 x 0.2881 and offsets under 0.4 px), and
# policies get that resized to the wrist stream's 640x480: flux3 lays cameras side by side only at one size.
# Its pixels are therefore not square, and the lens model carries over in normalized coordinates.
SCENE_CAMERA_MAIN = {
    "width": 2960,
    "height": 1668,
    "fx": 1701.9103281195205,
    "fy": 1712.218232011594,
    "cx": 1501.280864900228,
    "cy": 946.9408154665412,
    "distortion": (-0.33934284759922684, 0.08427571176878611, 0.0, 0.0, 0.0),
}
SCENE_STREAM_SIZE = (640, 480)  # (width, height) as policies see it
# Its optical frame (x right, y down, z forward) in the blue arm's base frame: the vision calibration's
# camera pose, through the robot-world calibration, to the left arm's base 0.2675 m from the rig origin.
SCENE_CAMERA_POSE = np.array([
    [0.7890498, -0.4475713, 0.4208091, 0.0428932],
    [-0.6140732, -0.5548522, 0.5612959, -0.5308106],
    [-0.0177331, -0.7012980, -0.7126476, 0.6055094],
    [0.0, 0.0, 0.0, 1.0],
])


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    model: str = "inverse_brown_conrady"  # librealsense's; "brown_conrady" is OpenCV's forward model

    @classmethod
    def left_wrist(cls) -> Intrinsics:
        return cls(**LEFT_WRIST_D405)

    @classmethod
    def scene_camera(cls) -> Intrinsics:
        """The scene camera as policies see it: the main-stream calibration scaled to 640x480 per axis."""
        main = SCENE_CAMERA_MAIN
        width, height = SCENE_STREAM_SIZE
        sx, sy = width / main["width"], height / main["height"]
        return cls(width=width, height=height, fx=main["fx"] * sx, fy=main["fy"] * sy,
                   cx=(main["cx"] + 0.5) * sx - 0.5, cy=(main["cy"] + 0.5) * sy - 0.5,
                   distortion=main["distortion"], model="brown_conrady")

    def jittered(self, rng: np.random.Generator, focal_frac: float, center_px: float) -> Intrinsics:
        """Per-episode calibration spread: focal scale and principal-point shift."""
        scale = 1.0 + rng.uniform(-focal_frac, focal_frac)
        return replace(self, fx=self.fx * scale, fy=self.fy * scale,
                       cx=self.cx + rng.uniform(-center_px, center_px),
                       cy=self.cy + rng.uniform(-center_px, center_px))

    def undistort_normalized(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distorted pixels to undistorted normalized coordinates (the lens model's deprojection)."""
        if self.model == "brown_conrady":
            return self._undistort_opencv(u, v)
        k1, k2, p1, p2, k3 = self.distortion
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        ux = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        uy = y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        return ux, uy

    def _undistort_opencv(self, u: np.ndarray, v: np.ndarray, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV's forward model inverted by fixed-point iteration, run to convergence: a wide lens needs far
        more than ``cv2.undistortPoints``' default five steps at its corners."""
        k1, k2, p1, p2, k3 = self.distortion
        xd = (np.asarray(u, dtype=np.float64) - self.cx) / self.fx
        yd = (np.asarray(v, dtype=np.float64) - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            nx = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial
            ny = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial
            step = max(float(np.abs(nx - x).max(initial=0.0)), float(np.abs(ny - y).max(initial=0.0)))
            x, y = nx, ny
            if step < 1e-12:
                break
        return x, y

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Undistorted pinhole projection of optical-frame points (+z forward) to pixels."""
        z = np.maximum(points_cam[..., 2], 1e-9)
        return np.stack([self.fx * points_cam[..., 0] / z + self.cx,
                         self.fy * points_cam[..., 1] / z + self.cy], axis=-1)


@dataclass(frozen=True)
class RenderPlan:
    """The pinhole frame MuJoCo renders and the remap onto the real grid."""

    width: int
    height: int
    focal: float
    map_x: np.ndarray
    map_y: np.ndarray

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.focal))


def render_plan(intr: Intrinsics, margin: float = 1.06, focal: float | None = None) -> RenderPlan:
    """Choose a centred pinhole render that covers every real pixel's ray, and its remap.

    ``focal`` (render pixels per unit of normalized coordinate) defaults to the stream's own mean focal.
    """
    u, v = np.meshgrid(np.arange(intr.width, dtype=np.float64), np.arange(intr.height, dtype=np.float64))
    ux, uy = intr.undistort_normalized(u, v)
    focal = 0.5 * (intr.fx + intr.fy) if focal is None else focal
    half_w = math.ceil(focal * float(np.abs(ux).max()) * margin)
    half_h = math.ceil(focal * float(np.abs(uy).max()) * margin)
    width, height = 2 * half_w, 2 * half_h
    map_x = (ux * focal + half_w - 0.5).astype(np.float32)
    map_y = (uy * focal + half_h - 0.5).astype(np.float32)
    return RenderPlan(width=width, height=height, focal=focal, map_x=map_x, map_y=map_y)


def to_real_grid(plan: RenderPlan, image: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Remap a pinhole render onto the real stream's pixel grid."""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, plan.map_x, plan.map_y, interp, borderMode=cv2.BORDER_REPLICATE)
