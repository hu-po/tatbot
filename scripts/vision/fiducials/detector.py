"""Reusable OpenCV AprilTag detector with an explicit inventory allowlist."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .config import DetectorProfile
from .geometry import normalize_corners


@dataclasses.dataclass(frozen=True)
class Detection:
    camera: str
    tag_id: int
    corners_px: np.ndarray
    timestamp_ns: int
    side_px: float


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    scale: float = 1.0
    adaptive_window_max: int = 45
    min_side_px: float = 12.0
    corner_refinement: bool = True

    @classmethod
    def from_profile(cls, profile: DetectorProfile) -> "DetectorConfig":
        return cls(**dataclasses.asdict(profile))


class FiducialDetector:
    def __init__(
        self,
        allowed_ids: set[int] | frozenset[int],
        config: DetectorConfig | None = None,
        family: str = "apriltag_16h5",
        keep_best_per_id: bool = False,
    ):
        if family != "apriltag_16h5":
            raise ValueError(f"unsupported OpenCV fiducial family {family!r}")
        self.allowed_ids = frozenset(int(tag_id) for tag_id in allowed_ids)
        self.config = config or DetectorConfig()
        self.keep_best_per_id = keep_best_per_id
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
            if self.config.corner_refinement
            else cv2.aruco.CORNER_REFINE_NONE
        )
        params.adaptiveThreshWinSizeMax = self.config.adaptive_window_max
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(
        self,
        camera: str,
        frame: np.ndarray,
        timestamp_ns: int,
        roi_xyxy: tuple[int, int, int, int] | None = None,
    ) -> list[Detection]:
        scale = self.config.scale
        if not 0 < scale <= 1.0:
            raise ValueError("detector scale must be in (0, 1]")
        offset = np.zeros(2, dtype=np.float64)
        work = frame
        if roi_xyxy is not None:
            x0, y0, x1, y1 = roi_xyxy
            height, width = frame.shape[:2]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"invalid detector ROI {roi_xyxy} for {width}x{height}")
            work = frame[y0:y1, x0:x1]
            offset[:] = (x0, y0)
        if scale != 1.0:
            work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        found: list[Detection] = []
        if ids is None:
            return []
        for candidate, raw_id in zip(corners, ids.flatten(), strict=True):
            tag_id = int(raw_id)
            if tag_id not in self.allowed_ids:
                continue
            pixels = normalize_corners(candidate) / scale + offset
            side_px = float(
                np.mean(
                    [np.linalg.norm(pixels[index] - pixels[(index + 1) % 4]) for index in range(4)]
                )
            )
            if side_px < self.config.min_side_px:
                continue
            detection = Detection(camera, tag_id, pixels, int(timestamp_ns), side_px)
            found.append(detection)
        if not self.keep_best_per_id:
            return found
        best: dict[int, Detection] = {}
        for detection in found:
            if detection.tag_id not in best or detection.side_px > best[detection.tag_id].side_px:
                best[detection.tag_id] = detection
        return list(best.values())
