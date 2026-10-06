"""Reusable OpenCV AprilTag detector with an explicit inventory allowlist."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from .config import DetectorProfile, load_inventory
from .geometry import normalize_corners


def tag_dictionary(family: str):
    dictionaries = {
        "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16H5,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36H11,
    }
    if family not in dictionaries:
        raise ValueError(f"unsupported OpenCV fiducial family {family!r}")
    return cv2.aruco.getPredefinedDictionary(dictionaries[family])


def _detector_parameters():
    # OpenCV 4.7 made the detector an object; the distribution's 4.6 (the ROS
    # node's system Python) has only the functional API.
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


class _FunctionalDetector:
    """`ArucoDetector.detectMarkers` over OpenCV < 4.7's `aruco.detectMarkers`."""

    def __init__(self, dictionary, params):
        self.dictionary, self.params = dictionary, params

    def detectMarkers(self, gray):  # noqa: N802 -- OpenCV's name
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)


def _marker_detector(dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return _FunctionalDetector(dictionary, params)


def refine_edges(gray: np.ndarray, corners, *, reach_px: float = 1.5, passes: int = 3,
                 max_shift_px: float = 3.0) -> np.ndarray | None:
    """A detected tag's corners fitted to its outer edge (pixel centres at integers), or None where a side shows no
    edge or the fit moves a corner more than max_shift_px. Along each side the edge between the dark border inside
    and the light margin outside is located on the side's normal: the centroid of the squared intensity step
    taken 1 px either side, over +-reach_px, sampled bilinearly (the AprilTag detector's refine_edges). A line
    through those points is the side; adjacent sides meet at the corners; repeated from the new corners."""
    image = np.ascontiguousarray(gray, dtype=np.float32)
    start = normalize_corners(corners)
    quad = start.copy()
    offsets = np.arange(-reach_px, reach_px + 1e-9, 0.25)
    along = np.linspace(0.15, 0.85, 12)          # clear of the corners, where the next side's edge blurs in

    def sample(points):
        return cv2.remap(image, np.ascontiguousarray(points[..., 0], np.float32),
                         np.ascontiguousarray(points[..., 1], np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    for _ in range(passes):
        centre, lines = quad.mean(axis=0), []
        for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True):
            normal = np.array([b[1] - a[1], a[0] - b[0]]) / np.linalg.norm(b - a)
            normal *= np.sign(normal @ ((a + b) / 2.0 - centre))           # outward
            base = a + np.outer(along, b - a)
            points = base[:, None, :] + offsets[None, :, None] * normal
            step = sample(points + normal) - sample(points - normal)       # light outside, dark inside
            weight = np.where(step > 0.0, step * step, 0.0)
            total = weight.sum(axis=1)
            seen = total > 0.0
            if np.count_nonzero(seen) < 4:
                return None
            edge = base[seen] + ((weight[seen] @ offsets) / total[seen])[:, None] * normal
            mean = edge.mean(axis=0)
            lines.append((mean, np.linalg.svd(edge - mean)[2][0]))
        for i in range(4):           # corner i joins side i - 1 (from corner i - 1) and side i (to corner i + 1)
            (p, u), (r, v) = lines[i - 1], lines[i]
            system = np.column_stack([u, -v])
            if abs(np.linalg.det(system)) < 1e-6:
                return None
            quad[i] = p + np.linalg.solve(system, r - p)[0] * u
    if float(np.max(np.linalg.norm(quad - start, axis=1))) > max_shift_px:
        return None
    return quad


@dataclasses.dataclass(frozen=True)
class Detection:
    camera: str
    tag_id: int
    corners_px: np.ndarray
    timestamp_ns: int
    side_px: float
    family: str | None = None


@dataclasses.dataclass(frozen=True)
class DetectorConfig:
    scale: float = 1.0
    adaptive_window_max: int = 45
    min_side_px: float = 12.0
    corner_refinement: bool = True
    # How corner_refinement refines: "subpix" (cornerSubPix) or "edges" (refine_edges on the contour pass's
    # quads). On the D555's 23 px wrist tags OpenCV 4.6's subpix leaves 89% of corner coordinates on whole
    # pixels; over synthetic tags of 20-26 px with known corners it is off by 0.82 px median, more than no
    # refinement (0.54), and the edge fit by 0.11, unbiased, finding every tag. OpenCV's AprilTag refinement is
    # not offered: its corners sit +0.44 px off in x and y on 4.6 and 5.0, and on 4.6 it misses 30% of such
    # tags (2026-09-29).
    refinement: str = "subpix"
    # The share of each code cell's margin left out when its bit is read (OpenCV's default 0.13). On the D555's 23
    # px roof tag, about 3 px a cell, a lamp over the palette bloomed the white cells into the black ones and 0.13
    # read the tag in 0 of 10 frames, 0.30 in 10 of 10; in a dimmed room 2 and 8 of 10; in daylight both read it
    # (2026-09-30).
    cell_margin: float = 0.13

    @classmethod
    def from_profile(cls, profile: DetectorProfile) -> "DetectorConfig":
        return cls(**dataclasses.asdict(profile))


class FiducialDetector:
    def __init__(
        self,
        allowed_ids: set[int] | frozenset[int],
        config: DetectorConfig | None = None,
        family: str | None = None,
        keep_best_per_id: bool = False,
        *,
        ids_by_family: dict[str, frozenset[int]] | None = None,
    ):
        self.allowed_ids = frozenset(int(tag_id) for tag_id in allowed_ids)
        self.config = config or DetectorConfig()
        self.keep_best_per_id = keep_best_per_id
        if ids_by_family is None:
            family = family or load_inventory().target("wrist").family
            ids_by_family = {family: self.allowed_ids}
        params = _detector_parameters()
        methods = {"subpix": cv2.aruco.CORNER_REFINE_SUBPIX, "edges": cv2.aruco.CORNER_REFINE_NONE}
        if self.config.refinement not in methods:
            raise ValueError(f"unknown corner refinement {self.config.refinement!r} (subpix or edges)")
        self.fit_edges = self.config.corner_refinement and self.config.refinement == "edges"
        params.cornerRefinementMethod = (
            methods[self.config.refinement]
            if self.config.corner_refinement
            else cv2.aruco.CORNER_REFINE_NONE
        )
        params.adaptiveThreshWinSizeMax = self.config.adaptive_window_max
        params.perspectiveRemoveIgnoredMarginPerCell = self.config.cell_margin
        self.detectors = [
            (name, _marker_detector(tag_dictionary(name), params), ids & self.allowed_ids)
            for name, ids in ids_by_family.items()
        ]

    @classmethod
    def from_inventory(cls, inventory, config=None, target=None, *, include_spares=False):
        families = inventory.ids_by_family(target, include_spares=include_spares)
        return cls(frozenset().union(*families.values()), config, ids_by_family=families)

    def _refined(self, gray: np.ndarray, candidates: list) -> list:
        """The candidates with their corners fitted to the tag's edges under refinement "edges", dropping one
        whose edges do not fit (no corner to trust there); else as detected."""
        if not self.fit_edges:
            return candidates
        fitted = ((family, refine_edges(gray, corners), tag_id) for family, corners, tag_id in candidates)
        return [candidate for candidate in fitted if candidate[1] is not None]

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
        found: list[Detection] = []
        candidates = []
        for family, detector, allowed in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                candidates.extend((family, corner, int(tag_id))
                                  for corner, tag_id in zip(corners, ids.flatten(), strict=True)
                                  if int(tag_id) in allowed)
        for family, candidate, tag_id in self._refined(gray, candidates):
            pixels = normalize_corners(candidate) / scale + offset
            side_px = float(
                np.mean(
                    [np.linalg.norm(pixels[index] - pixels[(index + 1) % 4]) for index in range(4)]
                )
            )
            if side_px < self.config.min_side_px:
                continue
            detection = Detection(camera, tag_id, pixels, int(timestamp_ns), side_px, family)
            found.append(detection)
        if not self.keep_best_per_id:
            return found
        best: dict[tuple[str | None, int], Detection] = {}
        for detection in found:
            key = detection.family, detection.tag_id
            if key not in best or detection.side_px > best[key].side_px:
                best[key] = detection
        return list(best.values())
