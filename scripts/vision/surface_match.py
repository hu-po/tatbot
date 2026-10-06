"""Bounded ORB/SIFT keyframe search, advisory only: no material IDs or robot pose.

A homography is a local image-consistency check, not a curved-surface model.
Even accepted candidates require independent material-identity validation.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchSettings:
    method: str = "orb"
    cpu_fraction: float = 0.1
    max_edge: int = 1280
    max_features: int = 1200
    interval_ms: int = 500
    ratio: float = 0.7
    min_inliers: int = 12
    min_coverage: float = 0.2


class PatchMatcher:
    def __init__(self, roi, settings=None):
        self.roi = tuple(roi)
        self.settings = settings or MatchSettings()
        s = self.settings
        if not (
            s.method in ("orb", "sift")
            and 0 < s.cpu_fraction <= 0.25
            and 64 <= s.max_edge <= 2048
            and 4 <= s.min_inliers <= s.max_features <= 2000
            and s.max_features >= 12
            and 100 <= s.interval_ms <= 10000
            and 0 < s.ratio < 1
            and 0 < s.min_coverage <= 1
        ):
            raise ValueError("invalid matcher workload or acceptance bounds")
        self.detector = (
            cv2.SIFT_create(nfeatures=s.max_features, contrastThreshold=0.02)
            if s.method == "sift"
            else cv2.ORB_create(nfeatures=s.max_features, edgeThreshold=15, fastThreshold=10)
        )
        self.norm = cv2.NORM_L2 if s.method == "sift" else cv2.NORM_HAMMING
        self.next_attempt_ns = None
        self.descriptors = None
        self.points = None
        self.last_attempt_ns = None
        self.shape = None
        self.last_seen_ns = None
        self.search_level = 0
        self.search_polygon = None

    def seed(self, gray):
        x, y, w, h = self.roi
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("ROI must fit inside the image")
        self.shape = gray.shape
        patch = gray[y : y + h, x : x + w]
        scale = min(1.0, self.settings.max_edge / max(patch.shape))
        if scale < 1:
            patch = cv2.resize(patch, (round(w * scale), round(h * scale)))
        keypoints, self.descriptors = self.detector.detectAndCompute(patch, None)
        self.points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2) / scale + (x, y)
        self.search_polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)
        self.search_level = 0
        return len(keypoints)

    def match(self, gray, timestamp_ns):
        out = {
            "status": "unavailable",
            "reason": None,
            "candidate_polygon_px": [],
            "reference_features": 0 if self.points is None else len(self.points),
            "matches": 0,
            "inliers": 0,
            "material_identity_verified": False,
            "motion_authority": False,
        }
        if gray.shape != self.shape:
            out["reason"] = "geometry_changed"
            return out
        if self.last_seen_ns is not None and timestamp_ns <= self.last_seen_ns:
            out["reason"] = "timestamp_regression"
            return out
        self.last_seen_ns = timestamp_ns
        if self.next_attempt_ns is not None and timestamp_ns < self.next_attempt_ns:
            out.update(status="skipped", reason="rate_limited")
            return out
        self.last_attempt_ns = timestamp_ns
        cpu_start = time.process_time()
        result = self._search(gray, out)
        if result["status"] == "candidate":
            # This only steers advisory image search, never material tracks.
            self.search_polygon = np.array(result["candidate_polygon_px"])
            self.search_level = 0
        else:
            self.search_level = (self.search_level + 1) % 3
        cpu_ms = (time.process_time() - cpu_start) * 1000
        delay_ms = max(self.settings.interval_ms, cpu_ms / self.settings.cpu_fraction)
        self.next_attempt_ns = timestamp_ns + round(delay_ms * 1_000_000)
        result.update(cpu_ms=cpu_ms, next_attempt_after_ms=delay_ms, method=self.settings.method)
        return result

    def _scene_features(self, gray):
        # A global strongest-N selection discarded nearly all paper features
        # in cluttered recordings. Reserve descriptor capacity spatially.
        rows, cols = 3, 4
        cap = max(1, self.settings.max_features // (rows * cols))
        detector = (
            cv2.SIFT_create(nfeatures=cap, contrastThreshold=0.02)
            if self.settings.method == "sift"
            else cv2.ORB_create(nfeatures=cap, edgeThreshold=15, fastThreshold=10)
        )
        points, descriptors = [], []
        height, width = gray.shape
        for row in range(rows):
            for col in range(cols):
                x0, x1 = col * width // cols, (col + 1) * width // cols
                y0, y1 = row * height // rows, (row + 1) * height // rows
                left, top = max(0, x0 - 32), max(0, y0 - 32)
                patch = gray[top : min(height, y1 + 32), left : min(width, x1 + 32)]
                keys, desc = detector.detectAndCompute(patch, None)
                if desc is None:
                    continue
                selected = sorted(range(len(keys)), key=lambda i: -keys[i].response)
                kept = 0
                for i in selected:
                    key = keys[i]
                    x, y = key.pt[0] + left, key.pt[1] + top
                    if x0 <= x < x1 and y0 <= y < y1:
                        key.pt = (x, y)
                        points.append(key)
                        descriptors.append(desc[i])
                        kept += 1
                        if kept == cap:
                            break
        return points, np.array(descriptors) if descriptors else None

    def _search_window(self, shape):
        """Expand after failures; perform only one bounded extraction per attempt."""
        height, width = shape
        if self.search_level == 2:
            return 0, 0, width, height
        low, high = self.search_polygon.min(axis=0), self.search_polygon.max(axis=0)
        center = (low + high) / 2
        # A square includes in-plane rotations of a long, narrow initial ROI.
        half = max(high - low) * (0.75 if self.search_level == 0 else 1.5)
        left, top = np.maximum(0, np.floor(center - half)).astype(int)
        right, bottom = np.minimum((width, height), np.ceil(center + half)).astype(int)
        return int(left), int(top), int(right - left), int(bottom - top)

    def _search(self, gray, out):
        if self.descriptors is None or len(self.descriptors) < self.settings.min_inliers:
            out["reason"] = "insufficient_reference_features"
            return out
        left, top, width, height = self._search_window(gray.shape)
        window = gray[top : top + height, left : left + width]
        scale = min(1.0, self.settings.max_edge / max(window.shape))
        reduced = cv2.resize(window, (round(width * scale), round(height * scale))) if scale < 1 else window
        out.update(
            search_stage=("local", "expanded", "global")[self.search_level],
            search_window_px=[left, top, width, height],
            search_scale=scale,
        )
        keypoints, descriptors = self._scene_features(reduced)
        if descriptors is None or len(descriptors) < 2:
            out["reason"] = "insufficient_scene_features"
            return out
        matcher = cv2.BFMatcher(self.norm)
        forward = matcher.knnMatch(self.descriptors, descriptors, k=2)
        reverse = matcher.knnMatch(descriptors, self.descriptors, k=2)
        # Bidirectional ratio checks reject ambiguous repeated descriptors;
        # one-to-one correspondence alone would still accept many grid aliases.
        reverse_good = {
            a.queryIdx: a.trainIdx
            for pair in reverse
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < self.settings.ratio * b.distance
        }
        good = [
            a
            for pair in forward
            if len(pair) == 2
            for a, b in [pair]
            if a.distance < self.settings.ratio * b.distance and reverse_good.get(a.trainIdx) == a.queryIdx
        ]
        out["matches"] = len(good)
        if len(good) < self.settings.min_inliers:
            out["reason"] = "insufficient_distinct_matches"
            return out
        src = self.points[[m.queryIdx for m in good]].astype(np.float32)
        dst = np.array([keypoints[m.trainIdx].pt for m in good], np.float32) / scale + (left, top)
        dst = dst.astype(np.float32)
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0 / scale, maxIters=1000, confidence=0.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            out["reason"] = "no_consistent_transform"
            return out
        selected = mask.ravel().astype(bool)
        out["inliers"] = int(selected.sum())
        x, y, w, h = self.roi
        coverage = (
            float(cv2.contourArea(cv2.convexHull(src[selected]))) / (w * h) if selected.sum() >= 3 else 0.0
        )
        out["reference_coverage"] = coverage
        if (
            selected.sum() < self.settings.min_inliers
            or selected.mean() < 0.6
            or coverage < self.settings.min_coverage
        ):
            out["reason"] = "weak_geometric_support"
            return out
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        polygon = cv2.perspectiveTransform(corners[None], matrix)[0]
        area = cv2.contourArea(polygon, oriented=True) / (w * h)
        inside = (
            (polygon >= -0.5).all()
            and (polygon[:, 0] <= gray.shape[1] + 0.5).all()
            and (polygon[:, 1] <= gray.shape[0] + 0.5).all()
        )
        if (
            not np.isfinite(polygon).all()
            or not cv2.isContourConvex(polygon)
            or not 0.25 <= area <= 4
            or not inside
        ):
            out["reason"] = "implausible_projection"
            return out
        out.update(status="candidate", reason="geometric_match_only", candidate_polygon_px=polygon.tolist())
        return out


class KeyframeBank:
    """Experimental, bounded appearance references; never a material-ID tracker.

    Admission reuses existing flow and requires direct agreement with the initial
    descriptor reference. One bank operation spends the shared CPU allowance.
    """

    def __init__(self, roi, settings=None):
        self.roi = tuple(roi)
        self.settings = settings or MatchSettings()
        self.references = []
        self.shape = None
        self.next_attempt_ns = None
        self.last_seen_ns = None
        self.cursor = 0
        self.last_admitted_polygon = None
        self.next_search_ns = None

    def seed(self, gray):
        first = PatchMatcher(self.roi, self.settings)
        count = first.seed(gray)
        self.references = [first]
        self.shape = gray.shape
        self.last_admitted_polygon = first.search_polygon.copy()
        return count

    def match(self, gray, timestamp_ns, tracking=None, anchors=None):
        result = {
            "status": "unavailable",
            "reason": None,
            "candidate_polygon_px": [],
            "material_identity_verified": False,
            "motion_authority": False,
            "keyframes": len(self.references),
        }
        if gray.shape != self.shape:
            return dict(result, reason="geometry_changed")
        if self.last_seen_ns is not None and timestamp_ns <= self.last_seen_ns:
            return dict(result, reason="timestamp_regression")
        self.last_seen_ns = timestamp_ns
        if self.next_attempt_ns is not None and timestamp_ns < self.next_attempt_ns:
            return dict(result, status="skipped", reason="rate_limited")
        start = time.process_time()
        polygon, reason = self._flow_polygon(tracking, anchors)
        if self.next_search_ns is None:
            self.next_search_ns = timestamp_ns + self.settings.interval_ms * 1_000_000
        waiting_for_view = (
            polygon is None
            and tracking
            and tracking.get("status") == "tracked"
            and timestamp_ns < self.next_search_ns
        )
        if waiting_for_view:
            result.update(status="skipped", reason=reason)
        elif polygon is not None:
            result = self._admit(gray, polygon, result)
        else:
            reference_index = self.cursor % len(self.references)
            reference = self.references[reference_index]
            # The bank owns the only scheduler, including admission CPU cost.
            reference.next_attempt_ns = None
            result = reference.match(gray, timestamp_ns)
            result.update(reference_index=reference_index, admission_reason=reason)
            self.cursor = (reference_index - 1) % len(self.references)
            self.next_search_ns = timestamp_ns + self.settings.interval_ms * 1_000_000
        elapsed = (time.process_time() - start) * 1000
        delay = max(
            50 if waiting_for_view else self.settings.interval_ms,
            elapsed / self.settings.cpu_fraction,
        )
        self.next_attempt_ns = timestamp_ns + round(delay * 1_000_000)
        result.update(
            cpu_ms=elapsed,
            next_attempt_after_ms=delay,
            keyframes=len(self.references),
            method=self.settings.method,
        )
        return result

    def _flow_polygon(self, tracking, anchors):
        if not tracking or tracking.get("status") != "tracked" or anchors is None:
            return None, "tracking_unavailable"
        if tracking.get("sharpness_ratio", 0) < 0.7:
            return None, "unclear_reference"
        ids = np.asarray(tracking["ids"], dtype=int)
        dst = np.asarray(tracking["points_px"], dtype=np.float32)
        if len(ids) < 24 or len(ids) < 0.8 * len(anchors):
            return None, "insufficient_original_tracks"
        if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(anchors)).any():
            return None, "invalid_original_ids"
        src = np.asarray(anchors, dtype=np.float32)[ids]
        if dst.shape != src.shape or not np.isfinite(dst).all():
            return None, "invalid_flow"
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 1.5, maxIters=500, confidence=0.995)
        if matrix is None or mask is None or not np.isfinite(matrix).all() or mask.mean() < 0.9:
            return None, "inconsistent_flow"
        x, y, w, h = self.roi
        if cv2.contourArea(cv2.convexHull(src[mask.ravel() != 0])) / (w * h) < 0.3:
            return None, "narrow_flow_support"
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        polygon = cv2.perspectiveTransform(corners[None], matrix)[0]
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return None, "invalid_flow_projection"
        if np.max(np.linalg.norm(polygon - self.last_admitted_polygon, axis=1)) < 5:
            return None, "view_unchanged"
        return polygon, None

    def _admit(self, gray, polygon, result):
        # Validate against the fixed first reference, not a chain of new guesses.
        initial = self.references[0]
        checkpoint = initial.search_polygon, initial.search_level
        initial.search_polygon, initial.search_level = polygon, 0
        try:
            check = initial._search(gray, dict(result))
        finally:
            initial.search_polygon, initial.search_level = checkpoint
        if check["status"] != "candidate":
            return dict(result, reason="initial_reference_rejected", admission_detail=check["reason"])
        error = float(np.max(np.linalg.norm(np.array(check["candidate_polygon_px"]) - polygon, axis=1)))
        if error > 2:
            return dict(result, reason="flow_descriptor_disagreement", admission_error_px=error)
        low = np.maximum(0, np.floor(polygon.min(axis=0))).astype(int)
        high = np.minimum((gray.shape[1], gray.shape[0]), np.ceil(polygon.max(axis=0))).astype(int)
        if (high - low < 32).any():
            return dict(result, reason="reference_too_small")
        reference = PatchMatcher((*map(int, low), *map(int, high - low)), self.settings)
        reference.seed(gray)
        # Exclude descriptor centers near the selected patch boundary.
        keep = np.array(
            [cv2.pointPolygonTest(polygon, tuple(map(float, p)), True) >= 16 for p in reference.points],
            dtype=bool,
        )
        if keep.sum() < self.settings.min_inliers:
            return dict(result, reason="insufficient_interior_features")
        x, y, w, h = self.roi
        original = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        to_original = cv2.getPerspectiveTransform(polygon.astype(np.float32), original)
        reference.points = cv2.perspectiveTransform(
            reference.points[keep].astype(np.float32)[None], to_original
        )[0]
        reference.descriptors = reference.descriptors[keep]
        reference.roi = self.roi
        reference.search_polygon = polygon.copy()
        if len(self.references) == 3:
            self.references.pop(1)
        self.references.append(reference)
        self.cursor = len(self.references) - 1
        self.last_admitted_polygon = polygon.copy()
        return dict(
            result,
            status="admitted",
            reason="flow_and_initial_reference_agree",
            admission_error_px=error,
            reference_features=len(reference.points),
        )
