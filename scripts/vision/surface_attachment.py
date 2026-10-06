"""Bounded original-reference appearance/depth observer; no motion authority.

Development settings describe software rejection criteria, not physical accuracy.
Every accepted sample matches the immutable initial appearance and measured points.
An identical-looking replacement cannot be distinguished by this sensor alone.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from surface_consistency import sample_xyz
from surface_match import MatchSettings, PatchMatcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import schemas  # noqa: E402


@dataclass(frozen=True)
class AttachmentSettings:
    max_features: int = 2000
    max_points: int = 128
    min_points: int = 12
    ratio: float = 0.65
    min_coverage: float = 0.12
    min_descriptor_distance: float = 120.0
    min_distinctive_fraction: float = 0.6
    residual_m: float = 0.0015
    min_inlier_fraction: float = 0.85
    max_age_ms: float = 250.0
    max_edge: int = 1280
    ransac_trials: int = 128

    def __post_init__(self):
        if not (
            12 <= self.min_points <= self.max_points <= 128
            and self.max_points <= self.max_features <= 2000
            and 0 < self.ratio < 1
            and 0 < self.min_coverage <= 1
            and 0 < self.min_descriptor_distance <= 512
            and 0 < self.min_distinctive_fraction <= 1
            and 0 < self.residual_m <= 0.02
            and 0.5 <= self.min_inlier_fraction <= 1
            and 0 < self.max_age_ms <= 10000
            and 64 <= self.max_edge <= 2048
            and 1 <= self.ransac_trials <= 512
        ):
            raise ValueError("invalid development observer settings")


def fit_rigid(source, target):
    """Least-squares proper rigid transform, source coordinates to target."""
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, b - rotation @ a
    return matrix


def rigid_residual(source, target, matrix):
    return np.linalg.norm(source @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)


def _tracking_image(gray):
    """Remove broad illumination changes without amplifying weak texture.

    Optical flow's brightness-constancy assumption otherwise moves correct
    descriptor matches toward unrelated pixels after exposure or shadow changes.
    Raw images still establish quality, descriptors, and the original reference.
    """
    raw = gray.astype(np.float32)
    illumination = cv2.GaussianBlur(raw, (21, 21), 7)
    return np.clip(raw - illumination + 128, 0, 255).astype(np.uint8)


def _bilinear(field, pixels):
    """Sample an array and its x/y derivatives, with explicit image bounds."""
    height, width = field.shape[:2]
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 0] < width - 1)
        & (pixels[:, 1] < height - 1)
    )
    safe = np.nan_to_num(pixels, nan=0, posinf=0, neginf=0)
    low = np.floor(np.clip(safe, [0, 0], [width - 2, height - 2])).astype(int)
    fraction = np.clip(safe - low, 0, 1)
    x, y = low.T
    a, b, c, d = field[y, x], field[y, x + 1], field[y + 1, x], field[y + 1, x + 1]
    fx, fy = fraction.T
    if field.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    value = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    dx, dy = (b - a) * (1 - fy) + (d - c) * fy, (c - a) * (1 - fx) + (d - b) * fx
    return value, dx, dy, valid


def measured_xyz(frame, pixels, discontinuity_m):
    """Subpixel measured depth; never fill a hole or interpolate across a step.

    Reuses nearest-pixel validity from surface_consistency. Interpolation uses
    four valid nearby measurements only; partial neighborhoods retain the
    measured nearest pixel. Depth steps invalidate the sample conservatively.
    """
    xyz, valid = sample_xyz(frame, pixels)
    depth = frame["depth_m"]
    height, width = depth.shape
    ij = np.rint(pixels).astype(int)
    x, y = np.clip(ij, [1, 1], [width - 2, height - 2]).T
    neighbors = np.stack([depth[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
    near, far = frame["depth_range"]
    supported = np.isfinite(neighbors) & (neighbors > near) & (neighbors < far)
    high = np.max(np.where(supported, neighbors, -np.inf), axis=1)
    low = np.min(np.where(supported, neighbors, np.inf), axis=1)
    valid &= high - low <= discontinuity_m
    z, _, _, inside = _bilinear(depth, pixels)
    corners = np.floor(np.clip(pixels, [0, 0], [width - 2, height - 2])).astype(int)
    cx, cy = corners.T
    four = np.stack([depth[cy + dy, cx + dx] for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))], axis=1)
    all_measured = (np.isfinite(four) & (four > near) & (four < far)).all(axis=1)
    rays, _, _, _ = _bilinear(frame["rays"], pixels)
    interpolate = valid & inside & all_measured
    xyz[interpolate] = rays[interpolate] * z[interpolate, None]
    return xyz, valid


def project_measured_rays(points, rays):
    """Invert the owner's calibrated ray field with bounded Newton iterations.

    This handles distortion through the same deprojection rays used for depth;
    it does not silently replace a distorted profile with pinhole projection.
    """
    height, width = rays.shape[:2]
    center = np.array([[(width - 1) / 2, (height - 1) / 2]])
    middle, dx, dy, _ = _bilinear(rays, center)
    jacobian = np.column_stack([dx[0, :2], dy[0, :2]])
    if abs(np.linalg.det(jacobian)) < 1e-12:
        return np.zeros((len(points), 2)), np.zeros(len(points), dtype=bool)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    normalized = points[:, :2] / np.maximum(points[:, 2, None], 1e-9)
    pixels = center + (normalized - middle[0, :2]) @ np.linalg.inv(jacobian).T
    for _ in range(8):
        value, dx, dy, inside = _bilinear(rays, pixels)
        determinant = dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]
        valid &= inside & (np.abs(determinant) > 1e-12)
        error = value[:, :2] - normalized
        # Calibrated pinhole fields are already solved by the initial Jacobian.
        # Distorted fields still iterate until a stricter convergence bound than
        # the unchanged final ray acceptance test, or the fixed work cap.
        if np.all(np.linalg.norm(error[valid], axis=1) < 1e-10):
            break
        divisor = np.where(np.abs(determinant) > 1e-12, determinant, 1)
        delta = np.column_stack(
            [
                (dy[:, 1] * error[:, 0] - dy[:, 0] * error[:, 1]) / divisor,
                (-dx[:, 1] * error[:, 0] + dx[:, 0] * error[:, 1]) / divisor,
            ]
        )
        pixels -= np.clip(delta, -32, 32)
    value, _, _, inside = _bilinear(rays, pixels)
    valid &= inside & (np.linalg.norm(value[:, :2] - normalized, axis=1) < 1e-5)
    return pixels, valid


class _ObservationRefusedError(Exception):
    """Expected advisory refusal, preserving the immutable reference."""


class SurfaceAttachmentObserver:
    """One retained reference, no queue/history, bounded descriptor and RANSAC work.

    Input ``rgbd`` follows surface_rgbd.recording_frames(include_images=True).
    ``now_ns`` is in the capture clock domain; offline adapters must explicitly
    supply replay time instead of relabeling historical capture times as fresh.
    Calibration/producer/profile changes latch invalidation until a new observer
    and explicit reference review are constructed. Lost appearance can recover.
    """

    def __init__(self, material_id, reference_id, roi, calibration_id=None, settings=None):
        if not material_id or not reference_id:
            raise ValueError("explicit material and reference identities required")
        self.material_id, self.reference_id = material_id, reference_id
        self.roi = tuple(roi)
        if (
            len(self.roi) != 4
            or any(not isinstance(v, int) or isinstance(v, bool) for v in self.roi)
            or min(self.roi[:2]) < 0
            or min(self.roi[2:]) <= 0
        ):
            raise ValueError("ROI must be integer x,y,width,height with positive area")
        self.calibration_id = calibration_id
        self.settings = settings or AttachmentSettings()
        self.matcher = PatchMatcher(
            self.roi,
            MatchSettings(
                method="sift",
                max_features=self.settings.max_features,
                max_edge=self.settings.max_edge,
                ratio=self.settings.ratio,
            ),
        )
        self.reference = None
        self.signature = None
        self.last_stamp = None
        self.last_sequence = None
        self.invalidated = False
        self.lost = False
        self.drops = 0
        self.observations = 0

    def reference_points(self):
        """Return the immutable original-camera inventory keyed by material ID."""
        if self.reference is None:
            return {}
        return {str(i): point.tolist() for i, point in enumerate(self.reference["xyz"])}

    def _output(self, frame, stamp):
        return {
            **schemas.stamp(schemas.SURFACE, "attachment"),
            "evidence_kind": frame.get("evidence_kind"),
            "material_id": self.material_id,
            "reference_id": self.reference_id,
            "capture_timestamp_ns": stamp,
            "sequence": frame.get("sequence"),
            "producer_id": frame.get("producer_id"),
            "profile_id": frame.get("profile_id"),
            "calibration_id": frame.get("calibration_id"),
            "provenance": frame.get("provenance", {}),
            "observer_profile": asdict(self.settings),
            "observer_algorithm": "original-sift-depth-rigid/7",
            "reference_provenance": None if self.reference is None else self.reference["provenance"],
            "reference_timestamp_ns": None if self.reference is None else self.reference["stamp"],
            "reference_digest": None if self.reference is None else self.reference["digest"],
            "state": "lost",
            "accepted": False,
            "reason": None,
            "motion_authority": False,
            "transform_reference_to_camera": None,
            "geometry_support": None,
            "diagnostics": {
                "observations": self.observations,
                "dropped_frames": self.drops,
                "reference_count": int(self.reference is not None),
            },
        }

    def observe(self, frame, now_ns=None):
        start = time.perf_counter()
        self.observations += 1
        stamp = frame.get("capture_timestamp_ns", frame.get("stamp"))
        result = self._output(frame, stamp)
        try:
            self._validate_clock(frame, now_ns, stamp, result)
            rgbd, signature = self._validate_capture(frame, result)
            if self.reference is None:
                self._seed(rgbd, frame, stamp, signature, result)
            result.update(
                reference_timestamp_ns=self.reference["stamp"],
                reference_provenance=self.reference["provenance"],
                reference_digest=self.reference["digest"],
            )
            result["diagnostics"]["reference_count"] = 1
            ids, source, current, pixels = self._matches(rgbd, result)
            current, pixels = self._reassociate(rgbd, ids, source, current, pixels, result)
            matrix, best, ref_pixels = self._fit(ids, source, current, result)
            self._check_geometry(rgbd, matrix, result)
            result.update(
                state="recovered" if self.lost else "tracked",
                accepted=True,
                reason="original_reference_rigid_match",
                transform_reference_to_camera=matrix.tolist(),
                geometry_support={
                    "ids": ids[best].tolist(),
                    "reference_points_m": source[best].tolist(),
                    "current_points_m": current[best].tolist(),
                    "reference_pixels": ref_pixels.tolist(),
                    "current_pixels": pixels[best].tolist(),
                },
            )
            self.lost = False

        except _ObservationRefusedError as error:
            result["reason"] = str(error)
            self.lost = self.reference is not None
        return self._finish(start, result)

    def _finish(self, start, result):
        result["diagnostics"].update(
            processing_ms=(time.perf_counter() - start) * 1000, dropped_frames=self.drops
        )
        if "capture_age_ms" in result["diagnostics"]:
            result["diagnostics"]["capture_to_result_age_ms"] = (
                result["diagnostics"]["capture_age_ms"] + result["diagnostics"]["processing_ms"]
            )
            if (
                result["accepted"]
                and result["evidence_kind"] == "live-rgbd"
                and result["diagnostics"]["capture_to_result_age_ms"] > self.settings.max_age_ms
            ):
                result.update(
                    accepted=False,
                    state="lost",
                    reason="stale_after_processing",
                    transform_reference_to_camera=None,
                    geometry_support=None,
                )
                self.lost = True
        return result

    def _validate_clock(self, frame, now_ns, stamp, result):
        if self.invalidated:
            result["state"] = "invalidated"
            raise _ObservationRefusedError("reference_review_required")
        if (
            not isinstance(stamp, int)
            or isinstance(stamp, bool)
            or stamp <= 0
            or not isinstance(now_ns, int)
            or isinstance(now_ns, bool)
        ):
            raise _ObservationRefusedError("missing_capture_clock_support")
        age = (now_ns - stamp) / 1e6
        result["diagnostics"]["capture_age_ms"] = age
        if age < 0 or age > self.settings.max_age_ms:
            self.drops += 1
            raise _ObservationRefusedError("future_capture" if age < 0 else "stale_capture")
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.drops += 1
            raise _ObservationRefusedError("duplicate_or_reordered_capture")
        self.last_stamp = stamp

    def _rgbd(self, frame):
        rgbd = frame.get("rgbd", {})
        gray = np.asarray(rgbd.get("gray"))
        depth = np.asarray(rgbd.get("depth_m"))
        rays = np.asarray(rgbd.get("rays"))
        if (
            gray.ndim != 2
            or gray.dtype != np.uint8
            or max(gray.shape) > self.settings.max_edge
            or min(gray.shape) < 4
            or depth.shape != gray.shape
            or not np.issubdtype(depth.dtype, np.number)
            or not np.issubdtype(rays.dtype, np.number)
            or rays.shape != (*gray.shape, 3)
            or not np.isfinite(rays).all()
            or not np.allclose(rays[..., 2], 1.0, atol=1e-6, rtol=0)
        ):
            raise _ObservationRefusedError("invalid_or_oversized_rgbd")
        bounds = rgbd.get("depth_range", ())
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] < float("inf"):
            raise _ObservationRefusedError("invalid_depth_range")
        return rgbd

    def _validate_capture(self, frame, result):
        if frame.get("invalid"):
            raise _ObservationRefusedError(frame["invalid"])
        if frame.get("evidence_kind") not in ("synthetic-render", "recorded-rgbd", "live-rgbd"):
            raise _ObservationRefusedError("missing_evidence_provenance")
        if not frame.get("producer_id") or not frame.get("profile_id"):
            raise _ObservationRefusedError("missing_producer_or_profile")
        if self.calibration_id is not None and frame.get("calibration_id") != self.calibration_id:
            self.invalidated = True
            result["state"] = "invalidated"
            raise _ObservationRefusedError("calibration_changed")
        rgbd = self._rgbd(frame)
        gray, rays, bounds = rgbd["gray"], rgbd["rays"], rgbd["depth_range"]
        signature = (
            frame.get("producer_id"),
            frame.get("profile_id"),
            frame.get("calibration_id"),
            frame["evidence_kind"],
            gray.shape,
            tuple(bounds),
            hashlib.sha256(np.ascontiguousarray(rays).tobytes()).hexdigest(),
        )
        if self.signature is not None and signature != self.signature:
            self.invalidated = True
            result["state"] = "invalidated"
            raise _ObservationRefusedError("capture_context_changed")
        sequence = frame.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise _ObservationRefusedError("missing_capture_sequence")
        if self.last_sequence is not None and sequence <= self.last_sequence:
            self.invalidated = True
            result["state"] = "invalidated"
            raise _ObservationRefusedError("producer_sequence_regression")
        self.last_sequence = sequence
        if np.std(gray) < 8 or np.mean(gray) < 8:
            raise _ObservationRefusedError("insufficient_surface_detail")
        return rgbd, signature

    def _reference_features(self, rgbd):
        gray, depth = rgbd["gray"], rgbd["depth_m"]
        x, y, w, h = self.roi
        if x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise _ObservationRefusedError("reference_roi_outside_image")
        keys, descriptors = self.matcher._scene_features(gray)
        if descriptors is None:
            raise _ObservationRefusedError("insufficient_reference_features")
        points = np.array([k.pt for k in keys], dtype=np.float32)
        inside = (points[:, 0] >= x) & (points[:, 0] < x + w) & (points[:, 1] >= y) & (points[:, 1] < y + h)
        # A silhouette can have valid center depth while its appearance window
        # mostly tracks background. Require measured support before committing
        # material identities; subsequent observations retain the entire bank.
        near, far = rgbd["depth_range"]
        measured = np.isfinite(depth) & (depth > near) & (depth < far)
        coverage = cv2.boxFilter(measured.astype(np.float32), -1, (21, 21),
                                 normalize=True, borderType=cv2.BORDER_CONSTANT)
        px, py = np.clip(np.rint(points).astype(int), [0, 0],
                         [depth.shape[1] - 1, depth.shape[0] - 1]).T
        inside &= coverage[py, px] >= 0.9
        if inside.sum() < self.settings.min_points:
            raise _ObservationRefusedError("insufficient_reference_depth_features")
        return points[inside], descriptors[inside]

    def _seed(self, rgbd, frame, stamp, signature, result):
        from surface_material_support import MaterialSupport, SupportProfile
        gray, depth = rgbd["gray"], rgbd["depth_m"]
        x, y, w, h = self.roi
        self.matcher.points, self.matcher.descriptors = self._reference_features(rgbd)
        xyz, valid = measured_xyz(rgbd, self.matcher.points, 2 * self.settings.residual_m)
        # Multiple scale/orientation keypoints at the same pixel do not
        # constitute independent material support.
        kept, occupied = [], set()
        for i in np.flatnonzero(valid):
            cell = tuple(np.rint(self.matcher.points[i] / 4).astype(int))
            if cell not in occupied:
                kept.append(i)
                occupied.add(cell)
        if len(kept) < self.settings.min_points:
            raise _ObservationRefusedError("insufficient_reference_depth_features")
        self.matcher.points = self.matcher.points[kept]
        self.matcher.descriptors = self.matcher.descriptors[kept]
        # Reject descriptors with near-identical alternatives elsewhere in
        # the original patch; bidirectional matching alone can retain grid aliases.
        desc = self.matcher.descriptors.astype(float)
        squared = np.sum(desc * desc, axis=1)
        distance = np.sqrt(np.maximum(0, squared[:, None] + squared[None, :] - 2 * desc @ desc.T))
        spatial = np.linalg.norm(self.matcher.points[:, None] - self.matcher.points[None, :], axis=2)
        distance[spatial < 12] = np.inf
        distinctive = distance.min(axis=1) >= self.settings.min_descriptor_distance
        result["diagnostics"]["distinctive_reference_fraction"] = float(distinctive.mean())
        if (
            distinctive.sum() < self.settings.min_points
            or distinctive.mean() < self.settings.min_distinctive_fraction
        ):
            raise _ObservationRefusedError("ambiguous_reference_appearance")
        kept = np.asarray(kept)[distinctive]
        self.matcher.points = self.matcher.points[distinctive]
        self.matcher.descriptors = self.matcher.descriptors[distinctive]
        # Farthest-point selection preserves broad image support with a
        # bounded immutable inventory, shared by all session observations.
        if len(kept) > self.settings.max_points:
            points = self.matcher.points
            selected = [int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))]
            distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
            for _ in range(1, self.settings.max_points):
                selected.append(int(np.argmax(distance)))
                distance = np.minimum(distance, np.sum((points - points[selected[-1]]) ** 2, axis=1))
            kept = kept[selected]
            self.matcher.points = self.matcher.points[selected]
            self.matcher.descriptors = self.matcher.descriptors[selected]
        grid_x, grid_y = np.meshgrid(np.linspace(x + 2, x + w - 3, 16), np.linspace(y + 2, y + h - 3, 16))
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        probes, supported = measured_xyz(rgbd, grid, 2 * self.settings.residual_m)
        if supported.sum() < 32:
            raise _ObservationRefusedError("insufficient_reference_geometry_probes")
        self.reference = {
            "gray": gray.copy(),
            "probes": probes[supported],
            "probe_pixels": grid[supported],
            "xyz": xyz[kept],
            "stamp": stamp,
            "provenance": json.loads(json.dumps(frame.get("provenance", {}))),
            "digest": hashlib.sha256(
                gray.tobytes()
                + depth.tobytes()
                + b"original-sift-depth-rigid/7"
                + self.matcher.descriptors.tobytes()
                + self.matcher.points.tobytes()
                + xyz[kept].tobytes()
                + probes[supported].tobytes()
                + json.dumps(asdict(self.settings), sort_keys=True).encode()
                + json.dumps(asdict(SupportProfile()), sort_keys=True).encode()
            ).hexdigest(),
        }
        original_rgbd = {key: np.array(rgbd[key], copy=True) for key in ('gray', 'depth_m', 'rays')}
        for array in original_rgbd.values():
            array.setflags(write=False)
        original_rgbd.update(depth_range=tuple(rgbd['depth_range']), material_roi=self.roi)
        self.reference['rgbd'] = original_rgbd
        try:
            self.material_support = MaterialSupport(original_rgbd, self.reference['xyz'],
                                                    self.reference['digest'], self.settings.residual_m)
        except (ValueError, cv2.error) as error:
            self.reference = None
            raise _ObservationRefusedError('original_material_support_initialization_failed') from error
        self.signature = signature

    def _matches(self, rgbd, result):
        gray = rgbd["gray"]
        keys, desc = self.matcher._scene_features(gray)
        if desc is None or len(desc) < 2:
            raise _ObservationRefusedError("insufficient_scene_features")
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        reference_desc = self.matcher.descriptors
        forward = matcher.knnMatch(reference_desc, desc, k=2)
        reverse = matcher.knnMatch(desc, reference_desc, k=2)
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
        result["diagnostics"]["distinct_matches"] = len(good)
        if len(good) < self.settings.min_points:
            raise _ObservationRefusedError("original_patch_not_recognized")
        ids = np.array([m.queryIdx for m in good])
        pixels = np.array([keys[m.trainIdx].pt for m in good])
        original_pixels = self.matcher.points[ids].astype(np.float32).reshape(-1, 1, 2)
        initial_pixels = pixels.astype(np.float32).reshape(-1, 1, 2)
        tracked, verified = self._verify_pixels(gray, original_pixels, initial_pixels)
        tracked, verified, extra_ids, extra_pixels = self._refine_material_matches(
            rgbd, ids, initial_pixels.reshape(-1, 2), tracked, verified, result, recover_unmatched=True)
        result["diagnostics"]["appearance_verified_matches"] = int(verified.sum()) + len(extra_ids)
        ids, pixels = ids[verified], tracked[verified]
        ids, pixels = np.concatenate([ids, extra_ids]), np.concatenate([pixels, extra_pixels])
        if len(ids) < self.settings.min_points:
            raise _ObservationRefusedError("insufficient_verified_original_matches")
        current, valid = measured_xyz(rgbd, pixels, 2 * self.settings.residual_m)
        ids, current, pixels = ids[valid], current[valid], pixels[valid]
        # Spatially ordered deterministic cap; all fits use original coordinates.
        if len(ids) > self.settings.max_points:
            take = np.linspace(0, len(ids) - 1, self.settings.max_points, dtype=int)
            ids, current, pixels = ids[take], current[take], pixels[take]
        source = self.reference["xyz"][ids]
        if len(ids) < self.settings.min_points:
            raise _ObservationRefusedError("insufficient_measured_depth_support")
        return ids, source, current, pixels

    def _refine_material_matches(self, rgbd, ids, initial, tracked, verified, result, matrix=None, *,
                                 recover_unmatched=False):
        """Rigid geometry proposes a view; raw original appearance verifies locations.

        Failed refinement retains the old verified identity and measurement, so
        a disagreeing point cannot disappear as a candidate-pose outlier. A
        biased LK location can change only after independent original-appearance
        verification, including competing starts and the same ambiguity margin.
        """
        from surface_appearance_warp import MeasuredAppearanceWarp
        unchanged = tracked, verified, np.empty(0, dtype=int), np.empty((0, 2))
        diagnostic = 'material_refined_matches' if matrix is None else 'reassociated_material_refined_matches'
        result['diagnostics'][diagnostic] = 0
        current, measured = measured_xyz(rgbd, initial, 2*self.settings.residual_m)
        if matrix is None:
            if measured.sum() < self.settings.min_points:
                return unchanged
            try:
                matrix, _ = self._candidate(self.reference['xyz'][ids[measured]], current[measured])
            except _ObservationRefusedError:
                # Unverified descriptor geometry only proposes a search view.
                # Its failure cannot veto independently verified original LK.
                return unchanged
        reference = self.reference['rgbd']
        try:
            warp = MeasuredAppearanceWarp(reference, rgbd, matrix, roi=self.roi,
                                          residual_m=self.settings.residual_m)
        except ValueError as error:
            raise _ObservationRefusedError('invalid_measured_appearance_projection') from error
        original = self.material_support._appearance
        active = np.flatnonzero(original.original['unique'] & self.material_support._original_window)
        if not recover_unmatched:
            active = np.intersect1d(active, ids)
        if not len(active):
            return unchanged
        if not warp.requires_warp(original.pixels[active], original.profile.window_px):
            if not recover_unmatched or verified.sum() >= self.settings.min_points:
                return unchanged
            warp = None
        expected = self.reference['xyz'] @ matrix[:3, :3].T + matrix[:3, 3]
        predicted, visible = project_measured_rays(expected, rgbd['rays'])
        eligible = np.full(len(expected), recover_unmatched)
        eligible[ids] = True
        eligible &= visible & self.material_support._original_window
        starts = np.full((len(expected), 2, 2), np.nan)
        starts[ids, 0] = initial
        starts[ids, 1] = np.where(verified[:, None], tracked, np.nan)
        evidence = self.material_support._appearance.match(
            rgbd, predicted, eligible, warp=warp, extra_pixels=starts)
        refined = evidence['pixels'][ids]
        qualified = evidence['unique'][ids] & (np.linalg.norm(refined-initial, axis=1) <= 3)
        tracked = np.where(qualified[:, None], refined, tracked)
        result['diagnostics'][diagnostic] = int(qualified.sum())
        # Descriptor redetection is only a proposal source. Search can recover
        # another retained original ID, but cannot add a reference or landmark.
        additional = eligible & evidence['unique']
        additional[ids] = False
        extra_ids = np.flatnonzero(additional)
        if recover_unmatched:
            result['diagnostics']['recovered_original_matches'] = len(extra_ids)
        return tracked, verified | qualified, extra_ids, evidence['pixels'][extra_ids]

    def _verify_pixels(self, gray, original_pixels, initial_pixels):
        """Recheck the original image in both directions; geometry only seeds search."""
        gray = _tracking_image(gray)
        reference = _tracking_image(self.reference["gray"])
        tracked, forward_valid, error = cv2.calcOpticalFlowPyrLK(
            reference,
            gray,
            original_pixels,
            initial_pixels.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            winSize=(21, 21),
            maxLevel=2,
        )
        if tracked is None:
            return initial_pixels.reshape(-1, 2), np.zeros(len(original_pixels), dtype=bool)
        back, backward_valid, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            reference,
            tracked,
            original_pixels.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            winSize=(21, 21),
            maxLevel=2,
        )
        if back is None:
            return initial_pixels.reshape(-1, 2), np.zeros(len(original_pixels), dtype=bool)
        verified = (
            forward_valid.ravel().astype(bool)
            & backward_valid.ravel().astype(bool)
            & (np.linalg.norm(back - original_pixels, axis=2).ravel() <= 0.75)
            & (np.linalg.norm(tracked - initial_pixels, axis=2).ravel() <= 3)
            & (error.ravel() <= 20)
        )
        return tracked.reshape(-1, 2), verified

    def _reassociate(self, rgbd, ids, source, current, pixels, result):
        """Recheck isolated grid aliases without discarding inconsistent material.

        The robust rigid candidate predicts a search location, never a measured
        correspondence. Original-image verification and fresh measured depth are
        required to replace a match. Every unrepaired match still reaches the
        strict rigidity check, including real local deformation.
        """
        matrix, _ = self._candidate(source, current)
        indices = np.flatnonzero(rigid_residual(source, current, matrix) > 2 * self.settings.residual_m)
        result["diagnostics"]["reassociated_matches"] = 0
        if not len(indices):
            return current, pixels
        predicted = source[indices] @ matrix[:3, :3].T + matrix[:3, 3]
        projected, visible = project_measured_rays(predicted, rgbd["rays"])
        original = self.matcher.points[ids[indices]].astype(np.float32).reshape(-1, 1, 2)
        initial = projected.astype(np.float32).reshape(-1, 1, 2)
        tracked, verified = self._verify_pixels(rgbd["gray"], original, initial)
        tracked, verified, _, _ = self._refine_material_matches(
            rgbd, ids[indices], initial.reshape(-1, 2), tracked, verified, result, matrix)
        verified &= visible & np.isfinite(tracked).all(axis=1)
        # Failed optical flow can contain nonfinite coordinates. Keep its
        # original correspondence rather than feeding invalid array indices
        # into measured depth sampling or silently dropping the material point.
        selected, tracked = indices[verified], tracked[verified]
        if not len(selected):
            return current, pixels
        xyz, supported = measured_xyz(rgbd, tracked, 2 * self.settings.residual_m)
        current[selected[supported]], pixels[selected[supported]] = xyz[supported], tracked[supported]
        result["diagnostics"]["reassociated_matches"] = int(supported.sum())
        return current, pixels

    def _candidate(self, source, current):
        best = np.zeros(len(source), bool)
        rng = np.random.default_rng(0)
        for _ in range(self.settings.ransac_trials):
            take = rng.choice(len(source), 3, replace=False)
            if (
                np.linalg.norm(np.cross(source[take[1]] - source[take[0]], source[take[2]] - source[take[0]]))
                < 1e-7
            ):
                continue
            candidate = fit_rigid(source[take], current[take])
            keep = rigid_residual(source, current, candidate) <= self.settings.residual_m
            if keep.sum() > best.sum():
                best = keep
        if best.sum() < self.settings.min_points:
            raise _ObservationRefusedError("rigidity_not_supported")
        return fit_rigid(source[best], current[best]), best

    def _fit(self, ids, source, current, result):
        matrix, best = self._candidate(source, current)
        # Depth-only Kabsch can turn small range noise in a narrow patch into
        # substantial tilt. Constrain the candidate with the measured image rays;
        # every original 3D residual and geometry probe remains mandatory below.
        normalized = current[:, :2] / current[:, 2, None]
        try:
            ok, rotation, translation = cv2.solvePnP(
                source[best], normalized[best], np.eye(3), None,
                cv2.Rodrigues(matrix[:3, :3])[0], matrix[:3, 3].copy(),
                True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as error:
            raise _ObservationRefusedError("image_constrained_pose_failed") from error
        if not ok or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise _ObservationRefusedError("image_constrained_pose_failed")
        matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
        # Image-only translation can trade range for a small bearing residual,
        # especially on narrow patches. For this rotation, the same measured
        # 3D consensus has an exact least-squares centroid translation. Do not
        # select a new subset or discard disagreements after refining the pose.
        matrix[:3, 3] = current[best].mean(axis=0) - matrix[:3, :3] @ source[best].mean(axis=0)
        result["diagnostics"]["pose_refinement"] = "calibrated_image_rotation_and_measured_centroid_translation"
        residual = rigid_residual(source, current, matrix)
        best = residual <= self.settings.residual_m
        result["diagnostics"].update(
            inliers=int(best.sum()),
            inlier_fraction=float(best.mean()),
            residual_p90_m=float(np.percentile(residual, 90)),
            residual_max_m=float(residual.max()),
        )
        # Do not explain away a locally bending part as RANSAC outliers. A
        # distinctive correspondence violating rigidity requires a pause, even
        # when most of a patch remains rigid. This also conservatively rejects
        # coherent mismatches; the observer cannot distinguish their cause.
        if np.any(residual > 2 * self.settings.residual_m):
            raise _ObservationRefusedError("deformation_or_inconsistent_correspondence")
        if best.sum() < self.settings.min_points or best.mean() < self.settings.min_inlier_fraction:
            raise _ObservationRefusedError("deformation_or_inconsistent_correspondence")
        ref_pixels = self.matcher.points[ids[best]].astype(np.float32)
        coverage = cv2.contourArea(cv2.convexHull(ref_pixels)) / (self.roi[2] * self.roi[3])
        result["diagnostics"]["reference_coverage"] = coverage
        if coverage < self.settings.min_coverage:
            raise _ObservationRefusedError("insufficient_spatial_support")
        return matrix, best, ref_pixels

    def _check_geometry(self, rgbd, matrix, result):
        try:
            appearance = self.material_support.evaluate(rgbd, matrix)
        except ValueError as error:
            raise _ObservationRefusedError('original_material_support_evaluation_failed') from error
        result['material_support'] = appearance
        if appearance['contradictory_count']:
            raise _ObservationRefusedError('original_material_appearance_contradiction')
        probes = self.reference["probes"] @ matrix[:3, :3].T + matrix[:3, 3]
        projected, visible = project_measured_rays(probes, rgbd["rays"])
        measured, supported = measured_xyz(rgbd, projected, 2 * self.settings.residual_m)
        supported &= visible
        probe_residual = np.linalg.norm(measured[supported] - probes[supported], axis=1)
        inconsistent = int(np.count_nonzero(probe_residual > 2 * self.settings.residual_m))
        result["diagnostics"]["geometry_probes"] = {
            "reference_count": len(probes),
            "supported_count": int(supported.sum()),
            "inconsistent_count": inconsistent,
            "residual_max_m": float(probe_residual.max()) if len(probe_residual) else None,
            "residual_p90_m": float(np.percentile(probe_residual, 90)) if len(probe_residual) else None,
        }
        if supported.sum() < max(32, 0.5 * len(probes)):
            raise _ObservationRefusedError("insufficient_original_geometry_support")
        if inconsistent >= 3:
            raise _ObservationRefusedError("deformation_or_inconsistent_geometry")
