#!/usr/bin/env python3
"""Vision-only multi-camera pose estimation for the follower end effector.

The wrist is one rigid target made from four AprilTag 16h5 faces.  Estimating
each planar tag independently throws that information away and re-introduces
the IPPE branch flips seen during field calibration.  This module instead
projects every known EE-frame corner into every calibrated Amcrest camera and
solves one ``world_from_ee`` transform with a robust loss.

Forward kinematics is deliberately absent.  Callers may compare this estimate
with FK after the solve, but FK is never an initializer, prior, or fallback;
the result remains an independent second measurement of the arm.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fiducials import (
    DEFAULT_INVENTORY_PATH,
    invert,
    load_inventory,
    matrix_from_pose,
    pose_vector,
    rotation_distance_deg,
    tag_model_corners,
    transform_from_vector,
    transform_points,
)
from fiducials.detector import Detection, DetectorConfig, FiducialDetector
from scipy.optimize import least_squares

WRIST_IDS = frozenset(load_inventory().target("wrist").ids)


@dataclasses.dataclass(frozen=True)
class CameraModel:
    name: str
    width: int
    height: int
    intrinsic: np.ndarray
    distortion: np.ndarray
    world_from_camera: np.ndarray

    @classmethod
    def from_bundle_entry(cls, name: str, entry: dict) -> "CameraModel":
        intr = entry["intrinsics"]
        intrinsic = np.array(
            [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pose = entry["world_from_camera"]
        return cls(
            name=name,
            width=int(intr["width"]),
            height=int(intr["height"]),
            intrinsic=intrinsic,
            distortion=np.asarray(entry.get("distortion", {}).get("coefficients", []), float),
            world_from_camera=matrix_from_pose(pose["rotation"], pose["translation_m"]),
        )

    def scaled(self, width: int, height: int, aspect_tolerance: float = 1e-3) -> "CameraModel":
        """Return intrinsics for an explicitly scaled, uncropped frame."""
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > aspect_tolerance:
            raise ValueError(
                f"{self.name}: {width}x{height} is not a uniform scale of "
                f"{self.width}x{self.height}; cropped frames need their own calibration"
            )
        intrinsic = self.intrinsic.copy()
        intrinsic[0, :] *= sx
        intrinsic[1, :] *= sy
        intrinsic[2, 2] = 1.0
        return dataclasses.replace(self, width=width, height=height, intrinsic=intrinsic)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pixels, self.intrinsic, self.distortion).reshape(-1, 2)

    def ray(self, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = self.undistort(np.asarray(pixel).reshape(1, 2))[0]
        direction = self.world_from_camera[:3, :3] @ np.array([xy[0], xy[1], 1.0])
        return self.world_from_camera[:3, 3], direction / np.linalg.norm(direction)

    def project(self, world_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_from_world = invert(self.world_from_camera)
        camera_points = transform_points(camera_from_world, world_points)
        projected, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic,
            self.distortion,
        )
        return projected.reshape(-1, 2), camera_points[:, 2]


@dataclasses.dataclass(frozen=True)
class WristLayout:
    edge_m: float
    ee_from_tag: dict[int, np.ndarray]
    layout_hash: str
    inventory_hash: str | None = None
    # The historical matrix/output names say "ee". This field makes their
    # concrete URDF meaning explicit; for the current hardware it is the
    # follower left jaw, not the flange-centered ee_gripper_link.
    parent_frame: str = "right/gripper_left"

    @classmethod
    def load(
        cls,
        path: str | Path,
        edge_m: float | None = None,
        inventory_path: str | Path = DEFAULT_INVENTORY_PATH,
    ) -> "WristLayout":
        path = Path(path).expanduser()
        raw = path.read_bytes()
        data = json.loads(raw)
        inventory = load_inventory(inventory_path)
        wrist = inventory.target("wrist")
        if data.get("schema_version") != 2:
            raise ValueError(
                f"{path}: unsupported wrist layout schema {data.get('schema_version')!r}"
            )
        if not wrist.parent_frame:
            raise ValueError(f"{inventory.source}: wrist target has no parent_frame")
        if data.get("parent_frame") != wrist.parent_frame:
            raise ValueError(
                f"{path}: wrist layout parent must be {wrist.parent_frame}"
            )
        expected_ids = frozenset(wrist.ids)
        declared_ids = tuple(int(tag_id) for tag_id in data.get("target_ids", []))
        if declared_ids != wrist.ids:
            raise ValueError(
                f"{path}: wrist target ids must be {list(wrist.ids)}, got {list(declared_ids)}; "
                "run a new calibration for the configured target"
            )
        recorded_inventory_hash = data.get("inventory_hash")
        if recorded_inventory_hash != inventory.inventory_hash:
            raise ValueError(
                f"{path}: inventory hash does not match {inventory.source}; regenerate the wrist layout"
            )
        transforms = {}
        for key, entry in data["tags"].items():
            tag_id = int(key)
            if tag_id not in expected_ids:
                continue
            if "ee_from_tag" not in entry:
                raise ValueError(f"tag {tag_id} has no ee_from_tag transform")
            transform = np.asarray(entry["ee_from_tag"], dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError(f"tag {tag_id} ee_from_tag must be a finite 4x4 matrix")
            if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
                raise ValueError(f"tag {tag_id} ee_from_tag has an invalid homogeneous row")
            rotation = transform[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
                np.linalg.det(rotation), 1.0, abs_tol=1e-5
            ):
                raise ValueError(f"tag {tag_id} ee_from_tag rotation is not rigid")
            transforms[tag_id] = transform
        missing = expected_ids - transforms.keys()
        if missing:
            raise ValueError(f"wrist layout is missing tag ids {sorted(missing)}")
        status = data.get("calibration_status")
        if status != "calibrated":
            raise ValueError(
                f"{path}: wrist layout is {status or 'unversioned'}, not calibrated; "
                "run the configured wrist calibration and export it before tracking"
            )
        configured_edge_m = float(data.get("edge_m", wrist.edge_m))
        if edge_m is not None and not math.isclose(float(edge_m), configured_edge_m, abs_tol=1e-9):
            raise ValueError(
                f"{path}: requested edge {float(edge_m):.6f} differs from layout {configured_edge_m:.6f}"
            )
        if not math.isclose(configured_edge_m, wrist.edge_m, abs_tol=1e-9):
            raise ValueError(
                f"{path}: edge {configured_edge_m:.6f} differs from inventory {wrist.edge_m:.6f}"
            )
        return cls(
            edge_m=configured_edge_m,
            ee_from_tag=transforms,
            layout_hash=hashlib.sha256(raw).hexdigest(),
            inventory_hash=inventory.inventory_hash,
            parent_frame=wrist.parent_frame,
        )

    def corners_ee(self, tag_id: int) -> np.ndarray:
        return transform_points(self.ee_from_tag[tag_id], tag_model_corners(self.edge_m))


def load_calibration(path: str | Path) -> tuple[dict[str, CameraModel], dict]:
    bundle = json.loads(Path(path).expanduser().read_text())
    cameras = {name: CameraModel.from_bundle_entry(name, entry) for name, entry in bundle["cameras"].items()}
    return cameras, bundle


class WristTagDetector(FiducialDetector):
    """Compatibility wrapper that constrains the shared detector to one layout."""

    def __init__(self, layout: WristLayout, config: DetectorConfig | None = None):
        self.layout = layout
        super().__init__(set(layout.ee_from_tag), config=config, keep_best_per_id=False)


@dataclasses.dataclass(frozen=True)
class EstimatorConfig:
    huber_px: float = 2.0
    max_source_rmse_px: float = 6.0
    # The accepted 2026-08-22 wrist sweep contains two accurate FK-consistent
    # poses at 4.07/4.16 px. 4.5 keeps those while rejecting the sparse 5.6+
    # px cross-view inconsistencies.
    max_total_rmse_px: float = 4.5
    min_corners: int = 8
    max_condition: float = 2e4
    max_translation_sigma_mm: float = 3.0
    max_rotation_sigma_deg: float = 1.5
    min_initial_tags: int = 2
    single_tag_reacquire_translation_m: float = 0.08
    single_tag_reacquire_rotation_deg: float = 30.0
    motion_compensation: bool = True
    max_motion_window_ms: float = 50.0
    # Weak zero-motion regularization.  These are deliberately wider than the
    # robot's normal motion envelope: timestamps and corners should determine
    # twist whenever they can; the prior only stabilizes a marginal window.
    velocity_prior_m_s: float = 5.0
    angular_velocity_prior_rad_s: float = 20.0


@dataclasses.dataclass
class PoseEstimate:
    status: str
    timestamp_ns: int
    world_from_ee: np.ndarray | None
    reprojection_rmse_px: float | None
    used_cameras: list[str]
    used_tags: list[int]
    rejected_sources: list[str]
    corner_count: int
    condition: float | None
    translation_sigma_mm: float | None
    rotation_sigma_deg: float | None
    reason: str | None = None
    twist: np.ndarray | None = None
    source_rmse_px: dict[str, float] = dataclasses.field(default_factory=dict)

    def as_dict(
        self,
        *,
        calibration_id: str,
        layout_hash: str,
        inventory_hash: str | None = None,
        tracking_frame: str | None = None,
        base_from_world: np.ndarray | None = None,
        sequence: int | None = None,
        maximum_skew_ns: int | None = None,
        latency_ms: float | None = None,
    ) -> dict:
        out = {
            "schema_version": 1,
            "sequence": sequence,
            "timestamp_ns": self.timestamp_ns,
            "status": self.status,
            "world_from_ee": self.world_from_ee.tolist() if self.world_from_ee is not None else None,
            "base_from_ee": (
                (base_from_world @ self.world_from_ee).tolist()
                if base_from_world is not None and self.world_from_ee is not None
                else None
            ),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "used_cameras": self.used_cameras,
            "used_tags": self.used_tags,
            "rejected_sources": self.rejected_sources,
            "corner_count": self.corner_count,
            "condition": self.condition,
            "translation_sigma_mm": self.translation_sigma_mm,
            "rotation_sigma_deg": self.rotation_sigma_deg,
            "reason": self.reason,
            "twist": self.twist.tolist() if self.twist is not None else None,
            "source_rmse_px": self.source_rmse_px,
            "maximum_skew_ns": maximum_skew_ns,
            "latency_ms": latency_ms,
            "calibration_id": calibration_id,
            "wrist_layout_hash": layout_hash,
            "inventory_hash": inventory_hash,
            "tracking_frame": tracking_frame,
        }
        return out


def _triangulate(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        matrix += projector
        rhs += projector @ origin
    if np.linalg.cond(matrix) > 1e6:
        return None
    return np.linalg.solve(matrix, rhs)


def _fit_rigid(model: np.ndarray, measured: np.ndarray) -> np.ndarray:
    model_center = model.mean(axis=0)
    measured_center = measured.mean(axis=0)
    u, _, vt = np.linalg.svd((model - model_center).T @ (measured - measured_center))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = measured_center - rotation @ model_center
    return out


class MultiCameraEstimator:
    def __init__(
        self,
        cameras: dict[str, CameraModel],
        layout: WristLayout,
        config: EstimatorConfig | None = None,
    ):
        self.cameras = cameras
        self.layout = layout
        self.config = config or EstimatorConfig()

    def _observable(self, detections: list[Detection]) -> bool:
        cameras = {item.camera for item in detections}
        if len(cameras) >= 2:
            return True
        if len(detections) < 2:
            return False
        points = np.concatenate([self.layout.corners_ee(item.tag_id) for item in detections])
        return np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-5) == 3

    def _bootstrap_multiview(self, detections: list[Detection]) -> np.ndarray | None:
        by_tag = defaultdict(list)
        for item in detections:
            by_tag[item.tag_id].append(item)
        candidates = []
        for tag_id, observations in by_tag.items():
            if len({item.camera for item in observations}) < 2:
                continue
            measured = []
            for corner_index in range(4):
                rays = [
                    self.cameras[item.camera].ray(item.corners_px[corner_index])
                    for item in observations
                    if item.camera in self.cameras
                ]
                point = _triangulate(rays)
                if point is None:
                    measured = []
                    break
                measured.append(point)
            if measured:
                candidates.append(
                    (len(observations), _fit_rigid(self.layout.corners_ee(tag_id), np.asarray(measured)))
                )
        return max(candidates, key=lambda pair: pair[0])[1] if candidates else None

    def _bootstrap_single_camera(self, detections: list[Detection]) -> np.ndarray | None:
        by_camera = defaultdict(list)
        for item in detections:
            by_camera[item.camera].append(item)
        for camera_name, observations in sorted(by_camera.items(), key=lambda pair: -len(pair[1])):
            if len(observations) < 2 or camera_name not in self.cameras:
                continue
            model = np.concatenate([self.layout.corners_ee(item.tag_id) for item in observations])
            if np.linalg.matrix_rank(model - model.mean(axis=0), tol=1e-5) < 3:
                continue
            pixels = np.concatenate([item.corners_px for item in observations])
            camera = self.cameras[camera_name]
            ok, rvec, tvec = cv2.solvePnP(
                model,
                pixels,
                camera.intrinsic,
                camera.distortion,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if ok:
                camera_from_ee = matrix_from_pose(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
                return camera.world_from_camera @ camera_from_ee
        return None

    def _planar_hypotheses(self, detections: list[Detection]) -> list[np.ndarray]:
        """Disambiguate planar IPPE branches by scoring all rigid-target views.

        A lone tag is rejected before bootstrap.  With multiple cameras/tags,
        however, each planar solution is a useful hypothesis: the wrong branch
        cannot reproject the other rigidly attached faces in their calibrated
        views.  This handles sparse frames where no two cameras see the same
        tag and no one camera sees two faces.
        """
        tag_model = tag_model_corners(self.layout.edge_m)
        candidates = []
        for item in detections:
            camera = self.cameras[item.camera]
            result = cv2.solvePnPGeneric(
                tag_model,
                item.corners_px,
                camera.intrinsic,
                camera.distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not result[0]:
                continue
            for rvec, tvec in zip(result[1], result[2], strict=True):
                camera_from_tag = matrix_from_pose(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
                candidates.append(
                    camera.world_from_camera @ camera_from_tag @ invert(self.layout.ee_from_tag[item.tag_id])
                )
        return candidates

    def _bootstrap_planar_candidates(self, detections: list[Detection]) -> np.ndarray | None:
        candidates = self._planar_hypotheses(detections)
        if not candidates:
            return None
        timestamp_ns = int(np.median([item.timestamp_ns for item in detections]))
        return min(
            candidates,
            key=lambda pose: np.sqrt(
                np.mean(self._residuals(pose_vector(pose), detections, timestamp_ns, False) ** 2)
            ),
        )

    def _bootstrap(self, detections: list[Detection]) -> np.ndarray | None:
        multiview = self._bootstrap_multiview(detections)
        if multiview is not None:
            return multiview
        single_camera = self._bootstrap_single_camera(detections)
        return single_camera if single_camera is not None else self._bootstrap_planar_candidates(detections)

    def _consensus(
        self,
        detections: list[Detection],
        timestamp_ns: int,
        tracked_pose: np.ndarray | None,
    ) -> tuple[np.ndarray, list[Detection], set[tuple[str, int]]] | None:
        """Find a rigid-target subset before nonlinear optimization.

        Wrist tag IDs also occur on loose calibration material in the live
        workspace. A global robust loss alone still spends hundreds of solver
        evaluations compromising between those physically unrelated targets.
        Pose hypotheses are cheap; score each against every source, require a
        multi-ID rigid consensus for acquisition, and allow one tag only near
        a recent vision-only track.
        """
        candidates = []
        if tracked_pose is not None:
            candidates.append(tracked_pose)
        multiview = self._bootstrap_multiview(detections)
        if multiview is not None:
            candidates.append(multiview)
        single_camera = self._bootstrap_single_camera(detections)
        if single_camera is not None:
            candidates.append(single_camera)
        candidates.extend(self._planar_hypotheses(detections))

        best = None
        for candidate in candidates:
            errors = self._source_errors(pose_vector(candidate), detections, timestamp_ns, False)
            kept = [
                item
                for item in detections
                if errors[(item.camera, item.tag_id)] <= self.config.max_source_rmse_px
            ]
            if not self._observable(kept) or len(kept) * 4 < self.config.min_corners:
                continue
            tags = {item.tag_id for item in kept}
            if tracked_pose is None and len(tags) < self.config.min_initial_tags:
                continue
            if len(tags) < self.config.min_initial_tags:
                translation = np.linalg.norm((invert(tracked_pose) @ candidate)[:3, 3])
                rotation = rotation_distance_deg(tracked_pose, candidate)
                if (
                    translation > self.config.single_tag_reacquire_translation_m
                    or rotation > self.config.single_tag_reacquire_rotation_deg
                ):
                    continue
            score = (
                len(tags),
                len(kept),
                -float(np.mean([errors[(item.camera, item.tag_id)] for item in kept])),
            )
            if best is None or score > best[0]:
                kept_sources = {(item.camera, item.tag_id) for item in kept}
                removed = {
                    (item.camera, item.tag_id)
                    for item in detections
                    if (item.camera, item.tag_id) not in kept_sources
                }
                best = score, candidate, kept, removed
        return None if best is None else best[1:]

    def _residuals(
        self,
        parameters: np.ndarray,
        detections: list[Detection],
        timestamp_ns: int,
        use_motion: bool,
    ) -> np.ndarray:
        reference = transform_from_vector(parameters[:6])
        twist = parameters[6:12] if use_motion else np.zeros(6)
        residuals = []
        for item in detections:
            dt = (item.timestamp_ns - timestamp_ns) / 1e9
            if use_motion:
                delta = transform_from_vector(np.r_[twist[:3] * dt, twist[3:] * dt])
                world_from_ee = delta @ reference
            else:
                world_from_ee = reference
            world_points = transform_points(world_from_ee, self.layout.corners_ee(item.tag_id))
            pixels, depths = self.cameras[item.camera].project(world_points)
            error = (pixels - item.corners_px).reshape(-1)
            if np.any(depths <= 1e-5):
                error += np.sign(error + 1e-9) * 1e4
            residuals.extend(error)
        if use_motion:
            residuals.extend(parameters[6:9] / self.config.angular_velocity_prior_rad_s)
            residuals.extend(parameters[9:12] / self.config.velocity_prior_m_s)
        return np.asarray(residuals, dtype=np.float64)

    def _source_errors(
        self, parameters: np.ndarray, detections: list[Detection], timestamp_ns: int, use_motion: bool
    ) -> dict[tuple[str, int], float]:
        errors = {}
        for item in detections:
            values = self._residuals(parameters, [item], timestamp_ns, use_motion)[:8]
            errors[(item.camera, item.tag_id)] = float(np.sqrt(np.mean(values**2)))
        return errors

    def estimate(
        self,
        detections: list[Detection],
        timestamp_ns: int,
        initial_world_from_ee: np.ndarray | None = None,
        initial_twist: np.ndarray | None = None,
    ) -> PoseEstimate:
        detections = [item for item in detections if item.camera in self.cameras]
        sources = [(item.camera, item.tag_id) for item in detections]
        duplicate_sources = sorted({source for source in sources if sources.count(source) > 1})
        if duplicate_sources:
            detail = ", ".join(f"{camera}:tag{tag_id}" for camera, tag_id in duplicate_sources)
            return PoseEstimate(
                "rejected",
                timestamp_ns,
                None,
                None,
                [],
                [],
                [],
                0,
                None,
                None,
                None,
                f"ambiguous duplicate wrist IDs in one camera: {detail}",
            )
        if not self._observable(detections):
            return PoseEstimate(
                "rejected",
                timestamp_ns,
                None,
                None,
                [],
                [],
                [],
                0,
                None,
                None,
                None,
                "need two cameras or two non-coplanar tag faces",
            )
        consensus = self._consensus(detections, timestamp_ns, initial_world_from_ee)
        if consensus is None:
            return PoseEstimate(
                "rejected",
                timestamp_ns,
                None,
                None,
                [],
                [],
                [],
                0,
                None,
                None,
                None,
                "no observable rigid multi-tag consensus",
            )
        initial, detections, consensus_rejected = consensus
        span_ms = (
            max(item.timestamp_ns for item in detections) - min(item.timestamp_ns for item in detections)
        ) / 1e6
        use_motion = self.config.motion_compensation and 0.5 < span_ms <= self.config.max_motion_window_ms
        x0 = pose_vector(initial)
        if use_motion:
            x0 = np.r_[x0, np.zeros(6) if initial_twist is None else initial_twist]
        result = least_squares(
            self._residuals,
            x0,
            args=(detections, timestamp_ns, use_motion),
            method="trf",
            loss="huber",
            f_scale=self.config.huber_px,
            max_nfev=300,
        )
        source_errors = self._source_errors(result.x, detections, timestamp_ns, use_motion)
        rejected = {
            source for source, error in source_errors.items() if error > self.config.max_source_rmse_px
        }
        kept = [item for item in detections if (item.camera, item.tag_id) not in rejected]
        culled = set(consensus_rejected)
        if rejected and self._observable(kept) and len(kept) * 4 >= self.config.min_corners:
            result = least_squares(
                self._residuals,
                result.x,
                args=(kept, timestamp_ns, use_motion),
                method="trf",
                loss="huber",
                f_scale=self.config.huber_px,
                max_nfev=300,
            )
            detections = kept
            culled.update(rejected)
        residual = self._residuals(result.x, detections, timestamp_ns, use_motion)
        final_source_errors = self._source_errors(result.x, detections, timestamp_ns, use_motion)
        pixel_residual = residual[: len(detections) * 8]
        rmse = float(np.sqrt(np.mean(pixel_residual**2)))
        normal = result.jac.T @ result.jac
        # Twist can be only weakly observed over a 10--40 ms shutter window;
        # its units and regularization should not make an otherwise strong
        # instantaneous pose fail the geometry gate. The full inverse below
        # still marginalizes that coupling into pose uncertainty.
        condition = float(np.linalg.cond(normal[:6, :6]))
        translation_sigma_mm = rotation_sigma_deg = None
        try:
            degrees_freedom = max(1, len(pixel_residual) - len(result.x))
            variance = float(pixel_residual @ pixel_residual / degrees_freedom)
            covariance = np.linalg.pinv(normal) * variance
            rotation_sigma_deg = float(np.degrees(np.sqrt(np.max(np.diag(covariance)[:3]))))
            translation_sigma_mm = float(1000 * np.sqrt(np.max(np.diag(covariance)[3:6])))
        except np.linalg.LinAlgError:
            condition = float("inf")
        reason = None
        if not result.success:
            reason = f"optimizer failed: {result.message}"
        elif len(detections) * 4 < self.config.min_corners:
            reason = f"only {len(detections) * 4} corners remain"
        elif rmse > self.config.max_total_rmse_px:
            reason = f"reprojection RMSE {rmse:.2f}px exceeds {self.config.max_total_rmse_px:.2f}px"
        elif not math.isfinite(condition) or condition > self.config.max_condition:
            reason = f"normal matrix condition {condition:.3g} is too weak"
        elif translation_sigma_mm is None or translation_sigma_mm > self.config.max_translation_sigma_mm:
            reason = (
                f"translation uncertainty {translation_sigma_mm} mm exceeds "
                f"{self.config.max_translation_sigma_mm:.2f} mm"
            )
        elif rotation_sigma_deg is None or rotation_sigma_deg > self.config.max_rotation_sigma_deg:
            reason = (
                f"rotation uncertainty {rotation_sigma_deg} deg exceeds "
                f"{self.config.max_rotation_sigma_deg:.2f} deg"
            )
        status = "measured" if reason is None else "rejected"
        pose = transform_from_vector(result.x[:6]) if status == "measured" else None
        return PoseEstimate(
            status=status,
            timestamp_ns=timestamp_ns,
            world_from_ee=pose,
            reprojection_rmse_px=rmse,
            used_cameras=sorted({item.camera for item in detections}),
            used_tags=sorted({item.tag_id for item in detections}),
            rejected_sources=[f"{camera}:tag{tag}" for camera, tag in sorted(culled)],
            corner_count=len(detections) * 4,
            condition=condition,
            translation_sigma_mm=translation_sigma_mm,
            rotation_sigma_deg=rotation_sigma_deg,
            reason=reason,
            twist=result.x[6:12] if use_motion else None,
            source_rmse_px={
                f"{camera}:tag{tag}": error for (camera, tag), error in sorted(final_source_errors.items())
            },
        )


class VisionOnlyTracker:
    """Stateful initializer and explicit measured/predicted/lost statuses."""

    def __init__(self, estimator: MultiCameraEstimator, prediction_timeout_s: float = 0.5):
        self.estimator = estimator
        self.prediction_timeout_ns = int(prediction_timeout_s * 1e9)
        self.last_measurement: PoseEstimate | None = None

    def update(self, detections: list[Detection], timestamp_ns: int) -> PoseEstimate:
        initial = self.last_measurement.world_from_ee if self.last_measurement else None
        twist = self.last_measurement.twist if self.last_measurement else None
        result = self.estimator.estimate(detections, timestamp_ns, initial, twist)
        if result.status == "measured":
            self.last_measurement = result
            return result
        if (
            self.last_measurement
            and timestamp_ns - self.last_measurement.timestamp_ns <= self.prediction_timeout_ns
        ):
            dt = (timestamp_ns - self.last_measurement.timestamp_ns) / 1e9
            predicted = self.last_measurement.world_from_ee.copy()
            if self.last_measurement.twist is not None:
                velocity = self.last_measurement.twist
                predicted = transform_from_vector(np.r_[velocity[:3] * dt, velocity[3:] * dt]) @ predicted
            result.status = "predicted"
            result.world_from_ee = predicted
            result.reason = f"vision prediction after rejected measurement: {result.reason}"
            return result
        result.status = "lost"
        return result
