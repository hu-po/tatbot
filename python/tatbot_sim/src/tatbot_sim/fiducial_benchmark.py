"""Deterministic five-Amcrest sim-to-real benchmark for EE fiducials.

This is deliberately separate from the high-throughput stroke data factory:
it renders one robot with the real calibrated camera rig, runs the same Python
detector/estimator as live shadow mode, and writes per-frame ground-truth
errors plus a compact report.

Example (on an x86_64 sim host):
  python -m tatbot_sim.fiducial_benchmark \
    --calibration ~/tatbot-logs/vision/calibration-current.json \
    --robot-world ~/tatbot-logs/vision/robot-world-current.json \
    --out-dir ~/tatbot-logs/vision/ee-tracking/sim-dev --split dev
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

import tatbot_sim  # noqa: F401  (registers the robot and environment)
from tatbot_sim.agent import TatbotWXAI
from tatbot_sim.config import DRConfig
from tatbot_sim.repo import repo_root
from tatbot_sim.urdf import rig_from_follower_base

REPO = repo_root()
sys.path.insert(0, str(REPO / "scripts/vision"))
from ee_fiducial import (  # noqa: E402
    DetectorConfig,
    EstimatorConfig,
    MultiCameraEstimator,
    VisionOnlyTracker,
    WristLayout,
    WristTagDetector,
    invert,
    load_calibration,
    rotation_distance_deg,
    transform_from_vector,
)
from fiducials import load_inventory  # noqa: E402

INVENTORY = load_inventory()
LIVE_DETECTOR_PROFILE = INVENTORY.detector_profiles["live"]


@dataclasses.dataclass
class Args:
    calibration: str
    robot_world: str
    out_dir: str
    wrist_layout: str = str(REPO / "config/wrist_tags_measured.json")
    pose_bank: str = str(REPO / "config/fiducial_benchmark_poses.json")
    """Observed follower poses that define the camera/tag visibility envelope."""
    split: str = "dev"
    """One of clean/dev/tune/holdout; each owns a fixed, disjoint seed."""
    num_frames: int = 120
    camera_scale: float = 0.5
    detector_scale: float = LIVE_DETECTOR_PROFILE.scale
    adaptive_window_max: int = LIVE_DETECTOR_PROFILE.adaptive_window_max
    min_side_px: float = LIVE_DETECTOR_PROFILE.min_side_px
    corner_refinement: bool = LIVE_DETECTOR_PROFILE.corner_refinement
    huber_px: float = 2.0
    max_source_rmse_px: float = 6.0
    max_total_rmse_px: float = 4.5
    max_condition: float = 2e4
    max_translation_sigma_mm: float = 3.0
    max_rotation_sigma_deg: float = 1.5
    motion_compensation: bool = True
    max_motion_window_ms: float = 50.0
    save_failure_frames: int = 20
    """Maximum failed frames to save as five annotated JPEGs; zero disables."""
    sim_backend: str = "auto"
    enforce_gates: bool = False


SPLIT_SEEDS = {"clean": 1009, "dev": 2017, "tune": 3011, "holdout": 4001}
POSE_JOINT_NAMES = tuple(f"right/joint_{index}" for index in range(6)) + (
    "right/left_carriage_joint",
)


def _load_pose_bank(path: str | Path) -> tuple[np.ndarray, dict, str]:
    source = Path(path).expanduser()
    raw = source.read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError(f"{source}: unsupported pose-bank schema {data.get('schema_version')!r}")
    if tuple(data.get("joint_names", ())) != POSE_JOINT_NAMES:
        raise ValueError(f"{source}: pose-bank joints must be {list(POSE_JOINT_NAMES)}")
    poses = np.asarray(data.get("poses", ()), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 3 or poses.shape[1] != len(POSE_JOINT_NAMES):
        raise ValueError(f"{source}: pose bank must contain at least three 7-joint poses")
    if not np.isfinite(poses).all():
        raise ValueError(f"{source}: pose bank contains non-finite values")
    return poses, data, hashlib.sha256(raw).hexdigest()


def _pose_path(
    poses: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Callable[[float], np.ndarray]]:
    """Return a split-specific smooth path through observed robot poses."""
    order = rng.permutation(len(poses))
    ordered = poses[order]

    def sample(phase: float) -> np.ndarray:
        progress = float(np.clip(phase, 0.0, 1.0 - np.finfo(float).eps)) * len(ordered)
        segment = min(int(progress), len(ordered) - 1)
        alpha = progress - segment
        # Smoothstep keeps velocities zero at the observed configurations.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - alpha) * ordered[segment] + alpha * ordered[(segment + 1) % len(ordered)]

    return order, sample


def _git_identity() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"source_commit": commit, "source_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": None, "source_dirty": None}


def _benchmark_dr() -> DRConfig:
    """Fixed renderer scene; stochastic sensor effects live in the manifest.

    The training environment intentionally redraws unseeded build-time assets.
    That is useful for data generation but would make identical benchmark
    splits incomparable.  Keep geometry/lighting fixed here and apply every
    benchmark nuisance explicitly through the split RNG below.
    """
    config = DRConfig()
    config.lighting.enabled = False
    config.background.enabled = False
    config.background.table_half_x = (0.35, 0.35)
    config.background.table_half_y = (0.45, 0.45)
    config.background.table_half_z = (0.015, 0.015)
    config.clutter.enabled = False
    config.sheet.enabled = False
    config.camera.mount_jitter_mm = 0.0
    config.camera.mount_jitter_deg = 0.0
    config.pad.xy_range = 0.0
    config.pad.yaw_range = 0.0
    config.pad.tilt_range = 0.0
    config.pad.z_range = (0.02, 0.02)
    return config


def _pose_matrix(raw_pose) -> np.ndarray:
    values = raw_pose.detach().cpu().numpy()[0]
    out = np.eye(4)
    out[:3, 3] = values[:3]
    # SAPIEN/ManiSkill raw poses are p + wxyz; scipy accepts xyzw.
    out[:3, :3] = Rotation.from_quat([values[4], values[5], values[6], values[3]]).as_matrix()
    return out


def _distortion_maps(camera):
    """Map each distorted output pixel back into the pinhole render."""
    xs, ys = np.meshgrid(np.arange(camera.width), np.arange(camera.height))
    distorted = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(distorted, camera.intrinsic, camera.distortion).reshape(-1, 2)
    map_x = (normalized[:, 0] * camera.intrinsic[0, 0] + camera.intrinsic[0, 2]).reshape(
        camera.height, camera.width
    )
    map_y = (normalized[:, 1] * camera.intrinsic[1, 1] + camera.intrinsic[1, 2]).reshape(
        camera.height, camera.width
    )
    return map_x.astype(np.float32), map_y.astype(np.float32)


def _motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    if length <= 1:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(float(kernel.sum()), 1e-9)
    return cv2.filter2D(image, -1, kernel)


def _corrupt(image: np.ndarray, rng: np.random.Generator, split: str) -> tuple[np.ndarray, dict]:
    if split == "clean":
        return image, {"gain": 1.0, "noise_sigma": 0.0, "blur_px": 0, "jpeg_quality": 100}
    gain = float(rng.uniform(0.65, 1.35 if split != "holdout" else 1.45))
    noise_sigma = float(rng.uniform(0.0, 5.0 if split != "holdout" else 7.0))
    blur_px = int(rng.choice([0, 0, 3, 5, 7 if split == "holdout" else 5]))
    jpeg_quality = int(rng.integers(55 if split == "holdout" else 65, 96))
    output = np.clip(image.astype(np.float32) * gain, 0, 255)
    if noise_sigma:
        output += rng.normal(0.0, noise_sigma, output.shape)
    output = np.clip(output, 0, 255).astype(np.uint8)
    if blur_px:
        output = _motion_blur(output, blur_px, float(rng.uniform(0, 180)))
    if rng.random() < (0.18 if split == "holdout" else 0.08):
        height, width = output.shape[:2]
        x = int(rng.uniform(0.15, 0.8) * width)
        y = int(rng.uniform(0.15, 0.8) * height)
        w = int(rng.uniform(0.03, 0.12) * width)
        h = int(rng.uniform(0.03, 0.12) * height)
        cv2.rectangle(output, (x, y), (min(width, x + w), min(height, y + h)), (20, 20, 20), -1)
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("OpenCV JPEG encoding failed")
    output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return output, {
        "gain": gain,
        "noise_sigma": noise_sigma,
        "blur_px": blur_px,
        "jpeg_quality": jpeg_quality,
    }


def _perturb_models(cameras, layout, rng, split):
    """Sample calibration/layout error for the estimator, not the renderer."""
    if split == "clean":
        return cameras, layout, {"camera": {}, "tag": {}}
    holdout = split == "holdout"
    camera_pos_sigma = 0.003 if holdout else 0.002
    camera_rot_sigma = math.radians(0.6 if holdout else 0.35)
    intrinsic_sigma = 0.005 if holdout else 0.003
    tag_pos_sigma = 0.004 if holdout else 0.0025
    tag_rot_sigma = math.radians(1.0 if holdout else 0.6)
    perturbed_cameras, perturbed_tags = {}, {}
    report = {"camera": {}, "tag": {}}
    for name, camera in cameras.items():
        delta = np.r_[
            rng.normal(0, camera_rot_sigma, 3),
            rng.normal(0, camera_pos_sigma, 3),
        ]
        intrinsic = camera.intrinsic.copy()
        focal_scale = rng.normal(1.0, intrinsic_sigma, 2)
        intrinsic[0, 0] *= focal_scale[0]
        intrinsic[1, 1] *= focal_scale[1]
        perturbed_cameras[name] = dataclasses.replace(
            camera,
            intrinsic=intrinsic,
            world_from_camera=transform_from_vector(delta) @ camera.world_from_camera,
        )
        report["camera"][name] = {
            "delta_rot_deg": np.degrees(delta[:3]).tolist(),
            "delta_pos_mm": (1000 * delta[3:]).tolist(),
            "focal_scale": focal_scale.tolist(),
        }
    for tag_id, transform in layout.ee_from_tag.items():
        delta = np.r_[rng.normal(0, tag_rot_sigma, 3), rng.normal(0, tag_pos_sigma, 3)]
        perturbed_tags[tag_id] = transform_from_vector(delta) @ transform
        report["tag"][str(tag_id)] = {
            "delta_rot_deg": np.degrees(delta[:3]).tolist(),
            "delta_pos_mm": (1000 * delta[3:]).tolist(),
        }
    return (
        perturbed_cameras,
        WristLayout(
            layout.edge_m,
            perturbed_tags,
            layout.layout_hash + f":sim-{split}",
            layout.inventory_hash,
            layout.parent_frame,
        ),
        report,
    )


def _percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def _summarize(records: list[dict], elapsed_s: float) -> dict:
    measured = [item for item in records if item["status"] == "measured"]
    translations = [item["translation_error_mm"] for item in measured]
    rotations = [item["rotation_error_deg"] for item in measured]
    false_accepted = [
        item for item in measured if item["translation_error_mm"] > 10.0 or item["rotation_error_deg"] > 2.0
    ]
    return {
        "frames": len(records),
        "measured": len(measured),
        "valid_rate": len(measured) / max(1, len(records)),
        "false_accepted_rate": len(false_accepted) / max(1, len(records)),
        "translation_error_mm": {
            "median": _percentile(translations, 50),
            "p95": _percentile(translations, 95),
            "max": max(translations) if translations else None,
        },
        "rotation_error_deg": {
            "median": _percentile(rotations, 50),
            "p95": _percentile(rotations, 95),
            "max": max(rotations) if rotations else None,
        },
        "throughput_fps": len(records) / max(elapsed_s, 1e-9),
    }


def _passes_gates(report: dict, split: str) -> tuple[bool, list[str]]:
    failures = []
    clean = split == "clean"
    min_valid = 0.99 if clean else 0.90
    max_translation = 2.0 if clean else 10.0
    max_rotation = 0.5 if clean else 2.0
    if report["valid_rate"] < min_valid:
        failures.append(f"valid rate {report['valid_rate']:.3f} < {min_valid:.3f}")
    if (
        report["translation_error_mm"]["p95"] is None
        or report["translation_error_mm"]["p95"] > max_translation
    ):
        failures.append(f"translation p95 exceeds {max_translation:.1f} mm")
    if report["rotation_error_deg"]["p95"] is None or report["rotation_error_deg"]["p95"] > max_rotation:
        failures.append(f"rotation p95 exceeds {max_rotation:.1f} deg")
    if report["false_accepted_rate"] >= 0.001:
        failures.append(f"false accepted rate {report['false_accepted_rate']:.4f} >= 0.001")
    return not failures, failures


def _annotate(image: np.ndarray, detections, status: str) -> np.ndarray:
    output = image.copy()
    for item in detections:
        corners = item.corners_px.astype(np.int32).reshape(1, 4, 2)
        cv2.polylines(output, corners, True, (0, 255, 0), 2)
        center = tuple(item.corners_px.mean(axis=0).astype(int))
        cv2.putText(output, f"id{item.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)
    return output


def main(args: Args):
    if args.split not in SPLIT_SEEDS:
        raise ValueError(f"unknown split {args.split}; choose {sorted(SPLIT_SEEDS)}")
    if args.num_frames <= 0 or args.save_failure_frames < 0:
        raise ValueError("num-frames must be positive and save-failure-frames non-negative")
    if not 0 < args.camera_scale <= 1 or not 0 < args.detector_scale <= 1:
        raise ValueError("camera-scale and detector-scale must be in (0, 1]")
    positive_tuning = (
        args.min_side_px,
        args.huber_px,
        args.max_source_rmse_px,
        args.max_total_rmse_px,
        args.max_condition,
        args.max_translation_sigma_mm,
        args.max_rotation_sigma_deg,
        args.max_motion_window_ms,
    )
    if args.adaptive_window_max < 3 or not all(
        math.isfinite(value) and value > 0 for value in positive_tuning
    ):
        raise ValueError("detector/estimator tuning values must be finite and positive")
    output = Path(args.out_dir).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark directory {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(SPLIT_SEEDS[args.split])
    calibrated, bundle = load_calibration(args.calibration)
    true_cameras = {
        name: camera.scaled(round(camera.width * args.camera_scale), round(camera.height * args.camera_scale))
        for name, camera in calibrated.items()
        if name.startswith("camera")
    }
    true_layout = WristLayout.load(args.wrist_layout, inventory_path=INVENTORY.source)
    pose_bank, pose_bank_metadata, pose_bank_hash = _load_pose_bank(args.pose_bank)
    cameras, layout, model_perturbations = _perturb_models(true_cameras, true_layout, rng, args.split)
    robot_world = json.loads(Path(args.robot_world).expanduser().read_text())
    world_from_rig = np.asarray(robot_world["world_from_base"], dtype=np.float64)
    world_from_base = world_from_rig @ rig_from_follower_base()
    distortion_maps = {name: _distortion_maps(camera) for name, camera in true_cameras.items()}

    env = gym.make(
        "TatbotDraw-v0",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sim_backend=args.sim_backend,
        num_textures=1,
        dr=_benchmark_dr(),
        fiducial_calibration=args.calibration,
        fiducial_robot_world=args.robot_world,
        fiducial_camera_scale=args.camera_scale,
    )
    base_env = env.unwrapped
    env.reset(seed=SPLIT_SEEDS[args.split])
    robot = base_env.agent.robot
    active_names = [joint.name for joint in robot.active_joints]
    controlled = [active_names.index(name) for name in TatbotWXAI.joint_names]
    q_limits = robot.get_qlimits().detach().cpu().numpy()
    if q_limits.ndim == 3:
        q_limits = q_limits[0]
    low = q_limits[controlled, 0] + 1e-4
    high = q_limits[controlled, 1] - 1e-4
    if np.any(pose_bank < low) or np.any(pose_bank > high):
        raise ValueError(f"{args.pose_bank}: pose bank exceeds simulator joint limits")
    pose_order, sample_pose = _pose_path(pose_bank, rng)
    detector_config = DetectorConfig(
        scale=args.detector_scale,
        adaptive_window_max=args.adaptive_window_max,
        min_side_px=args.min_side_px,
        corner_refinement=args.corner_refinement,
    )
    estimator_config = EstimatorConfig(
        huber_px=args.huber_px,
        max_source_rmse_px=args.max_source_rmse_px,
        max_total_rmse_px=args.max_total_rmse_px,
        max_condition=args.max_condition,
        max_translation_sigma_mm=args.max_translation_sigma_mm,
        max_rotation_sigma_deg=args.max_rotation_sigma_deg,
        min_initial_tags=INVENTORY.target("wrist").minimum_acquisition_ids or 1,
        motion_compensation=args.motion_compensation,
        max_motion_window_ms=args.max_motion_window_ms,
    )
    detector = WristTagDetector(layout, detector_config)
    tracker = VisionOnlyTracker(MultiCameraEstimator(cameras, layout, estimator_config))
    frame_log = (output / "frames.jsonl").open("x")
    manifest = {
        "schema_version": 1,
        "split": args.split,
        "seed": SPLIT_SEEDS[args.split],
        "num_frames": args.num_frames,
        "camera_scale": args.camera_scale,
        "detector_scale": args.detector_scale,
        "detector_config": dataclasses.asdict(detector_config),
        "estimator_config": dataclasses.asdict(estimator_config),
        "calibration_id": bundle["bundle_id"],
        "wrist_layout_hash": true_layout.layout_hash,
        "fiducial_inventory": str(INVENTORY.source),
        "inventory_hash": INVENTORY.inventory_hash,
        "calibration": str(Path(args.calibration).expanduser()),
        "robot_world": str(Path(args.robot_world).expanduser()),
        "pose_bank": str(Path(args.pose_bank).expanduser()),
        "pose_bank_hash": pose_bank_hash,
        "pose_bank_source_session": pose_bank_metadata.get("source_session"),
        "pose_bank_order": pose_order.tolist(),
        "model_perturbations": model_perturbations,
        **_git_identity(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    diagnostics_saved = 0
    duration_s = max((args.num_frames - 1) * 0.1, 0.1)

    def joint_pose(sample_time_s: float):
        return sample_pose(sample_time_s / duration_s)

    def set_joint_pose(q):
        full = robot.get_qpos().clone()
        full[:, controlled] = torch.as_tensor(q, device=base_env.device, dtype=full.dtype)
        robot.set_qpos(full)

    started = time.perf_counter()
    try:
        for index in range(args.num_frames):
            detections = []
            scenario = {}
            diagnostic_images = {}
            timestamp_ns = int(index * 1e8)  # nominal 10 Hz benchmark timeline
            for name in true_cameras:
                skew_ns = int(rng.uniform(-20e6, 20e6)) if args.split != "clean" else 0
                set_joint_pose(joint_pose(index * 0.1 + skew_ns / 1e9))
                obs = base_env.get_obs()
                rgb = obs["sensor_data"][name]["rgb"][0].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                map_x, map_y = distortion_maps[name]
                distorted = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                distorted, scenario[name] = _corrupt(distorted, rng, args.split)
                dropout = args.split != "clean" and rng.random() < (0.10 if args.split == "holdout" else 0.04)
                scenario[name]["dropout"] = dropout
                scenario[name]["timestamp_skew_ns"] = skew_ns
                found = []
                if not dropout:
                    found = detector.detect(name, distorted, timestamp_ns + skew_ns)
                    detections.extend(found)
                diagnostic_images[name] = (distorted, found)
            set_joint_pose(joint_pose(index * 0.1))
            sim_tracking_link = true_layout.parent_frame.rsplit("/", 1)[-1]
            base_from_ee = _pose_matrix(robot.links_map[sim_tracking_link].pose.raw_pose)
            world_from_ee = world_from_base @ base_from_ee
            estimate = tracker.update(detections, timestamp_ns)
            record = {
                "index": index,
                "status": estimate.status,
                "detected_tags": sorted({item.tag_id for item in detections}),
                "detected_cameras": sorted({item.camera for item in detections}),
                "reprojection_rmse_px": estimate.reprojection_rmse_px,
                "used_cameras": estimate.used_cameras,
                "used_tags": estimate.used_tags,
                "rejected_sources": estimate.rejected_sources,
                "corner_count": estimate.corner_count,
                "condition": estimate.condition,
                "translation_sigma_mm": estimate.translation_sigma_mm,
                "rotation_sigma_deg": estimate.rotation_sigma_deg,
                "reason": estimate.reason,
                "source_rmse_px": estimate.source_rmse_px,
                "scenario": scenario,
                "translation_error_mm": None,
                "rotation_error_deg": None,
            }
            if estimate.status == "measured":
                delta = invert(world_from_ee) @ estimate.world_from_ee
                record["translation_error_mm"] = float(1000 * np.linalg.norm(delta[:3, 3]))
                record["rotation_error_deg"] = rotation_distance_deg(world_from_ee, estimate.world_from_ee)
            error_limit_mm = 2.0 if args.split == "clean" else 10.0
            error_limit_deg = 0.5 if args.split == "clean" else 2.0
            diagnostic_failure = (
                estimate.status != "measured"
                or record["translation_error_mm"] > error_limit_mm
                or record["rotation_error_deg"] > error_limit_deg
            )
            if diagnostic_failure and diagnostics_saved < args.save_failure_frames:
                diagnostic_dir = output / "diagnostics"
                diagnostic_dir.mkdir(exist_ok=True)
                record["diagnostic_files"] = []
                for name, (image, found) in diagnostic_images.items():
                    path = diagnostic_dir / f"{index:05d}_{name}.jpg"
                    cv2.imwrite(str(path), _annotate(image, found, estimate.status))
                    record["diagnostic_files"].append(str(path.relative_to(output)))
                diagnostics_saved += 1
            records.append(record)
            frame_log.write(json.dumps(record, separators=(",", ":")) + "\n")
            frame_log.flush()
    finally:
        frame_log.close()
        env.close()
    report = _summarize(records, time.perf_counter() - started)
    passed, failures = _passes_gates(report, args.split)
    report.update({"split": args.split, "passed": passed, "gate_failures": failures})
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.enforce_gates and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main(tyro.cli(Args))
