#!/usr/bin/env python3
"""Offline pairwise Open3D benchmark. Candidate transforms never authorize motion.

Transforms map the previous camera coordinates into the current camera coordinates.
This is local registration, not a persistent material tracker or a robot-world pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path

# Bound native pools before importing numerical libraries, including when run directly.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# The stationary-depth report is intentionally usable on the aarch64 camera
# owner, where Open3D does not publish wheels. Registration still imports it.
if "--stationary-depth" in sys.argv:
    o3d = None
else:
    import open3d as o3d  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import tatbot_rerun as tr  # noqa: E402
import tatbot_runlog  # noqa: E402
from surface_consistency import consistency, correspondences  # noqa: E402
from visiond_wire import read_evidence_frame  # noqa: E402


@dataclass(frozen=True)
class Settings:
    voxel_m: float = 0.002
    correspondence_m: float = 0.006
    min_fitness: float = 0.5
    max_rmse_m: float = 0.003
    max_gap_ms: float = 150.0
    error_mm: float = 2.0
    error_deg: float = 2.0
    max_points: int = 20000


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("expected a finite rigid 4x4 transform")
    return matrix


def pose_error(estimate, truth, probes):
    delta = estimate[:3, :3] @ truth[:3, :3].T
    return {"translation_mm": float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) * 1000),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))),
            "material_rms_mm": float(np.sqrt(np.mean(np.sum(
                (transform(probes, estimate) - transform(probes, truth)) ** 2, axis=1))) * 1000)}


def stats(values):
    return {k: float(np.percentile(values, p)) if values else None
            for k, p in (("p50", 50), ("p95", 95), ("max", 100))}


def cloud(points, colors, settings):
    # Deterministic input cap bounds normal estimation and registration work.
    take = np.linspace(0, len(points) - 1, min(len(points), settings.max_points), dtype=int)
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points[take])
    result.colors = o3d.utility.Vector3dVector(colors[take])
    result = result.voxel_down_sample(settings.voxel_m)
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=settings.voxel_m * 4, max_nn=30))
    result.orient_normals_towards_camera_location(np.zeros(3))
    return result


def register(source, target, method, settings):
    reg = o3d.pipelines.registration
    estimator = (reg.TransformationEstimationForColoredICP() if method == "colored"
                 else reg.TransformationEstimationPointToPlane())
    fn = reg.registration_colored_icp if method == "colored" else reg.registration_icp
    result = fn(source, target, settings.correspondence_m, np.eye(4), estimator,
                reg.ICPConvergenceCriteria(max_iteration=30))
    matrix = rigid_matrix(result.transformation)
    reverse = reg.evaluate_registration(target, source, settings.correspondence_m, np.linalg.inv(matrix))
    fitness = min(float(result.fitness), float(reverse.fitness))
    rmse = max(float(result.inlier_rmse), float(reverse.inlier_rmse))
    enough = len(result.correspondence_set) >= 30
    accepted = enough and fitness >= settings.min_fitness and rmse <= settings.max_rmse_m
    return {"status": "candidate" if accepted else "rejected",
            "reason": None if accepted else "insufficient_overlap_or_large_residual",
            "transform_previous_to_current": matrix.tolist(), "bidirectional_fitness": fitness,
            "geometric_inlier_rmse_m": rmse, "correspondences": len(result.correspondence_set)}


def entries(index):
    with index.open() as stream:
        previous = -1
        for line in stream:
            row = json.loads(line)
            stamp = row["metadata"]["timestamps"].get("normalized_unix_ns")
            if not isinstance(stamp, int) or stamp <= previous or stamp <= 0:
                raise ValueError("recording needs strictly increasing normalized timestamps")
            previous = stamp
            yield row


def intrinsics(metadata):
    value = metadata.get("attributes", {}).get("intrinsics")
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("missing active camera intrinsics")
    profile = metadata["profile"]
    if any(value[k] != profile[k] for k in ("width", "height")):
        raise ValueError("intrinsics dimensions differ from recorded profile")
    coefficients = value.get("distortion_coefficients")
    if (coefficients is None or len(coefficients) != 5 or not np.isfinite(coefficients).all()
            or value.get("distortion_model") not in ("None", "BrownConradyInverse")):
        raise ValueError("unsupported or unknown distortion metadata")
    numbers = np.array([value[k] for k in ("fx", "fy", "ppx", "ppy")], dtype=float)
    if not np.isfinite(numbers).all() or min(numbers[:2]) <= 0:
        raise ValueError("invalid focal length or principal point")
    return numbers


@lru_cache(maxsize=4)
def owner_rays(intrinsics_json):
    # Reuse the drawing mapper's SDK convention and bounded profile cache.
    from capture_geometry import _owner_rays
    return _owner_rays(json.dumps({"intrinsics": json.loads(intrinsics_json)}))


def rgbd_points(depth, bgr, metadata, roi, depth_range):
    units = float(metadata.get("attributes", {}).get("depth_units_m", "nan"))
    if not np.isfinite(units) or units <= 0:
        raise ValueError("missing or invalid measured depth units")
    fx, fy, cx, cy = intrinsics(metadata)
    x, y, w, h = roi
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > depth.shape[1] or y + h > depth.shape[0]:
        raise ValueError("ROI outside depth image")
    if bgr.shape != (*depth.shape, 3):
        raise ValueError("aligned color/depth dimensions differ")
    z = depth[y:y+h, x:x+w].astype(float) * units
    v, u = np.mgrid[y:y+h, x:x+w]
    valid = np.isfinite(z) & (z > depth_range[0]) & (z < depth_range[1]) & (depth[y:y+h, x:x+w] < 65535)
    active = metadata["attributes"]["intrinsics"]
    active = json.loads(active) if isinstance(active, str) else active
    if np.any(np.abs(active["distortion_coefficients"]) > 0):
        rays = owner_rays(json.dumps(active, sort_keys=True))
        points = rays[y:y+h, x:x+w][valid] * z[valid, None]
    else:
        points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)[valid]
    colors = bgr[y:y+h, x:x+w, ::-1][valid].astype(float) / 255
    return points, colors


def recording_frames(root, sensor, roi, depth_range, pair_ms=10.0, include_images=False, include_depth_roi=False):
    """Merge two evidence indexes with a bounded one-to-one timestamp pairing.

    Explicit ROI/depth bounds isolate the working object. Background segmentation
    and moving-camera compensation remain the caller's experimental responsibility.
    """
    if not sensor or Path(sensor).name != sensor or sensor in (".", ".."):
        raise ValueError("invalid sensor name")
    color_name, depth_name = sensor + "_color", sensor + "_depth"
    color_dir, depth_dir = root / color_name, root / depth_name
    colors = iter(entries(color_dir / "frames.jsonl"))
    color = next(colors, None)
    calibration = None
    previous_drops = None
    for depth in entries(depth_dir / "frames.jsonl"):
        dm = depth["metadata"]
        stamp = dm["timestamps"]["normalized_unix_ns"]
        while color and color["metadata"]["timestamps"]["normalized_unix_ns"] < stamp - pair_ms * 1e6:
            color = next(colors, None)
        if color is None or abs(color["metadata"]["timestamps"]["normalized_unix_ns"] - stamp) > pair_ms * 1e6:
            yield {"stamp": stamp, "invalid": "unpaired_rgbd"}
            continue
        cm = color["metadata"]
        if dm["sensor_name"] != depth_name or cm["sensor_name"] != color_name:
            raise ValueError("sensor identity mismatch")
        if dm.get("attributes", {}).get("aligned_to") != color_name:
            raise ValueError("depth is not explicitly aligned to selected color stream")
        if ((not include_depth_roi or include_images)
                and not np.allclose(intrinsics(dm), intrinsics(cm), rtol=0, atol=1e-6)):
            raise ValueError("aligned color/depth intrinsics disagree")
        signature = json.dumps([{k: m.get(k) for k in ("profile", "calibration_id")}
                                | ({"units": m.get("attributes", {}).get("depth_units_m")}
                                   if include_depth_roi and not include_images else
                                   {"intrinsics": m.get("attributes", {}).get("intrinsics"),
                                    "units": m.get("attributes", {}).get("depth_units_m")})
                                for m in (dm, cm)], sort_keys=True)
        if calibration is not None and calibration != signature:
            raise ValueError("camera calibration/profile changed during recording")
        calibration = signature
        if dm["profile"]["format"] != "z16":
            raise ValueError("depth stream is not Z16")
        # The owner records lifetime drop totals, not a per-frame gap count.
        drops = tuple(int(m.get("dropped_before", 0)) for m in (dm, cm))
        new_drops = previous_drops is not None and drops != previous_drops
        previous_drops = drops
        allowed = {"real_sense_global": {"timestamp_domain=Global Time"},
                   "host_unix": {"timestamp_domain=System Time"},
                   "real_sense_hardware": {"timestamp_domain=Hardware Clock",
                                           "hardware_clock_host_normalized"}}
        warnings = [flag for m in (dm, cm) for flag in m.get("flags", [])
                    if flag not in allowed.get(m["timestamps"].get("source_domain"), set())]
        if warnings or new_drops:
            yield {"stamp": stamp, "invalid": "capture_drop_counter_changed" if new_drops else "flagged_capture"}
        else:
            d = read_evidence_frame(depth_dir, depth)["depth"]
            extra = {}
            if include_depth_roi:
                x, y, w, h = roi
                raw = d[y:y+h, x:x+w]
                z = raw.astype(np.float32) * float(dm["attributes"]["depth_units_m"])
                z[(raw == 65535) | (z <= depth_range[0]) | (z >= depth_range[1])] = np.nan
                extra["depth_roi_m"] = z
            if not include_depth_roi or include_images:
                bgr = read_evidence_frame(color_dir, color)["image"]
                points, rgb = rgbd_points(d, bgr, dm, roi, depth_range)
                extra.update({"points": points, "colors": rgb})
            if include_images:
                active = dm["attributes"]["intrinsics"]
                active = json.loads(active) if isinstance(active, str) else active
                if np.any(np.abs(active["distortion_coefficients"]) > 0):
                    rays = owner_rays(json.dumps(active, sort_keys=True))
                else:
                    fx, fy, cx, cy = intrinsics(dm)
                    v, u = np.mgrid[:d.shape[0], :d.shape[1]]
                    rays = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones_like(u)), axis=-1)
                depth_m = d.astype(float) * float(dm["attributes"]["depth_units_m"])
                depth_m[d == 65535] = 0
                extra.update({"rgbd": {"gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                                       "depth_m": depth_m, "rays": rays, "depth_range": depth_range},
                              "roi": roi})
            yield {"stamp": stamp, **extra,
                   "provenance": {"depth_sha256": depth["sha256"], "color_sha256": color["sha256"],
                                  "depth_sequence": dm["sequence"], "color_sequence": cm["sequence"],
                                  "owner_cumulative_drops": list(drops)}}
        color = next(colors, None)


def evaluate_stationary_depth(frames, path, max_frames=300):
    """Operator-asserted stationary ROI; repeatability, never absolute accuracy.

    Retain only cropped depth, with a 64 MiB input-stack budget. Missing/flagged
    frames break consecutive comparisons and count against coverage.
    """
    samples, deltas, levels = [], [], []
    previous = None
    count = 0
    shape = None
    first_stamp = last_stamp = None
    start, cpu = time.perf_counter(), time.process_time()
    with path.open("w") as output:
        for frame in islice(frames, max_frames):
            stamp = frame["stamp"]
            if last_stamp is not None and stamp <= last_stamp:
                raise ValueError("nonmonotonic frame timestamp")
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            count += 1
            row = {"timestamp_ns": stamp, "provenance": frame.get("provenance"),
                   "invalid": frame.get("invalid"), "motion_authority": False}
            if frame.get("invalid"):
                previous = None
            else:
                z = np.asarray(frame["depth_roi_m"], dtype=np.float32)
                if z.ndim != 2 or z.size == 0 or (shape is not None and z.shape != shape):
                    raise ValueError("stationary depth ROI shape changed or is empty")
                shape = z.shape
                if (len(samples) + 1) * z.nbytes > 64 * 1024 * 1024:
                    raise ValueError("stationary depth stack exceeds 64 MiB; reduce ROI or max frames")
                z = z.copy()
                z[~np.isfinite(z) | (z <= 0)] = np.nan
                valid = np.isfinite(z)
                row["valid_fraction"] = float(valid.mean())
                row["median_depth_mm"] = float(np.median(z[valid]) * 1000) if valid.any() else None
                if row["median_depth_mm"] is not None:
                    levels.append(row["median_depth_mm"])
                if previous is not None:
                    dt = (stamp - previous[0]) / 1e6
                    if dt <= 150:
                        diff = np.abs(z - previous[1]) * 1000
                        finite = diff[np.isfinite(diff)]
                        row["consecutive_delta_p95_mm"] = float(np.percentile(finite, 95)) if finite.size else None
                        if finite.size:
                            deltas.append(row["consecutive_delta_p95_mm"])
                samples.append(z)
                previous = (stamp, z)
            output.write(json.dumps(row, allow_nan=False) + "\n")
    if count < 3 or len(samples) < 3:
        raise ValueError("stationary depth needs at least three usable frames")
    stack = np.stack(samples)
    valid = np.isfinite(stack)
    support = valid.sum(axis=0)
    # A quiet statistic must not hide pixels that rarely return depth.
    eligible = support >= max(3, int(np.ceil(count * .9)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(stack, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(stack - median), axis=0) * 1000
    return {"mode": "stationary_depth", "motion_authority": False,
            "meaning": "operator-asserted stationary scene; temporal repeatability, not absolute accuracy or tracking error",
            "frames": count, "usable_frames": len(samples), "invalid_frames": count - len(samples),
            "roi_pixels": int(np.prod(shape)), "support_required_fraction": .9,
            "valid_fraction_including_invalid_frames": float(valid.sum() / (count * np.prod(shape))),
            "pixels_with_90_percent_support_fraction": float(eligible.mean()),
            "temporal_robust_sigma_mm": stats(sigma[eligible].tolist()),
            "per_pair_depth_delta_p95_mm": stats(deltas),
            "frame_median_depth_mm": stats(levels),
            "consecutive_pairs": len(deltas), "capture_duration_s": (last_stamp - first_stamp) / 1e9,
            "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - start}


def with_ground_truth(frames, path):
    """Independently measured object-to-camera poses, never estimated ICP outputs."""
    document = json.loads(path.read_text())
    if document.get("schema") != "tatbot.surface-rgbd-ground-truth/1" or not document.get("provenance"):
        raise ValueError("ground truth needs a supported schema and measurement provenance")
    poses = {}
    for row in document["frames"]:
        stamp = row["timestamp_ns"]
        if not isinstance(stamp, int) or stamp <= 0 or stamp in poses:
            raise ValueError("ground truth timestamps must be positive and unique")
        poses[stamp] = rigid_matrix(row["object_to_camera"])
    for frame in frames:
        if frame["stamp"] not in poses:
            raise ValueError("ground truth missing an evaluated frame timestamp")
        yield dict(frame, pose=poses[frame["stamp"]])


def synthetic_frames(case, count=30):
    """Sampled 3D controls, not simulated sensor images or physical accuracy evidence."""
    v, u = np.mgrid[-1:1:70j, -1:1:90j]
    x, y = u * .055, v * .045
    z = .25 + .012 * np.sin(u * 3) * np.cos(v * 2) + .006 * u * v
    base = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = np.stack((.5 + .4 * np.sin(u * 13), .5 + .4 * np.cos(v * 17),
                       .5 + .4 * np.sin(u * 9 + v * 11)), axis=-1).reshape(-1, 3)
    if case == "flat_textured":
        base[:, 2] = .25
    if case == "symmetric_alias":
        # Integer angular steps leave an untextured cylinder's samples identical.
        theta, height = np.meshgrid(np.arange(120) * 2 * np.pi / 120, np.linspace(-.04, .04, 45))
        base = np.stack((.04 * np.cos(theta), height, .25 + .04 * np.sin(theta)), axis=-1).reshape(-1, 3)
        colors = np.full_like(base, .6)
    for i in range(count):
        matrix = np.eye(4)
        angle = np.radians(i * (3 if case == "symmetric_alias" else .25))
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle, 0])
        center = np.array([0, 0, .25])
        matrix[:3, 3] = center - matrix[:3, :3] @ center + [i * .0003, 0, 0]
        if case == "flat_textured":
            matrix = np.eye(4)
            matrix[0, 3] = i * .0035
        points = transform(base, matrix)
        if case == "noisy_motion":
            points = points + np.random.default_rng(i).normal(0, .00015, points.shape)
        rgb = colors.copy()
        if case == "symmetric_alias":
            matrix[:3, 3] = center - matrix[:3, :3] @ center
            points = base.copy()
        if case == "dropout" and i >= count // 2:
            points, rgb = points[:0], rgb[:0]
        if case == "no_overlap" and i >= count // 2:
            points = points + [0.2 * (i - count // 2 + 1), 0, 0]
            matrix[:3, 3] += [0.2 * (i - count // 2 + 1), 0, 0]
        stamp = 1_000_000_000 + i * 50_000_000
        if case == "gap" and i >= count // 2:
            stamp += 500_000_000
        yield {"stamp": stamp, "points": points, "colors": rgb, "pose": matrix,
               "expected_reject": case in ("dropout", "no_overlap") and i >= count // 2
                                  or case == "gap" and i == count // 2}


def evaluate(frames, settings, path, max_frames=300, rr=None, name="recording", methods=("point_to_plane", "colored")):
    stage_times = {stage: [] for stage in ("read", "prepare", "image_correspondence", "point_to_plane", "colored", "output", "total")}
    method_cpu = {m: [] for m in methods}
    errors = {m: [] for m in methods}
    confusion = {m: dict.fromkeys(("true_accept", "true_reject", "false_accept", "false_reject", "unlabelled"), 0)
                 for m in errors}
    image_residuals = {m: [] for m in errors}
    image_unavailable = dict.fromkeys(errors, 0)
    previous = None
    first_stamp = last_stamp = None
    next_visual = 0
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    frame_count, pair_count = 0, 0
    iterator = iter(frames)
    with path.open("w") as output:
        for _ in range(max_frames):
            start = time.perf_counter()
            try:
                frame = next(iterator)
            except StopIteration:
                break
            stage_times["read"].append((time.perf_counter() - start) * 1000)
            frame_count += 1
            stamp = frame["stamp"]
            if last_stamp is not None and stamp <= last_stamp:
                raise ValueError("nonmonotonic frame timestamp")
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            prep = time.perf_counter()
            invalid = frame.get("invalid")
            current = None
            if not invalid and len(frame["points"]) >= 30:
                current = cloud(frame["points"], frame["colors"], settings)
            if current is None or len(current.points) < 30:
                invalid = invalid or "insufficient_depth"
            stage_times["prepare"].append((time.perf_counter() - prep) * 1000)
            row = {"timestamp_ns": stamp, "motion_authority": False, "methods": {},
                   "provenance": frame.get("provenance"), "input_reason": invalid}
            if previous is not None:
                pair_count += 1
                old_frame, old_cloud = previous
                dt_ms = (stamp - old_frame["stamp"]) / 1e6
                reason = invalid or ("previous_input_unavailable" if old_cloud is None else None)
                reason = reason or ("timestamp_gap" if dt_ms > settings.max_gap_ms else None)
                matches = None
                if not reason and "rgbd" in frame and "rgbd" in old_frame:
                    check_start = time.perf_counter()
                    matches = correspondences(old_frame["rgbd"], frame["rgbd"], frame["roi"])
                    stage_times["image_correspondence"].append((time.perf_counter() - check_start) * 1000)
                for method in errors:
                    tick, cpu = time.perf_counter(), time.process_time()
                    if reason:
                        result = {"status": "rejected", "reason": reason}
                    else:
                        try:
                            result = register(old_cloud, current, method, settings)
                        except (RuntimeError, ValueError) as exc:
                            result = {"status": "rejected", "reason": "solver_failure", "detail": str(exc)}
                    elapsed = (time.perf_counter() - tick) * 1000
                    result.update(wall_ms=elapsed, cpu_ms=(time.process_time() - cpu) * 1000)
                    result["solver_attempted"] = reason is None
                    if reason is None:
                        stage_times[method].append(elapsed)
                        method_cpu[method].append(result["cpu_ms"])
                    truth_error = None
                    if "pose" in frame and "pose" in old_frame and "transform_previous_to_current" in result:
                        truth = frame["pose"] @ np.linalg.inv(old_frame["pose"])
                        truth_error = pose_error(np.asarray(result["transform_previous_to_current"]), truth, old_frame["points"])
                        errors[method].append(truth_error["material_rms_mm"])
                    result["ground_truth_error"] = truth_error
                    if matches is not None and "transform_previous_to_current" in result:
                        check = consistency(matches, result["transform_previous_to_current"])
                        result["image_depth_consistency"] = check
                        if check["status"] == "available":
                            image_residuals[method].append(check["residual_mm"]["p50"])
                        else:
                            image_unavailable[method] += 1
                    bad = frame.get("expected_reject")
                    if truth_error is not None:
                        bad = bool(bad or truth_error["material_rms_mm"] > settings.error_mm
                                   or truth_error["rotation_deg"] > settings.error_deg)
                    if bad is None:
                        label = "unlabelled"
                    else:
                        accepted = result["status"] == "candidate"
                        label = ("false_accept" if bad else "true_accept") if accepted else ("true_reject" if bad else "false_reject")
                    confusion[method][label] += 1
                    result["evaluation_label"] = label
                    row["methods"][method] = result
            out_start = time.perf_counter()
            if rr is not None and stamp >= next_visual:
                rr.set_time("surface_rgbd_seconds", duration=(stamp - first_stamp) / 1e9)
                entity = f"surface/rgbd/{name}"
                if current is not None:
                    rr.log(entity + "/current", rr.Points3D(np.asarray(current.points), colors=np.asarray(current.colors)))
                else:
                    rr.log(entity + "/current", rr.Clear(recursive=True))
                for method, result in row["methods"].items():
                    if "transform_previous_to_current" in result:
                        aligned = transform(np.asarray(previous[1].points), np.asarray(result["transform_previous_to_current"]))
                        rr.log(entity + "/" + method, rr.Points3D(aligned, colors=[255, 80, 80]))
                    else:
                        rr.log(entity + "/" + method, rr.Clear(recursive=True))
                rr.log(entity + "/status", rr.TextLog(json.dumps(row)))
                next_visual = stamp + 200_000_000
            output.write(json.dumps(row, allow_nan=False) + "\n")
            stage_times["output"].append((time.perf_counter() - out_start) * 1000)
            stage_times["total"].append((time.perf_counter() - start) * 1000)
            # Independent consecutive-pair benchmark; no hidden recovery or pose chaining.
            previous = (frame, current if not invalid else None)
    if pair_count == 0:
        raise ValueError("benchmark needs at least two frames")
    duration = (last_stamp - first_stamp) / 1e9
    cpu_s = time.process_time() - cpu_start
    return {"frames": frame_count, "pairs": pair_count, "stage_wall_ms": {k: stats(v) for k, v in stage_times.items()},
            "image_depth_consistency": {m: {"pairs": len(v), "unavailable_pairs": image_unavailable[m], "not_checked_pairs": pair_count - len(v) - image_unavailable[m], "pair_median_residual_mm": stats(v)} for m, v in image_residuals.items()},
            "method_cpu_ms": {k: stats(v) for k, v in method_cpu.items()},
            "material_rms_mm": {k: stats(v) for k, v in errors.items()}, "failure_detection": confusion,
            "cpu_seconds": cpu_s, "wall_seconds": time.perf_counter() - wall_start,
            "source_duration_s": duration, "estimated_combined_core_fraction": cpu_s / duration,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--recording", type=Path, help="visiond evidence root with SENSOR_color and SENSOR_depth")
    parser.add_argument("--sensor")
    parser.add_argument("--method", choices=("both", "colored", "point_to_plane"), default="both")
    parser.add_argument("--max-points", type=int, default=20000, help="deterministic point cap before voxel sampling")
    parser.add_argument("--image-check", action="store_true", help="cross-check with RGB optical flow and depth; not independent ground truth")
    parser.add_argument("--ground-truth", type=Path, help="independently measured timestamped object-to-camera poses")
    parser.add_argument("--roi", type=int, nargs=4)
    parser.add_argument("--depth-range", type=float, nargs=2, metavar=("NEAR_M", "FAR_M"))
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--stationary-depth", action="store_true", help="measure stationary ROI repeatability without registration")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.max_frames <= 10000:
        parser.error("require 3–10000 frames")
    if args.stationary_depth and (not args.recording or args.image_check or args.ground_truth or args.rerun):
        parser.error("--stationary-depth requires recording and cannot combine with image-check, ground-truth or rerun")
    if args.image_check and not args.recording:
        parser.error("--image-check requires --recording")
    if args.ground_truth and not args.recording:
        parser.error("--ground-truth requires --recording")
    if args.recording and (not args.sensor or args.roi is None or args.depth_range is None):
        parser.error("recording requires --sensor, --roi and --depth-range")
    if args.depth_range and (not np.isfinite(args.depth_range).all() or not 0 < args.depth_range[0] < args.depth_range[1]):
        parser.error("depth range must satisfy 0 < near < far")
    if not 1000 <= args.max_points <= 20000:
        parser.error("--max-points must be 1000–20000")
    cv2.setNumThreads(1)
    settings = Settings(max_points=args.max_points)
    methods = ("point_to_plane", "colored") if args.method == "both" else (args.method,)
    with threadpool_limits(limits=1), tatbot_runlog.init("surface-rgbd", prune_first=False) as run:
        rr = tr.start("surface_rgbd", output=run.dir / "replay.rrd", recording_id=run.run_id) if args.rerun else None
        report = {"schema_version": 1, "settings": asdict(settings), "methods": methods,
                  "image_check": {"enabled": args.image_check, "max_points": 120, "forward_backward_limit_px": 1., "photometric_limit": 20., "min_matches": 12,
                                  "meaning": "RGB-flow/depth consistency, not independent ground truth; includes depth noise and pixel quantization"}, "open3d": None if o3d is None else o3d.__version__,
                  "python": platform.python_version(), "machine": platform.machine(),
                  "thread_pools": threadpool_info(), "motion_authority": False,
                  "input_kind": "sampled_3d_controls" if args.synthetic else "recorded_rgbd_without_ground_truth",
                  "latency_basis": "offline; includes decoding and optional capped RRD output; excludes imports, live transport and control",
                  "limits": "pairwise rigid registration only; residual is not pose error; quality gates cannot prove material identity",
                  "implementation_sha256": {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                             for n in ("surface_rgbd.py", "visiond_wire.py", "surface_consistency.py")}, "cases": {}}
        report["implementation_sha256"]["capture_geometry.py"] = hashlib.sha256(
            Path(__file__).resolve().parents[1].joinpath("capture_geometry.py").read_bytes()).hexdigest()
        if args.synthetic:
            sources = {case: synthetic_frames(case, args.max_frames)
                       for case in ("motion", "noisy_motion", "flat_textured", "dropout", "no_overlap", "gap", "symmetric_alias")}
        else:
            root = args.recording.expanduser().resolve()
            report["input"] = {"path": str(root), "sensor": args.sensor, "roi": args.roi, "depth_range_m": args.depth_range,
                               "indexes_sha256": {kind: hashlib.sha256((root / (args.sensor + "_" + kind) / "frames.jsonl").read_bytes()).hexdigest()
                                                  for kind in ("color", "depth")}}
            frames = recording_frames(root, args.sensor, args.roi, args.depth_range, include_images=args.image_check, include_depth_roi=args.stationary_depth)
            if args.ground_truth:
                gt = args.ground_truth.expanduser().resolve()
                report["input"]["ground_truth_sha256"] = hashlib.sha256(gt.read_bytes()).hexdigest()
                report["input"]["ground_truth_path"] = str(gt)
                report["input_kind"] = "recorded_rgbd_with_supplied_ground_truth"
                frames = with_ground_truth(frames, gt)
            sources = {"recording": frames}
        if args.stationary_depth:
            report["methods"] = []
            report["limits"] = "stationary-scene assertion supplied by operator; missing depth retained; no accuracy or motion authority"
        for name, frames in sources.items():
            if args.stationary_depth:
                report["cases"][name] = evaluate_stationary_depth(frames, run.dir / (name + ".jsonl"), args.max_frames)
                continue
            report["cases"][name] = evaluate(islice(frames, args.max_frames), settings, run.dir / (name + ".jsonl"), args.max_frames, rr, name, methods)
        if rr is not None:
            rr.get_global_data_recording().flush()
        (run.dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Report: {run.dir / 'report.json'}")


if __name__ == "__main__":
    main()
