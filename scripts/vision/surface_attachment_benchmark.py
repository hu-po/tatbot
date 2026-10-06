"""Offline observer comparison against immutable original material references.

No session lifecycle or motion authority lives here. Generated truth is consumed
only by scoring; real recordings produce consistency evidence without true error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from surface_attachment import SurfaceAttachmentObserver  # noqa: E402
from surface_attachment_inputs import pose, recorded_frames, render_fixture  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

# Reporting threshold for known synthetic controls, never a physical tolerance.
SCORE_MAX_ERROR_MM = 2.0


def implementation_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name("surface_attachment.py"),
             Path(__file__).with_name("surface_material_support.py"),
             Path(__file__).with_name("surface_attachment_inputs.py"),
             Path(__file__).with_name("surface_rgbd.py"), Path(__file__).with_name("visiond_wire.py"),
             Path(__file__).with_name("surface_match.py"), Path(__file__).with_name("surface_consistency.py"),
             Path(__file__).resolve().parents[1]/"capture_geometry.py"]
    dependencies = {}
    for name in ("open3d", "opencv-python-headless", "opencv-python", "pyrealsense2", "threadpoolctl"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"dependencies": dependencies, "opencv_threads": cv2.getNumThreads(),
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_tree_status": subprocess.check_output(
                ["git", "status", "--short", "--", *[str(p.relative_to(root)) for p in paths]], cwd=root, text=True).strip(),
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__}


def material_error(matrix, original_truth, current_truth):
    """Score original material locations, including local deformation displacement."""
    initial = original_truth["material_points"]
    current = current_truth["material_points"]
    reference_pose = original_truth["camera_from_material"]
    current_pose = current_truth["camera_from_material"]
    source = initial @ reference_pose[:3, :3].T + reference_pose[:3, 3]
    target = current @ current_pose[:3, :3].T + current_pose[:3, 3]
    predicted = source @ matrix[:3, :3].T + matrix[:3, 3]
    error = np.linalg.norm(predicted-target, axis=1)*1000
    return {"rms_mm": float(np.sqrt(np.mean(error**2))), "max_mm": float(error.max())}


def _dense_cloud(frame, settings):
    from surface_rgbd import cloud
    rgbd = frame["rgbd"]
    depth = rgbd["depth_m"]
    valid = np.isfinite(depth) & (depth > rgbd["depth_range"][0]) & (depth < rgbd["depth_range"][1])
    if frame.get("roi"):
        x, y, w, h = frame["roi"]
        crop = np.zeros_like(valid)
        crop[y:y+h, x:x+w] = True
        valid &= crop
    points = (rgbd["rays"]*depth[..., None])[valid]
    if "bgr" in frame:
        bgr = np.asarray(frame["bgr"])
        if bgr.shape != (*depth.shape, 3) or bgr.dtype != np.uint8:
            raise ValueError("invalid aligned BGR geometry")
        colors = bgr[..., ::-1][valid].astype(float)/255
    else:
        colors = np.repeat(rgbd["gray"][valid, None], 3, axis=1).astype(float)/255
    if len(points) < 30:
        raise ValueError("insufficient_depth_points")
    return cloud(points, colors, settings)


def _overlay(path, frame, results):
    canvas = cv2.cvtColor(frame["rgbd"]["gray"], cv2.COLOR_GRAY2BGR)
    support = results["sparse"].get("material_support")
    if support:
        for point in support["points"]:
            pixel = point.get("current_pixel")
            if pixel is not None:
                color = (40, 210, 50) if point["supported"] else (40, 40, 230)
                cv2.circle(canvas, tuple(np.rint(pixel).astype(int)), 2, color, 1)
    # Observer correspondences, never truth points, supply the overlay.
    strip = np.zeros((90, canvas.shape[1], 3), np.uint8)
    for i, (name, result) in enumerate(results.items()):
        label = f'{name}: {"accepted" if result["accepted"] else "withheld"}'
        cv2.putText(strip, label, (5, 17+20*i), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
    if support:
        label = f'Original samples: {support["supported_count"]}/{support["point_count"]} supported'
        cv2.putText(strip, label, (5, 80), cv2.FONT_HERSHEY_SIMPLEX, .35, (230, 230, 230), 1)
    cv2.imwrite(str(path), np.vstack((canvas, strip)))


def _dense_results(reference_cloud, frame, settings, reference_setup_ms):
    from surface_rgbd import register
    prepare_start = time.perf_counter()
    current_cloud = _dense_cloud(frame, settings)
    current_setup_ms = (time.perf_counter()-prepare_start)*1000
    results = {}
    for method in ("colored", "point_to_plane"):
        start = time.perf_counter()
        try:
            result = register(reference_cloud, current_cloud, method, settings)
            result["accepted"] = result["status"] == "candidate"
            result["transform_reference_to_camera"] = result["transform_previous_to_current"]
        except (RuntimeError, ValueError) as error:
            result = {"accepted": False, "reason": str(error), "transform_reference_to_camera": None}
        result["registration_ms"] = (time.perf_counter()-start)*1000
        result["current_preprocessing_ms"] = current_setup_ms
        result["reference_preprocessing_ms"] = reference_setup_ms
        result["processing_ms"] = result["registration_ms"] + current_setup_ms + reference_setup_ms
        results[method] = result
    return results


def _condition(truth, previous, expected, index):
    if truth is None:
        return "recorded_motion_unknown"
    if expected is False:
        return "ambiguous_or_unavailable"
    if index == 0 or previous is None:
        return "reference"
    if (not np.allclose(truth["camera_from_material"], previous["camera_from_material"])
            or not np.allclose(truth["material_points"], previous["material_points"])):
        return "during_motion"
    return "stationary"


def _score_method(result, expected, reference_truth, truth, matches):
    from surface_consistency import consistency
    matrix = result.get("transform_reference_to_camera")
    record = {"accepted": result["accepted"], "reason": result.get("reason"),
              "false_accept": expected is False and result["accepted"],
              "false_reject": expected is True and not result["accepted"],
              "processing_ms": result.get("processing_ms", result.get("diagnostics", {}).get("processing_ms")),
              "diagnostics": result.get("diagnostics"), "state": result.get("state"),
              "reference_digest": result.get("reference_digest"),
              "material_support": result.get("material_support"),
              "original_material_error": None, "sensor_consistency": None,
              "excessive_error_accept": False,
              "dense_timing": {k: result[k] for k in ("registration_ms", "current_preprocessing_ms",
                                                     "reference_preprocessing_ms") if k in result}}
    if matrix is not None:
        matrix = np.asarray(matrix)
        record["transform_reference_to_camera"] = matrix.tolist()
        if truth is not None:
            record["original_material_error"] = material_error(matrix, reference_truth, truth)
            record["excessive_error_accept"] = bool(result["accepted"] and expected is True
                and record["original_material_error"]["rms_mm"] > SCORE_MAX_ERROR_MM)
        if matches is not None:
            record["sensor_consistency"] = consistency(matches, matrix)
    return record


def compare_frames(samples, *, dense=True, output=None, label="case", max_visuals=3):
    """Compare frame, withheld truth or None, expected validity or None samples.

    Dense outputs are geometric candidates, intentionally lacking an appearance
    identity gate, so false acceptances expose why low ICP residual is inadequate.
    All methods register directly to frame zero; none chains pose estimates.
    """
    from surface_consistency import correspondences
    if dense:
        from surface_rgbd import Settings
        settings = Settings(max_points=8000)
    reference_frame = reference_truth = reference_cloud = observer = previous_truth = None
    rows = []
    for index, (frame, truth, expected) in enumerate(samples):
        row = {"index": index, "label": label, "capture_timestamp_ns": frame.get("capture_timestamp_ns"),
               "evidence_kind": frame["evidence_kind"], "expected_attachment": expected,
               "provenance": frame.get("provenance"), "methods": {},
               "condition": _condition(truth, previous_truth, expected, index)}
        previous_truth = truth
        rows.append(row)
        if frame.get("invalid"):
            row["invalid"] = frame["invalid"]
            if observer is not None:
                observer.observe(frame, now_ns=frame.get("capture_timestamp_ns"))
            continue
        reference_setup_ms = 0.0
        if reference_frame is None:
            reference_frame, reference_truth = frame, truth
            h, w = frame["rgbd"]["gray"].shape
            observer = SurfaceAttachmentObserver("benchmark-material", "original-reference",
                                                 frame.get("roi", (0, 0, w, h)))
            if dense:
                reference_start = time.perf_counter()
                reference_cloud = _dense_cloud(frame, settings)
                reference_setup_ms = (time.perf_counter()-reference_start)*1000
        results = {"sparse": observer.observe(frame, now_ns=frame["capture_timestamp_ns"])}
        if dense:
            results.update(_dense_results(reference_cloud, frame, settings, reference_setup_ms))
        matches = None
        if truth is None:
            matches = correspondences(reference_frame["rgbd"], frame["rgbd"], observer.roi)
        row["methods"] = {method: _score_method(result, expected, reference_truth, truth, matches)
                          for method, result in results.items()}
        if output is not None and index < max_visuals:
            _overlay(output/f"{label}-{index:03d}.png", frame, results)
    return rows


def synthetic_cases():
    """Positive and ambiguous controls share calibrated inputs and fixed ROI."""
    surface = {"world_from_material": pose(xyz=(.003, -.002, .321), angles_deg=(.3, -.3, .6))}
    camera = {"world_from_camera": pose(xyz=(-.002, .001, -.001), angles_deg=(-.2, .2, -.4))}
    alias = {"world_from_material": pose(xyz=(.005, 0, .32))}
    cases = {"surface_motion": surface, "camera_motion": camera, "both_motion": surface | camera,
             "partial_occlusion": surface | {"occlusion_fraction": .2}, "full_occlusion": {"occlusion_fraction": 1},
             "darkness": {"dark": True}, "depth_holes": {"depth_hole_fraction": .08},
             "blur": {"blur_sigma": 4}, "alternate": {"appearance": "alternate"},
             "deformation": {"deformation_m": .012}, "smooth": alias, "repeating": alias, "recovery": surface}
    negatives = {"full_occlusion", "darkness", "alternate", "deformation", "smooth", "repeating", "blur"}
    for shape in ("plane", "curved"):
        for case, changes in cases.items():
            start = {"shape": shape, "appearance": case if case in ("smooth", "repeating") else "distinctive"}
            first, first_truth = render_fixture(**start, sequence=0)
            second, second_truth = render_fixture(**(start | changes), stamp=1_033_333_333, sequence=1)
            samples = [(first, first_truth, case not in ("smooth", "repeating")),
                       (second, second_truth, case not in negatives)]
            if case == "recovery":
                lost, lost_truth = render_fixture(**start, dark=True, stamp=1_016_666_666, sequence=1)
                second["sequence"] = 2
                samples.insert(1, (lost, lost_truth, False))
                settled, settled_truth = render_fixture(**(start | changes), stamp=1_066_666_666, sequence=3)
                samples.append((settled, settled_truth, True))
            yield f"{shape}-{case}", samples


def sustained_samples(count):
    for i in range(count):
        fraction = i/30
        dark = i % 100 in range(70, 76)
        frame, truth = render_fixture(shape="curved", sequence=i, stamp=1_000_000_000+i*33_333_333,
                                     world_from_material=pose(xyz=(.003*np.sin(fraction), .002*np.cos(fraction), .32),
                                                              angles_deg=(0, 0, .5*np.sin(fraction))), dark=dark)
        yield frame, truth, not dark


def summarize(rows):
    summary = {}
    for name in sorted({m for row in rows for m in row["methods"]}):
        values = [r["methods"][name] for r in rows if name in r["methods"]]
        errors = [v["original_material_error"]["rms_mm"] for v in values
                  if v["accepted"] and v["original_material_error"] is not None]
        times = [v["processing_ms"] for v in values if v["processing_ms"] is not None]
        labeled = sum(r["expected_attachment"] is not None for r in rows if name in r["methods"])
        summary[name] = {"samples": len(values), "labeled_samples": labeled,
                         "accepted": sum(v["accepted"] for v in values),
                         "false_accepts": sum(v["false_accept"] for v in values) if labeled else None,
                         "false_rejects": sum(v["false_reject"] for v in values) if labeled else None,
                         "excessive_error_accepts": sum(v["excessive_error_accept"] for v in values) if labeled else None,
                         "material_rms_mm_p50_p95_max": np.percentile(errors, [50, 95, 100]).tolist() if errors else None,
                         "processing_ms_p50_p95_max": np.percentile(times, [50, 95, 100]).tolist() if times else None}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--recording", type=Path, action="append", default=[])
    parser.add_argument("--sensor", default="realsense1")
    parser.add_argument("--roi", nargs=4, type=int, default=None)
    parser.add_argument("--depth-range", nargs=2, type=float, default=(.03, 2.))
    parser.add_argument("--recording-max-frames", type=int, default=30)
    parser.add_argument("--recording-stride", type=int, default=1)
    parser.add_argument("--sustained-frames", type=int, default=300)
    args = parser.parse_args(argv)
    cv2.setNumThreads(1)
    if not 0 <= args.sustained_frames <= 1000:
        parser.error("sustained frame count must be in [0, 1000]")
    if len(args.recording) > 8:
        parser.error("at most eight bounded recordings per report")
    if not (1 <= args.recording_max_frames <= 1000 and 1 <= args.recording_stride <= 100
            and args.recording_max_frames*args.recording_stride <= 10000):
        parser.error("recording count/stride exceeds bounded input budget")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("comparison artifacts must be outside the repository")
    if args.output.exists():
        parser.error("comparison output already exists; use a new evidence directory")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tatbot.surface-attachment-comparison/1", "implementation": implementation_manifest(),
              "score_error_threshold_mm": SCORE_MAX_ERROR_MM,
              "false_accept_definition": "accepted on a labelled negative condition; excessive positive-control error is counted separately",
              "limits": ["Software development controls, not physical tolerances.",
                         "Timing includes per-frame preprocessing and registration; first frame includes reference setup.",
                         "Timings exclude shared rendering/decoding, imports, and artifact output; offline capture age uses original replay time.",
                         "Dense ICP candidate does not establish material identity.",
                         "Synthetic ideal depth is not a physical noise model."], "cases": {}}
    rows = []
    for name, samples in ([] if args.skip_synthetic else synthetic_cases()):
        result = compare_frames(samples, dense=not args.sparse_only, output=args.output, label=name)
        report["cases"][name] = summarize(result)
        rows.extend(result)
    for i, recording in enumerate(args.recording):
        source = recorded_frames(recording, args.sensor,
                                 max_frames=args.recording_max_frames*args.recording_stride,
                                 roi=args.roi, depth_range=args.depth_range)
        samples = ((f, None, None) for j, f in enumerate(source) if j % args.recording_stride == 0)
        name = f"recording-{i}"
        result = compare_frames(samples, dense=not args.sparse_only, output=args.output, label=name)
        report["cases"][name] = summarize(result)
        report["cases"][name]["recording_path"] = str(recording.resolve())
        report["cases"][name]["invalid_frames"] = sum("invalid" in r for r in result)
        rows.extend(result)
    if args.sustained_frames and not args.skip_synthetic:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        sustained = compare_frames(sustained_samples(args.sustained_frames), dense=False,
                                   output=args.output, label="sustained")
        report["sustained"] = summarize(sustained)
        report["sustained"]["maxrss_before_kib"] = before
        report["sustained"]["maxrss_after_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["sustained"]["distinct_reference_digests"] = len({r["methods"]["sparse"]["reference_digest"] for r in sustained})
        rows.extend(sustained)
    report["summary"] = summarize(rows)
    report["unavailable_input_frames"] = sum("invalid" in r for r in rows)
    report["metrics_by_condition"] = {condition: summarize([r for r in rows if r["condition"] == condition])
                                      for condition in sorted({r["condition"] for r in rows})}
    (args.output/"observations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output / 'report.json'}")
    return 0


def run_cli():
    import tatbot_runlog
    with tatbot_runlog.init("surface-attachment-benchmark", prune_first=False):
        return main()


if __name__ == "__main__":
    raise SystemExit(run_cli())
