#!/usr/bin/env python3
"""Offline sparse material-point experiment. No pose estimate or motion authority.

Reads one sensor's visiond frames.jsonl, or a deterministic synthetic suite.
Tracks are seeded once inside the explicit ROI. Default loss is latched;
--recover pauses on blur and retries the last clear reference for at most 750 ms.
Newly detected corners never inherit old material identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
import tatbot_rerun as tr  # noqa: E402
import tatbot_runlog  # noqa: E402
from surface_match import KeyframeBank, MatchSettings, PatchMatcher  # noqa: E402
from visiond_wire import read_evidence_frame  # noqa: E402


@dataclass(frozen=True)
class Settings:
    max_points: int = 200
    min_points: int = 12
    max_gap_ms: float = 150.0
    fb_limit_px: float = 1.0
    patch_error_limit: float = 20.0
    crop_margin_px: int = 96


class SparsePatchTracker:
    def __init__(self, roi, settings=None):
        self.roi = tuple(roi)
        self.settings = settings or Settings()
        self.previous = None
        self.points = np.empty((0, 1, 2), np.float32)
        self.ids = np.empty(0, dtype=int)
        self.anchors = np.empty((0, 2), np.float32)
        self.timestamp_ns = None
        self.loss = None

    def seed_points(self, image, timestamp_ns, points):
        """Initialize externally verified image correspondences without redetection."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xy = np.asarray(points, np.float32).reshape(-1, 2)
        if gray.dtype != np.uint8 or gray.ndim != 2 or not np.isfinite(xy).all():
            raise ValueError("verified seed requires an 8-bit image and finite points")
        if not self.settings.min_points <= len(xy) <= self.settings.max_points:
            raise ValueError("verified seed count is outside tracker bounds")
        if np.any(xy < 0) or np.any(xy >= (gray.shape[1], gray.shape[0])):
            raise ValueError("verified seed points must be inside the image")
        self.previous = gray.copy()
        self.points = xy.reshape(-1, 1, 2).copy()
        self.ids = np.arange(len(xy))
        self.anchors = xy.copy()
        self.timestamp_ns = int(timestamp_ns)
        self.loss = None

    def step(self, image, timestamp_ns):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("tracker requires an 8-bit image")
        if self.timestamp_ns is not None:
            gap_ms = (timestamp_ns - self.timestamp_ns) / 1e6
            if gap_ms <= 0 or gap_ms > self.settings.max_gap_ms:
                self.loss = self.loss or "timestamp_gap"
            if gray.shape != self.previous.shape:
                self.loss = self.loss or "geometry_changed"
        if self.loss:
            return self._result("lost", self.loss)
        if self.previous is None:
            x, y, w, h = self.roi
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                raise ValueError("ROI must fit inside the image")
            mask = np.zeros_like(gray)
            mask[y:y+h, x:x+w] = 255
            points = cv2.goodFeaturesToTrack(gray, self.settings.max_points, 0.02, 7, mask=mask, blockSize=7)
            self.points = points if points is not None else self.points
            self.ids = np.arange(len(self.points))
            self.anchors = self.points.reshape(-1, 2).copy()
            if len(self.points) < self.settings.min_points:
                self.loss = "insufficient_texture"
            status = "lost" if self.loss else "initialized"
        else:
            options = {"winSize": (21, 21), "maxLevel": 3,
                       "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)}
            # Build optical-flow pyramids only around the surviving patch. The
            # common crop origin keeps old/new pixel coordinates comparable.
            points_xy = self.points.reshape(-1, 2)
            margin = self.settings.crop_margin_px
            x0, y0 = np.maximum(0, np.floor(points_xy.min(axis=0) - margin)).astype(int)
            x1, y1 = np.minimum((gray.shape[1], gray.shape[0]),
                               np.ceil(points_xy.max(axis=0) + margin + 1)).astype(int)
            origin = np.array([x0, y0], dtype=np.float32)
            local = self.points - origin
            before, after = self.previous[y0:y1, x0:x1], gray[y0:y1, x0:x1]
            forward, ok, error = cv2.calcOpticalFlowPyrLK(before, after, local, None, **options)
            backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, **options)
            forward += origin
            xy = forward.reshape(-1, 2)
            fb = np.linalg.norm(backward - local, axis=2).ravel()
            keep = (ok.ravel() != 0) & (back_ok.ravel() != 0)
            keep &= np.isfinite(xy).all(axis=1) & np.isfinite(fb)
            keep &= (fb <= self.settings.fb_limit_px) & (error.ravel() <= self.settings.patch_error_limit)
            keep &= (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
            self.points, self.ids = forward[keep], self.ids[keep]
            if len(self.points) < self.settings.min_points:
                self.loss = "insufficient_tracks"
            status = "lost" if self.loss else "tracked"
        self.previous = gray.copy()
        self.timestamp_ns = timestamp_ns
        return self._result(status, self.loss)

    def _result(self, status, reason):
        valid = status != "lost"
        return {"status": status, "reason": reason,
                "ids": self.ids.tolist() if valid else [],
                "points_px": self.points.reshape(-1, 2).tolist() if valid else [],
                "surviving_points": len(self.points), "motion_authority": False}


class RecoveringPatchTracker:
    """Bounded retries against a clear reference, never a fresh feature detection.

    Coordinates are withheld during recovery. Two direct reference matches are
    required before publishing the old feature IDs again. This is image evidence,
    not proof of physical identity on repeating textures or motion authority.
    """
    def __init__(self, roi, settings=None, recovery_ms=750.0):
        self.settings = settings or Settings()
        if not np.isfinite(recovery_ms) or not 150 <= recovery_ms <= 2000:
            raise ValueError("recovery_ms must be finite and within 150..2000")
        self.recovery_ms = recovery_ms
        self.core = SparsePatchTracker(roi, replace(self.settings, max_gap_ms=recovery_ms))
        self.last_input_ns = None
        self.last_attempt_ns = None
        self.loss = None
        self.recovering = False
        self.confirmations = 0
        self.reference_sharpness = None
        self.sharpness_ratio = None

    @property
    def anchors(self):
        return self.core.anchors

    @staticmethod
    def sharpness(gray, points):
        xy = points.reshape(-1, 2)
        low = np.maximum(0, np.floor(xy.min(axis=0))).astype(int)
        high = np.minimum((gray.shape[1], gray.shape[0]), np.ceil(xy.max(axis=0) + 1)).astype(int)
        patch = gray[low[1]:high[1], low[0]:high[0]]
        return float(cv2.Laplacian(patch, cv2.CV_32F).var()) if patch.size else 0.0

    def unavailable(self, status, reason):
        return {"status": status, "reason": reason, "ids": [], "points_px": [],
                "surviving_points": 0, "reference_points": len(self.core.points), "motion_authority": False,
                "sharpness_ratio": self.sharpness_ratio,
                "reference_timestamp_ns": self.core.timestamp_ns,
                "recovery_confirmations": self.confirmations}

    def step(self, image, timestamp_ns):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("tracker requires an 8-bit image")
        if self.last_input_ns is not None:
            gap = (timestamp_ns - self.last_input_ns) / 1e6
            if gap <= 0 or gap > self.settings.max_gap_ms:
                self.loss = self.loss or "timestamp_gap"
            if gray.shape != self.core.previous.shape:
                self.loss = self.loss or "geometry_changed"
        self.last_input_ns = timestamp_ns
        if self.loss:
            return self.unavailable("lost", self.loss)
        if self.core.previous is None:
            result = self.core.step(gray, timestamp_ns)
            self.loss = self.core.loss
            if self.loss is None:
                self.reference_sharpness = self.sharpness(gray, self.core.points)
            return result
        reference_age_ms = (timestamp_ns - self.core.timestamp_ns) / 1e6
        if reference_age_ms > self.recovery_ms:
            self.loss = "recovery_timeout"
            return self.unavailable("lost", self.loss)
        sharpness = self.sharpness(gray, self.core.points)
        self.sharpness_ratio = sharpness / max(self.reference_sharpness, 1e-6)
        # Refuse to turn a blurred image into the next reference. Variance is a
        # cheap warning signal, not a universal calibrated blur measurement.
        if self.sharpness_ratio < 0.35:
            self.recovering = True
            self.confirmations = 0
            return self.unavailable("paused", "blur_or_visibility_loss")
        if self.recovering and self.last_attempt_ns is not None and timestamp_ns - self.last_attempt_ns < 100_000_000:
            return self.unavailable("paused", "recovery_rate_limited")
        self.last_attempt_ns = timestamp_ns
        # step() replaces these arrays; retain references without duplicating
        # full-resolution pixels on every failed recovery attempt.
        checkpoint = self.core.__dict__.copy()
        result = self.core.step(gray, timestamp_ns)
        retained = len(result["ids"]) / max(len(checkpoint["ids"]), 1)
        valid = result["status"] == "tracked" and retained >= 0.6
        if not valid:
            self.core.__dict__.update(checkpoint)
            self.recovering = True
            self.confirmations = 0
            return self.unavailable("paused", "reference_match_failed")
        if self.recovering:
            self.confirmations += 1
            if self.confirmations < 2:
                self.core.__dict__.update(checkpoint)
                return self.unavailable("paused", "recovery_confirmation")
            result["status"] = "reacquired"
            result["reason"] = "two_reference_matches"
        self.recovering = False
        self.confirmations = 0
        self.reference_sharpness = self.sharpness(gray, self.core.points)
        result.update(sharpness_ratio=self.sharpness_ratio, reference_timestamp_ns=timestamp_ns)
        return result


def recording_frames(index):
    """Stream a single sensor index; do not decode unused cameras or open sockets."""
    sensor = None
    with index.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            metadata = entry["metadata"]
            if sensor is not None and metadata["sensor_name"] != sensor:
                raise ValueError("index changes sensor identity")
            sensor = metadata["sensor_name"]
            timestamp = metadata["timestamps"].get("normalized_unix_ns")
            if timestamp is None or int(timestamp) <= 0:
                raise ValueError("replay requires normalized_unix_ns; no clock-domain guessing")
            frame = read_evidence_frame(index.parent, entry, preserve_luma=True)
            attributes = metadata.get("attributes", {})
            yield frame["image"], int(timestamp), {"sensor": sensor, "sequence": metadata["sequence"],
                  "sha256": entry["sha256"], "calibration_id": metadata.get("calibration_id"),
                  "capture_epoch": attributes.get("capture_epoch"), "profile": metadata.get("profile"),
                  "intrinsics": attributes.get("intrinsics")}, None


def synthetic_frames(case, width=640, height=480, count=90):
    """Known image-space motion, not a simulated camera or physical surface claim."""
    rng = np.random.default_rng(23)
    base = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (5, 5), 0)
    if case == "alias":
        base = np.tile(base[:, :16], (1, (width + 15) // 16))[:, :width]
        count = min(count, 10)
    for i in range(count):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), (90 if i >= count // 3 else 0) if case == "turn" else i * 0.03, 1.0)
        if case == "scale_turn":
            phase = min(1.0, i / max(count // 2, 1))
            angle, scale = (60, 0.55) if i >= 2 * count // 3 else (30 * phase, 1 - 0.2 * phase)
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += (i * 0.7, i * 0.25)
        if case == "alias":
            matrix = np.array([[1.0, 0.0, i * 16.0], [0.0, 1.0, 0.0]])
        frame = cv2.warpAffine(base, matrix, (width, height),
                               borderMode=cv2.BORDER_CONSTANT if case == "scale_turn" else cv2.BORDER_WRAP)
        if case == "blank":
            frame[:] = 127
        elif case == "occlusion" and count // 3 <= i < 2 * count // 3:
            frame[:] = 0
        elif case == "blur" and count // 3 <= i < count // 3 + 6:
            frame = cv2.GaussianBlur(frame, (41, 41), 0)
        elif case == "short_occlusion" and count // 3 <= i < count // 3 + 5:
            frame[:] = 0
        elif case == "brightness":
            frame = np.clip(frame.astype(float) + 12 * np.sin(i / 8), 0, 255).astype(np.uint8)
        timestamp = 1_000_000_000 + i * 33_333_333
        if case == "gap" and i >= count // 3:
            timestamp += 500_000_000
        yield frame, timestamp, {"synthetic": case, "sequence": i}, matrix


def percentiles(values):
    return {key: float(np.percentile(values, q)) if values else None
            for key, q in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


def evaluate(frames, roi, settings, output, rr=None, name="replay", max_frames=300, recover=False, match=False, keyframes=False):
    tracker = RecoveringPatchTracker(roi, settings) if recover else SparsePatchTracker(roi, settings)
    times, active_times, decode_times, errors, states = [], [], [], [], Counter()
    state_wall, state_cpu = {}, {}
    matcher = (KeyframeBank if keyframes else PatchMatcher)(roi, MatchSettings(method=match if isinstance(match, str) else "orb")) if match else None
    match_times, match_errors, match_states = [], [], Counter()
    admission_reasons = Counter()
    peak_keyframes = 1 if matcher else 0
    first_unavailable = None
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    next_visual_ns = 0
    first_stamp = last_stamp = None
    iterator = iter(frames)
    with output.open("w") as stream:
        for _ in range(max_frames):
            start = time.perf_counter()
            try:
                image, stamp, provenance, matrix = next(iterator)
            except StopIteration:
                break
            decode_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            step_cpu_start = time.process_time()
            result = tracker.step(image, stamp)
            step_cpu_ms = (time.process_time() - step_cpu_start) * 1000
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            state_wall.setdefault(result["status"], []).append(elapsed)
            state_cpu.setdefault(result["status"], []).append(step_cpu_ms)
            states[result["status"]] += 1
            if result["status"] in ("tracked", "reacquired"):
                active_times.append(elapsed)
            if first_stamp is None:
                first_stamp = stamp
            last_stamp = stamp
            if first_unavailable is None and result["status"] in ("paused", "lost"):
                first_unavailable = {"source_time_s": (stamp - first_stamp) / 1e9,
                                     "status": result["status"], "reason": result["reason"]}
            result.update(timestamp_ns=stamp, processing_ms=elapsed, source=provenance)
            if matcher is not None:
                gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                match_start = time.perf_counter()
                if matcher.shape is None:
                    features = matcher.seed(gray)
                    matcher.next_attempt_ns = stamp + (100 if keyframes else matcher.settings.interval_ms) * 1_000_000
                    candidate = {"status": "seeded", "reference_features": features}
                else:
                    candidate = matcher.match(gray, stamp, result, tracker.anchors) if keyframes else matcher.match(gray, stamp)
                match_ms = (time.perf_counter() - match_start) * 1000
                candidate["processing_ms"] = match_ms
                if matrix is not None and candidate.get("candidate_polygon_px"):
                    x, y, w, h = roi
                    corners = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                    expected = corners @ matrix[:,:2].T + matrix[:,2]
                    corner_errors = np.linalg.norm(np.array(candidate["candidate_polygon_px"])-expected, axis=1)
                    match_errors.extend(corner_errors.tolist())
                    candidate["ground_truth_corner_error_px"] = percentiles(corner_errors.tolist())
                result["keyframe_match"] = candidate
                match_states[candidate["status"]] += 1
                peak_keyframes = max(peak_keyframes, candidate.get("keyframes", 1))
                if keyframes:
                    reason = candidate.get("admission_reason") or candidate.get("reason")
                    if reason and reason != "rate_limited":
                        admission_reasons[reason] += 1
                if candidate["status"] != "skipped":
                    match_times.append(match_ms)
            if matrix is not None and result["ids"]:
                anchors = tracker.anchors[result["ids"]]
                expected = anchors @ matrix[:, :2].T + matrix[:, 2]
                frame_errors = np.linalg.norm(np.asarray(result["points_px"]) - expected, axis=1)
                errors.extend(frame_errors.tolist())
                result["ground_truth_error_px"] = percentiles(frame_errors.tolist())
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            visual_ready = (stamp >= next_visual_ns if matcher is None
                            else result["keyframe_match"]["status"] != "skipped")
            if rr is not None and visual_ready and (not keyframes or stamp >= next_visual_ns):
                # Synthetic relative timestamps must not masquerade as fleet wall time.
                rr.set_time("surface_replay_seconds", duration=(stamp - first_stamp) / 1e9)
                path = f"surface/replay/{name}"
                rr.log(path + "/image", rr.Image(image))
                rr.log(path + "/image/tracks", rr.Points2D(np.asarray(result["points_px"]).reshape(-1, 2),
                                                         labels=[str(i) for i in result["ids"]]))
                rr.log(path + "/status", rr.TextLog(f"{result['status']}: {result['reason']}"))
                rr.log(path + "/processing_ms", rr.Scalars(elapsed))
                if matcher is not None:
                    candidate = result["keyframe_match"]
                    polygon = candidate.get("candidate_polygon_px", [])
                    strips = [polygon + [polygon[0]]] if polygon else []
                    rr.log(path + "/image/candidate", rr.LineStrips2D(strips))
                    rr.log(path + "/match", rr.TextLog(json.dumps(candidate)))
                next_visual_ns = stamp + 200_000_000
    if not times:
        raise ValueError("recording contains no frames")
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    duration = (last_stamp - first_stamp) / 1e9
    return {"frames": len(times), "states": dict(states), "tracking_ms": percentiles(times),
            "active_tracking_ms": percentiles(active_times),
            "first_tracking_unavailable": first_unavailable,
            "appearance_bank_enabled": keyframes, "peak_keyframes": peak_keyframes,
            "appearance_bank_reasons": dict(admission_reasons),
            "keyframe_match_settings": asdict(matcher.settings) if matcher else None,
            "keyframe_match_states": dict(match_states), "keyframe_match_ms": percentiles(match_times),
            "keyframe_corner_error_px": percentiles(match_errors),
            "state_processing": {state: {"samples": len(values), "wall_ms": percentiles(values),
                                         "cpu_ms": percentiles(state_cpu[state])}
                                 for state, values in state_wall.items()},
            "input_read_decode_or_generate_ms": percentiles(decode_times),
            "ground_truth_point_error_px": percentiles(errors), "cpu_seconds": cpu, "wall_seconds": wall,
            "mean_cpu_ms_per_frame": cpu * 1000 / len(times),
            "estimated_core_fraction_at_source_rate": cpu / duration if duration > 0 else None,
            "source_duration_s": duration, "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "final_reason": tracker.loss, "metric_3d_evaluated": False,
            "latency_basis": "offline processing; excludes live capture, transport and competing services"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording", type=Path, help="one sensor's visiond frames.jsonl")
    source.add_argument("--synthetic", action="store_true", help="deterministic motion and failure suite")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640, help="synthetic image width")
    parser.add_argument("--height", type=int, default=480, help="synthetic image height")
    parser.add_argument("--match-keyframes", action="store_true", help="experimental three-reference appearance bank")
    parser.add_argument("--match", choices=("orb", "sift"), help="periodic advisory keyframe search")
    parser.add_argument("--recover", action="store_true", help="bounded blur pause and reference-only reacquisition")
    parser.add_argument("--rerun", action="store_true", help="save a capped replay RRD in the run directory")
    args = parser.parse_args()
    if args.match_keyframes and not (args.match and args.recover):
        parser.error("--match-keyframes requires --match and --recover")
    if args.max_frames < 3 or args.max_frames > 10000 or min(args.width, args.height) < 128 or max(args.width, args.height) > 4096:
        parser.error("require 3–10000 frames and synthetic dimensions 128–4096")
    if args.recording and not args.roi:
        parser.error("--recording requires an explicit --roi X Y W H")
    cv2.setNumThreads(1)
    settings = Settings()
    with tatbot_runlog.init("surface-replay", prune_first=False) as run:
        rr = tr.start("surface_replay", output=run.dir / "replay.rrd", recording_id=run.run_id) if args.rerun else None
        report = {"schema_version": 1, "settings": asdict(settings), "opencv": cv2.__version__,
                  "opencv_threads": cv2.getNumThreads(), "python": platform.python_version(),
                  "machine": platform.machine(), "processor": platform.processor(),
                  "input_kind": "synthetic_suite" if args.synthetic else "unverified_recording",
                  "cases": {}, "motion_authority": False, "recovery_enabled": args.recover, "keyframe_search_enabled": args.match, "appearance_bank_enabled": args.match_keyframes}
        if args.synthetic:
            roi = args.roi or (args.width // 4, args.height // 4, args.width // 2, args.height // 2)
            sources = {case: synthetic_frames(case, args.width, args.height, args.max_frames)
                       for case in ("motion", "brightness", "blur", "short_occlusion", "turn", "scale_turn", "occlusion", "blank", "gap", "alias")}
        else:
            roi = args.roi
            index = args.recording.expanduser().resolve()
            report["input"] = {"path": str(index), "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
            sources = {"recording": recording_frames(index)}
        report["implementation_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("surface_replay.py", "visiond_wire.py", "surface_match.py")
        }
        report["roi"] = roi
        report["dimensions"] = [args.width, args.height] if args.synthetic else None
        for name, frames in sources.items():
            report["cases"][name] = evaluate(frames, roi, settings, run.dir / f"{name}.jsonl", rr, name, args.max_frames, args.recover, args.match, args.match_keyframes)
        if rr is not None:
            rr.get_global_data_recording().flush()
        (run.dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2))
        print(f"Report: {run.dir / 'report.json'}")


if __name__ == "__main__":
    main()
