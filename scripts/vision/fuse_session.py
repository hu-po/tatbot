#!/usr/bin/env python3
"""Fuse a sweep session's streams into solver-ready samples (the missing link).

    fuse_session.py <session_dir> [--calibration bundle.json]

The sweep writes three artifacts that share nothing but wall-clock time: tag
corners (`shot_*/detections.json` + `timing.json`), the follower flight log
(`teleop.wxtl`), and the narration events (`events.jsonl`). This joins them at
the still intervals — the only moments worth sampling — and classifies each by
evidence, not speech:

  wrist sample    detections of a configured wrist tag inside a guided wrist
                  phase -> pose_NNNN.json for solve_robot_world.py. Phase
                  routing is required because the EE and board reuse IDs.
                  EVERY sighting is recorded as raw normalized corners — one
                  camera is enough for the solver's corner-reprojection mode;
                  >=2 cameras additionally yield a triangulated tag pose.
  tip hold        guided tip-phase windows -> touches.json tip_holds, the
                  stills with the pen tip planted on one point. Since
                  2026-08-26 that point is on the paper pad and every hold
                  counts; archived sessions planted on the palette tag, with
                  their "pad" holds hovering (waypoint only, never a touch)
  neither         kept in intervals.json for the audit trail

Triangulated poses come from CORNER TRIANGULATION, not per-camera PnP
(single-tag IPPE branch-flips; the board calibration measured 20-29 mm from
exactly that), then a Procrustes fit of the known square. Phantom defense for
corner observations is the few-sightings filter plus the static-tag guards —
the old >=2-camera rule no longer gates recording.

Speech is notes-only; 'scratch that' (discard) is the one surviving control.
Effort-based contact classification and pivot windows remain as legacy paths
for pre-2026-08-22 sessions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from tatbot_runlog import log_root  # noqa: E402
from teleop_log import TeleopLog  # noqa: E402
from urdf_kinematics import driver_joint_names  # noqa: E402

_INVENTORY = load_inventory()
BOARD_AND_PALETTE_IDS = set(_INVENTORY.target("board").ids) | set(
    _INVENTORY.target("palette").ids
)
PAD_WORDS = re.compile(r"\b(pad|paper)\b")


# --- camera model ----------------------------------------------------------

class Camera:
    def __init__(self, name, cal):
        self.name = name
        intr = cal["intrinsics"]
        self.fx, self.fy = float(intr["fx"]), float(intr["fy"])
        self.cx, self.cy = float(intr["cx"]), float(intr["cy"])
        self.dist = np.asarray(cal.get("distortion", {}).get("coefficients", []), float)
        pose = cal["world_from_camera"]
        self.rotation = np.asarray(pose["rotation"], float).reshape(3, 3)
        self.translation = np.asarray(pose["translation_m"], float)

    def undistort(self, pixel):
        """Pixel -> normalized image coordinates, inverting Brown-Conrady.

        Implemented here so the fuser needs numpy only (it runs where
        the vision venv has no cv2 promise). The forward model matches what
        calibrateCamera fitted; 8 fixed-point iterations converge far below a
        millipixel at these distortion magnitudes.
        """
        x = (pixel[0] - self.cx) / self.fx
        y = (pixel[1] - self.cy) / self.fy
        if not len(self.dist):
            return np.array([x, y])
        k = np.zeros(8)
        k[:len(self.dist)] = self.dist[:8]
        xd, yd = x, y
        for _ in range(8):
            r2 = x * x + y * y
            radial = (1 + k[0] * r2 + k[1] * r2 ** 2 + k[4] * r2 ** 3)
            if len(self.dist) > 5:
                radial /= (1 + k[5] * r2 + k[6] * r2 ** 2 + k[7] * r2 ** 3)
            dx = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x)
            dy = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return np.array([x, y])

    def ray(self, pixel):
        """World-frame origin and unit direction through an (undistorted) pixel."""
        normalized = self.undistort(pixel)
        direction = self.rotation @ np.array([normalized[0], normalized[1], 1.0])
        return self.translation, direction / np.linalg.norm(direction)

    def reproject_error_px(self, world_point, pixel):
        cam_point = self.rotation.T @ (world_point - self.translation)
        if cam_point[2] <= 1e-6:
            return float("inf")
        predicted = cam_point[:2] / cam_point[2]
        measured = self.undistort(pixel)
        delta = predicted - measured
        return float(np.hypot(self.fx * delta[0], self.fy * delta[1]))


def triangulate(rays):
    """Point minimizing distance to all rays: A p = b with A = sum(I - d d^T)."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in rays:
        m = np.eye(3) - np.outer(direction, direction)
        a += m
        b += m @ origin
    # Near-parallel rays (cameras looking along one line) leave A rank-2.
    if np.linalg.cond(a) > 1e6:
        return None
    return np.linalg.solve(a, b)


def fit_square(world_corners, edge_m):
    """Procrustes-fit the tag's model square to 4 triangulated corners.

    Corner order is the detector's (TL, TR, BR, BL in the tag frame). The
    convention only has to be consistent across observations — AX=ZB absorbs
    any fixed choice into X.
    """
    model = tag_model_corners(edge_m)
    world = np.asarray(world_corners, float)
    centroid_w = world.mean(axis=0)
    centroid_m = model.mean(axis=0)
    h = (model - centroid_m).T @ (world - centroid_w)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = centroid_w - rotation @ centroid_m
    fitted = (rotation @ model.T).T + pose[:3, 3]
    rms_mm = float(np.sqrt(np.mean(np.sum((fitted - world) ** 2, axis=1))) * 1000)
    return pose, rms_mm


def measured_edge_m(world_corners):
    sides = [np.linalg.norm(world_corners[i] - world_corners[(i + 1) % 4])
             for i in range(4)]
    return float(np.mean(sides))


# --- session loading -------------------------------------------------------

def load_bundle(path):
    bundle = json.loads(Path(path).expanduser().read_text())
    return {name: Camera(name, cal) for name, cal in bundle["cameras"].items()}


def load_shots(session):
    shots = []
    for shot_dir in sorted(Path(session).glob("shot_*")):
        detections = shot_dir / "detections.json"
        timing = shot_dir / "timing.json"
        if not (detections.is_file() and timing.is_file()):
            continue
        shots.append({
            "name": shot_dir.name,
            "unix_seconds": json.loads(timing.read_text())["unix_seconds"],
            "detections": json.loads(detections.read_text()),
        })
    return shots


def load_events(session):
    path = Path(session) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_wrist_config(repo):
    """Return wrist identity and size from the canonical fiducial inventory."""
    inventory = load_inventory(Path(repo) / "config" / "fiducials.json")
    wrist = inventory.target("wrist")
    return {"ids": list(wrist.ids), "edge_m": wrist.edge_m}


# --- fusion ----------------------------------------------------------------

def fuse_tag(observations, cameras, edge_m, max_reproj_px, min_cameras):
    """observations: list of (camera_name, corners 4x2). Returns (pose, meta)."""
    world_corners = []
    used_cameras = set()
    for corner_index in range(4):
        rays, sources = [], []
        for camera_name, corners in observations:
            camera = cameras.get(camera_name)
            if camera is None:
                continue
            rays.append(camera.ray(np.asarray(corners[corner_index], float)))
            sources.append((camera, np.asarray(corners[corner_index], float)))
        if len({id(s[0]) for s in sources}) < min_cameras:
            return None, {"reason": f"corner {corner_index}: fewer than "
                                    f"{min_cameras} cameras"}
        point = triangulate(rays)
        if point is None:
            return None, {"reason": f"corner {corner_index}: degenerate ray geometry"}
        # One bad camera (blur, mis-detection) should not poison the corner:
        # drop observations that disagree with the consensus point, as long as
        # two cameras remain.
        keep = [(camera, pixel) for camera, pixel in sources
                if camera.reproject_error_px(point, pixel) <= max_reproj_px]
        if len({id(c) for c, _ in keep}) < min_cameras:
            return None, {"reason": f"corner {corner_index}: reprojection "
                                    f"> {max_reproj_px} px"}
        if len(keep) != len(sources):
            point = triangulate([camera.ray(pixel) for camera, pixel in keep])
            if point is None:
                return None, {"reason": f"corner {corner_index}: degenerate after cull"}
        world_corners.append(point)
        used_cameras.update(camera.name for camera, _ in keep)
    world_corners = np.array(world_corners)
    estimated_edge = measured_edge_m(world_corners)
    pose, rms_mm = fit_square(world_corners, edge_m or estimated_edge)
    return pose, {"rms_mm": rms_mm, "edge_est_m": round(estimated_edge, 5),
                  "cameras": sorted(used_cameras)}


def in_any_window(unix_seconds, windows):
    return any(start <= unix_seconds <= end for start, end in windows)


def phase_windows(timeline, phase, lead_s=0.4, tail_s=0.2, settle_s=0.0):
    """Capture-time windows for one guided phase.

    ID-reusing hardware must be routed by these windows rather than by
    subtracting another target's IDs. ``settle_s`` deliberately drops frames
    near a guided hold boundary: RTSP decode can deliver an earlier buffered
    frame with a current receive timestamp, so sharpness alone does not prove
    that an arrival frame matches the encoder pose.
    """
    if not timeline:
        return []
    return [
        (entry["start_unix"] + settle_s - lead_s,
         entry["end_unix"] + tail_s)
        for entry in timeline.get("entries", [])
        if entry.get("phase") == phase
    ]


def filter_shot_dirs(shot_dirs, window_start, window_end, windows=None):
    """Shots whose capture time falls inside [window_start, window_end].

    Used by calibrate_board_session.py to solve ONLY on board-phase shots: in
    a guided session the board is put away for the arm phases, and without
    its board-only context a lone shared-ID sighting could be classified as
    the 44 mm board copy and contaminate the solve. Shots with
    no timing.json (the pose-at-a-time capture path) are kept — that path
    never mixes phases. Lives here, not in the board session, so it stays
    importable and testable without cv2.
    """
    kept = []
    for shot_dir in shot_dirs:
        timing = Path(shot_dir) / "timing.json"
        if not timing.is_file():
            kept.append(shot_dir)
            continue
        unix = json.loads(timing.read_text()).get("unix_seconds", 0.0)
        if windows is not None:
            if in_any_window(unix, windows):
                kept.append(shot_dir)
        elif window_start <= unix <= window_end:
            kept.append(shot_dir)
    return kept


def load_timeline(session):
    """session_guide.py's script of the session, when it conducted one."""
    path = Path(session) / "guide_timeline.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def overlap_s(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def guided_touches(timeline, log, events):
    """Touch samples straight from the guide's paced windows.

    The 2026-08-21 session showed why classification cannot do this: pen-press
    effort (~1 Nm) sits inside the pose-dependent gravity-comp residual
    (0.9-7 Nm free-space on the same log), and pressing trembles past the
    free-space stillness gate. The script says WHEN; the log supplies the
    quietest joints in that window; the plate-fit residual (with per-touch
    listing) is what catches a touch that never landed. 'scratch that' still
    drops the touch it points at.
    """
    windows = [e for e in timeline.get("entries", []) if e.get("kind") == "touch"]
    touches, missed = [], []
    for window in windows:
        sample = log.window_sample(window["start_unix"] - 0.5,
                                   window["end_unix"] + 1.0)
        if sample is None:
            missed.append({"index": window.get("index"),
                           "label": window.get("label")})
            continue
        touches.append({
            # all seven driver joints (2026-08-30): the tool mount rides the
            # carriage, so the tip's FK needs its reading too
            "joints": sample["joints"][:7],
            "start_unix": sample["start_unix"],
            "end_unix": sample["end_unix"],
            "duration_s": sample["duration_s"],
            "arm_eff_med_nm": sample["arm_eff_med_nm"],
            "spread_rad": sample["spread_rad"],
            "label": window.get("label", "plate"),
            "source": "guided",
        })
    for index in sorted(discarded_indices(touches, events, max_gap_s=10.0),
                        reverse=True):
        touches.pop(index)
    return touches, missed


def guided_pivots(timeline, log, sample_hz=20.0, min_span_s=3.0):
    """Continuous joint samples from each planted-tip pivot window.

    No stillness wanted here — the whole point is that the wrist ROLLS while
    the tip stays on the palette tag center, so every tick is an observation.
    The guide stamps the window after its go-instruction, so the approach is
    already excluded; a small pad skips the last of the announcement audio.
    """
    pivots, missed = [], []
    stride = max(1, int(round(1.0 / (sample_hz * max(log.period_s, 1e-4)))))
    for entry in timeline.get("entries", []):
        if entry.get("kind") != "pivot":
            continue
        mask = (log.unix_seconds >= entry["start_unix"] + 0.3) \
            & (log.unix_seconds <= entry["end_unix"])
        indices = np.nonzero(mask)[0][::stride]
        span = (float(log.unix_seconds[indices[-1]] - log.unix_seconds[indices[0]])
                if len(indices) > 1 else 0.0)
        seq = log.follower_pos[indices][:, :6]
        travel = float((seq.max(axis=0) - seq.min(axis=0)).max()) if len(seq) else 0.0
        if span < min_span_s or len(indices) < 20:
            missed.append({"index": entry.get("index"), "reason": "window too short"})
            continue
        if travel < 0.05:
            missed.append({"index": entry.get("index"),
                           "reason": f"wrist barely moved ({travel:.3f} rad) — "
                                     "roll more next time"})
            continue
        pivots.append({
            "joints_seq": [row.tolist() for row in seq],
            "start_unix": float(log.unix_seconds[indices[0]]),
            "end_unix": float(log.unix_seconds[indices[-1]]),
            "n": int(len(seq)),
            "travel_rad": travel,
        })
    return pivots, missed


# Tip-hold labels whose sample is the tip PLANTED on a surface, and which
# surface that is. "pad" is absent on purpose: in sessions before 2026-08-26 it
# meant a hover above the pad, so counting one as a touch would put a waypoint's
# worth of air into the solve and, worse, report a paper_plane_z that nothing
# ever touched. Archived sessions must keep fusing exactly as they did.
PLANTED_TIP_LABELS = {"pad_planted": "pad", "palette": "palette"}


def guided_tip(timeline, log):
    """Tip-phase holds. Every planted hold is a pen-tip observation; the label
    says which surface it was planted on, and pre-2026-08-26 "pad" hovers stay
    waypoint-only. Stillness starts AT the high beep, which is the window's
    start stamp."""
    tip_holds = []
    waypoints = {"pad": [], "palette": []}
    surfaces = set()
    missed = []
    for entry in timeline.get("entries", []):
        if entry.get("kind") != "tip_hold":
            continue
        sample = log.window_sample(entry["start_unix"],
                                   entry["end_unix"] + 0.5)
        label = entry.get("label", "pad")
        if sample is None:
            missed.append({"index": entry.get("index"),
                           "label": label,
                           "reason": "no quiet window — hold stiller after "
                                     "the high beep"})
            continue
        if label in waypoints:
            waypoints[label].append(sample["joints"][:6])
        surface = PLANTED_TIP_LABELS.get(label)
        if surface is None:
            continue
        surfaces.add(surface)
        tip_holds.append({
            "joints": sample["joints"][:7],
            "start_unix": sample["start_unix"],
            "end_unix": sample["end_unix"],
            "duration_s": sample["duration_s"],
            "spread_rad": sample["spread_rad"],
            "arm_eff_med_nm": sample["arm_eff_med_nm"],
            "surface": surface,
        })
    # One surface per session, or none we can name: a mixed set would make
    # the solved pivot a point on neither surface.
    surface = surfaces.pop() if len(surfaces) == 1 else None
    return tip_holds, waypoints, missed, surface


def extract_poses(timeline, intervals, min_overlap_s=0.3):
    """Named waypoints: each guided pose window takes the still interval it
    overlaps most. Returns ({name: sample}, [missing names])."""
    poses, missing = {}, []
    if not timeline:
        return poses, missing
    for entry in timeline.get("entries", []):
        if entry.get("kind") != "pose":
            continue
        best, best_overlap = None, min_overlap_s
        for interval in intervals:
            got = overlap_s(interval["start_unix"], interval["end_unix"],
                            entry["start_unix"], entry["end_unix"])
            if got > best_overlap:
                best, best_overlap = interval, got
        if best is None:
            missing.append(entry.get("name"))
            continue
        poses[entry["name"]] = {
            "label": entry.get("label", entry["name"]),
            "joints": best["follower_pos"][:6],
            "start_unix": best["start_unix"],
            "duration_s": best["duration_s"],
            "in_contact": bool(best.get("contact")),
        }
    return poses, missing


def interval_events(interval, events):
    """Events said DURING the interval — labels need strict overlap."""
    return [e for e in events
            if e["start_unix"] <= interval["end_unix"]
            and e["end_unix"] >= interval["start_unix"]]


def discarded_indices(intervals, events, max_gap_s=None):
    """'scratch that' drops the interval it overlaps, else the nearest one
    that ended before it — it points backwards at the thing just done.

    `max_gap_s` bounds that backwards reach for SPARSE lists (the guided
    touches): without it, a discard said during the pose tour would delete a
    plate touch from minutes earlier."""
    discarded = set()
    for event in events:
        if "discard" not in event.get("kinds", []):
            continue
        target = None
        for index, interval in enumerate(intervals):
            if (interval["start_unix"] <= event["end_unix"]
                    and interval["end_unix"] >= event["start_unix"]):
                target = index
        if target is None:
            prior = [i for i, interval in enumerate(intervals)
                     if interval["end_unix"] <= event["start_unix"]
                     and (max_gap_s is None or
                          event["start_unix"] - interval["end_unix"] <= max_gap_s)]
            target = prior[-1] if prior else None
        if target is not None:
            discarded.add(target)
    return discarded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--wxtl", default=None, help="flight log (default <session>/teleop.wxtl)")
    ap.add_argument("--calibration", default=None,
                    help="CalibrationBundle (default <session>/calibration.json, "
                         "then ~/tatbot-logs/vision/calibration-current.json)")
    ap.add_argument("--tolerance-rad", type=float, default=0.003)
    ap.add_argument("--min-still", type=float, default=0.5)
    ap.add_argument("--max-reproj-px", type=float, default=3.0)
    ap.add_argument("--min-cameras", type=int, default=2)
    ap.add_argument("--static-tag-mm", type=float, default=3.0,
                    help="a candidate wrist tag whose world position spreads "
                         "less than this across >=3 poses is a stray "
                         "environment tag and is excluded")
    args = ap.parse_args()

    session = Path(args.session_dir).expanduser()
    repo = Path(__file__).resolve().parents[2]

    wxtl = Path(args.wxtl).expanduser() if args.wxtl else session / "teleop.wxtl"
    if not wxtl.is_file():
        sys.exit(f"no flight log at {wxtl} — arm phases need teleop attached")
    log = TeleopLog(wxtl)
    intervals = log.still_intervals(args.tolerance_rad, args.min_still)
    touch_info = log.classify_contacts(intervals)
    events = load_events(session)
    shots = load_shots(session)
    timeline = load_timeline(session)

    wrist_config = load_wrist_config(repo)
    if not wrist_config["ids"]:
        raise ValueError("canonical fiducial inventory has no wrist ids")
    wrist_ids = set(wrist_config["ids"])
    edge_m = wrist_config["edge_m"]
    shared_scene_ids = wrist_ids & BOARD_AND_PALETTE_IDS
    # Wait 0.4 s into the high-beep hold and admit no tail. In the first new
    # four-tag field session, a frame 0.12 s before hold 7 and the frame at the
    # end of that same hold differed by 150-200 px despite identical encoder
    # joints. The early image was a buffered arrival frame and poisoned plain
    # AX=ZB. Four tenths still leaves one capture in 11/12 real full-profile
    # holds and in the synthetic fixture's 0.9 s holds.
    wrist_windows = phase_windows(
        timeline, "wrist", lead_s=0.0, tail_s=0.0, settle_s=0.4
    )
    if shared_scene_ids and timeline is None and shots:
        sys.exit(
            "wrist ids "
            f"{sorted(shared_scene_ids)} overlap board/palette ids; an unguided "
            "mixed-session fuse cannot assign physical targets. Run the guided "
            "wrist phase so guide_timeline.json provides phase windows."
        )

    # Everything below is reproducible from detections + teleop + timeline.
    # Clear prior derived files before regenerating so a stricter second pass
    # cannot leave pose_0012.json from an earlier 13-pose pass beside its new
    # 11-pose output. Stale derived files are more dangerous than no output.
    for path in session.glob("pose_*.json"):
        path.unlink()
    for name in (
        "intervals.json", "poses.json", "report.json", "report.md",
        "robot_world.json", "touches.json", "touchoff_report.json",
    ):
        (session / name).unlink(missing_ok=True)

    cameras = {}
    bundle_path = None
    for candidate in ([args.calibration] if args.calibration else
                      [session / "calibration.json",
                       log_root() / "vision/calibration-current.json"]):
        if candidate and Path(candidate).expanduser().is_file():
            bundle_path = Path(candidate).expanduser()
            cameras = load_bundle(bundle_path)
            break

    print(f"{len(intervals)} still intervals "
          f"({sum(i['contact'] for i in intervals)} in contact, threshold "
          f"{touch_info['threshold_nm']:.2f} Nm), {len(shots)} shots, "
          f"{len(events)} narration events")
    print("wrist tags: config/fiducials.json; edge "
          f"{'%.4f m' % edge_m if edge_m else 'UNMEASURED — will estimate'}")
    if not cameras and shots:
        print("WARNING: no calibration bundle found — wrist samples cannot be "
              "fused (contact samples are unaffected)")

    pose_count = 0
    touches = []
    missed_touches = []
    audit = []
    pending_wrist = []
    discovered_ids = set()
    discarded = discarded_indices(intervals, events)
    pivots, missed_pivots = [], []
    tip_holds, tip_waypoints, missed_tip = [], {"pad": [], "palette": []}, []
    tip_surface = None
    guided_contact = bool(timeline and any(
        entry.get("phase") in {"tip", "touch", "pivot"}
        for entry in timeline.get("entries", [])
    ))
    if guided_contact:
        touches, missed_touches = guided_touches(timeline, log, events)
        pivots, missed_pivots = guided_pivots(timeline, log)
        tip_holds, tip_waypoints, missed_tip, tip_surface = guided_tip(timeline, log)
    for index, interval in enumerate(intervals):
        record = {**interval, "kind": "still", "events": []}
        overlapping = interval_events(interval, events)
        record["events"] = [{"kinds": e.get("kinds", []), "text": e.get("text", "")}
                            for e in overlapping]
        if index in discarded:
            record["kind"] = "discarded"
            audit.append(record)
            continue

        if interval["contact"] and not guided_contact:
            # Unguided fallback only: effort classification worked in
            # synthetic tests but real pen presses hide inside the
            # pose-dependent gravity-comp residual — guided sessions take
            # their touches from the script's windows instead (above).
            label = "plate"
            for event in overlapping:
                if "touch" in event.get("kinds", []) and PAD_WORDS.search(
                        event.get("text", "").lower()):
                    label = "pad"
            touches.append({
                "joints": interval["follower_pos"][:7],
                "start_unix": interval["start_unix"],
                "end_unix": interval["end_unix"],
                "duration_s": interval["duration_s"],
                "arm_eff_med_nm": interval["arm_eff_med_nm"],
                "label": label,
            })
            record["kind"] = f"contact:{label}"
            audit.append(record)
            continue

        # Wrist sample: accepted detection sets inside the interval — plus a
        # short lead-in, because the sweep's motion gate fires on ARRIVAL at
        # a new pose, often a beat before the stillness detector opens the
        # interval (the 2026-08-22 session banked 4 holds but only 2 samples
        # survived this join). Lead-in shots passed the sharpness gate, so
        # the arm was already essentially still when they were taken.
        lead_in = 0.4
        margin = 0.1
        in_window = [s for s in shots
                     if interval["start_unix"] - lead_in <= s["unix_seconds"]
                     <= interval["end_unix"] - margin]
        if timeline is not None:
            in_window = [s for s in in_window
                         if in_any_window(s["unix_seconds"], wrist_windows)]
        observations = {}
        for shot in in_window:
            for camera_name, info in shot["detections"].items():
                candidates = info.get("candidates")
                if candidates is not None:
                    for tag_id_str, groups in candidates.items():
                        tag_id = int(tag_id_str)
                        if tag_id not in wrist_ids or len(groups) != 1:
                            # A guided wrist scene promises one physical copy.
                            # More than one means the board/palette was not
                            # isolated, so refusing this camera/id beats a swap.
                            continue
                        observations.setdefault(tag_id, []).append(
                            (camera_name, groups[0])
                        )
                else:
                    # Sessions captured before the candidate-preserving schema.
                    for tag_id_str, corners in info.get("corners", {}).items():
                        tag_id = int(tag_id_str)
                        if tag_id not in wrist_ids:
                            continue
                        observations.setdefault(tag_id, []).append((camera_name, corners))
        world_from_tag = {}
        corner_obs = {}
        fuse_meta = {}
        for tag_id, obs in sorted(observations.items()):
            discovered_ids.add(tag_id)
            # Raw normalized corners per camera — ONE camera is a usable
            # observation for the corner-reprojection AX=ZB (the 2026-08-21
            # rig never showed a wrist tag to two cameras at once, and the
            # mechanical fix is deferred). Triangulated poses remain the
            # premium product when >=2 cameras do line up.
            for camera_name, corners in obs:
                camera = cameras.get(camera_name)
                if camera is None:
                    continue
                corner_obs.setdefault(str(tag_id), []).append({
                    "camera": camera_name, "fx": camera.fx,
                    "normalized": [camera.undistort(
                        np.asarray(c, float)).tolist() for c in corners],
                })
            pose, meta = fuse_tag(obs, cameras, edge_m,
                                  args.max_reproj_px, args.min_cameras)
            fuse_meta[str(tag_id)] = meta
            if pose is not None:
                world_from_tag[str(tag_id)] = pose.tolist()
        if world_from_tag or corner_obs:
            # Written after the static-tag check below, once every interval
            # has been seen.
            pending_wrist.append({"record": record, "interval": interval,
                                  "world_from_tag": world_from_tag,
                                  "corner_obs": corner_obs,
                                  "shots": [s["name"] for s in in_window],
                                  "meta": fuse_meta})
        elif observations:
            record["kind"] = "wrist-rejected"
            record["reasons"] = {str(k): v.get("reason", v) for k, v in fuse_meta.items()}
        audit.append(record)

    # A tag whose fused WORLD position never moves while the joints change is
    # not on the wrist — it is a stray tag in the environment (or the palette
    # misread), and one X can never fit it, so it would poison the AX=ZB
    # solve. Wrist tags travel centimetres between guided orientations.
    positions_by_tag = {}
    for sample in pending_wrist:
        for tag_id, matrix in sample["world_from_tag"].items():
            positions_by_tag.setdefault(tag_id, []).append(
                np.asarray(matrix)[:3, 3])
    static_ids = set()
    for tag_id, points in positions_by_tag.items():
        if len(points) < 3:
            continue
        spread_mm = max(np.linalg.norm(a - b) * 1000.0
                        for i, a in enumerate(points) for b in points[i + 1:])
        if spread_mm < args.static_tag_mm:
            static_ids.add(tag_id)
    # The pixel-space analogue for tags never triangulated (single-camera
    # sightings): a tag whose corners sit still in EVERY camera that watched
    # it, across >=3 intervals of changing joints, is scenery.
    pixels_by_tag_cam = {}
    for sample in pending_wrist:
        for tag_id, obs_list in sample.get("corner_obs", {}).items():
            for obs in obs_list:
                center = np.mean(obs["normalized"], axis=0) * obs["fx"]
                pixels_by_tag_cam.setdefault(tag_id, {}).setdefault(
                    obs["camera"], []).append(center)
    rare_ids = set()
    for tag_id, by_camera in pixels_by_tag_cam.items():
        if tag_id in static_ids:
            continue
        sightings = [c for centers in by_camera.values() for c in centers]
        # A 16h5 phantom flickers once or twice; a wrist tag the guide waited
        # for shows up per accepted hold. Without this, a single bogus
        # sighting rides into the solver as its own tag (the >=2-camera rule
        # no longer guards corner observations — one camera is legitimate).
        if len(sightings) < 3 and tag_id not in positions_by_tag \
                and (wrist_ids is None or int(tag_id) not in wrist_ids):
            # Auto-discovery only: a pinned id is the operator vouching for
            # the tag — two real sightings of a known wrist tag are data
            # (the 2026-08-21 session had exactly two, both genuine).
            rare_ids.add(tag_id)
            continue
        if tag_id in positions_by_tag:
            continue  # triangulated evidence already decided static-ness
        multi = [centers for centers in by_camera.values() if len(centers) > 1]
        if not multi:
            continue  # single sightings across cameras prove nothing
        if all(max(np.linalg.norm(a - b)
                   for i, a in enumerate(centers) for b in centers[i + 1:])
               < 3.0 for centers in multi):
            static_ids.add(tag_id)
    if rare_ids:
        print(f"dropped tag(s) {sorted(int(t) for t in rare_ids)}: fewer than "
              "3 sightings in the whole session — 16h5 flicker, not a wrist tag")
    if static_ids:
        print(f"WARNING: tag(s) {sorted(int(t) for t in static_ids)} never moved "
              "while the joints changed — a static tag in the scene, not a "
              "wrist tag. Excluded; verify config/fiducials.json")
    excluded_ids = static_ids | rare_ids
    for sample in pending_wrist:
        kept_tags = {tag_id: matrix
                     for tag_id, matrix in sample["world_from_tag"].items()
                     if tag_id not in excluded_ids}
        kept_corners = {tag_id: obs
                        for tag_id, obs in sample.get("corner_obs", {}).items()
                        if tag_id not in excluded_ids}
        if not kept_tags and not kept_corners:
            sample["record"]["kind"] = "wrist-rejected"
            sample["record"]["reasons"] = dict.fromkeys(
                set(sample["world_from_tag"]) | set(sample.get("corner_obs", {})),
                "static in the world — not a wrist tag")
            continue
        pose_count += 1
        interval = sample["interval"]
        (session / f"pose_{pose_count:04d}.json").write_text(json.dumps({
            # Wrist tags are mounted to the follower's left jaw. Unlike the
            # flange, that frame depends on the seventh (prismatic carriage)
            # encoder, so wrist calibration must not discard it.
            "joint_names": driver_joint_names("right", len(interval["follower_pos"])),
            "joints": interval["follower_pos"],
            "world_from_tag": kept_tags,
            "corner_obs": kept_corners,
            "meta": {"start_unix": interval["start_unix"],
                     "end_unix": interval["end_unix"],
                     "duration_s": interval["duration_s"],
                     "shots": sample["shots"],
                     "tags": sample["meta"]},
        }, indent=2))
        seen = sorted(int(k) for k in set(kept_tags) | set(kept_corners))
        sample["record"]["kind"] = f"wrist:{seen}"

    if touches or pivots or tip_holds:
        (session / "touches.json").write_text(json.dumps(
            {"source": str(wxtl), "threshold": touch_info, "touches": touches,
             "pivots": pivots, "tip_holds": tip_holds,
             # Which surface the tip was planted on. il_touchoff records the
             # solved pivot as paper_plane_z, and only a pad session may claim
             # that is the paper the pen draws on.
             "tip_surface": tip_surface}, indent=2))
    named_poses, missing_poses = extract_poses(timeline, intervals)
    # Waypoints from the LEGACY tip phase only: pad hovers -> paper_pad_over,
    # palette holds -> palette_center. Since 2026-08-26 every tip hold is
    # planted on the pad, so a current session publishes neither — a pose named
    # "above the paper pad" must never be filled with one that is touching it.
    # stage_poses merges bounded by the tour, so the existing entries survive
    # untouched rather than being retired.
    for slug, label, key in (("paper_pad_over", "above the paper pad", "pad"),
                             ("palette_center", "on the palette tag", "palette")):
        samples = tip_waypoints.get(key, [])
        if samples:
            named_poses[slug] = {
                "label": label,
                "joints": np.median(np.asarray(samples), axis=0).tolist(),
                "start_unix": 0.0, "duration_s": 0.0,
                "in_contact": key == "palette",
            }
    if named_poses or missing_poses:
        (session / "poses.json").write_text(json.dumps(
            {"poses": named_poses, "missing": missing_poses}, indent=2))
    (session / "intervals.json").write_text(json.dumps(
        {"threshold": touch_info, "bundle": str(bundle_path),
         "guided": bool(timeline), "intervals": audit}, indent=2))

    print(f"\nfused: {pose_count} wrist poses, {len(tip_holds)} tip holds"
          + (f", {len(named_poses)} named poses" if named_poses else "")
          + (f", {len(touches)} legacy touches" if touches else "")
          + (f", {len(pivots)} legacy pivot windows" if pivots else ""))
    for miss in missed_pivots:
        print(f"NOTE: pivot {miss['index']} unusable: {miss['reason']}")
    for miss in missed_tip:
        print(f"NOTE: tip hold {miss['index']} ({miss['label']}) unusable: "
              f"{miss['reason']}")
    if tip_holds:
        print(f"tip: {len(tip_holds)} palette holds, "
              f"{len(tip_waypoints['pad'])} pad hovers")
    if missed_touches:
        print(f"NOTE: {len(missed_touches)} paced touch(es) had no quiet "
              f"window in the log (arm kept moving): {missed_touches} — "
              "press steadier; the plate-fit residual will police the rest")
    if missing_poses:
        print(f"named poses with no still interval: {missing_poses} — "
              "hold longer at each waypoint next time")
    if discovered_ids:
        print(f"wrist tag ids seen: {sorted(discovered_ids)}"
              + ("" if wrist_config["ids"] else
                 " — record them in config/fiducials.json"))
    if pose_count:
        print("next:\n  python3 scripts/vision/solve_robot_world.py "
              f"{session} --urdf urdf/tatbot.urdf --out {session / 'robot_world.json'}")
    if touches:
        print(f"  python3 scripts/il_touchoff.py {session} "
              "--tool-id <the tool in the gripper> --write")
    if not pose_count and not touches:
        print("nothing fused — see intervals.json for why each interval was rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
