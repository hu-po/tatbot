#!/usr/bin/env python3
"""Palette calibration: measure where the ink rack (palette_root) actually is
in the arm base frame, so il_dip dips into caps that are where the rack is, not
where the URDF nominally puts it. Two sources, written into
config/palette_calibration.yaml (scripts/lib/ink_spec.py owns the file):

  tip     AUTHORITATIVE. The ballpoint tip planted on palette_tag8 and rolled
          to several wrist angles (a tip phase, but on the tag). The pivot
          solve recovers the tag centre in the base frame by forward
          kinematics — no camera in the loop, so it is limited only by the
          tool. Reuses scripts/il_touchoff.py's solver.

            palette_cal.py tip <teleop session dir> --ee-tool <id> [--write]

  vision  QUICK, hands-off. palette_tag8 as the cameras see it, carried into
          the base frame through the two calibration bundles
          (~/tatbot-logs/vision/calibration-current.json for the per-camera
          intrinsics/extrinsics, robot-world-current.json for world<-base). No
          better than those bundles (~7 mm), so it is a sanity check, not the
          truth. Consumes a `tatbot vision tags scan` JSON.

            palette_cal.py vision --from-scan <tag_scan.json> [--write]

Neither moves the arm. --write merges its one source into the calibration
file, leaving the other source untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))
import ink_spec  # noqa: E402
from tatbot_runlog import log_root  # noqa: E402

PALETTE_TAG_ID = 8


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _root_from_tag_center(tag_center_base: np.ndarray) -> np.ndarray:
    """palette_root = observed tag8 centre - (tag8 relative to root, from the URDF)."""
    return np.asarray(tag_center_base, float) - np.asarray(ink_spec.tag8_in_palette_root(REPO), float)


# --- tip: the planted-tip pivot solve, on the tag -------------------------------------

def _tip_hold_joints(session: Path):
    """The follower joints at each tip hold, and where they came from.
    Prefer the fused touches.json; fall back to the raw teleop.wxtl windowed by
    guide_timeline.json, so a capture-only run (calib sweep --no-pipeline)
    needs no camera pipeline for a palette tip calibration."""
    touches = session / "touches.json"
    if touches.is_file():
        holds = json.loads(touches.read_text()).get("tip_holds") or []
        if holds:
            return [list(h["joints"]) for h in holds], "touches.json"
    timeline = session / "guide_timeline.json"
    wxtl = session / "teleop.wxtl"
    if timeline.is_file() and wxtl.is_file():
        import numpy as _np
        from teleop_log import TeleopLog
        tl = json.loads(timeline.read_text())
        log = TeleopLog(str(wxtl))
        us = log.unix_seconds
        seqs = []
        for e in tl.get("entries", []):
            if e.get("kind") != "tip_hold":
                continue
            m = (us >= e["start_unix"]) & (us <= e["end_unix"])
            if int(m.sum()) >= 5:
                seqs.append(list(_np.median(log.follower_pos[m][:, :6], axis=0)))
        if seqs:
            return seqs, "teleop.wxtl + guide_timeline.json"
    return [], "no touches.json, teleop.wxtl or guide_timeline"


def cal_tip(args) -> int:
    import il_touchoff as touchoff
    from urdf_kinematics import UrdfChain

    session = Path(args.session).expanduser()
    joint_seqs, provenance = _tip_hold_joints(session)
    if len(joint_seqs) < 4:
        sys.exit(f"need >= 4 tip holds, found {len(joint_seqs)} ({provenance}) — "
                 "roll the planted tip to several wrist angles")

    tool_id, spec = touchoff.resolve_tool(args)
    chain = UrdfChain(args.urdf)
    names = chain.driver_joint_names("right")  # the tip frame rides the carriage
    poses = [touchoff.ee_pose(chain, names, j) for j in joint_seqs]
    fit = touchoff.solve_pivot_trimmed(poses)
    # touchoff.ee_pose / the pivot are in the rig ROOT frame (that is how the
    # pad touch-off's pivot_point is stored); il_dip's caps are in the right
    # base, so cross root -> base before deriving palette_root.
    base_from_root = ink_spec.base_from_root_matrix(REPO, "right")
    tag_center = (base_from_root @ np.append(fit["pivot"], 1.0))[:3]
    root = _root_from_tag_center(tag_center)
    gate_mm = touchoff.residual_gate_mm(spec)

    print(f"tip palette calibration — {len(poses)} holds from {provenance} ({fit['dropped']} trimmed)")
    print(f"  tag8 centre {np.round(tag_center * 1000, 1)} mm (base)   palette_root {np.round(root * 1000, 1)} mm")
    print(f"  cond {fit['cond']:.1f}  spread {fit['spread_deg']:.1f} deg  rms {fit['rms_mm']:.2f} mm (gate {gate_mm:.2f})")
    problems = []
    if fit["spread_deg"] < touchoff.PIVOT_SPREAD_MIN_DEG:
        problems.append(f"orientation spread {fit['spread_deg']:.0f} deg < {touchoff.PIVOT_SPREAD_MIN_DEG:.0f} — roll the tip more")
    if fit["cond"] > touchoff.COND_MAX:
        problems.append(f"condition {fit['cond']:.0f} > {touchoff.COND_MAX:.0f} — too uniform")
    if fit["rms_mm"] > gate_mm:
        problems.append(f"rms {fit['rms_mm']:.2f} mm > {gate_mm:.2f} — the tip slid; press firmer, keep it on the tag centre")
    return _finish("tip", args, root, fit["rms_mm"], problems,
                   extra={"tool_id": tool_id, "cond": round(fit["cond"], 1),
                          "spread_deg": round(fit["spread_deg"], 1),
                          "tag_id": PALETTE_TAG_ID, "session": str(session)})


# --- vision: the tag as the cameras see it, through the bundles -----------------------

def _bundle(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"missing calibration bundle {path}")
    return json.loads(path.read_text())


def _mat(rows) -> np.ndarray:
    return np.array(rows, float)


def _world_from_camera(cam: dict) -> np.ndarray:
    wfc = cam["world_from_camera"]
    m = np.eye(4)
    rot = _mat(wfc["rotation"])
    m[:3, :3] = rot.reshape(3, 3) if rot.ndim == 1 else rot
    # the bundle spells it translation_m; older/synthetic ones say translation
    m[:3, 3] = wfc.get("translation_m", wfc.get("translation"))
    return m


def _cluster(candidates: list, radius_m: float = 0.02) -> list:
    """The largest agreeing set of tag observations, at most one per camera.
    Each candidate seeds a cluster of the nearest observation from every OTHER
    camera within radius_m; the biggest cluster wins. This drops a camera's
    false-positive id-8 (it sits far from the real tag) while keeping its true
    one."""
    best: list = []
    for _, seed in candidates:
        by_cam: dict = {}
        for cam, w in candidates:
            if float(np.linalg.norm(w - seed)) <= radius_m and (cam not in by_cam
                    or np.linalg.norm(w - seed) < np.linalg.norm(by_cam[cam][1] - seed)):
                by_cam[cam] = (cam, w)
        if len(by_cam) > len(best):
            best = list(by_cam.values())
    return best


def cal_vision(args) -> int:
    import cv2

    scan = json.loads(Path(args.from_scan).expanduser().read_text())
    calib = _bundle(Path(args.calibration).expanduser())
    rw = _bundle(Path(args.robot_world).expanduser())
    # The robot-world bundle solves in the rig ROOT frame ("base" is the
    # historical key; it is the root — docs/ee_fiducial_tracking.md). il_dip's
    # caps are in the right-arm base, so cross root -> base after world -> root.
    root_from_world = np.linalg.inv(_mat(rw["world_from_base"]))
    base_from_root = ink_spec.base_from_root_matrix(REPO, "right")
    base_from_world = base_from_root @ root_from_world

    from fiducials import load_inventory
    palette_edge = load_inventory().target("palette").edge_m
    half = palette_edge / 2.0
    objp = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], np.float64)

    # id 16h5 8 is reused (palette 41 mm, calibration board 56 mm) and has a
    # known false-positive rate, so a camera can report more than one "8".
    # Take EVERY id-8 candidate, PnP each to a world centre, and keep the set
    # that agrees across cameras (one per camera) — a false positive lands
    # somewhere random and is left out, the real tag clusters.
    candidates = []  # (cam_name, world_centre)
    for rep in scan.get("reports", []):
        cam_name = rep.get("camera")
        cam = calib.get("cameras", {}).get(cam_name)
        if cam is None:
            continue
        intr = cam["intrinsics"]
        cam_k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
        dist = np.array(cam.get("distortion", {}).get("coefficients", [0, 0, 0, 0, 0]), float)
        for det in rep.get("detections", []):
            if int(det["id"]) != PALETTE_TAG_ID:
                continue
            img = np.array(det["corners_px"], np.float64).reshape(-1, 2)
            ok, rvec, tvec = cv2.solvePnP(objp, img, cam_k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            cam_from_tag = np.eye(4)
            cam_from_tag[:3, :3], _ = cv2.Rodrigues(rvec)
            cam_from_tag[:3, 3] = tvec.ravel()
            candidates.append((cam_name, (_world_from_camera(cam) @ cam_from_tag @ np.array([0, 0, 0, 1.0]))[:3]))

    if not candidates:
        sys.exit(f"palette_tag8 (id {PALETTE_TAG_ID}) not detected in {args.from_scan} on any calibrated camera")
    consensus = _cluster(candidates, radius_m=0.02)
    used = [c for c, _ in consensus]
    centers_world = np.array([w for _, w in consensus])
    center_world = centers_world.mean(axis=0)
    spread_mm = float(np.linalg.norm(centers_world - center_world, axis=1).max() * 1000) if len(centers_world) > 1 else 0.0
    center_base = (base_from_world @ np.append(center_world, 1.0))[:3]
    root = _root_from_tag_center(center_base)
    bundle_mm = float(rw.get("residual_mm_median") or 0.0)
    residual_mm = max(bundle_mm, spread_mm)

    print(f"vision palette calibration — tag8 on {len(used)} camera(s): {', '.join(used)}")
    print(f"  tag8 centre {np.round(center_base * 1000, 1)} mm (base)   palette_root {np.round(root * 1000, 1)} mm")
    print(f"  cross-camera spread {spread_mm:.1f} mm; robot-world residual {bundle_mm:.1f} mm -> ±{residual_mm:.1f} mm")
    problems = []
    if len(used) < 2:
        problems.append(f"only {len(used)} camera saw the tag — a single-camera pose is unverified; want >= 2")
    if spread_mm > 15.0:
        problems.append(f"cameras disagree by {spread_mm:.0f} mm — a detection is wrong or the bundle is stale")
    return _finish("vision", args, root, residual_mm, problems,
                   extra={"tag_id": PALETTE_TAG_ID, "cameras": used,
                          "cross_camera_mm": round(spread_mm, 1),
                          "robot_world_residual_mm": round(bundle_mm, 1),
                          "calibration_id": calib.get("bundle_id"),
                          "robot_world_utc": rw.get("utc"), "scan": str(args.from_scan)})


# --- shared: report, gate, write ------------------------------------------------------

def _finish(source: str, args, root: np.ndarray, residual_mm: float, problems: list[str], extra: dict) -> int:
    rec = {"root_xyz_m": [round(float(x), 6) for x in root],
           "residual_mm": round(float(residual_mm), 3), "utc": _utc(), **extra}
    if problems:
        for p in problems:
            print(f"  REFUSED: {p}", file=sys.stderr)
        rec["note"] = "refused: " + "; ".join(problems)
        print("  not written (refused). Fix the above and re-measure.", file=sys.stderr)
        return 2
    if not args.write:
        print("  dry run — pass --write to update config/palette_calibration.yaml")
        return 0
    path = ink_spec.write_palette_cal(source, rec, REPO)
    print(f"  wrote {source} calibration to {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("tip", help="authoritative: planted-tip pivot solve on palette_tag8")
    t.set_defaults(fn=cal_tip)
    t.add_argument("session", help="teleop session dir with touches.json (tip planted on palette_tag8)")
    t.add_argument("--ee-tool", "--tool-id", dest="tool_id", required=True, help="the tool in the gripper, stated")
    t.add_argument("--urdf", default=str(REPO / "urdf" / "tatbot.urdf"))
    t.add_argument("--workspace", default=str(REPO / "config" / "workspace.yaml"))
    t.add_argument("--write", action="store_true", help="update config/palette_calibration.yaml")

    v = sub.add_parser("vision", help="quick: palette_tag8 through the camera bundles")
    v.set_defaults(fn=cal_vision)
    v.add_argument("--from-scan", required=True, help="a `tatbot vision tags scan` JSON")
    v.add_argument("--calibration", default=str(log_root() / "vision/calibration-current.json"))
    v.add_argument("--robot-world", default=str(log_root() / "vision/robot-world-current.json"))
    v.add_argument("--write", action="store_true", help="update config/palette_calibration.yaml")

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
