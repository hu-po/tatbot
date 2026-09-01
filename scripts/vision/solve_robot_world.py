#!/usr/bin/env python3
"""Tie the camera rig to the robot using tags on the wrist (robot-world calibration).

The palette tag fixes the world origin but says nothing about where the robot
is; the URDF's answer to that is hand-authored. The arm's encoders plus its
URDF are the one metric reference in the scene that was not guessed, so drive
the wrist carrying the configured 16h5 tags to several poses, observe the tags
with the calibrated PoE rig, and solve for the rest.

For every pose i, with Z = world_from_base and X = link_from_tag unknown:

    world_from_tag_i  =  Z @ base_from_link_i @ X
                            ^ forward kinematics from the encoders

the classic AX = ZB problem. Z is what aligns the calibration to the robot —
measured rather than assumed. X falls out as a bonus: it is where the tag sits
on the wrist, the first half of D405 hand-eye.

  python3 solve_robot_world.py --self-test
  python3 solve_robot_world.py <session_dir> --calibration <bundle.json> \
      --urdf <tatbot.urdf> [--link right/realsense_link] [--arm-prefix right]

The session directory holds one JSON per pose, written by fuse_session.py:
joint values plus per-tag observations. With --calibration the solver runs in
corner-reprojection mode (raw single-camera corner sightings, no IPPE
ambiguity — the production path); without it, only >=2-camera triangulated
tag poses are used.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from urdf_kinematics import UrdfChain, driver_joint_names  # noqa: E402

# Rotation error is reported in metres of arc at this radius, so a rotation
# and a translation residual are comparable in the same least-squares problem.
ROTATION_SCALE_M = 0.05
# Five normalized-image milliradians is roughly 5-15 px on this rig. Huber
# downweights a buffered/mis-timed camera frame without letting it drag every
# tag transform and world_from_base tens of millimetres.
ROBUST_SCALE = 0.005


def pose_joint_values(pose, arm_prefix):
    joints = pose["joints"]
    names = pose.get("joint_names") or driver_joint_names(arm_prefix, len(joints))
    if len(names) != len(joints):
        raise ValueError(
            f"pose has {len(joints)} joint values but {len(names)} joint names"
        )
    return dict(zip(names, joints, strict=True))


def rotation_to_vector(rotation):
    """Log map SO(3) -> R^3 (axis * angle)."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    return axis * (angle / (2 * np.sin(angle)))


def vector_to_rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def pack(transform):
    return np.concatenate([rotation_to_vector(transform[:3, :3]), transform[:3, 3]])


def unpack(vector):
    transform = np.eye(4)
    transform[:3, :3] = vector_to_rotation(vector[:3])
    transform[:3, 3] = vector[3:6]
    return transform


def solve(base_from_link, world_from_tag, tag_ids=None, initial_z=None,
          corner_obs=None, edge_m=None, initial_x_by_tag=None, restarts=8):
    """Solve world_from_base (shared) and link_from_tag (one per tag id).

    Two observation kinds, mixable:
    - pose observations: a fused `world_from_tag` per (pose, tag) — needs the
      tag triangulated from >=2 cameras;
    - corner observations (`corner_obs`: list of (base_from_link, tag_id,
      cam_rotation, cam_translation, normalized 4x2)): raw undistorted
      corners from ONE camera. Each contributes 8 reprojection residuals in
      normalized image coordinates, constrained through the kinematics — so a
      single camera is enough, and there is no IPPE branch ambiguity because
      FK anchors the orientation. A normalized-coordinate error of 0.003 is
      ~3 px at these lenses and commensurate with 3 mm of pose residual, so
      the two kinds share one least-squares problem unweighted.
    """
    edge_m = edge_m if edge_m is not None else load_inventory().target("wrist").edge_m
    tag_ids = tag_ids if tag_ids is not None else [0] * len(base_from_link)
    corner_obs = corner_obs or []
    unique = sorted(set(tag_ids) | {obs[1] for obs in corner_obs})
    index_of = {tag: i for i, tag in enumerate(unique)}
    model = np.concatenate([tag_model_corners(edge_m).T,
                            np.ones((1, 4))])  # 4x4 homogeneous columns

    def residuals(params):
        z = unpack(params[:6])
        x_by_tag = [unpack(params[6 + 6 * i:12 + 6 * i]) for i in range(len(unique))]
        out = []
        for b, w, tag in zip(base_from_link, world_from_tag, tag_ids, strict=True):
            predicted = z @ b @ x_by_tag[index_of[tag]]
            error = np.linalg.inv(predicted) @ w
            out.append(np.concatenate([
                error[:3, 3],
                rotation_to_vector(error[:3, :3]) * ROTATION_SCALE_M,
            ]))
        for b, tag, cam_rotation, cam_translation, normalized in corner_obs:
            corners_world = (z @ b @ x_by_tag[index_of[tag]] @ model)[:3].T
            cam_points = (corners_world - cam_translation) @ cam_rotation
            projected = cam_points[:, :2] / np.maximum(cam_points[:, 2:3], 1e-6)
            out.append((projected - normalized).ravel())
        return np.concatenate(out)

    best = None
    # The problem is non-convex; a handful of fast linear-loss LM starts find
    # the basin. Refine only the winner with Huber: running every random start
    # through numeric-Jacobian TRF took >75 s on the 105-observation field
    # session, while LM basin search + one robust refinement produces the same
    # answer in seconds. A warm start (triangulated fits or a full solve) still
    # makes attempt 0 the usual winner.
    rng = np.random.default_rng(0)
    for attempt in range(restarts):
        start = np.zeros(6 + 6 * len(unique))
        if attempt == 0 and initial_z is not None:
            start[:6] = pack(initial_z)
            for tag, x in (initial_x_by_tag or {}).items():
                if tag in index_of:
                    start[6 + 6 * index_of[tag]:12 + 6 * index_of[tag]] = pack(x)
        elif attempt > 0:
            start[:3] = rng.normal(0, 1.0, 3)
            start[3:6] = rng.normal(0, 0.3, 3)
            for i in range(len(unique)):
                start[6 + 6 * i:9 + 6 * i] = rng.normal(0, 1.0, 3)
                start[9 + 6 * i:12 + 6 * i] = rng.normal(0, 0.05, 3)
        result = least_squares(residuals, start, method="lm", max_nfev=20000)
        if best is None or result.cost < best.cost:
            best = result
    best = least_squares(
        residuals,
        best.x,
        method="trf",
        loss="huber",
        f_scale=ROBUST_SCALE,
        max_nfev=20000,
    )
    z = unpack(best.x[:6])
    x_by_tag = {tag: unpack(best.x[6 + 6 * index_of[tag]:12 + 6 * index_of[tag]])
                for tag in unique}
    return z, x_by_tag, best


def report(base_from_link, world_from_tag, tag_ids, z, x_by_tag):
    position_errors, angle_errors = [], []
    for b, w, tag in zip(base_from_link, world_from_tag, tag_ids, strict=True):
        predicted = z @ b @ x_by_tag[tag]
        position_errors.append(np.linalg.norm(predicted[:3, 3] - w[:3, 3]) * 1000)
        delta = np.linalg.inv(predicted[:3, :3]) @ w[:3, :3]
        angle_errors.append(np.degrees(np.linalg.norm(rotation_to_vector(delta))))
    position_errors = np.array(position_errors)
    angle_errors = np.array(angle_errors)
    print(f"  poses: {len(position_errors)}")
    print(f"  tag position residual: median {np.median(position_errors):.1f} mm, "
          f"max {position_errors.max():.1f} mm")
    print(f"  tag rotation residual: median {np.median(angle_errors):.2f} deg, "
          f"max {angle_errors.max():.2f} deg")
    return position_errors, angle_errors


def corner_residuals_px(corner_obs, corner_fx, z, x_by_tag, edge_m):
    """Per-sighting worst corner error in pixels, for the report."""
    if not corner_obs:
        return None
    model = np.concatenate([tag_model_corners(edge_m).T, np.ones((1, 4))])
    errors = []
    for (b, tag, rotation, translation, normalized), fx in zip(
            corner_obs, corner_fx, strict=True):
        corners_world = (z @ b @ x_by_tag[tag] @ model)[:3].T
        cam_points = (corners_world - translation) @ rotation
        projected = cam_points[:, :2] / np.maximum(cam_points[:, 2:3], 1e-6)
        errors.append(float(np.abs(projected - normalized).max() * fx))
    return np.asarray(errors)


def self_test():
    """Recover a known Z and X from synthetic poses with realistic noise."""
    rng = np.random.default_rng(7)
    true_z = np.eye(4)
    true_z[:3, :3] = vector_to_rotation(np.array([0.02, -0.01, 1.57]))
    true_z[:3, 3] = [0.126, 0.0, 0.0885]
    true_x = np.eye(4)
    true_x[:3, :3] = vector_to_rotation(np.array([0.1, 0.2, -0.3]))
    true_x[:3, 3] = [0.03, -0.01, 0.05]

    for noise_mm, noise_deg in ((0.0, 0.0), (3.0, 0.3)):
        base_from_link, world_from_tag = [], []
        for _ in range(16):
            b = np.eye(4)
            b[:3, :3] = vector_to_rotation(rng.normal(0, 0.8, 3))
            b[:3, 3] = rng.uniform([-0.1, -0.4, 0.05], [0.4, 0.1, 0.4])
            w = true_z @ b @ true_x
            if noise_mm:
                w[:3, 3] += rng.normal(0, noise_mm / 1000.0, 3)
                w[:3, :3] = vector_to_rotation(
                    rng.normal(0, np.radians(noise_deg), 3)) @ w[:3, :3]
            base_from_link.append(b)
            world_from_tag.append(w)
        z, x_by_tag, _ = solve(base_from_link, world_from_tag)
        x = x_by_tag[0]
        position_error = np.linalg.norm(z[:3, 3] - true_z[:3, 3]) * 1000
        angle_error = np.degrees(np.linalg.norm(rotation_to_vector(
            np.linalg.inv(z[:3, :3]) @ true_z[:3, :3])))
        x_error = np.linalg.norm(x[:3, 3] - true_x[:3, 3]) * 1000
        print(f"noise {noise_mm:.1f} mm / {noise_deg:.1f} deg -> "
              f"world_from_base off by {position_error:.2f} mm, {angle_error:.3f} deg; "
              f"link_from_tag off by {x_error:.2f} mm")
        limit = 1.0 if noise_mm == 0 else 6.0
        assert position_error < limit, f"Z translation error {position_error} mm"
        assert x_error < limit, f"X translation error {x_error} mm"
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?")
    ap.add_argument("--urdf")
    ap.add_argument("--link", default=None,
                    help="URDF link the wrist tag is rigidly attached to "
                         "(default: targets.wrist.parent_frame)")
    ap.add_argument("--arm-prefix", default="right")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ablate", action="store_true",
                    help="also solve on chronological prefixes (n=4..N) and "
                         "report how the answer moves — accuracy vs pose count")
    ap.add_argument("--calibration", default=None,
                    help="CalibrationBundle json: switches to raw corner "
                         "observations (single-camera sightings usable; "
                         "wrist tags rarely reach two cameras on this rig)")
    ap.add_argument("--tag-edge-m", type=float, default=None,
                    help="wrist tag edge (default: the session's own "
                         "triangulated estimates when any exist, else the "
                         "ruler-measured 56 mm)")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.session_dir or not args.urdf:
        sys.exit("need <session_dir> --urdf, or --self-test")

    session = Path(args.session_dir).expanduser()
    chain = UrdfChain(args.urdf)
    wrist = load_inventory().target("wrist")
    args.link = args.link or wrist.parent_frame
    if not args.link:
        sys.exit("targets.wrist.parent_frame is required")
    cameras = {}
    calibration_id = None
    if args.calibration and Path(args.calibration).expanduser().is_file():
        bundle = json.loads(Path(args.calibration).expanduser().read_text())
        calibration_id = bundle.get("bundle_id")
        cameras = {
            name: (np.asarray(cal["world_from_camera"]["rotation"],
                              float).reshape(3, 3),
                   np.asarray(cal["world_from_camera"]["translation_m"], float))
            for name, cal in bundle["cameras"].items()}

    base_from_link, world_from_tag, tag_ids = [], [], []
    corner_obs, corner_fx = [], []
    edge_estimates = []
    pose_observations_by_tag = {}
    for pose_file in sorted(session.glob("pose_*.json")):
        pose = json.loads(pose_file.read_text())
        values = pose_joint_values(pose, args.arm_prefix)
        b = chain.link_pose(args.link, values)
        for tag_meta in pose.get("meta", {}).get("tags", {}).values():
            if isinstance(tag_meta, dict) and tag_meta.get("edge_est_m"):
                edge_estimates.append(float(tag_meta["edge_est_m"]))
        if cameras:
            # Corner mode: raw single-camera sightings, every one usable.
            # Never mixed with the fused poses of the same sightings — that
            # would count the same pixels twice.
            tags_in_pose = set()
            for tag_id, obs_list in pose.get("corner_obs", {}).items():
                for obs in obs_list:
                    if obs["camera"] not in cameras:
                        continue
                    tags_in_pose.add(int(tag_id))
                    rotation, translation = cameras[obs["camera"]]
                    corner_obs.append((b, int(tag_id), rotation, translation,
                                       np.asarray(obs["normalized"], float)))
                    corner_fx.append(float(obs.get("fx", 900.0)))
        else:
            tags_in_pose = set()
            for tag_id, matrix in pose.get("world_from_tag", {}).items():
                tags_in_pose.add(int(tag_id))
                base_from_link.append(b)
                world_from_tag.append(np.asarray(matrix, float))
                tag_ids.append(int(tag_id))
        for tag_id in tags_in_pose:
            pose_observations_by_tag[tag_id] = pose_observations_by_tag.get(tag_id, 0) + 1
    observations = len(base_from_link) + len(corner_obs)
    if observations < 4:
        sys.exit(f"only {observations} tag observations; need at least 4 "
                 f"(and they must differ in orientation, not just position)")
    # Every distinct tag adds 6 unknowns (its X); refuse when the residuals
    # cannot outnumber the parameters with margin — a solve that CAN overfit
    # will, and reports a beautiful residual for a meaningless answer.
    tags_seen = set(tag_ids) | {o[1] for o in corner_obs}
    residual_rows = 6 * len(base_from_link) + 8 * len(corner_obs)
    params = 6 + 6 * len(tags_seen)
    if residual_rows < params + 6:
        sys.exit(f"{observations} observations of {len(tags_seen)} tags give "
                 f"{residual_rows} residuals for {params} parameters — "
                 "underdetermined. Capture more wrist holds (or pass "
                 "--calibration so single-camera corner sightings count).")
    # Distinct ARM POSES are what separate Z from the per-tag X's: the
    # 2026-08-22 two-pose session had residuals to spare and still placed
    # every tag ~580 mm from the wrist along a degenerate Z/X direction.
    distinct_poses = {tuple(np.round(b[:3, 3], 6)) for b in base_from_link}         | {tuple(np.round(o[0][:3, 3], 6)) for o in corner_obs}
    if len(distinct_poses) < 3:
        sys.exit(f"only {len(distinct_poses)} distinct arm poses — Z and "
                 "link_from_tag cannot be separated with fewer than 3. "
                 "Capture more wrist holds.")
    seen_tags = sorted(set(tag_ids) | {o[1] for o in corner_obs})
    print(f"{observations} observations of tags {seen_tags}"
          + (f" ({len(corner_obs)} raw corner sightings)" if corner_obs else ""))
    print(f"distinct arm poses per tag: {dict(sorted(pose_observations_by_tag.items()))}")

    # The tag edge scales depth in corner mode, so wrong = biased. Priority:
    # explicit flag > the session's own triangulated edge estimates > the
    # caliper-measured size in the canonical inventory.
    if args.tag_edge_m:
        edge_m = args.tag_edge_m
    elif edge_estimates:
        edge_m = float(np.median(edge_estimates))
        print(f"tag edge from session: {edge_m * 1000:.1f} mm "
              f"({len(edge_estimates)} triangulated estimates)")
    else:
        edge_m = load_inventory().target("wrist").edge_m
    nominal = chain.link_pose("palette_tag8", {})
    # Seed each tag's X from any triangulated sighting (X = (Z_nom B)^-1 W):
    # rough, but it puts attempt 0 in the right basin.
    x_seeds = {}
    for pose_file in sorted(session.glob("pose_*.json")):
        pose = json.loads(pose_file.read_text())
        values = pose_joint_values(pose, args.arm_prefix)
        b = chain.link_pose(args.link, values)
        for tag_id, matrix in pose.get("world_from_tag", {}).items():
            x_seeds.setdefault(int(tag_id), np.linalg.inv(nominal @ b)
                               @ np.asarray(matrix, float))
    z, x_by_tag, _ = solve(base_from_link, world_from_tag, tag_ids,
                           initial_z=nominal, corner_obs=corner_obs,
                           edge_m=edge_m, initial_x_by_tag=x_seeds)
    print("\nresiduals:")
    if base_from_link:
        position_errors, angle_errors = report(base_from_link, world_from_tag,
                                               tag_ids, z, x_by_tag)
    else:
        position_errors = angle_errors = np.array([float("nan")])
    corner_errors_px = corner_residuals_px(corner_obs, corner_fx, z, x_by_tag,
                                           edge_m)
    if corner_errors_px is not None:
        print(f"  corner reprojection: median {np.median(corner_errors_px):.2f} px, "
              f"max {corner_errors_px.max():.2f} px over {len(corner_obs)} sightings")

    print("\nworld_from_base (this is what aligns the rig to the robot):")
    print(f"  translation (m): {np.round(z[:3, 3], 4)}")
    print(f"  rotation (deg):  {np.round(np.degrees(rotation_to_vector(z[:3, :3])), 2)}")
    print("\nnominal from the URDF's palette_tag8 link, for comparison:")
    print(f"  translation (m): {np.round(nominal[:3, 3], 4)}")
    delta = np.linalg.inv(nominal) @ z
    print(f"  measured minus nominal: {np.linalg.norm(delta[:3, 3]) * 1000:.1f} mm, "
          f"{np.degrees(np.linalg.norm(rotation_to_vector(delta[:3, :3]))):.2f} deg")
    for tag, x in sorted(x_by_tag.items()):
        print(f"\nlink_from_tag[{tag}] ({args.link} -> tag):")
        print(f"  translation (m): {np.round(x[:3, 3], 4)}")
        print(f"  rotation (deg):  {np.round(np.degrees(rotation_to_vector(x[:3, :3])), 2)}")

    ablation = []
    if args.ablate and observations > 4:
        # How much does each extra observation buy? Chronological prefixes,
        # each solved fresh; delta is the base translation's distance from
        # the full-set answer. This is what sizes the debug-vs-full profiles.
        step = max(1, observations // 8)
        counts = sorted(set(range(4, observations + 1, step)) | {observations})
        for count in counts:
            if corner_obs:
                z_n, x_n, _ = solve([], [], [], initial_z=z,
                                    corner_obs=corner_obs[:count],
                                    edge_m=edge_m, initial_x_by_tag=x_by_tag,
                                    restarts=1)
                errors_px = corner_residuals_px(corner_obs[:count],
                                                corner_fx[:count], z_n, x_n,
                                                edge_m)
                residual = float(np.median(errors_px))
            else:
                z_n, x_n, _ = solve(base_from_link[:count], world_from_tag[:count],
                                    tag_ids[:count], initial_z=z,
                                    initial_x_by_tag=x_by_tag, restarts=1)
                errors = [np.linalg.norm((z_n @ b @ x_n[tag])[:3, 3] - w[:3, 3]) * 1000
                          for b, w, tag in zip(base_from_link[:count],
                                               world_from_tag[:count],
                                               tag_ids[:count], strict=True)]
                residual = float(np.median(errors))
            ablation.append({
                "n": count,
                "residual_mm_median": round(residual, 3),
                "delta_vs_full_mm": round(
                    float(np.linalg.norm(z_n[:3, 3] - z[:3, 3])) * 1000, 3),
            })
        unit = "px" if corner_obs else "mm"
        print("\naccuracy vs observation count:")
        for row in ablation:
            print(f"  n={row['n']:2d}  residual {row['residual_mm_median']:6.2f} {unit}"
                  f"  delta vs full {row['delta_vs_full_mm']:6.2f} mm")

    if corner_obs is not None and len(corner_obs):
        # Normalized error * 1000 ~= mm at 1 m range — commensurate enough to
        # share the residual fields the report card reads.
        position_errors = corner_errors_px / np.asarray(corner_fx) * 1000.0
        angle_errors = np.array([float("nan")])
    if args.out:
        # Residuals ride along so the report card never has to parse stdout.
        Path(args.out).expanduser().write_text(json.dumps({
            "world_from_base": z.tolist(),
            # This transform is expressed in the exact camera world defined
            # by the selected bundle; the two artifacts are one calibration.
            "calibration_id": calibration_id,
            "link": args.link,
            "link_from_tag": {str(k): v.tolist() for k, v in x_by_tag.items()},
            "observations": observations,
            "pose_observations_by_tag": {
                str(tag_id): count for tag_id, count in sorted(pose_observations_by_tag.items())
            },
            "mode": "corner_reprojection" if corner_obs else "fused_pose",
            "loss": "huber",
            "robust_scale": ROBUST_SCALE,
            "corner_px_median": (round(float(np.median(corner_errors_px)), 2)
                                 if corner_obs else None),
            "residual_mm_median": round(float(np.median(position_errors)), 3),
            "residual_mm_max": round(float(position_errors.max()), 3),
            "residual_deg_median": (None if np.isnan(np.median(angle_errors))
                                    else round(float(np.median(angle_errors)), 4)),
            "nominal_delta_mm": round(np.linalg.norm(delta[:3, 3]) * 1000, 2),
            "nominal_delta_deg": round(float(np.degrees(np.linalg.norm(
                rotation_to_vector(delta[:3, :3])))), 3),
            "ablation": ablation,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
