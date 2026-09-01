#!/usr/bin/env python3
"""Full-rig calibration from a guided board session (runs on the camera node).

Input: a calib_session.py session directory of shots, each holding
camera*.png + detections.json. The calibration board is a 3x3 grid of nine
44 mm 16h5 tags (ids 3 through 11) with measured-by-solver layout; the final
anchor shot shows the 41 mm palette tag (also id 8) with the board removed.

Pipeline:
  1. self-calibrate the board layout from all nine board tags.  IDs 3/6/7/8
     are reused on other physical targets, so they are board observations
     only because this solver consumes explicitly isolated board-phase shots;
  2. per-camera intrinsics + Brown-Conrady distortion via calibrateCamera
     over all shots that camera saw;
  3. camera-to-camera extrinsics from shots seen by 2+ cameras (pose-graph
     spanning tree from the best-connected camera);
  4. world anchoring from the palette-tag shot (cameras that cannot see the
     palette — camera1 — get their world pose through the graph);
  5. draft CalibrationBundle -> finalize-calibration -> validate-calibration.

  python3 calibrate_board_session.py <session_dir> [--board-size-m 0.044]
      [--palette-size-m 0.041]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory, tag_model_corners  # noqa: E402
from fiducials.board import rigid_board_instance_ids  # noqa: E402

_INVENTORY = load_inventory()
_BOARD = _INVENTORY.target("board")
_PALETTE = _INVENTORY.target("palette")
BOARD_IDS = list(_BOARD.ids)
# A numeric ID is not a physical-instance identity in this rig.  The caller
# has already selected board-only holds (and the guide tells the operator to
# cover the wrist and palette), so all nine detections belong to the board.
# Restricting this to globally-exclusive IDs discarded the entire middle row
# and 44% of the available corners from the 2026-08-26 field dataset.
CALIBRATION_IDS = BOARD_IDS
BOARD_ROOT = _BOARD.calibration_root_id
MAX_REPROJECTION_PX = _BOARD.max_calibration_reprojection_px or 5.0
PALETTE_ID = _PALETTE.ids[0]
IMAGE_SIZE = (2960, 1668)
VISIOND = Path(__file__).resolve().parents[2] / "rust/target/release/tatbot-visiond"
# A duplicate wrist/palette instance can still leak into an otherwise isolated
# board shot.  The exclusive root tag defines that shot's board homography;
# any same-number detection far from its rigid board location is another
# physical instance, not a noisy board corner.
MAX_BOARD_INSTANCE_ERROR_M = 0.025

if BOARD_ROOT is None or BOARD_ROOT not in CALIBRATION_IDS:
    raise ValueError(
        "board calibration_root_id must be one of its physically exclusive ids"
    )


tag_object_points = tag_model_corners


def nominal_intrinsics():
    w, h = IMAGE_SIZE
    f = w * 2.8 / 5.37
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def to_matrix(rvec, tvec):
    m = np.eye(4)
    m[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    m[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return m


def consensus_transform(mats, rot_tol_deg=10.0, trans_tol_m=0.05):
    """Average only the largest mutually-agreeing subset of transforms.

    Per-shot IPPE board poses branch-flip on near-frontal views, and a blind
    mean over flipped candidates poisons the extrinsics tree (2026-08-22:
    bundle adjust started at 324 px on a session whose intrinsics and layout
    were sub-millimetre). Vote instead: each candidate proposes itself as the
    hypothesis, inliers agree within tolerance, best hypothesis's inliers
    are averaged. Falls back to the plain mean when nothing agrees."""
    if len(mats) <= 2:
        return average_transforms(mats)
    best_inliers = []
    for hypothesis in mats:
        h_inv_rot = hypothesis[:3, :3].T
        inliers = []
        for m in mats:
            delta = h_inv_rot @ m[:3, :3]
            angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
            dist = np.linalg.norm(m[:3, 3] - hypothesis[:3, 3])
            if angle <= rot_tol_deg and dist <= trans_tol_m:
                inliers.append(m)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < max(2, len(mats) // 3):
        return average_transforms(mats)
    return average_transforms(best_inliers)


def average_transforms(mats):
    """Chordal-mean rotation (quaternion eigen average) + median translation."""
    quats = []
    for m in mats:
        q = rotation_to_quat(m[:3, :3])
        if quats and np.dot(q, quats[0]) < 0:
            q = -q
        quats.append(q)
    a = np.zeros((4, 4))
    for q in quats:
        a += np.outer(q, q)
    _, vecs = np.linalg.eigh(a)
    mean_q = vecs[:, -1]
    out = np.eye(4)
    out[:3, :3] = quat_to_rotation(mean_q)
    out[:3, 3] = np.median([m[:3, 3] for m in mats], axis=0)
    return out


def rotation_to_quat(rot):
    w = np.sqrt(max(0.0, 1 + rot[0, 0] + rot[1, 1] + rot[2, 2])) / 2
    if w < 1e-6:
        q = cv2.Rodrigues(rot)[0].reshape(3)
        angle = np.linalg.norm(q)
        axis = q / angle if angle > 0 else np.array([1.0, 0, 0])
        return np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])
    return np.array([w, (rot[2, 1] - rot[1, 2]) / (4 * w),
                     (rot[0, 2] - rot[2, 0]) / (4 * w),
                     (rot[1, 0] - rot[0, 1]) / (4 * w)])


def quat_to_rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def solve_tag(corners, size_m, intrinsics, dist):
    """Ambiguity-aware single-tag pose: returns list of (err, 4x4 cam_T_tag)."""
    obj = tag_object_points(size_m)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, np.asarray(corners, np.float64), intrinsics, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [(float(np.asarray(errs).flatten()[i]), to_matrix(rvecs[i], tvecs[i]))
            for i in range(n)]


def board_layout(board_shots, size_m, intrinsics, dist):
    """Self-calibrate the board's tag layout, exploiting its planarity.

    All tags lie in the board plane, so the board plane and the image are
    related by a homography — and a homography between two planes does not
    depend on the camera intrinsics. Anchoring that homography on the root
    tag's known edge length therefore maps every other tag's corners
    straight into board-plane metres, with none of the out-of-plane noise or
    IPPE branch-flipping that single-tag PnP suffers from.

    Corners are undistorted first whenever a distortion model is known, so
    the estimate sharpens as the outer loop's intrinsics improve.

    Returns (layout, spreads_mm, view_count) where layout maps tag id ->
    (4,3) corner positions in board coordinates (z = 0).
    """
    half = size_m / 2.0
    root_plane = np.array([[-half, half], [half, half],
                           [half, -half], [-half, -half]], np.float64)
    samples = {i: [] for i in CALIBRATION_IDS}
    views = 0
    for shot in board_shots:
        for camera, tags in shot["cameras"].items():
            if BOARD_ROOT not in tags or len(set(tags) & set(CALIBRATION_IDS)) < 2:
                continue
            k_matrix, d = intrinsics.get(camera), dist.get(camera)

            def rectify(corners, k_matrix=k_matrix, d=d):
                if d is None:
                    return corners
                # Undistorted pixels in the same (normalized->pixel) frame.
                out = cv2.undistortPoints(corners.reshape(-1, 1, 2), k_matrix, d, P=k_matrix)
                return out.reshape(-1, 2)

            root_px = rectify(tags[BOARD_ROOT])
            homography, _ = cv2.findHomography(root_px, root_plane)
            if homography is None:
                continue
            views += 1
            for tag_id, corners in tags.items():
                if tag_id not in CALIBRATION_IDS:
                    continue
                plane = cv2.perspectiveTransform(
                    rectify(corners).reshape(1, -1, 2), homography).reshape(-1, 2)
                samples[tag_id].append(plane)
    layout, spreads = {}, {}
    for tag_id, observations in samples.items():
        if not observations:
            continue
        stack = np.stack(observations)
        median = np.median(stack, axis=0)
        spreads[tag_id] = float(np.median(np.abs(stack - median)) * 1000)
        layout[tag_id] = np.column_stack([median, np.zeros(len(median))])
    return layout, spreads, views


def board_instance_ids(tags, layout, intrinsics, dist):
    return rigid_board_instance_ids(
        tags,
        layout,
        intrinsics,
        dist,
        board_ids=CALIBRATION_IDS,
        root_id=BOARD_ROOT,
        max_error_m=MAX_BOARD_INSTANCE_ERROR_M,
    )


def bundle_adjust(world_from_camera, board_shots, anchor_shots, layout,
                  intrinsics, dist, palette_size_m, fixed_camera=None):
    """Jointly refine every camera pose and every board pose.

    The spanning tree that seeds this chains pairwise transforms, so error
    accumulates along the chain and a camera with few shared views (camera1)
    drags. Here every observed corner in the session pulls on every pose at
    once, which distributes that error instead of propagating it.

    Board poses are free nuisance parameters; only the camera poses are
    returned. The world frame is held fixed by the palette-tag observations,
    which are included as residuals — without them the whole rig could drift
    as a rigid body (gauge freedom).
    """
    from scipy.optimize import least_squares

    names = sorted(world_from_camera)
    free_names = [name for name in names if name != fixed_camera]
    camera_index = {name: i for i, name in enumerate(names)}

    # Observations: (camera_index, pose_index, object_points, image_points).
    # pose_index indexes the free board poses; -1 marks the fixed world frame
    # (the palette tag), whose "board" is the palette square itself.
    observations, initial_poses = [], []
    palette_object = tag_object_points(palette_size_m)
    for shot in board_shots:
        seen = {}
        for camera, tags in shot["cameras"].items():
            if camera not in camera_index:
                continue
            ids = board_instance_ids(
                tags, layout, intrinsics[camera], dist[camera]
            )
            if len(ids) < 2:
                continue
            seen[camera] = (np.concatenate([layout[i] for i in ids]),
                            np.concatenate([tags[i] for i in ids]))
        if not seen:
            continue
        # Seed this shot's board pose from the best-reprojecting camera.
        camera = next(iter(seen))
        obj, img = seen[camera]
        ok, rvec, tvec = cv2.solvePnP(obj, img, intrinsics[camera], dist[camera],
                                      flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        pose_index = len(initial_poses)
        initial_poses.append(world_from_camera[camera] @ to_matrix(rvec, tvec))
        for camera, (obj, img) in seen.items():
            observations.append((camera_index[camera], pose_index, obj, img))
    for shot in anchor_shots:
        for camera, tags in shot["cameras"].items():
            if camera in camera_index and PALETTE_ID in tags:
                observations.append((camera_index[camera], -1,
                                     palette_object, tags[PALETTE_ID]))
    if not observations:
        print("  bundle adjust: no observations, keeping seed poses")
        return world_from_camera

    def pack(matrix):
        return np.concatenate([cv2.Rodrigues(matrix[:3, :3])[0].flatten(), matrix[:3, 3]])

    def unpack(vector):
        return to_matrix(vector[:3], vector[3:6])

    x0 = np.concatenate(
        [pack(np.linalg.inv(world_from_camera[n])) for n in free_names]   # cam_T_world
        + [pack(p) for p in initial_poses])                               # world_T_board

    def residuals(x):
        cam_by_name = {
            name: unpack(x[6 * i:6 * i + 6]) for i, name in enumerate(free_names)
        }
        if fixed_camera is not None:
            cam_by_name[fixed_camera] = np.linalg.inv(world_from_camera[fixed_camera])
        cam = [cam_by_name[name] for name in names]
        base = 6 * len(free_names)
        board = [unpack(x[base + 6 * j:base + 6 * j + 6]) for j in range(len(initial_poses))]
        out = []
        for ci, pi, obj, img in observations:
            transform = cam[ci] if pi < 0 else cam[ci] @ board[pi]
            rvec = cv2.Rodrigues(transform[:3, :3])[0]
            projected, _ = cv2.projectPoints(obj, rvec, transform[:3, 3],
                                             intrinsics[names[ci]], dist[names[ci]])
            out.append((projected.reshape(-1, 2) - img).ravel())
        return np.concatenate(out)

    before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
    result = least_squares(residuals, x0, method="trf", loss="huber", f_scale=2.0,
                           max_nfev=60, verbose=0)
    after = float(np.sqrt(np.mean(result.fun ** 2)))
    print(f"  bundle adjust: {len(observations)} views, {len(initial_poses)} board poses; "
          f"reprojection {before:.3f} -> {after:.3f} px")
    if after > before:
        print("  bundle adjust made it worse — keeping seed poses")
        return world_from_camera
    refined = {
        name: np.linalg.inv(unpack(result.x[6 * i:6 * i + 6]))
        for i, name in enumerate(free_names)
    }
    if fixed_camera is not None:
        refined[fixed_camera] = world_from_camera[fixed_camera]
    return refined


def load_session(session_dir, window_start=None, window_end=None,
                 windows_file=None):
    shot_dirs = sorted(session_dir.glob("shot_*/"))
    if windows_file:
        # Per-hold STILL windows from the guide: a moving board on these
        # rolling-shutter cameras yields sharp-but-sheared corners whose PnP
        # poses poison the extrinsics tree (2026-08-22: bundle adjust started
        # at 324 px and stalled at 16 px). Only still shots may calibrate.
        from fuse_session import filter_shot_dirs
        windows = json.loads(Path(windows_file).read_text())
        before = len(shot_dirs)
        shot_dirs = filter_shot_dirs(shot_dirs, None, None, windows=windows)
        print(f"still windows ({len(windows)} holds): "
              f"{len(shot_dirs)}/{before} shots kept")
    elif window_start is not None and window_end is not None:
        # A guided session mixes phases in one directory. Outside the board
        # phase the board-only context is away, so a lone palette sighting
        # (41 mm, also id 8) reads as a board tag — restrict
        # the solve to the board window and the ambiguity never arises.
        from fuse_session import filter_shot_dirs
        before = len(shot_dirs)
        shot_dirs = filter_shot_dirs(shot_dirs, window_start, window_end)
        print(f"board window {window_start:.0f}..{window_end:.0f}: "
              f"{len(shot_dirs)}/{before} shots kept")
    shots = []
    for shot_dir in shot_dirs:
        detections_path = shot_dir / "detections.json"
        if not detections_path.is_file():
            continue
        detections = json.loads(detections_path.read_text())
        shots.append({
            "name": shot_dir.name,
            "anchor": "anchor" in shot_dir.name,
            "cameras": {
                camera: {int(i): np.asarray(c, np.float64)
                         for i, c in info.get("corners", {}).items()}
                for camera, info in detections.items()
            },
        })
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--board-size-m", type=float, default=_BOARD.edge_m)
    ap.add_argument("--palette-size-m", type=float, default=_PALETTE.edge_m)
    ap.add_argument("--min-views", type=int, default=8)
    ap.add_argument("--layout-iterations", type=int, default=2,
                    help="layout/intrinsics alternations; two is the field-validated "
                         "default for the nine-tag board. More can overfit a "
                         "self-measured planar layout and must still pass the "
                         "reprojection and consensus gates")
    ap.add_argument("--no-bundle-adjust", action="store_true",
                    help="skip the joint refinement (debug/comparison)")
    ap.add_argument("--world-yaw-deg", type=float, default=90.0,
                    help="in-plane rotation from the printed palette tag to the "
                         "URDF palette_tag8 link convention (measured 90 deg for this "
                         "rig: the URDF spreads the camera bar along Y, the raw "
                         "tag frame along X). Set 0 to keep the raw tag frame.")
    ap.add_argument("--window-start", type=float, default=None,
                    help="unix time: only solve on shots captured after this "
                         "(the guided session's board phase — see the report "
                         "card for the exact window)")
    ap.add_argument("--window-end", type=float, default=None)
    ap.add_argument("--windows-file", default=None,
                    help="json list of [start_unix, end_unix] still windows; "
                         "only shots inside them are used")
    args = ap.parse_args()
    if not 1 <= args.layout_iterations <= 4:
        ap.error("--layout-iterations must be between 1 and 4")
    session_dir = Path(args.session_dir).expanduser()
    shots = load_session(session_dir, args.window_start, args.window_end,
                         args.windows_file)
    board_shots = [s for s in shots if not s["anchor"]]
    anchor_shots = [s for s in shots if s["anchor"]]
    print(f"{len(board_shots)} board shots, {len(anchor_shots)} anchor shots")
    if not board_shots:
        sys.exit("no board shots")

    cameras = sorted({c for s in shots for c in s["cameras"]})
    intrinsics = {c: nominal_intrinsics() for c in cameras}
    dist = dict.fromkeys(cameras)

    layout = None
    rmse = {}
    for iteration in range(args.layout_iterations):
        layout, spreads, views = board_layout(
            board_shots, args.board_size_m, intrinsics, dist)
        missing = [i for i in CALIBRATION_IDS if i not in layout]
        if missing:
            print(f"  WARNING: no layout for tags {missing}")
        # -- per-camera intrinsics from the (planar) layout.
        for camera in cameras:
            object_points, image_points = [], []
            for shot in board_shots:
                tags = shot["cameras"].get(camera, {})
                seen = board_instance_ids(
                    tags, layout, intrinsics[camera], dist[camera]
                )
                if len(seen) < 2:
                    continue
                obj = np.concatenate([layout[i] for i in seen])
                img = np.concatenate([tags[i] for i in seen])
                object_points.append(obj.astype(np.float32))
                image_points.append(img.astype(np.float32))
            if len(object_points) < args.min_views:
                print(f"  {camera}: only {len(object_points)} usable views — keeping current model")
                continue
            # Planar target (Zhang): let OpenCV find K itself, and hold the
            # weakly-observed higher-order terms at zero for stability.
            # Do NOT add CALIB_FIX_ASPECT_RATIO: these sensors are not quite
            # square-pixel (the four well-covered cameras independently agree
            # fy/fx ~ 0.99), and forcing 1.0 tripled the p95 cross-camera
            # error by pushing that bias into the good cameras.
            err, k_matrix, d, _, _ = cv2.calibrateCamera(
                object_points, image_points, IMAGE_SIZE,
                intrinsics[camera].copy(),
                (np.zeros(5, np.float64) if dist[camera] is None
                 else np.asarray(dist[camera], np.float64).copy()),
                flags=(cv2.CALIB_USE_INTRINSIC_GUESS
                       | cv2.CALIB_FIX_K3
                       | cv2.CALIB_ZERO_TANGENT_DIST))
            intrinsics[camera], dist[camera], rmse[camera] = k_matrix, d, float(err)
        print(f"iter {iteration}: rmse(px)={ {c: round(e, 3) for c, e in rmse.items()} }")
        print(f"          layout spread(mm)={ {i: round(s, 2) for i, s in spreads.items()} }"
              f"  from {views} views")

    if not layout:
        sys.exit("REFUSE board bundle: failed to estimate board layout from shots")

    quality_failures = []
    width, height = IMAGE_SIZE
    for camera in cameras:
        if camera not in rmse:
            quality_failures.append(f"{camera}: did not reach the minimum usable views")
            continue
        k_matrix = intrinsics[camera]
        fx, fy = float(k_matrix[0, 0]), float(k_matrix[1, 1])
        cx, cy = float(k_matrix[0, 2]), float(k_matrix[1, 2])
        if rmse[camera] > MAX_REPROJECTION_PX:
            quality_failures.append(
                f"{camera}: reprojection {rmse[camera]:.2f} px > {MAX_REPROJECTION_PX:.2f} px"
            )
        if not (0.25 * width <= fx <= 2.0 * width
                and 0.25 * width <= fy <= 2.0 * width
                and 0.8 <= fy / fx <= 1.2
                and 0 <= cx <= width and 0 <= cy <= height):
            quality_failures.append(
                f"{camera}: implausible intrinsics fx={fx:.1f} fy={fy:.1f} "
                f"cx={cx:.1f} cy={cy:.1f}"
            )
    if quality_failures:
        sys.exit(
            "REFUSE board bundle: " + "; ".join(quality_failures)
            + ". The numeric ids are reused across physical targets; keep the board, "
              "wrist, and palette isolated and capture more still tilted board views."
        )

    # -- per-shot board pose per camera, then pairwise camera extrinsics.
    board_pose = {}
    for si, shot in enumerate(board_shots):
        for camera, tags in shot["cameras"].items():
            seen = board_instance_ids(
                tags, layout, intrinsics[camera], dist[camera]
            )
            if len(seen) < 2 or camera not in rmse:
                continue
            obj = np.concatenate([layout[i] for i in seen])
            img = np.concatenate([tags[i] for i in seen])
            ok, rvec, tvec = cv2.solvePnP(obj, img, intrinsics[camera], dist[camera],
                                          flags=cv2.SOLVEPNP_IPPE)
            if ok:
                board_pose[(si, camera)] = to_matrix(rvec, tvec)
    pair_obs = {}
    for si in range(len(board_shots)):
        present = [c for c in cameras if (si, c) in board_pose]
        for a in present:
            for b in present:
                if a < b:
                    t = board_pose[(si, a)] @ np.linalg.inv(board_pose[(si, b)])
                    pair_obs.setdefault((a, b), []).append(t)
    print("pairwise observations:",
          {f"{a}-{b}": len(v) for (a, b), v in pair_obs.items()})
    # Spanning tree from the camera with the most pairwise support.
    support = {c: sum(len(v) for (a, b), v in pair_obs.items() if c in (a, b))
               for c in cameras}
    if not support:
        sys.exit("REFUSE board bundle: no camera support found")
    root = max(support, key=lambda c: support[c])
    root_from = {root: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for (a, b), mats in sorted(pair_obs.items(), key=lambda kv: -len(kv[1])):
            t_ab = consensus_transform(mats)  # a_T_b
            if a in root_from and b not in root_from:
                root_from[b] = root_from[a] @ t_ab
                changed = True
            elif b in root_from and a not in root_from:
                root_from[a] = root_from[b] @ np.linalg.inv(t_ab)
                changed = True
    missing = [c for c in cameras if c not in root_from]
    if missing:
        print(f"WARNING: no graph connection for {missing} — they will be "
              f"omitted from the bundle. Add shared-view shots and rerun.")

    # -- world anchor: palette tag (41 mm id 8), board absent.
    anchor_mats = []
    for shot in anchor_shots:
        for camera, tags in shot["cameras"].items():
            if PALETTE_ID in tags and camera in root_from and camera in rmse:
                candidates = solve_tag(tags[PALETTE_ID], args.palette_size_m,
                                       intrinsics[camera], dist[camera])
                # physical prior: camera above the palette plane
                valid = [m for _, m in candidates
                         if (np.linalg.inv(m))[2, 3] > 0] or [candidates[0][1]]
                # Frames: root_from[c] is root_T_c, solve_tag returns c_T_world.
                # root_T_world = root_T_c @ c_T_world.
                cam_t_world = valid[0]
                anchor_mats.append(root_from[camera] @ cam_t_world)
    if anchor_mats:
        world_from_root = average_transforms([np.linalg.inv(m) for m in anchor_mats])
        world_frame = "palette_tag8"
    else:
        print("WARNING: no anchor shot usable — world frame is the root camera")
        world_from_root, world_frame = np.eye(4), f"{root}_optical"

    # -- joint refinement of every camera pose and every board pose at once.
    world_from_camera_by_name = {
        camera: world_from_root @ root_from[camera]  # world_T_c = world_T_root @ root_T_c
        for camera in cameras
        if camera in root_from and camera in rmse
    }
    if not args.no_bundle_adjust:
        world_from_camera_by_name = bundle_adjust(
            world_from_camera_by_name, board_shots, anchor_shots, layout,
            intrinsics, dist, args.palette_size_m,
            fixed_camera=None if anchor_shots else root)

    # The printed tag's in-plane orientation is not the URDF palette_tag8 link's, so
    # rotate the finished world frame into the robot's convention here rather
    # than making every consumer know about it. This must come AFTER bundle
    # adjustment, which models the palette square as axis-aligned at the
    # origin — rotating first invalidates that anchor.
    if anchor_mats and args.world_yaw_deg:
        angle = np.radians(args.world_yaw_deg)
        yaw = np.eye(4)
        yaw[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
        world_from_camera_by_name = {
            name: yaw @ pose for name, pose in world_from_camera_by_name.items()}
        world_frame = f"{world_frame}_urdf_yaw{int(args.world_yaw_deg)}"

    bundle_cameras = {}
    for camera in cameras:
        if camera not in world_from_camera_by_name:
            continue
        world_from_camera = world_from_camera_by_name[camera]
        # Re-orthonormalize after averaging so visiond's rigidity check passes.
        u, _, vt = np.linalg.svd(world_from_camera[:3, :3])
        rot = u @ vt
        k_matrix = intrinsics[camera]
        bundle_cameras[camera] = {
            "sensor_name": camera,
            "profile": {"stream": "main", "width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1],
                        "fps_num": 20, "fps_den": 1, "format": "h264"},
            "intrinsics": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1],
                           "fx": k_matrix[0, 0], "fy": k_matrix[1, 1],
                           "cx": k_matrix[0, 2], "cy": k_matrix[1, 2]},
            "distortion": {"model": "brown_conrady",
                           "coefficients": np.asarray(dist[camera]).flatten().tolist()},
            "world_from_camera": {"rotation": rot.flatten().tolist(),
                                  "translation_m": world_from_camera[:3, 3].tolist()},
            "depth_to_color": None,
            "metadata": {
                "method": f"board_session_{args.board_size_m * 1000:g}mm_bundle",
                "session": session_dir.name,
                "reproj_rmse_px": f"{rmse[camera]:.4f}",
                "board_layout_ids": ",".join(str(i) for i in sorted(layout)),
                "board_inventory_ids": ",".join(str(i) for i in BOARD_IDS),
                "instance_policy": "phase_isolated_board_target",
            },
        }
    if not bundle_cameras:
        # Every camera fell at the min-views intrinsics gate (rmse gates all
        # downstream membership) — finalize-calibration would just crash on an
        # empty bundle, so refuse here with the actual remedy.
        sys.exit(f"0 cameras calibrated: no camera reached --min-views "
                 f"{args.min_views} usable board views over {len(board_shots)} "
                 f"shots. Capture more board holds, or lower --min-views "
                 f"(coarser intrinsics, honestly reported by verify_calibration).")
    bundle = {"schema_version": 1, "bundle_id": "", "world_frame": world_frame,
              "cameras": bundle_cameras}
    draft = session_dir / "calibration_draft.json"
    draft.write_text(json.dumps(bundle, indent=2))
    layout_out = {str(i): m.tolist() for i, m in layout.items()}
    (session_dir / "board_layout.json").write_text(json.dumps(layout_out, indent=2))
    print(f"draft: {draft}")
    final = session_dir / "calibration.json"
    subprocess.run([str(VISIOND), "finalize-calibration", str(draft),
                    "--output", str(final)], check=True)
    subprocess.run([str(VISIOND), "validate-calibration", str(final)], check=True)
    print(f"final bundle: {final}")


if __name__ == "__main__":
    main()
