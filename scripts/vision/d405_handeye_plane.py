#!/usr/bin/env python3
"""Plane-route hand-eye for the wrist D405s (surface-first drawing, decision 7).

    d405_handeye_plane.py <capture-dir>... [--out DIR] [--roi 0.5] [--paper-z Z]
    d405_handeye_plane.py --self-test

Input: capture-*.npz files (docs/draw.md "Capture handshake") taken with the
D405s looking at the touched-off paper plane from many wrist orientations —
a `tatbot draw scan` orbit, or `draw_capture.py once` at hand-guided poses.
For each capture and camera the central depth ROI is fitted with a plane
(RANSAC + least squares) in the camera optical frame. The nominal URDF chain
root_from_camera(joints) predicts where that plane is in root; the truth is
z = paper_plane_z (config/workspace.yaml) with normal +z. Per camera, one
correction

    link6_from_camera = link6_from_camera_nominal @ Exp(delta),  delta = (dr, dt)

is solved by Gauss-Newton on [tangent components of the predicted normal (2),
predicted plane offset error along +z (1)] over every capture.

What a plane cannot see: translation IN the plane (the two components of dt
orthogonal to the camera-frame normal) — those are regularised to zero and
reported as unobservable. With varied tilts the rotation is fully determined.

Writes <out>/d405_handeye.json (link6_from_depth_optical per role, provenance,
residuals) and prints the equivalent <origin xyz rpy> for a human to fold into
urdf/tatbot.urdf. It never edits the URDF.

Any numpy python; the URDF chain comes from scripts/vision/urdf_kinematics.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from urdf_kinematics import UrdfChain, driver_joint_names  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
URDF = REPO / "urdf/tatbot.urdf"
WORKSPACE = REPO / "config/workspace.yaml"
LINK6 = "right/link_6"
OPTICAL_FRAME = {
    "wrist_upper": "right/realsense_depth_optical_frame",
    "wrist_lower": "right/realsense_lower_depth_optical_frame",
}
MOUNT_JOINT = {"wrist_upper": "right/realsense_mount_joint", "wrist_lower": "right/realsense_lower_mount_joint"}
TRUE_NORMAL = np.array([0.0, 0.0, 1.0])
D405_RANGE_M = (0.07, 0.5)
OFFSET_SCALE_M = 0.15  # 1 rad of normal error ~ 150 mm of offset at the working standoff
INPLANE_WEIGHT = 1e3  # Tikhonov on the unobservable in-plane translation (rad-equivalent per metre)

# --- small SO(3)/SE(3) ------------------------------------------------------------------


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def exp_so3(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    kk = skew(k)
    return np.eye(3) + np.sin(th) * kk + (1 - np.cos(th)) * (kk @ kk)


def log_so3(rot):
    c = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])


def exp_se3(delta):
    """delta = (dr (3), dt (3)); rotation then translation, first-order coupling ignored on purpose:
    the solve is a few-mrad correction and the parameterisation only needs to be a chart."""
    tf = np.eye(4)
    tf[:3, :3] = exp_so3(delta[:3])
    tf[:3, 3] = delta[3:]
    return tf


def rpy_from_matrix(rot):
    """URDF rpy (Rz(y) Ry(p) Rx(r)); inverse of urdf_kinematics._rpy_to_matrix."""
    pitch = -np.arcsin(np.clip(rot[2, 0], -1, 1))
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw])


# --- plane fit in the camera frame -------------------------------------------------------


def deproject_roi(depth, intrinsics, units_m, roi):
    fx, fy, ppx, ppy = (float(x) for x in np.asarray(intrinsics)[:4])
    h, w = depth.shape
    dh, dw = int(h * roi / 2), int(w * roi / 2)
    sub = depth[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw]
    v, u = np.nonzero((sub > 0) & (sub < 65535))
    z = sub[v, u].astype(np.float64) * units_m
    u = u + (w // 2 - dw)
    v = v + (h // 2 - dh)
    keep = (z >= D405_RANGE_M[0]) & (z <= D405_RANGE_M[1])
    u, v, z = u[keep], v[keep], z[keep]
    return np.stack([(u - ppx) / fx * z, (v - ppy) / fy * z, z], -1)


def fit_plane(points, iters=60, inlier_m=0.0015, seed=0):
    """(n, d) with n unit and n . x = d for x on the plane, n oriented toward the camera (d > 0 side
    flipped so the camera origin is on the negative side: n points from the plane back at the camera)."""
    if len(points) < 50:
        raise ValueError(f"too few valid depth points for a plane ({len(points)})")
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        d = n @ p0
        count = int((np.abs(points @ n - d) < inlier_m).sum())
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    inl = np.abs(points @ n - d) < inlier_m
    q = points[inl]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[2]
    d = float(n @ c)
    if d > 0:  # camera at the origin; put the camera on the positive side of the plane
        n, d = -n, -d
    resid = float(np.sqrt(np.mean((q @ n - d) ** 2)))
    return n, d, resid, int(inl.sum()), len(points)


# --- the solve ----------------------------------------------------------------------------


def transform_plane(tf, n_cam, d_cam):
    """Plane {n.x = d} from the camera frame into the frame tf maps to."""
    rot, t = tf[:3, :3], tf[:3, 3]
    n = rot @ n_cam
    return n, d_cam + n @ t


def residuals(delta, obs, paper_z):
    """obs: list of (root_from_link6, link6_from_cam_nominal, n_cam, d_cam). Returns (r, per-capture rows)."""
    corr = exp_se3(delta)
    rows = []
    for root_from_link6, l6_from_cam, n_cam, d_cam in obs:
        tf = root_from_link6 @ l6_from_cam @ corr
        n, d = transform_plane(tf, n_cam, d_cam)
        if n @ TRUE_NORMAL < 0:
            n, d = -n, -d
        rows.append([n[0], n[1], (d - paper_z) / OFFSET_SCALE_M])
    return np.asarray(rows).ravel(), np.asarray(rows)


def solve(obs, paper_z, iters=30):
    """Gauss-Newton with numeric Jacobian (6 params, cheap) and in-plane translation regularised."""
    delta = np.zeros(6)
    # mean camera-frame plane normal: the direction of dt a plane CAN see
    n_bar = np.mean([n for _, _, n, _ in obs], axis=0)
    n_bar /= np.linalg.norm(n_bar)
    p_inplane = np.eye(3) - np.outer(n_bar, n_bar)
    reg_rows = INPLANE_WEIGHT * np.hstack([np.zeros((3, 3)), p_inplane])
    eps = 1e-6
    for _ in range(iters):
        r0, _ = residuals(delta, obs, paper_z)
        jac = np.zeros((len(r0), 6))
        for j in range(6):
            dp = np.zeros(6)
            dp[j] = eps
            jac[:, j] = (residuals(delta + dp, obs, paper_z)[0] - r0) / eps
        a_mat = np.vstack([jac, reg_rows])
        b = np.concatenate([-r0, -reg_rows @ delta])
        step, *_ = np.linalg.lstsq(a_mat, b, rcond=None)
        delta = delta + step
        if np.linalg.norm(step) < 1e-10:
            break
    return delta, n_bar


def summarize(rows):
    ang = np.degrees(np.arcsin(np.clip(np.linalg.norm(rows[:, :2], axis=1), 0, 1)))
    off = rows[:, 2] * OFFSET_SCALE_M * 1000.0
    return {"angle_deg_rms": float(np.sqrt(np.mean(ang ** 2))), "angle_deg_max": float(ang.max()),
            "offset_mm_rms": float(np.sqrt(np.mean(off ** 2))), "offset_mm_max": float(np.abs(off).max())}


# --- inputs ---------------------------------------------------------------------------------


def paper_plane_z(path=WORKSPACE):
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("paper_plane_z:"):
            return float(s.split(":", 1)[1].split("#")[0])
    raise ValueError(f"no paper_plane_z in {path}")


def joint_map(joints):
    return dict(zip(driver_joint_names("right", 6), [float(j) for j in joints], strict=True))


def nominal_link6_from_cam(chain, role):
    zero = {}
    return np.linalg.inv(chain.link_pose(LINK6, zero)) @ chain.link_pose(OPTICAL_FRAME[role], zero)


def load_observations(capture_dirs, chain, roi):
    """role -> (obs list, per-capture notes)."""
    per_role = {r: [] for r in OPTICAL_FRAME}
    notes = {r: [] for r in OPTICAL_FRAME}
    files = []
    for d in capture_dirs:
        d = Path(d).expanduser()
        files += sorted((d / "capture").glob("capture-*.npz")) if (d / "capture").is_dir() else []
        files += sorted(d.glob("capture-*.npz"))
    if not files:
        sys.exit(f"d405_handeye_plane: no capture-*.npz under {[str(d) for d in capture_dirs]}")
    for f in files:
        z = np.load(f)
        joints = np.asarray(z["joints"], float)
        root_from_link6 = chain.link_pose(LINK6, joint_map(joints))
        for role in OPTICAL_FRAME:
            if f"depth_{role}" not in z:
                continue
            pts = deproject_roi(z[f"depth_{role}"], z[f"intrinsics_{role}"], float(z[f"units_m_{role}"]), roi)
            try:
                n, d, resid, inl, tot = fit_plane(pts)
            except ValueError as error:
                notes[role].append(f"{f.name}: skipped ({error})")
                continue
            per_role[role].append((root_from_link6, nominal_link6_from_cam(chain, role), n, d))
            notes[role].append(f"{f.name}: plane {d * 1000:.1f} mm, fit rms {resid * 1000:.2f} mm, "
                               f"{inl}/{tot} inliers")
    return per_role, notes, [str(f) for f in files]


# --- report -----------------------------------------------------------------------------------


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def solve_role(role, obs, paper_z, chain):
    nominal = obs[0][1]
    before = summarize(residuals(np.zeros(6), obs, paper_z)[1])
    delta, n_bar = solve(obs, paper_z)
    after = summarize(residuals(delta, obs, paper_z)[1])
    corrected = nominal @ exp_se3(delta)
    # Equivalent single fixed joint link_6 -> optical, and the same correction
    # folded into the existing mount joint (its child chain stays as in the URDF).
    zero = {}
    mount_child = chain.joints[MOUNT_JOINT[role]]["child"]
    mount_from_cam = np.linalg.inv(chain.link_pose(mount_child, zero)) @ chain.link_pose(OPTICAL_FRAME[role], zero)
    link6_from_mount = corrected @ np.linalg.inv(mount_from_cam)
    return {
        "urdf_frame": OPTICAL_FRAME[role],
        "captures": len(obs),
        "link6_from_depth_optical": corrected.tolist(),
        "link6_from_depth_optical_nominal": nominal.tolist(),
        "delta_rotvec_deg": np.degrees(delta[:3]).tolist(),
        "delta_translation_mm": (delta[3:] * 1000).tolist(),
        "delta_translation_along_normal_mm": float(delta[3:] @ n_bar * 1000),
        "unobservable": "in-plane translation (the 2 components of dt orthogonal to the mean camera-frame "
                        "plane normal) is not constrained by a plane; regularised to zero",
        "camera_frame_mean_normal": n_bar.tolist(),
        "residual_before": before,
        "residual_after": after,
        "origin_link6_to_optical": {"xyz": corrected[:3, 3].tolist(),
                                    "rpy": rpy_from_matrix(corrected[:3, :3]).tolist()},
        "origin_mount_joint": {"joint": MOUNT_JOINT[role], "xyz": link6_from_mount[:3, 3].tolist(),
                               "rpy": rpy_from_matrix(link6_from_mount[:3, :3]).tolist()},
    }


def print_role(role, rec):
    b, a = rec["residual_before"], rec["residual_after"]
    print(f"\n{role} ({rec['urdf_frame']}), {rec['captures']} captures")
    print(f"  before: normal {b['angle_deg_rms']:.3f} deg rms / {b['angle_deg_max']:.3f} max; "
          f"offset {b['offset_mm_rms']:.2f} mm rms / {b['offset_mm_max']:.2f} max")
    print(f"  after : normal {a['angle_deg_rms']:.3f} deg rms / {a['angle_deg_max']:.3f} max; "
          f"offset {a['offset_mm_rms']:.2f} mm rms / {a['offset_mm_max']:.2f} max")
    dr, dt = rec["delta_rotvec_deg"], rec["delta_translation_mm"]
    print(f"  delta : rot ({dr[0]:+.3f}, {dr[1]:+.3f}, {dr[2]:+.3f}) deg; "
          f"trans ({dt[0]:+.2f}, {dt[1]:+.2f}, {dt[2]:+.2f}) mm "
          f"[along normal {rec['delta_translation_along_normal_mm']:+.2f} mm; in-plane regularised to 0]")
    o = rec["origin_mount_joint"]
    print(f"  URDF  : <joint name=\"{o['joint']}\"> <origin xyz=\"{o['xyz'][0]:.8f} {o['xyz'][1]:.8f} "
          f"{o['xyz'][2]:.8f}\" rpy=\"{o['rpy'][0]:.8f} {o['rpy'][1]:.8f} {o['rpy'][2]:.8f}\"/>  "
          "(keeps the mount->optical chain below it as-is)")
    o = rec["origin_link6_to_optical"]
    print(f"          equivalent single fixed joint link_6 -> {rec['urdf_frame']}: "
          f"xyz=\"{o['xyz'][0]:.8f} {o['xyz'][1]:.8f} {o['xyz'][2]:.8f}\" "
          f"rpy=\"{o['rpy'][0]:.8f} {o['rpy'][1]:.8f} {o['rpy'][2]:.8f}\"")


def run(capture_dirs, out, roi, paper_z, urdf):
    chain = UrdfChain(urdf)
    z_true = paper_z if paper_z is not None else paper_plane_z()
    per_role, notes, files = load_observations(capture_dirs, chain, roi)
    result = {"schema": "tatbot.d405-handeye/1", "roles": {},
              "provenance": {"captures": files, "paper_plane_z": z_true, "urdf": str(urdf),
                             "workspace": str(WORKSPACE), "roi": roi, "git": git_sha(), "t_wall": time.time(),
                             "notes": notes}}
    print(f"paper plane z = {z_true * 1000:.2f} mm (normal +z); {len(files)} capture file(s)")
    for role, obs in per_role.items():
        for line in notes[role]:
            print(f"  {role} {line}")
        if len(obs) < 3:
            print(f"\n{role}: only {len(obs)} usable capture(s); need >= 3 at distinct tilts — not solved")
            continue
        rec = solve_role(role, obs, z_true, chain)
        result["roles"][role] = rec
        print_role(role, rec)
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "d405_handeye.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {path}  (the URDF is NOT edited; fold the origin above in by hand and re-run to verify)")
    return 0


# --- self-test ---------------------------------------------------------------------------------


def _look_at_plane(standoff, tilt_x, tilt_y, spin, paper_z):
    """root_from_cam for a camera whose optical +z looks down at the plane (through tilts) from standoff."""
    rot = exp_so3([np.pi, 0, 0])  # optical z -> root -z
    rot = exp_so3([tilt_x, 0, 0]) @ exp_so3([0, tilt_y, 0]) @ rot @ exp_so3([0, 0, spin])
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = [0.3, -0.2, paper_z + standoff]
    return tf


def _synth_depth(root_from_cam, paper_z, intr, units_m, noise_m, rng):
    fx, fy, ppx, ppy, w, h = intr
    cam_from_root = np.linalg.inv(root_from_cam)
    n_c, d_c = transform_plane(cam_from_root, TRUE_NORMAL, paper_z)
    v, u = np.mgrid[0:int(h), 0:int(w)]
    rays = np.stack([(u - ppx) / fx, (v - ppy) / fy, np.ones_like(u, float)], -1)
    z = d_c / (rays @ n_c)
    z = z + rng.normal(0, noise_m, z.shape) if noise_m else z
    raw = np.round(z / units_m)
    raw[(z <= 0) | ~np.isfinite(z)] = 0
    return np.clip(raw, 0, 65534).astype(np.uint16)


def self_test(tmp=None, noise_m=0.0):
    import tempfile

    rng = np.random.default_rng(3)
    chain = UrdfChain(URDF)
    paper_z = 0.005945
    intr = np.array([385.0, 385.0, 320.0, 240.0, 640.0, 480.0])
    units_m = 1e-4
    truth = {"wrist_upper": np.array([np.radians(0.5), np.radians(-0.3), np.radians(0.2), 0, 0, 0.002]),
             "wrist_lower": np.array([np.radians(-0.4), np.radians(0.6), np.radians(-0.25), 0, 0, -0.0015])}
    tilts = [(0, 0, 0), (0.25, 0, 0.3), (-0.25, 0, -0.3), (0, 0.25, 1.0), (0, -0.25, -1.0), (0.18, 0.18, 2.0),
             (-0.18, 0.18, -2.0)]
    tmp = Path(tmp or tempfile.mkdtemp(prefix="d405-handeye-selftest-"))
    for k, (tx, ty, spin) in enumerate(tilts, 1):
        arrays = {"joints": np.zeros(6), "carriage_m": 0.002, "k": k, "t_wall": 0.0}
        # The synthetic rig has no encoder path: generate root_from_link6 so that the TRUE
        # camera pose looks at the plane, and stash it in the file for the test harness.
        for role, delta in truth.items():
            nominal = nominal_link6_from_cam(chain, role)
            l6_from_cam_true = nominal @ exp_se3(delta)
            root_from_cam = _look_at_plane(0.15, tx, ty, spin, paper_z)
            root_from_link6 = root_from_cam @ np.linalg.inv(l6_from_cam_true)
            arrays[f"_root_from_link6_{role}"] = root_from_link6
            arrays[f"depth_{role}"] = _synth_depth(root_from_cam, paper_z, intr, units_m, noise_m, rng)
            arrays[f"intrinsics_{role}"] = intr
            arrays[f"units_m_{role}"] = units_m
        np.savez_compressed(tmp / f"capture-{k}.npz", **arrays)

    ok = True
    for role, delta_true in truth.items():
        obs = []
        for f in sorted(tmp.glob("capture-*.npz")):
            z = np.load(f)
            pts = deproject_roi(z[f"depth_{role}"], z[f"intrinsics_{role}"], float(z[f"units_m_{role}"]), 0.5)
            n, d, *_ = fit_plane(pts)
            obs.append((z[f"_root_from_link6_{role}"], nominal_link6_from_cam(chain, role), n, d))
        delta, n_bar = solve(obs, paper_z)
        after = summarize(residuals(delta, obs, paper_z)[1])
        rot_err = np.degrees(np.linalg.norm(log_so3(exp_so3(delta[:3]).T @ exp_so3(delta_true[:3]))))
        t_err = abs((delta[3:] - delta_true[3:]) @ n_bar) * 1000
        good = rot_err <= 0.05 and t_err <= 0.1
        ok &= good
        print(f"self-test {role}: rotation error {rot_err:.4f} deg, translation-along-normal error "
              f"{t_err:.4f} mm, residual after {after['angle_deg_rms']:.4f} deg / {after['offset_mm_rms']:.4f} mm "
              f"-> {'PASS' if good else 'FAIL'}")
    print("self-test: in-plane translation is unobservable with a plane and is not scored")
    print(f"self-test: {'PASS' if ok else 'FAIL'} (fixtures in {tmp})")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*", type=Path, help="draw dirs or capture dirs holding capture-*.npz")
    ap.add_argument("--out", type=Path, help="where d405_handeye.json goes (default: first capture dir)")
    ap.add_argument("--roi", type=float, default=0.5, help="central fraction of the depth image to fit")
    ap.add_argument("--paper-z", type=float, help="override config/workspace.yaml paper_plane_z (metres)")
    ap.add_argument("--urdf", type=Path, default=URDF)
    ap.add_argument("--self-test", action="store_true", help="fabricate captures from a known perturbation and recover it")
    ap.add_argument("--self-test-noise-mm", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test(noise_m=a.self_test_noise_mm / 1000.0)
    if not a.captures:
        ap.error("give at least one capture dir (or --self-test)")
    return run(a.captures, a.out or a.captures[0], a.roi, a.paper_z, a.urdf)


if __name__ == "__main__":
    sys.exit(main())
