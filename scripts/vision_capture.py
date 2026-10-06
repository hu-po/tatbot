#!/usr/bin/env python3
"""Wrist D405 captures without an executor (`tatbot vision capture`).

Each capture is capture-<k>.npz (per-pixel median depth of >= 8 frames per
camera, the depth intrinsics, the depth unit, one colour frame, the joints it
was asked to stamp) followed by capture-<k>.done. docs/surface-formats.md
"Wrist captures" is the contract; the keys written here are exactly the ones
listed there.

    vision_capture.py once <dir> --arm left --k 1 --joints j0..j5 --carriage 0.002
                                                # --arm names the physical arm whose wrist
                                                # cameras answer (left|right, never a default);
                                                # the arm's registry roles must be ones this
                                                # checkout retains geometry for (ROLES)
    vision_capture.py once ... --fake          # tilted-plane synthetic depth

Runs in the LeRobot venv (numpy, pyrealsense2, lerobot) because that is the
interpreter that owns the configured D405s on the camera host; `--fake` needs
only numpy. The cameras
are opened through LeRobot's RealSenseCamera, including its 0.1 mm depth-unit
trap (scripts/il_patch_lerobot.py patch 6; the retired depth_probe.py found it): `units_m_<role>` is the metres per
raw unit of what this process actually stored, patch-aware, so a reader never
has to guess.

This process never touches the arm.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30
MIN_FRAMES = 8
D405_UNITS_M = 0.0001
D405_RANGE_M = (0.07, 0.5)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

from wrist_cameras import capture_roles, optical_frames  # noqa: E402

# Current capture roles are a view of the camera registry, not a fixed list of
# arm colours or wrist mount names.
ROLES = tuple(role for roles in capture_roles(REPO).values() for role in roles)

# One deprecation warning per camera per read drowns the capture lines.
logging.getLogger("lerobot.cameras.realsense.camera_realsense").setLevel(logging.ERROR)


def registry_cameras(arm: str) -> dict[str, str]:
    """role -> serial of `arm`'s wrist cameras from the visiond sensor registry
    (never serials in code); the arm is the program's, never a default."""
    from tatbot_cli import nodes
    from wrist_cameras import for_arm

    try:
        cams = for_arm(REPO, arm, nodes.this_node(nodes.load(REPO)))
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"vision_capture: wrist ownership refused: {e}")
    roles = {c.get("role"): c["serial"] for c in cams if c.get("role")}
    try:
        retained = optical_frames(REPO, arm=arm, stream='depth')
    except (OSError, ValueError) as error:
        sys.exit(f'vision_capture: {arm}-arm camera roles lack retained wrist geometry: {error}')
    if not roles or not set(roles).issubset(retained):
        sys.exit(f'vision_capture: {arm}-arm camera roles lack retained wrist geometry '
                 f'(registry {sorted(roles)}; retained {list(retained)})')
    # A camera assigned to the other physical arm cannot borrow this arm's
    # extrinsic. Capture only the remaining manifested, locally owned views.
    return {r: roles[r] for r in retained if r in roles}


def valid_mask(raw: np.ndarray) -> np.ndarray:
    """0 means no measurement; 65535 means the stereo match saturated."""
    return (raw > 0) & (raw < 65535)


# --- cameras -----------------------------------------------------------------


class RealSense:
    """One D405 through LeRobot's RealSenseCamera, depth on, 640x480@30."""

    def __init__(self, role: str, serial: str):
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        self.role, self.serial = role, serial
        cfg = RealSenseCameraConfig(serial, fps=DEPTH_FPS, width=DEPTH_W, height=DEPTH_H, use_depth=True)
        self.cam = RealSenseCamera(cfg)
        self.cam.connect()
        # Priming read: a patched install stamps its unit marker on the first
        # frame, and the unit must be decided after that (the patch-6 trap).
        self.cam.read_depth()
        self.units_m = self._units_m()
        self.intrinsics = self._intrinsics()

    def _units_m(self) -> float:
        if getattr(self.cam, "_tatbot_mm_per_unit", None) is not None:
            return 0.001  # il_patch_lerobot patch 6: read_depth() already returns millimetres
        try:
            sensor = self.cam.rs_profile.get_device().first_depth_sensor()
            return float(sensor.get_depth_scale())
        except Exception:  # noqa: BLE001
            return D405_UNITS_M

    def _intrinsics(self) -> np.ndarray:
        import pyrealsense2 as rs

        i = self.cam.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return np.array([i.fx, i.fy, i.ppx, i.ppy, i.width, i.height], dtype=np.float64)

    def read_depth(self) -> np.ndarray:
        return np.squeeze(np.asarray(self.cam.read_depth())).astype(np.uint16)

    def read_color(self) -> np.ndarray | None:
        try:
            return np.asarray(self.cam.read()).astype(np.uint8)
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        disconnect = getattr(self.cam, "disconnect", None)
        if disconnect:
            with contextlib.suppress(Exception):
                disconnect()


# The synthetic plane's tilt per role: the two right mounts look at it from
# either side, the left wrist from a third.
FAKE_TILT_RAD = {"wrist_upper": 0.12, "wrist_lower": -0.08, "wrist_left": 0.10}


class FakeCamera:
    """A tilted plane ~150 mm away with a hole and D405-like noise, for dry runs."""

    def __init__(self, role: str, seed: int):
        self.role, self.serial = role, f"fake-{role}"
        self.units_m = D405_UNITS_M
        self.intrinsics = np.array([385.0, 385.0, DEPTH_W / 2, DEPTH_H / 2, DEPTH_W, DEPTH_H])
        self.rng = np.random.default_rng(seed)
        tilt = FAKE_TILT_RAD.get(role, 0.10 + 0.02 * seed)
        fx, fy, ppx, ppy = self.intrinsics[:4]
        v, u = np.mgrid[0:DEPTH_H, 0:DEPTH_W]
        rays = np.stack([(u - ppx) / fx, (v - ppy) / fy, np.ones_like(u, dtype=np.float64)], -1)
        n = np.array([np.sin(tilt), 0.3 * np.sin(tilt), np.cos(tilt)])
        n /= np.linalg.norm(n)
        self.z_m = 0.15 / (rays @ n)  # ray-plane intersection, plane 150 mm along its normal
        self.hole = (u - 200) ** 2 + (v - 300) ** 2 < 40**2

    def read_depth(self) -> np.ndarray:
        noise = self.rng.normal(0.0, 0.0004, self.z_m.shape)
        raw = np.round((self.z_m + noise) / self.units_m).astype(np.uint16)
        raw[self.hole] = 0
        raw[self.rng.random(raw.shape) < 0.02] = 0
        return raw

    def read_color(self) -> np.ndarray:
        return np.full((DEPTH_H, DEPTH_W, 3), 200, dtype=np.uint8)

    def close(self) -> None:
        pass


def open_cameras(fake: bool, from_owner: bool, arm: str) -> list:
    """The named arm's wrist cameras: the owner's stream, the synthetic pair
    for that arm's roles, or the devices themselves. The arm is never a default."""
    if fake and from_owner:
        raise ValueError('fake and owner capture are mutually exclusive')
    configured = capture_roles(REPO)
    if arm not in configured:
        raise ValueError(f'capture arm {arm!r} is not configured')
    if not configured[arm]:
        raise ValueError(f'{arm}: no wrist camera is configured')
    if from_owner:
        from capture_owner import OwnerCameras
        cams = OwnerCameras(registry_cameras(arm), repo=REPO)
    elif fake:
        cams = [FakeCamera(r, seed=i) for i, r in enumerate(configured[arm])]
    else:
        cams = [RealSense(r, s) for r, s in registry_cameras(arm).items()]
    for c in cams:
        fx, fy, ppx, ppy, w, h = c.intrinsics
        print(f"vision_capture: {c.role} ({c.serial}) depth {int(w)}x{int(h)} fx {fx:.1f} fy {fy:.1f} "
              f"pp ({ppx:.1f}, {ppy:.1f}); {c.units_m * 1000:.4f} mm per unit", flush=True)
    return cams


# --- one capture ---------------------------------------------------------------


def median_depth(cam, frames: int = MIN_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of the valid samples of `frames` depth frames, and the valid count."""
    stack = np.stack([cam.read_depth() for _ in range(frames)]).astype(np.float32)
    cam.raw_depth = stack.astype(np.uint16)
    valid = valid_mask(stack)
    stack[~valid] = np.nan
    count = valid.sum(axis=0).astype(np.uint8)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are holes, handled below
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack-med[None]), axis=0)
    cam.temporal_mad_m = np.nan_to_num(mad, nan=0.0)*cam.units_m
    med = np.where(count > 0, np.nan_to_num(med, nan=0.0), 0.0)
    return np.clip(np.round(med), 0, 65534).astype(np.uint16), count


def _camera_arrays(cam, frames: int, arrays: dict) -> str:
    """Fill `arrays` with one camera's evidence; return its report line."""
    depth, count = median_depth(cam, frames)
    if hasattr(cam, "evidence_arrays"):
        arrays.update(cam.evidence_arrays())
    color = cam.read_color()
    arrays[f"depth_{cam.role}"] = depth
    arrays[f"valid_{cam.role}"] = count
    arrays[f"temporal_mad_m_{cam.role}"] = cam.temporal_mad_m.astype(np.float32)
    arrays[f"raw_depth_{cam.role}"] = cam.raw_depth
    arrays[f"frames_{cam.role}"] = np.int64(frames)
    arrays[f"units_m_{cam.role}"] = np.float64(cam.units_m)
    arrays[f"intrinsics_{cam.role}"] = np.asarray(cam.intrinsics, dtype=np.float64)
    if color is not None and color.ndim == 3 and color.shape[2] == 3:
        arrays[f"color_{cam.role}"] = color
    good = count > 0
    med_mm = float(np.median(depth[good])) * cam.units_m * 1000.0 if good.any() else float("nan")
    flag = "" if D405_RANGE_M[0] * 1000 <= med_mm <= D405_RANGE_M[1] * 1000 else " !range"
    return f"{cam.role} valid {100 * good.mean():5.1f}% median {med_mm:6.1f} mm{flag}"


def _write_wrist_window(cams, path: Path) -> None:
    """The intersection of the selected wrist intervals, not their union."""
    timestamps = [[r['metadata']['timestamps'].get('normalized_unix_ns') for r in cam.records]
                  for cam in cams]
    if not all(all(type(t) is int and t > 0 for t in row) for row in timestamps):
        return
    window = {'after_ns': max(min(row) for row in timestamps),
              'before_ns': min(max(row) for row in timestamps)}
    if window['after_ns'] > window['before_ns']:
        raise ValueError('wrist capture windows do not overlap')
    path.write_text(json.dumps(window)+'\n')


def capture(cams: list, out_dir: Path, k: int, joints, carriage_m: float, t_wall: float,
            frames: int = MIN_FRAMES) -> Path:
    if hasattr(cams, "begin_capture"):
        cams.begin_capture(frames)
    arrays: dict[str, np.ndarray] = {
        "camera_roles": np.array(json.dumps([cam.role for cam in cams])),
        "joints": np.asarray(joints, dtype=np.float64).reshape(6),
        "carriage_m": np.float64(carriage_m),
        "k": np.int64(k),
        "t_wall": np.float64(t_wall),
    }
    report = []
    for cam in cams:
        report.append(_camera_arrays(cam, frames, arrays))

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"capture-{k}.npz"
    tmp = out_dir / f".capture-{k}.tmp.npz"
    with open(tmp, "wb") as fh:  # a file object keeps numpy from appending a second .npz
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    if hasattr(cams, 'begin_capture'):
        _write_wrist_window(cams, out_dir / f'capture-{k}.window.json')
    (out_dir / f"capture-{k}.done").touch()
    print(f"capture {k}: " + " | ".join(report), flush=True)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("once", help="one capture without the executor (bench aid)")
    o.add_argument("dir", type=Path)
    o.add_argument("--k", type=int, default=1)
    o.add_argument("--joints", type=float, nargs=6, default=[0.0] * 6, metavar="J")
    o.add_argument("--carriage", type=float, default=0.002, help="carriage reading, metres")
    o.add_argument("--arm", choices=tuple(capture_roles(REPO)), required=True,
                   help="the physical arm whose wrist cameras answer (never a default)")
    source = o.add_mutually_exclusive_group()
    source.add_argument("--fake", action="store_true", help="synthetic tilted-plane depth; no cameras opened")
    source.add_argument("--from-owner", action="store_true", help="subscribe to the existing D405 owner; never open a camera")
    o.add_argument("--frames", type=int, default=MIN_FRAMES, help=f"frames per camera per capture (>= {MIN_FRAMES})")
    a = ap.parse_args(argv)
    if a.frames < MIN_FRAMES:
        ap.error(f"--frames must be >= {MIN_FRAMES} (docs/surface-formats.md)")
    cams = open_cameras(a.fake, a.from_owner, a.arm)
    try:
        path = capture(cams, a.dir, a.k, a.joints, a.carriage, time.time(), a.frames)
    finally:
        for c in cams:
            c.close()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
