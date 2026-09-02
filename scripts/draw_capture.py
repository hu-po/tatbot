#!/usr/bin/env python3
"""Capture server for `tatbot draw`: the wrist D405s, answered over a directory.

The C++ executor (wxai_teleop --draw-dir D) holds the arm still at each orbit
pose and writes D/capture/request-<k>.json; this process answers with
capture-<k>.npz (per-pixel median depth of >= 8 frames per camera, the depth
intrinsics, the depth unit, one colour frame, the joints it was asked to
stamp) and then capture-<k>.done. docs/draw.md "Capture handshake" is the
contract; the keys written here are exactly the ones listed there.

    draw_capture.py serve <dir>                 # started by draw_run.sh
    draw_capture.py once <dir> --k 1 --joints j0..j5 --carriage 0.002
    draw_capture.py serve|once ... --fake       # tilted-plane synthetic depth

Runs in the LeRobot venv (numpy, pyrealsense2, lerobot) because that is the
interpreter that owns the configured D405s on the camera host; `--fake` needs
only numpy. The cameras
are opened through LeRobot's RealSenseCamera exactly as scripts/depth_probe.py
does, including its 0.1 mm depth-unit trap: `units_m_<role>` is the metres per
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
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np

SCHEMA_REQUEST = "tatbot.draw-capture-request/1"
ROLES = ("wrist_upper", "wrist_lower")
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 30
MIN_FRAMES = 8
POLL_HZ = 20.0
D405_UNITS_M = 0.0001
D405_RANGE_M = (0.07, 0.5)
REPO = Path(__file__).resolve().parent.parent

# One deprecation warning per camera per read drowns the capture lines.
logging.getLogger("lerobot.cameras.realsense.camera_realsense").setLevel(logging.ERROR)


def registry_cameras() -> dict[str, str]:
    """role -> serial from the visiond sensor registry (never serials in code)."""
    import tomllib

    toml = REPO / "rust/visiond/config/vision.toml"
    try:
        cams = tomllib.loads(toml.read_text()).get("cameras", {}).get("realsense", [])
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"draw_capture: cannot read sensor registry {toml}: {e}")
    roles = {c.get("role"): c["serial"] for c in cams if c.get("role")}
    missing = [r for r in ROLES if r not in roles]
    if missing:
        sys.exit(f"draw_capture: {toml} names no realsense with role {missing}")
    return {r: roles[r] for r in ROLES}


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
        # frame, and the unit must be decided after that (depth_probe.py).
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


class FakeCamera:
    """A tilted plane ~150 mm away with a hole and D405-like noise, for dry runs."""

    def __init__(self, role: str, seed: int):
        self.role, self.serial = role, f"fake-{role}"
        self.units_m = D405_UNITS_M
        self.intrinsics = np.array([385.0, 385.0, DEPTH_W / 2, DEPTH_H / 2, DEPTH_W, DEPTH_H])
        self.rng = np.random.default_rng(seed)
        tilt = 0.12 if role == "wrist_upper" else -0.08
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


def open_cameras(fake: bool) -> list:
    if fake:
        cams = [FakeCamera(r, seed=i) for i, r in enumerate(ROLES)]
    else:
        cams = [RealSense(r, s) for r, s in registry_cameras().items()]
    for c in cams:
        fx, fy, ppx, ppy, w, h = c.intrinsics
        print(f"draw_capture: {c.role} ({c.serial}) depth {int(w)}x{int(h)} fx {fx:.1f} fy {fy:.1f} "
              f"pp ({ppx:.1f}, {ppy:.1f}); {c.units_m * 1000:.4f} mm per unit", flush=True)
    return cams


# --- one capture ---------------------------------------------------------------


def median_depth(cam, frames: int = MIN_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of the valid samples of `frames` depth frames, and the valid count."""
    stack = np.stack([cam.read_depth() for _ in range(frames)]).astype(np.float32)
    valid = valid_mask(stack)
    stack[~valid] = np.nan
    count = valid.sum(axis=0).astype(np.uint8)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are holes, handled below
        med = np.nanmedian(stack, axis=0)
    med = np.where(count > 0, np.nan_to_num(med, nan=0.0), 0.0)
    return np.clip(np.round(med), 0, 65534).astype(np.uint16), count


def capture(cams: list, out_dir: Path, k: int, joints, carriage_m: float, t_wall: float,
            frames: int = MIN_FRAMES) -> Path:
    arrays: dict[str, np.ndarray] = {
        "joints": np.asarray(joints, dtype=np.float64).reshape(6),
        "carriage_m": np.float64(carriage_m),
        "k": np.int64(k),
        "t_wall": np.float64(t_wall),
    }
    report = []
    for cam in cams:
        depth, count = median_depth(cam, frames)
        color = cam.read_color()
        arrays[f"depth_{cam.role}"] = depth
        arrays[f"valid_{cam.role}"] = count
        arrays[f"units_m_{cam.role}"] = np.float64(cam.units_m)
        arrays[f"intrinsics_{cam.role}"] = np.asarray(cam.intrinsics, dtype=np.float64)
        if color is not None and color.ndim == 3 and color.shape[2] == 3:
            arrays[f"color_{cam.role}"] = color
        good = count > 0
        med_mm = float(np.median(depth[good])) * cam.units_m * 1000.0 if good.any() else float("nan")
        flag = "" if D405_RANGE_M[0] * 1000 <= med_mm <= D405_RANGE_M[1] * 1000 else " !range"
        report.append(f"{cam.role} valid {100 * good.mean():5.1f}% median {med_mm:6.1f} mm{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"capture-{k}.npz"
    tmp = out_dir / f".capture-{k}.tmp.npz"
    with open(tmp, "wb") as fh:  # a file object keeps numpy from appending a second .npz
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    (out_dir / f"capture-{k}.done").touch()
    print(f"capture {k}: " + " | ".join(report), flush=True)
    return final


# --- serve -----------------------------------------------------------------------


def read_request(path: Path) -> dict | None:
    """The executor may still be writing; a partial file is retried on the next poll."""
    try:
        req = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if req.get("schema") != SCHEMA_REQUEST:
        print(f"draw_capture: {path.name}: unknown schema {req.get('schema')!r}; ignored", file=sys.stderr)
        return {}
    return req


def serve(out_dir: Path, fake: bool, frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = out_dir / "server.stop"
    stop_flag.unlink(missing_ok=True)
    stopping = []

    def _stop(signum, _frame):
        stopping.append(signum)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    cams = open_cameras(fake)
    answered: set[int] = set()
    try:
        (out_dir / "server.ready").write_text(json.dumps({
            "pid": os.getpid(), "fake": fake, "frames": frames, "t_wall": time.time(),
            "cameras": {c.role: {"serial": c.serial, "units_m": c.units_m,
                                 "intrinsics": [float(x) for x in c.intrinsics]} for c in cams},
        }) + "\n")
        print(f"draw_capture: serving {out_dir} ({'FAKE cameras' if fake else 'both D405s streaming'})", flush=True)
        period = 1.0 / POLL_HZ
        while not stopping and not stop_flag.exists():
            t0 = time.monotonic()
            for req_path in sorted(out_dir.glob("request-*.json")):
                try:
                    k = int(req_path.stem.split("-", 1)[1])
                except ValueError:
                    continue
                if k in answered or (out_dir / f"capture-{k}.done").exists():
                    answered.add(k)
                    continue
                req = read_request(req_path)
                if req is None:
                    continue
                if not req:
                    answered.add(k)
                    continue
                capture(cams, out_dir, k, req.get("joints", [0.0] * 6), float(req.get("carriage_m", 0.0)),
                        float(req.get("t_wall", time.time())), frames)
                answered.add(k)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        for c in cams:
            c.close()
        (out_dir / "server.ready").unlink(missing_ok=True)
    why = "server.stop" if stop_flag.exists() else f"signal {stopping[0]}" if stopping else "done"
    print(f"draw_capture: stopped ({why}); {len(answered)} capture(s) answered", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="answer request-<k>.json in <dir> until server.stop or SIGTERM")
    s.add_argument("dir", type=Path)
    o = sub.add_parser("once", help="one capture without the executor (bench aid)")
    o.add_argument("dir", type=Path)
    o.add_argument("--k", type=int, default=1)
    o.add_argument("--joints", type=float, nargs=6, default=[0.0] * 6, metavar="J")
    o.add_argument("--carriage", type=float, default=0.002, help="carriage reading, metres")
    for p in (s, o):
        p.add_argument("--fake", action="store_true", help="synthetic tilted-plane depth; no cameras opened")
        p.add_argument("--frames", type=int, default=MIN_FRAMES, help=f"frames per camera per capture (>= {MIN_FRAMES})")
    a = ap.parse_args(argv)
    if a.frames < MIN_FRAMES:
        ap.error(f"--frames must be >= {MIN_FRAMES} (docs/draw.md)")
    if a.cmd == "serve":
        return serve(a.dir, a.fake, a.frames)
    cams = open_cameras(a.fake)
    try:
        path = capture(cams, a.dir, a.k, a.joints, a.carriage, time.time(), a.frames)
    finally:
        for c in cams:
            c.close()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
