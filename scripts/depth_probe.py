#!/usr/bin/env python3
"""Phase 0 gate: can the wrist D405s actually see the paper?

Everything in the depth plan rests on this. The D405 is a short-range STEREO
camera, and stereo needs texture — near-white paper is the classic failure
case. The printed grid may supply enough; it may not. Find out in half an hour
rather than after re-recording a dataset.

Run one labelled capture per pose, teleoping between them:

    scripts/depth_probe.py hover 15     # tool held clear of the paper
    scripts/depth_probe.py touch 15     # tool down on the paper, held still

Then `scripts/depth_probe.py --report` prints the comparison, or hand the
output directory to whoever is analysing.

Reads the cameras only. It never touches the arm, so it is safe to run
alongside teleop.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# One deprecation warning per camera per read drowns the live readout.
logging.getLogger("lerobot.cameras.realsense.camera_realsense").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import tatbot_runlog  # noqa: E402


def _registry_cameras() -> dict[str, str]:
    """RealSense name -> serial from the visiond sensor registry — the same
    file the vision daemon trusts, so no serials live in code (plan Phase 3)."""
    import tomllib

    toml = Path(__file__).resolve().parent.parent / "rust/visiond/config/vision.toml"
    try:
        cams = tomllib.loads(toml.read_text()).get("cameras", {}).get("realsense", [])
    except (OSError, tomllib.TOMLDecodeError) as e:
        sys.exit(f"depth_probe: cannot read sensor registry {toml}: {e}")
    if not cams:
        sys.exit(f"depth_probe: no realsense cameras in {toml} — describe yours "
                 "there (see rust/visiond/config/vision.example.toml)")
    # Key by role where the registry states one (wrist_upper/wrist_lower),
    # else by name — the labels in the readout should mean something.
    return {(c.get("role") or c["name"]): c["serial"] for c in cams}


def _cameras() -> dict[str, str]:
    global _CAMERAS
    if _CAMERAS is None:
        _CAMERAS = _registry_cameras()
    return _CAMERAS


_CAMERAS = None
OUT = tatbot_runlog.log_root() / "depth_probe"
ROI = 0.4  # central fraction of the frame used for the paper statistics
D405_RANGE_MM = (70.0, 500.0)  # datasheet working range; outside it, depth is noise

# LeRobot's read_depth() docstring says the uint16 it returns is millimetres.
# Upstream that is true of most D400s but NOT of the D405, whose depth units
# are 0.0001 m — which is why il_patch_lerobot.py patch 6 now converts to mm
# inside the camera itself. A patched camera returns true millimetres and
# scaling again here would understate every distance 10x (that bug cost the
# first phase-0 read on 2026-08-21: "median 16 mm" was really 160). Detect the
# patch at runtime — it stamps _tatbot_mm_per_unit on the camera after the
# first frame — and only apply the device scale on an unpatched install.
DEFAULT_SCALE_MM = 0.1


def depth_scale_mm(cam) -> float:
    """Millimetres per unit of what read_depth() returns, patch-aware.

    Call only after at least one depth read, so a patched camera has had the
    chance to stamp its marker.
    """
    if getattr(cam, "_tatbot_mm_per_unit", None) is not None:
        return 1.0  # patch 6 already converted to millimetres
    try:
        sensor = cam.rs_profile.get_device().first_depth_sensor()
        return float(sensor.get_depth_scale()) * 1000.0
    except Exception:
        return DEFAULT_SCALE_MM


def _valid(raw: np.ndarray) -> np.ndarray:
    """0 means no measurement; 65535 means the stereo match saturated."""
    return (raw > 0) & (raw < 65535)


def _roi(img: np.ndarray) -> np.ndarray:
    # LeRobot returns depth as (H, W, 1); squeeze so downstream indexing is 2-D.
    img = np.squeeze(img)
    h, w = img.shape[:2]
    dh, dw = int(h * ROI / 2), int(w * ROI / 2)
    return img[h // 2 - dh : h // 2 + dh, w // 2 - dw : w // 2 + dw]


def capture(label: str, seconds: float, hz: float = 10.0):
    cams = {}
    for name, serial in _cameras().items():
        cfg = RealSenseCameraConfig(serial, fps=30, width=640, height=480, use_depth=True)
        cam = RealSenseCamera(cfg)
        cam.connect()
        cams[name] = cam
        print(f"connected {name} ({serial})")

    # one priming read per camera, so a patched install has stamped its
    # unit marker before the scale is decided (see depth_scale_mm)
    for cam in cams.values():
        cam.read_depth()
    scales = {n: depth_scale_mm(c) for n, c in cams.items()}
    for n, mm in scales.items():
        note = " (patched camera, already millimetres)" if mm == 1.0 else ""
        print(f"  {n}: {mm:.4f} mm per depth unit{note}")

    frames = {n: [] for n in cams}
    rgb0 = {}
    t0 = time.monotonic()
    period = 1.0 / hz
    print(f"\ncapturing '{label}' for {seconds:.0f}s — hold the pose still\n")
    print(f"{'t':>5s}  " + "  ".join(f"{n:>28s}" for n in cams))
    try:
        nxt = t0
        while (t := time.monotonic() - t0) < seconds:
            line = f"{t:5.1f}  "
            for n, cam in cams.items():
                d = np.asarray(cam.read_depth())
                frames[n].append(d.astype(np.uint16))
                if n not in rgb0:
                    with contextlib.suppress(Exception):
                        rgb0[n] = np.asarray(cam.read())
                r = _roi(d)
                valid = _valid(r)
                med = float(np.median(r[valid])) * scales[n] if valid.any() else float("nan")
                flag = " " if D405_RANGE_MM[0] <= med <= D405_RANGE_MM[1] else "!"
                line += f"  valid {100 * valid.mean():5.1f}%  med {med:6.1f}mm{flag}"
            print(line, flush=True)
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        print("\ninterrupted — saving what we have")
    finally:
        for cam in cams.values():
            with_ = getattr(cam, "disconnect", None)
            if with_:
                with_()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.npz"
    np.savez_compressed(
        path,
        **{f"depth_{n}": np.stack(v) for n, v in frames.items() if v},
        **{f"rgb_{n}": v for n, v in rgb0.items()},
        **{f"scale_{n}": np.array(v) for n, v in scales.items()},
        roi=np.array(ROI),
    )
    print(f"\nsaved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    summarise(label)


def summarise(label: str):
    path = OUT / f"{label}.npz"
    z = np.load(path)
    print(f"\n=== {label} ===")
    for n in _cameras():
        key = f"depth_{n}"
        if key not in z:
            continue
        scale = float(z[f"scale_{n}"]) if f"scale_{n}" in z else DEFAULT_SCALE_MM
        d = z[key]
        r = np.stack([_roi(f) for f in d])
        valid = _valid(r)
        r = r.astype(np.float32) * scale
        print(f"  {n}")
        print(f"    valid pixels over paper : {100 * valid.mean():5.1f}%")
        if not valid.any():
            print("    NO VALID DEPTH — the sensor cannot see this surface")
            continue
        vals = r[valid]
        med_all = float(np.median(vals))
        if not D405_RANGE_MM[0] <= med_all <= D405_RANGE_MM[1]:
            print(f"    OUT OF RANGE — median {med_all:.0f} mm is outside the D405's "
                  f"{D405_RANGE_MM[0]:.0f}-{D405_RANGE_MM[1]:.0f} mm window; "
                  "the camera is not pointed at the work surface")
        print(f"    distance                : median {np.median(vals):6.1f} mm  "
              f"p05 {np.percentile(vals, 5):6.1f}  p95 {np.percentile(vals, 95):6.1f}")
        # temporal noise: per-pixel std across frames, where the pixel is always valid
        always = valid.all(axis=0)
        if always.sum() > 50:
            noise = r[:, always].std(axis=0)
            print(f"    temporal noise (static) : median {np.median(noise):5.2f} mm  "
                  f"p95 {np.percentile(noise, 95):5.2f} mm  over {int(always.sum())} px")
        else:
            print("    temporal noise          : too few consistently-valid pixels to measure")
        # flatness: how well the middle frame's ROI fits a plane
        mid = r[len(r) // 2]
        m = valid[len(r) // 2]
        if m.sum() > 100:
            yy, xx = np.mgrid[0 : mid.shape[0], 0 : mid.shape[1]]
            a = np.c_[xx[m], yy[m], np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(a, mid[m], rcond=None)
            resid = mid[m] - a @ coef
            print(f"    flatness (plane fit)    : residual std {resid.std():5.2f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", help="name for this capture, e.g. hover or touch")
    ap.add_argument("seconds", nargs="?", type=float, default=15.0)
    ap.add_argument("--report", action="store_true", help="re-print stats for every saved capture")
    a = ap.parse_args()
    if a.report:
        for p in sorted(OUT.glob("*.npz")):
            summarise(p.stem)
        sys.exit(0)
    if not a.label:
        ap.error("give a label, e.g. 'hover' or 'touch' (or use --report)")
    capture(a.label, a.seconds)
