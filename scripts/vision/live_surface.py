#!/usr/bin/env python3
"""Live surface reconstruction streamed to a Rerun viewer (runs on the camera node).

Receive -> reconstruct -> log, on a loop, so you can put a hand (or a fake
skin, or a stencil) in the working zone and watch the surface the rig actually
recovers. Each pass logs the coloured mesh, the raw points, and a per-pass
height/score image so it is obvious whether a poor surface came from bad
matching or from nothing being there to match.

  python3 live_surface.py --socket /tmp/tatbot-surface.sock \
      --calibration <bundle.json> \
      --connect rerun+http://<viewer-host>:9876/proxy \
      [--center X Y] [--size W H] [--height-range LO HI] \
      [--xy-step-mm 3] [--iterations 0]

Speed comes from a small zone and a coarse-to-fine sweep; a pass is a few
seconds, not interactive video. Bare silicone is nearly textureless and will
reconstruct poorly by nature — a hand, a stencil, or an inked design gives the
matcher something to work with.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruct_surface import (  # noqa: E402
    load_cameras,
    sweep_coarse_to_fine,
)
from rerun_metadata import log_producer_metadata  # noqa: E402
from visiond_wire import latest_socket_sets  # noqa: E402


def log_status(message: str) -> None:
    rr.log("surface/status", rr.TextLog(message))
    print(message, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", type=Path, required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--connect", required=True, help="rerun+http://HOST:9876/proxy")
    ap.add_argument("--center", nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument("--size", nargs=2, type=float, default=[0.22, 0.16])
    ap.add_argument("--height-range", nargs=2, type=float, default=[-0.06, 0.10])
    ap.add_argument("--xy-step-mm", type=float, default=3.0)
    ap.add_argument("--coarse-mm", type=float, default=4.0)
    ap.add_argument("--fine-mm", type=float, default=1.0)
    ap.add_argument("--patch-mm", type=float, default=4.0)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=0, help="0 = run until stopped")
    ap.add_argument("--recording-id", help="join visiond's live recording so the "
                                           "surface overlays the camera stream")
    ap.add_argument("--log-frustums", action="store_true",
                    help="draw camera frustums here (visiond already does when sharing)")
    args = ap.parse_args()

    world_frame, cameras = load_cameras(args.calibration)
    rr.init("tatbot_vision_v2", recording_id=args.recording_id)
    rr.connect_grpc(args.connect)
    log_producer_metadata(
        rr,
        "live_surface",
        args.recording_id,
        Path(args.calibration),
    )
    log_status(f"starting; waiting for synchronized frames on {args.socket}")
    try:
        # Cosmetic (tells the 3D view which way is up). The rerun-sdk wheel on
        # some nodes are built against a different numpy ABI and reject this one
        # batch type; everything else logs fine, so do not let it stop us.
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    except Exception as error:  # noqa: BLE001
        print(f"  (view-coordinates hint skipped: {error})")

    if args.log_frustums:
        for name, cam in sorted(cameras.items()):
            world_from_cam = np.linalg.inv(cam["cam_from_world"])
            rr.log(f"world/cameras/{name}",
                   rr.Transform3D(translation=world_from_cam[:3, 3],
                                  mat3x3=world_from_cam[:3, :3]), static=True)
            rr.log(f"world/cameras/{name}",
                   rr.Pinhole(image_from_camera=cam["K"], width=2960, height=1668,
                              camera_xyz=rr.ViewCoordinates.RDF,
                              image_plane_distance=0.08), static=True)

    step = args.xy_step_mm / 1000.0
    xs = np.arange(args.center[0] - args.size[0] / 2, args.center[0] + args.size[0] / 2, step)
    ys = np.arange(args.center[1] - args.size[1] / 2, args.center[1] + args.size[1] / 2, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = args.patch_mm / 1000.0
    offsets = np.array([[dx * radius, dy * radius]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)], np.float64)
    print(f"world frame {world_frame}; grid {grid_x.shape[1]}x{grid_x.shape[0]} "
          f"@ {args.xy_step_mm} mm; heights {args.height_range[0]:+.3f}..{args.height_range[1]:+.3f} m")

    # Zone outline, so the operator can see where to put their hand.
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    for z, path in ((args.height_range[0], "floor"), (args.height_range[1], "ceiling")):
        rr.log(f"world/zone/{path}",
               rr.LineStrips3D([[[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]]],
                               colors=[[120, 120, 160]]), static=True)

    iteration = 0
    for frame_set in latest_socket_sets(args.socket.expanduser()):
        if args.iterations and iteration >= args.iterations:
            break
        iteration += 1
        # Share visiond's `capture_time` timeline so the surface lines up with
        # the live camera frames instead of living on a timeline of its own.
        rr.set_time("capture_time", timestamp=frame_set["timestamp_ns"] / 1e9)
        rr.set_time("pass", sequence=iteration)
        frames = {name: frame["image"] for name, frame in frame_set["frames"].items()}
        if len(frames) < 3:
            log_status(f"pass {iteration}: only {len(frames)} cameras returned frames; skipped")
            continue
        captured = time.monotonic()
        log_status(f"pass {iteration}: reconstructing synchronized set {frame_set['sequence']}")

        names, images, colors = [], {}, {}
        for name in sorted(frames):
            if name not in cameras:
                continue
            undistorted = cv2.undistort(frames[name], cameras[name]["K"], cameras[name]["dist"])
            images[name] = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY).astype(np.float32)
            colors[name] = undistorted
            names.append(name)

        height, score, _ = sweep_coarse_to_fine(
            images, cameras, names, grid_x, grid_y, tuple(args.height_range),
            args.coarse_mm, args.fine_mm, offsets)
        height = height.reshape(grid_x.shape)
        score = score.reshape(grid_x.shape)
        good = np.isfinite(height) & (score >= args.min_score)

        points = np.stack([grid_x[good], grid_y[good], height[good]], axis=-1)
        if len(points):
            # Colour from the most face-on camera, cheaply: nearest camera.
            rgb = np.zeros((len(points), 3), np.uint8)
            best = np.full(len(points), np.inf)
            world = np.concatenate([points, np.ones((len(points), 1))], axis=1).T
            for name in names:
                cam = cameras[name]
                local = cam["cam_from_world"] @ world
                depth = local[2]
                u = cam["K"][0, 0] * local[0] / depth + cam["K"][0, 2]
                v = cam["K"][1, 1] * local[1] / depth + cam["K"][1, 2]
                h_px, w_px = images[name].shape
                inside = (u >= 0) & (u < w_px - 1) & (v >= 0) & (v < h_px - 1) & (depth > 0)
                take = inside & (depth < best)
                if take.any():
                    rgb[take] = colors[name][v[take].astype(int), u[take].astype(int)][:, ::-1]
                    best[take] = depth[take]
            rr.log("world/surface", rr.Points3D(points, colors=rgb, radii=step * 0.5))
        else:
            rr.log("world/surface", rr.Points3D(np.zeros((0, 3))))

        # Height and confidence as images: shows WHY a pass looks the way it does.
        shown = np.where(good, height, np.nan)
        if np.isfinite(shown).any():
            lo, hi = np.nanpercentile(shown, 2), np.nanpercentile(shown, 98)
            span = max(hi - lo, 1e-3)
            normalized = np.clip((np.nan_to_num(shown, nan=lo) - lo) / span, 0, 1)
            tinted = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            tinted[~good] = (40, 40, 40)
        else:
            tinted = np.full((*shown.shape, 3), 40, dtype=np.uint8)
        rr.log("surface/height", rr.Image(cv2.flip(tinted, 0)[:, :, ::-1]))
        rr.log("surface/confidence", rr.Image(cv2.flip(
            (np.clip(score, 0, 1) * 255).astype(np.uint8), 0)))
        # rerun-sdk's ScalarBatch faults on some numpy ABIs; TextLog works.
        age_s = max(0.0, time.time() - frame_set["timestamp_ns"] / 1e9)
        log_status(
            f"pass {iteration}: {len(names)} cams, {good.sum()} pts "
            f"({100 * good.mean():.1f}%), median NCC "
            f"{float(np.median(score[good])) if good.any() else 0:.3f}, "
            f"compute {time.monotonic() - captured:.1f}s, source age {age_s:.1f}s"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
