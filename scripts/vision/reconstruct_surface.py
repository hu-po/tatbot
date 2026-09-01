#!/usr/bin/env python3
"""Multi-view surface reconstruction over the tattoo working volume (camera node).

Consumes a calibrated PoE rig (`calibrate_board_session.py`) and one
synchronized decoded shot, and produces a metric height field of whatever lies
in the working zone, expressed in the calibration's world frame — the shape the
downstream stroke mapper wants (a triangle mesh with per-vertex normals, which
it walks geodesically to keep the needle perpendicular to the surface).

Method: plane sweep. The skin rests on a table and is therefore 2.5D, so rather
than matching pixels between image pairs and triangulating, we hypothesise the
answer directly: for every (x, y) in a world-frame grid, try a range of heights,
project each candidate 3D point into all five cameras, and keep the height whose
image patches agree best (mean pairwise NCC). Sweeping in world space rather
than image space means every camera contributes symmetrically, occlusion is just
a missing vote, and the output is already a metric height field — no rectifying,
no per-pair disparity maps, no fusion step.

Patches are sampled on the world-frame tangent plane, so a patch covers the same
physical millimetres in every camera regardless of its distance or obliquity.

  python3 reconstruct_surface.py <shot_dir> --calibration <bundle.json> \
      [--auto-zone <calib_session>] [--center X Y] [--size W H] \
      [--height-range LO HI] [--xy-step-mm 1.0] [--z-step-mm 0.5] --out <dir>

Textureless surfaces (bare silicone) have nothing to match and will come back
sparse or noisy; a stencilled or otherwise textured surface is the intended
input. See docs/vision.md for the accuracy this rig can support.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiducials import load_inventory  # noqa: E402

BOARD_IDS = list(load_inventory().target("board").ids)


def load_cameras(bundle_path):
    bundle = json.loads(Path(bundle_path).expanduser().read_text())
    cameras = {}
    for name, cal in bundle["cameras"].items():
        k = cal["intrinsics"]
        matrix = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float64)
        world_from_cam = np.eye(4)
        world_from_cam[:3, :3] = np.asarray(cal["world_from_camera"]["rotation"]).reshape(3, 3)
        world_from_cam[:3, 3] = cal["world_from_camera"]["translation_m"]
        cameras[name] = {
            "K": matrix,
            "dist": np.asarray(cal["distortion"]["coefficients"], np.float64),
            "cam_from_world": np.linalg.inv(world_from_cam),
            "position": world_from_cam[:3, 3],
        }
    return bundle["world_frame"], cameras


def auto_zone(session_dir, cameras, margin_m=0.02):
    """Working volume = where the calibration board actually went.

    The operator swept the board through the region they care about, so its
    observed extent is a better zone than any guessed constant.
    """
    session_dir = Path(session_dir).expanduser()
    layout = {int(i): np.asarray(c, np.float64)
              for i, c in json.loads((session_dir / "board_layout.json").read_text()).items()}
    centers = []
    for shot in sorted(session_dir.glob("shot_*/")):
        if "anchor" in shot.name:
            continue
        detections = json.loads((shot / "detections.json").read_text())
        for name, info in detections.items():
            if name not in cameras:
                continue
            tags = {int(i): np.asarray(c, np.float64)
                    for i, c in info.get("corners", {}).items()}
            ids = [i for i in BOARD_IDS if i in tags and i in layout]
            if len(ids) < 2:
                continue
            obj = np.concatenate([layout[i] for i in ids])
            img = np.concatenate([tags[i] for i in ids])
            ok, rvec, tvec = cv2.solvePnP(obj, img, cameras[name]["K"], cameras[name]["dist"],
                                          flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            cam_from_board = np.eye(4)
            cam_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
            cam_from_board[:3, 3] = tvec.reshape(3)
            world_from_cam = np.linalg.inv(cameras[name]["cam_from_world"])
            centers.append((world_from_cam @ cam_from_board)[:3, 3])
            break
    if not centers:
        sys.exit("auto-zone: no board poses recovered")
    points = np.stack(centers)
    low = points.min(axis=0) - margin_m
    high = points.max(axis=0) + margin_m
    return low, high


def sample_bilinear(image, u, v):
    """Bilinear gather with an out-of-bounds mask."""
    height, width = image.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    valid = (u0 >= 0) & (u0 < width - 1) & (v0 >= 0) & (v0 < height - 1)
    u0c = np.clip(u0, 0, width - 2)
    v0c = np.clip(v0, 0, height - 2)
    fu = (u - u0c).astype(np.float32)
    fv = (v - v0c).astype(np.float32)
    top = image[v0c, u0c] * (1 - fu) + image[v0c, u0c + 1] * fu
    bottom = image[v0c + 1, u0c] * (1 - fu) + image[v0c + 1, u0c + 1] * fu
    return top * (1 - fv) + bottom * fv, valid


def sweep_candidates(images, cameras, names, flat_x, flat_y, candidates, patch_offsets):
    """Score a per-point set of height hypotheses.

    `candidates` is (n_hypotheses, n_points), so each grid point can be given
    its own heights — which is what makes a coarse-to-fine sweep possible:
    scan the whole volume coarsely, then re-scan a narrow band around each
    point's winner. Returns (best_height, best_score, votes).
    """
    count = flat_x.size
    patch = patch_offsets.shape[0]
    best_score = np.full(count, -np.inf, np.float32)
    best_height = np.full(count, np.nan, np.float32)
    best_votes = np.zeros(count, np.int16)

    for row in candidates:
        z_row = np.broadcast_to(row, (count,)) if row.ndim else np.full(count, float(row))
        px = (flat_x[None, :] + patch_offsets[:, 0:1]).ravel()
        py = (flat_y[None, :] + patch_offsets[:, 1:2]).ravel()
        pz = np.tile(z_row, patch)
        world = np.stack([px, py, pz, np.ones_like(px)])

        patches, masks = [], []
        for name in names:
            cam = cameras[name]
            local = cam["cam_from_world"] @ world
            depth = local[2]
            in_front = depth > 1e-6
            depth = np.where(in_front, depth, 1.0)
            u = cam["K"][0, 0] * local[0] / depth + cam["K"][0, 2]
            v = cam["K"][1, 1] * local[1] / depth + cam["K"][1, 2]
            values, inside = sample_bilinear(images[name], u, v)
            ok = inside & in_front
            patches.append(values.reshape(patch, count))
            masks.append(ok.reshape(patch, count).all(axis=0))

        stack = np.stack(patches)
        visible = np.stack(masks)
        mean = stack.mean(axis=1, keepdims=True)
        centered = stack - mean
        norm = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
        visible &= norm[:, 0, :] > 1e-3
        normalized = centered / np.maximum(norm, 1e-6)

        score = np.zeros(count, np.float32)
        pairs = np.zeros(count, np.float32)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                both = visible[a] & visible[b]
                if not both.any():
                    continue
                ncc = (normalized[a] * normalized[b]).sum(axis=0)
                score += np.where(both, ncc, 0.0).astype(np.float32)
                pairs += both
        votes = visible.sum(axis=0).astype(np.int16)
        mean_ncc = np.where(pairs > 0, score / np.maximum(pairs, 1), -np.inf)
        mean_ncc = np.where(votes >= 3, mean_ncc, -np.inf)

        better = mean_ncc > best_score
        best_score = np.where(better, mean_ncc, best_score)
        best_height = np.where(better, z_row, best_height)
        best_votes = np.where(better, votes, best_votes)
    return best_height, best_score, best_votes


def sweep_coarse_to_fine(images, cameras, names, grid_x, grid_y, height_range,
                         coarse_mm, fine_mm, patch_offsets):
    """Full-volume coarse pass, then a narrow fine pass around each winner."""
    flat_x, flat_y = grid_x.ravel(), grid_y.ravel()
    coarse = np.arange(height_range[0], height_range[1], coarse_mm / 1000.0)
    height, score, votes = sweep_candidates(
        images, cameras, names, flat_x, flat_y,
        [np.float64(z) for z in coarse], patch_offsets)
    if fine_mm >= coarse_mm:
        return height, score, votes
    steps = int(np.ceil(coarse_mm / fine_mm))
    seed = np.where(np.isfinite(height), height, np.mean(height_range))
    offsets = (np.arange(-steps, steps + 1) * fine_mm / 1000.0)
    fine_height, fine_score, fine_votes = sweep_candidates(
        images, cameras, names, flat_x, flat_y,
        [seed + offset for offset in offsets], patch_offsets)
    better = fine_score > score
    return (np.where(better, fine_height, height),
            np.where(better, fine_score, score),
            np.where(better, fine_votes, votes))


def plane_sweep(images, cameras, names, grid_x, grid_y, heights, patch_offsets):
    """Return (best_height, best_score, votes) over the grid."""
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    count = flat_x.size
    patch = patch_offsets.shape[0]

    best_score = np.full(count, -np.inf, np.float32)
    best_height = np.full(count, np.nan, np.float32)
    best_votes = np.zeros(count, np.int16)

    for z in heights:
        # Patch samples on the world tangent plane: same physical area in
        # every camera, so NCC compares like with like.
        px = (flat_x[None, :] + patch_offsets[:, 0:1]).ravel()
        py = (flat_y[None, :] + patch_offsets[:, 1:2]).ravel()
        pz = np.full(px.shape, z, np.float64)
        world = np.stack([px, py, pz, np.ones_like(px)])

        patches, masks = [], []
        for name in names:
            cam = cameras[name]
            local = cam["cam_from_world"] @ world
            depth = local[2]
            in_front = depth > 1e-6
            depth = np.where(in_front, depth, 1.0)
            u = cam["K"][0, 0] * local[0] / depth + cam["K"][0, 2]
            v = cam["K"][1, 1] * local[1] / depth + cam["K"][1, 2]
            values, inside = sample_bilinear(images[name], u, v)
            ok = inside & in_front
            patches.append(values.reshape(patch, count))
            masks.append(ok.reshape(patch, count).all(axis=0))

        stack = np.stack(patches)                    # (cam, patch, count)
        visible = np.stack(masks)                    # (cam, count)
        mean = stack.mean(axis=1, keepdims=True)
        centered = stack - mean
        norm = np.sqrt((centered ** 2).sum(axis=1, keepdims=True))
        # A patch with no contrast carries no information; drop it rather
        # than let 0/0 masquerade as a perfect match.
        textured = norm[:, 0, :] > 1e-3
        visible &= textured
        normalized = centered / np.maximum(norm, 1e-6)

        score = np.zeros(count, np.float32)
        pairs = np.zeros(count, np.float32)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                both = visible[a] & visible[b]
                if not both.any():
                    continue
                ncc = (normalized[a] * normalized[b]).sum(axis=0)
                score += np.where(both, ncc, 0.0).astype(np.float32)
                pairs += both
        votes = visible.sum(axis=0).astype(np.int16)
        mean_ncc = np.where(pairs > 0, score / np.maximum(pairs, 1), -np.inf)
        # Require agreement from at least three cameras: two can agree by
        # coincidence along an epipolar line, three rarely do.
        mean_ncc = np.where(votes >= 3, mean_ncc, -np.inf)

        better = mean_ncc > best_score
        best_score = np.where(better, mean_ncc, best_score)
        best_height = np.where(better, z, best_height)
        best_votes = np.where(better, votes, best_votes)
    return best_height, best_score, best_votes


def write_ply(path, vertices, normals, colors, faces):
    with open(path, "wb") as f:
        header = [
            "ply", "format binary_little_endian 1.0",
            f"element vertex {len(vertices)}",
            "property float x", "property float y", "property float z",
            "property float nx", "property float ny", "property float nz",
            "property uchar red", "property uchar green", "property uchar blue",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices", "end_header",
        ]
        f.write(("\n".join(header) + "\n").encode())
        vertex = np.empty(len(vertices), dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        vertex["x"], vertex["y"], vertex["z"] = vertices.T
        vertex["nx"], vertex["ny"], vertex["nz"] = normals.T
        vertex["red"], vertex["green"], vertex["blue"] = colors.T
        f.write(vertex.tobytes())
        if len(faces):
            face = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
            face["n"] = 3
            face["a"], face["b"], face["c"] = faces.T
            f.write(face.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shot_dir", help="directory of cameraN.png stills")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--auto-zone", help="calibration session whose board sweep defines the zone")
    ap.add_argument("--center", nargs=2, type=float, help="zone centre x y (m, world frame)")
    ap.add_argument("--size", nargs=2, type=float, default=[0.25, 0.20], help="zone width height (m)")
    ap.add_argument("--height-range", nargs=2, type=float, default=[-0.03, 0.12],
                    help="height search range (m, world frame)")
    ap.add_argument("--xy-step-mm", type=float, default=1.0)
    ap.add_argument("--z-step-mm", type=float, default=0.5)
    ap.add_argument("--patch-mm", type=float, default=3.0, help="tangent-plane patch radius")
    ap.add_argument("--min-score", type=float, default=0.5, help="minimum mean pairwise NCC")
    ap.add_argument("--cameras", help="comma-separated subset to use (default: all calibrated)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shot_dir = Path(args.shot_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    world_frame, cameras = load_cameras(args.calibration)
    if args.cameras:
        keep = {name.strip() for name in args.cameras.split(",")}
        cameras = {name: cal for name, cal in cameras.items() if name in keep}

    # Undistort once, then treat every camera as an exact pinhole.
    images, colors = {}, {}
    for name in list(cameras):
        path = shot_dir / f"{name}.png"
        if not path.is_file():
            print(f"  {name}: no image in shot, skipping")
            del cameras[name]
            continue
        bgr = cv2.imread(str(path))
        undistorted = cv2.undistort(bgr, cameras[name]["K"], cameras[name]["dist"])
        images[name] = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY).astype(np.float32)
        colors[name] = undistorted
    names = sorted(cameras)
    if len(names) < 3:
        sys.exit("need at least three calibrated cameras with images")
    print(f"world frame: {world_frame}; cameras: {', '.join(names)}")

    if args.auto_zone:
        low, high = auto_zone(args.auto_zone, cameras)
        center = (low[:2] + high[:2]) / 2
        size = np.maximum(high[:2] - low[:2], 0.05)
        height_range = (low[2], high[2])
        print(f"auto zone: centre ({center[0]:+.3f},{center[1]:+.3f}) m, "
              f"size {size[0]:.3f}x{size[1]:.3f} m, heights {height_range[0]:+.3f}..{height_range[1]:+.3f} m")
    else:
        center = np.asarray(args.center if args.center else [0.0, 0.0], float)
        size = np.asarray(args.size, float)
        height_range = tuple(args.height_range)

    step = args.xy_step_mm / 1000.0
    xs = np.arange(center[0] - size[0] / 2, center[0] + size[0] / 2, step)
    ys = np.arange(center[1] - size[1] / 2, center[1] + size[1] / 2, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    heights = np.arange(height_range[0], height_range[1], args.z_step_mm / 1000.0)
    radius = args.patch_mm / 1000.0
    offsets = np.array([[dx * radius, dy * radius]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)], np.float64)
    print(f"grid {grid_x.shape[1]}x{grid_x.shape[0]} @ {args.xy_step_mm} mm, "
          f"{len(heights)} height hypotheses @ {args.z_step_mm} mm, 3x3 patch @ {args.patch_mm} mm")

    height, score, votes = plane_sweep(images, cameras, names, grid_x, grid_y, heights, offsets)
    height = height.reshape(grid_x.shape)
    score = score.reshape(grid_x.shape)
    votes = votes.reshape(grid_x.shape)

    good = np.isfinite(height) & (score >= args.min_score)
    print(f"accepted {good.sum()} / {good.size} grid points "
          f"({100 * good.mean():.1f}%) at NCC >= {args.min_score}")
    if not good.any():
        sys.exit("nothing reconstructed — is the surface textured, and is the zone right?")
    # A lone accepted point surrounded by rejects is noise, not surface.
    filled = np.where(good, height, np.nan)
    smoothed = cv2.medianBlur(np.nan_to_num(filled, nan=0).astype(np.float32), 3)
    height = np.where(good, smoothed, np.nan)

    # Normals from the height-field gradient (surface is z = h(x, y)).
    dzdx = cv2.Sobel(np.nan_to_num(height), cv2.CV_32F, 1, 0, ksize=3) / (8 * step)
    dzdy = cv2.Sobel(np.nan_to_num(height), cv2.CV_32F, 0, 1, ksize=3) / (8 * step)
    normals = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    index = -np.ones(grid_x.shape, np.int32)
    index[good] = np.arange(good.sum())
    vertices = np.stack([grid_x[good], grid_y[good], height[good]], axis=-1)
    vertex_normals = normals[good]

    # Colour each vertex from the camera whose view is most face-on.
    world = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1).T
    best_cos = np.full(len(vertices), -1.0)
    vertex_colors = np.zeros((len(vertices), 3), np.uint8)
    for name in names:
        cam = cameras[name]
        local = cam["cam_from_world"] @ world
        depth = local[2]
        u = cam["K"][0, 0] * local[0] / depth + cam["K"][0, 2]
        v = cam["K"][1, 1] * local[1] / depth + cam["K"][1, 2]
        direction = cam["position"][None, :] - vertices
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        cosine = (direction * vertex_normals).sum(axis=1)
        height_px, width_px = images[name].shape
        inside = (u >= 0) & (u < width_px - 1) & (v >= 0) & (v < height_px - 1) & (depth > 0)
        take = inside & (cosine > best_cos)
        if take.any():
            ui = np.clip(u[take].astype(int), 0, width_px - 1)
            vi = np.clip(v[take].astype(int), 0, height_px - 1)
            vertex_colors[take] = colors[name][vi, ui][:, ::-1]  # BGR -> RGB
            best_cos[take] = cosine[take]

    faces = []
    rows, cols = grid_x.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            quad = (index[r, c], index[r, c + 1], index[r + 1, c + 1], index[r + 1, c])
            if min(quad) < 0:
                continue
            # Do not bridge a depth discontinuity with a face.
            corner_heights = [height[r, c], height[r, c + 1],
                              height[r + 1, c + 1], height[r + 1, c]]
            if max(corner_heights) - min(corner_heights) > 0.01:
                continue
            faces.append((quad[0], quad[1], quad[2]))
            faces.append((quad[0], quad[2], quad[3]))
    faces = np.asarray(faces, np.int32).reshape(-1, 3)

    mesh_path = out_dir / "surface_mesh.ply"
    write_ply(mesh_path, vertices, vertex_normals, vertex_colors, faces)
    write_ply(out_dir / "surface_points.ply", vertices, vertex_normals, vertex_colors,
              np.zeros((0, 3), np.int32))
    np.savez(out_dir / "heightfield.npz", height=height, score=score, votes=votes,
             x=grid_x, y=grid_y)
    summary = {
        "world_frame": world_frame,
        "cameras": names,
        "grid_shape": list(grid_x.shape),
        "xy_step_mm": args.xy_step_mm,
        "z_step_mm": args.z_step_mm,
        "accepted_points": int(good.sum()),
        "coverage_fraction": float(good.mean()),
        "median_score": float(np.median(score[good])),
        "height_span_mm": [float(np.nanmin(height) * 1000), float(np.nanmax(height) * 1000)],
        "faces": int(len(faces)),
    }
    (out_dir / "reconstruction.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"mesh: {mesh_path}")


if __name__ == "__main__":
    main()
