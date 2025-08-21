"""VGGT runner utilities and COLMAP writer.

This module provides a light wrapper around VGGT inference outputs and
utilities to serialize results (PLY point clouds, COLMAP text format,
and frustum JSON) without forcing a heavy dependency footprint when
VGGT is not available at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import jaxlie

from tatbot.utils.log import get_logger
from tatbot.utils.plymesh import save_ply

log = get_logger("cam.vggt_runner", "📐")


@dataclass
class VGGTResult:
    images: np.ndarray  # (S, 3, H, W) float32 [0,1]
    extrinsic: np.ndarray  # (S, 3, 4) camera-from-world (OpenCV convention)
    intrinsic: np.ndarray  # (S, 3, 3)
    depth: Optional[np.ndarray] = None  # (S, H, W, 1)
    depth_conf: Optional[np.ndarray] = None  # (S, H, W)
    world_points: Optional[np.ndarray] = None  # (S, H, W, 3)


def save_world_points_as_ply(
    world_points: np.ndarray,
    colors_uint8: np.ndarray,
    output_path: str | Path,
    conf: Optional[np.ndarray] = None,
    conf_threshold: Optional[float] = None,
) -> None:
    """Save dense world points to a PLY file with optional confidence filtering.

    world_points: (N, 3) or (S, H, W, 3)
    colors_uint8: (N, 3) or (S, H, W, 3) uint8
    conf: (N,) or (S, H, W) optional confidence
    """
    pts = world_points
    cols = colors_uint8
    if pts.ndim == 4:
        pts = pts.reshape(-1, 3)
    if cols.ndim == 4:
        cols = cols.reshape(-1, 3)
    if conf is not None:
        c = conf.reshape(-1)
        if conf_threshold is not None:
            mask = c >= conf_threshold
            pts = pts[mask]
            cols = cols[mask]
    save_ply(str(output_path), pts.astype(np.float32), cols.astype(np.uint8))


def write_colmap_text(
    intrinsic_list: Sequence[np.ndarray],
    extrinsic_list: Sequence[np.ndarray],
    image_names: Sequence[str],
    output_dir: str | Path,
    shared_camera: bool = False,
) -> None:
    """Write COLMAP text files (cameras.txt, images.txt). Points3D optional.

    - extrinsic_list: (S, 3, 4) camera-from-world (R|t) in OpenCV convention
    - intrinsic_list: (S, 3, 3) or singleton when shared_camera=True
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cameras: assign IDs
    cams_lines: List[str] = ["# Camera list with one line of data per camera:\n",
                             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
    imgs_lines: List[str] = ["# Image list with two lines of data per image:\n",
                             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
                             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]

    camera_id_map: List[int] = []
    camera_params_cache: dict[Tuple[float, ...], int] = {}

    def _intr_to_params(K: np.ndarray) -> Tuple[float, ...]:
        # Use PINHOLE model: fx, fy, cx, cy
        return (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

    for i, K in enumerate(intrinsic_list if not shared_camera else [intrinsic_list[0]]):
        params = _intr_to_params(K)
        if shared_camera and i > 0:
            break
        # Placeholder width/height unknown here; set to 0 to indicate unknown
        camera_id = len(camera_params_cache) + 1
        camera_params_cache[params] = camera_id
        cams_lines.append(f"{camera_id} PINHOLE 0 0 {params[0]} {params[1]} {params[2]} {params[3]}\n")

    for img_id, (name, E) in enumerate(zip(image_names, extrinsic_list), start=1):
        R = E[:, :3]
        t = E[:, 3]
        # Convert to quaternion in COLMAP order (qw, qx, qy, qz) robustly
        qw, qx, qy, qz = tuple(jaxlie.SO3.from_matrix(R).wxyz)
        if shared_camera:
            cam_id = 1
        else:
            params = _intr_to_params(intrinsic_list[min(img_id - 1, len(intrinsic_list) - 1)])
            cam_id = camera_params_cache.setdefault(params, len(camera_params_cache) + 1)
            if f"{cam_id} PINHOLE" not in "".join(cams_lines):
                cams_lines.append(f"{cam_id} PINHOLE 0 0 {params[0]} {params[1]} {params[2]} {params[3]}\n")
        imgs_lines.append(f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n")
        imgs_lines.append("\n")

    (out / "cameras.txt").write_text("".join(cams_lines))
    (out / "images.txt").write_text("".join(imgs_lines))
    # points3D optional; leave empty file for compatibility
    (out / "points3D.txt").write_text("# No points exported\n")


def write_frustums_json(
    extrinsic_list: Sequence[np.ndarray],
    intrinsic_list: Sequence[np.ndarray],
    names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Write a JSON file with camera frustum data for later visualization.

    Stores camera-from-world (3x4) extrinsics and intrinsics (3x3) per name.
    """
    payload = []
    for name, E, K in zip(names, extrinsic_list, intrinsic_list):
        payload.append({
            "name": name,
            "extrinsic_3x4": np.asarray(E, dtype=float).tolist(),
            "intrinsic_3x3": np.asarray(K, dtype=float).tolist(),
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2))


def run_vggt_from_images(
    image_paths: Sequence[str],
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
) -> VGGTResult:
    """Run VGGT on a list of image file paths and return predictions.

    This function defers heavy imports and raises a clear error if the VGGT
    library is not available. It is intended to be called on a GPU node.
    """
    try:
        import torch  # noqa: F401
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "VGGT dependencies not available. Install vggt/torch on GPU node.") from e

    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Prefer HF cache if weights_path not supplied
    if weights_path:
        log.info(f"Loading VGGT weights from {weights_path}")
        model = VGGT()
        state_dict = torch.load(weights_path, map_location=dev)
        model.load_state_dict(state_dict)
    else:
        log.info("Loading VGGT from HuggingFace cache: facebook/VGGT-1B")
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to(dev)

    images = load_and_preprocess_images(list(image_paths)).to(dev)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = model(images)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    # Convert to numpy and squeeze batch
    result = VGGTResult(
        images=images.detach().cpu().numpy().squeeze(0),
        extrinsic=extrinsic.detach().cpu().numpy().squeeze(0),
        intrinsic=intrinsic.detach().cpu().numpy().squeeze(0),
    )

    # Optional depth
    try:
        depth = preds.get("depth")
        depth_conf = preds.get("depth_conf")
        if depth is not None:
            result.depth = depth.detach().cpu().numpy().squeeze(0)
        if depth_conf is not None:
            result.depth_conf = depth_conf.detach().cpu().numpy().squeeze(0)
        if result.depth is not None:
            from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
            result.world_points = unproject_depth_map_to_point_map(result.depth, result.extrinsic, result.intrinsic)
    except Exception as e:
        log.warning(f"Failed to compute world points from depth: {e}")

    return result


def camera_centers_from_extrinsics(extrinsic_list: Sequence[np.ndarray]) -> np.ndarray:
    """Compute camera centers (world coords) from camera-from-world extrinsics.

    For E = [R|t] mapping X_world -> X_cam, camera center C_world = -R^T t.
    Returns array (S, 3).
    """
    centers = []
    for E in extrinsic_list:
        R = E[:, :3]
        t = E[:, 3]
        C = -R.T @ t
        centers.append(C)
    return np.stack(centers, axis=0)


def median_baseline_scale(ref_centers: np.ndarray, pred_centers: np.ndarray) -> float:
    """Compute a robust scale factor aligning pred to ref using median pairwise baselines.

    s = median(||ref_i - ref_j||) / median(||pred_i - pred_j||)
    Returns 1.0 if any degenerate case occurs.
    """
    n = min(len(ref_centers), len(pred_centers))
    if n < 2:
        return 1.0
    ref_d = []
    pred_d = []
    for i in range(n):
        for j in range(i + 1, n):
            ref_d.append(np.linalg.norm(ref_centers[i] - ref_centers[j]))
            pred_d.append(np.linalg.norm(pred_centers[i] - pred_centers[j]))
    ref_med = float(np.median(ref_d)) if ref_d else 0.0
    pred_med = float(np.median(pred_d)) if pred_d else 0.0
    if ref_med <= 1e-9 or pred_med <= 1e-9:
        return 1.0
    return ref_med / pred_med
