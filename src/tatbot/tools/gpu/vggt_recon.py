"""GPU-accelerated VGGT dense reconstruction tool."""

import glob
import os
from pathlib import Path

import hydra
import numpy as np
import yaml

from tatbot.cam.vggt_runner import (
    VGGTResult,
    camera_centers_from_extrinsics,
    median_baseline_scale,
    run_vggt_from_images,
    save_world_points_as_ply,
    write_colmap_text,
    write_frustums_json,
)
from tatbot.tools.base import ToolContext
from tatbot.tools.gpu.models import VGGTReconInput, VGGTReconOutput
from tatbot.tools.registry import tool
from tatbot.utils.exceptions import TatbotError
from tatbot.utils.log import get_logger

log = get_logger("tools.vggt_recon", "🧠⚡")


@tool(
    name="vggt_reconstruct",
    nodes=["ook"],
    description="Run VGGT on a set of images to estimate cameras and dense points.",
    input_model=VGGTReconInput,
    output_model=VGGTReconOutput,
    requires=["gpu"],
)
async def vggt_reconstruct(input_data: VGGTReconInput, ctx: ToolContext):
    """
    Perform dense reconstruction using VGGT on a GPU node.

    Inputs are NFS paths; outputs are written to NFS for downstream consumption.
    """
    # Verify GPU support on this node
    cfg = hydra.compose(config_name="config")

    # Load node-specific MCP config
    config_dir = Path(__file__).parent.parent.parent.parent / "conf" / "mcp"
    node_config_file = config_dir / f"{ctx.node_name}.yaml"

    if node_config_file.exists():
        with open(node_config_file, 'r') as f:
            node_cfg = yaml.safe_load(f)
    else:
        node_cfg = cfg.mcp

    if "gpu" not in node_cfg.get("extras", []):
        yield VGGTReconOutput(
            success=False,
            message=f"Node {ctx.node_name} does not have GPU support",
        )
        return

    try:
        img_dir = Path(input_data.image_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        image_paths = sorted([p for p in glob.glob(os.path.join(str(img_dir), "*")) if os.path.isfile(p)])
        if not image_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        yield {"progress": 0.1, "message": f"Loaded {len(image_paths)} images"}

        # Preflight GPU info
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                log.info(f"CUDA VRAM free/total: {free/1e9:.2f}GB/{total/1e9:.2f}GB")
        except Exception:
            pass

        # Run VGGT
        yield {"progress": 0.2, "message": "Running VGGT inference on GPU..."}
        result: VGGTResult = run_vggt_from_images(image_paths, weights_path=input_data.weights_path)

        # Compute optional scale alignment using AprilTag frustums or URDF
        scale_factor = 1.0
        try:
            # Expect images dir under dataset root
            dataset_root = img_dir.parent
            apriltag_json = dataset_root / "metadata" / "apriltag_frustums.json"
            if apriltag_json.exists():
                import json as _json
                apr = _json.loads(apriltag_json.read_text())
                # Match order by image_names
                name_to_pos = {c["name"]: np.array(c["pose"]["position"], dtype=float) for c in apr}
                names = [Path(p).name for p in image_paths]
                matched = [n for n in names if n in name_to_pos]
                if len(matched) >= 2:
                    ref_centers = np.stack([name_to_pos[n] for n in matched], axis=0)
                    pred_centers_full = camera_centers_from_extrinsics(list(result.extrinsic))
                    # Align order by matched subset length
                    assert pred_centers_full.shape[0] >= len(matched), "VGGT returned fewer cameras than matched AprilTag frustums"
                    pred_centers = pred_centers_full[: len(matched)]
                    scale_factor = median_baseline_scale(ref_centers, pred_centers)
                    # Sanity bounds
                    if not (1e-3 <= scale_factor <= 1e3):
                        log.warning(f"Unreasonable scale factor {scale_factor}, resetting to 1.0")
                        scale_factor = 1.0
                log.info(f"Computed scale factor from AprilTag frustums: {scale_factor:.4f}")
            else:
                log.info("No AprilTag frustums found; skipping scale alignment")
        except Exception as e:
            log.warning(f"Scale alignment failed: {e}")

        # Save outputs
        out_ply = Path(input_data.output_pointcloud_path)
        out_json = Path(input_data.output_frustums_path)
        out_colmap = Path(input_data.output_colmap_dir)
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        out_colmap.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        colors = (result.images.transpose(0, 2, 3, 1) * 255.0).astype("uint8")
        if result.world_points is not None and result.depth_conf is not None:
            yield {"progress": 0.6, "message": "Saving VGGT dense point cloud..."}
            # Only scale camera poses (below), do not double-scale points
            save_world_points_as_ply(result.world_points, colors, out_ply, result.depth_conf, input_data.vggt_conf_threshold)
            # Count points after confidence thresholding
            mask = result.depth_conf >= float(input_data.vggt_conf_threshold)
            point_count = int(mask.sum())
        else:
            # If world_points missing, fallback: no points
            point_count = 0

        # Frustums JSON
        yield {"progress": 0.7, "message": "Saving camera frustums JSON..."}
        names = [Path(p).name for p in image_paths]
        # Scale only translation component of extrinsics
        E_scaled = []
        for E in result.extrinsic:
            E2 = E.copy()
            E2[:, 3] = E2[:, 3] * scale_factor
            E_scaled.append(E2)
        write_frustums_json(E_scaled, result.intrinsic, names, out_json)

        # COLMAP text
        yield {"progress": 0.8, "message": "Writing COLMAP text files..."}
        write_colmap_text(list(result.intrinsic), E_scaled, names, out_colmap, shared_camera=input_data.shared_camera)

        # Save metadata/metrics
        try:
            import json as _json
            metrics_path = dataset_root / "metadata" / "metrics.json"
            metrics = {
                "vggt_scale_factor": float(scale_factor),
                "vggt_point_count": int(point_count),
                "vggt_frustum_count": int(len(names)),
                "vggt_conf_threshold": float(input_data.vggt_conf_threshold),
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(_json.dumps(metrics, indent=2))
        except Exception as e:
            log.warning(f"Failed to write metrics.json: {e}")

        # Save confidence maps for optional analysis
        try:
            if result.depth_conf is not None:
                import numpy as _np
                conf_path = dataset_root / "metadata" / "vggt_confidence.npz"
                _np.savez_compressed(conf_path, depth_conf=result.depth_conf)
        except Exception as e:
            log.warning(f"Failed to save confidence maps: {e}")

        yield VGGTReconOutput(
            success=True,
            message=f"VGGT reconstruction complete. PLY: {out_ply}",
            point_count=point_count,
            frustum_count=len(image_paths),
        )

    except TatbotError as e:
        log.error(f"Tatbot error during VGGT reconstruction: {e}")
        yield VGGTReconOutput(success=False, message=f"Tatbot error: {e}")
    except Exception as e:
        log.error(f"Unexpected error in VGGT reconstruction: {e}")
        yield VGGTReconOutput(success=False, message=f"Unexpected error: {e}")
