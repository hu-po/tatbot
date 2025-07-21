"""
Surface Mapping

This module provides functionality to map flat 2D strokes to 3D positions on a point cloud surface 
representing the skin. It uses the potpourri3d library to compute intrinsic surface properties like tangent frames 
(including normals) and logarithmic maps for low-distortion parameterization. This allows "wrapping" the strokes onto 
the curved surface while preserving approximate geodesic paths, and provides per-point normals for orienting the 
end effector perpendicular to the surface.
"""

import numpy as np
from scipy.spatial import KDTree
import potpourri3d as pp3d

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.utils.log import get_logger

log = get_logger("gen.map", "🗺️")


def map_strokes_to_surface(
    ply_file: str,
    strokes: StrokeList,
    design_origin: Pose,
    stroke_length: int = 100,
) -> list[Stroke]:
    assert strokes, "No strokes provided"
    try:
        log.info(f"Loading pointcloud from {ply_file}")
        P = pp3d.read_point_cloud(ply_file)
    except Exception as e:
        raise ValueError(f"Failed to load point cloud from {ply_file}: {e}")
    
    if P.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3), got {P.shape}")

    # Compute solver and tangent frames
    solver = pp3d.PointCloudHeatSolver(P)
    basisX, basisY, basisN = solver.get_tangent_frames()  # basisN are the normals

    # Find source index closest to design origin position
    p_xy_tree = KDTree(P[:, :2])
    design_xy = design_origin.pos.xyz[:2]  # Extract XY coordinates from design origin
    dist, source_idx = p_xy_tree.query(np.array([design_xy]), k=1)
    source_idx = source_idx[0]

    # Compute log map: 2D params U for all points
    U = solver.compute_log_map(source_idx)  # Nx2

    # Build KDTree on U for querying
    u_tree = KDTree(U)

    def map_stroke(stroke: Stroke) -> Stroke:
        # inkdip or rest strokes do not need to be mapped
        if stroke.is_inkdip or stroke.is_rest:
            return stroke
        
        pts_flat = stroke.meter_coords[:, :2]  # (N,2)
        dists, indices = u_tree.query(pts_flat, k=1)
        indices = indices.flatten()  # (N,)

        # Map to 3D positions and normals
        pts_mapped = P[indices]  # (N,3)
        normals = basisN[indices]  # (N,3)

        # Optional: Resample along 3D arc-length for even surface spacing
        # Compute cumulative 3D lengths
        seg = np.diff(pts_mapped, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.insert(np.cumsum(seg_len), 0, 0.0)
        total_len = cum_len[-1]

        if total_len > 0:
            target_lens = np.linspace(0.0, total_len, stroke_length)
            resampled_3d = np.zeros((stroke_length, 3), dtype=np.float32)
            resampled_normals = np.zeros((stroke_length, 3), dtype=np.float32)

            seg_idx = 0
            for i, t in enumerate(target_lens):
                while seg_idx < len(cum_len) - 2 and t > cum_len[seg_idx + 1]:
                    seg_idx += 1

                t0, t1 = cum_len[seg_idx], cum_len[seg_idx + 1]
                if t1 == t0:  # duplicate
                    resampled_3d[i] = pts_mapped[seg_idx]
                    resampled_normals[i] = normals[seg_idx]
                else:
                    alpha = (t - t0) / (t1 - t0)
                    resampled_3d[i] = (1 - alpha) * pts_mapped[seg_idx] + alpha * pts_mapped[seg_idx + 1]
                    resampled_normals[i] = (1 - alpha) * normals[seg_idx] + alpha * normals[seg_idx + 1]
                    # Normalize the interpolated normal
                    norm_len = np.linalg.norm(resampled_normals[i])
                    if norm_len > 0:
                        resampled_normals[i] /= norm_len

            pts_mapped = resampled_3d
            normals = resampled_normals

        # Create a new Stroke object with the mapped 3D positions and normals
        return Stroke(
            description=stroke.description,
            arm=stroke.arm,
            meter_coords=pts_mapped,
            ee_rot=stroke.ee_rot,
            dt=stroke.dt,
            pixel_coords=stroke.pixel_coords,
            gcode_text=stroke.gcode_text,
            inkcap=stroke.inkcap,
            is_inkdip=stroke.is_inkdip,
            color=stroke.color,
            frame_path=stroke.frame_path,
            normals=normals,
        )

    mapped_strokes: StrokeList = StrokeList(strokes=[])
    for stroke_l, stroke_r in strokes.strokes:
        mapped_stroke_l = map_stroke(stroke_l)
        mapped_stroke_r = map_stroke(stroke_r)
        mapped_strokes.strokes.append((mapped_stroke_l, mapped_stroke_r))
    return mapped_strokes