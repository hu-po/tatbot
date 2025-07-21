"""
Surface Mapper Module
=====================

This module provides functionality to map flat 2D strokes to 3D positions on a point cloud surface 
representing the skin. It uses the potpourri3d library to compute intrinsic surface properties like tangent frames 
(including normals) and logarithmic maps for low-distortion parameterization. This allows "wrapping" the strokes onto 
the curved surface while preserving approximate geodesic paths, and provides per-point normals for orienting the 
end effector perpendicular to the surface.

Process Overview:
-----------------
1. **Load Point Cloud**: Load the point cloud from a file (assumed to be a NumPy array of shape (N, 3) in meters).

2. **Compute Tangent Frames**: Use PointCloudHeatSolver to estimate per-point normals (basisN) and other basis vectors.

3. **Select Source Point**: Find the point in the cloud closest to the design origin pose position to serve as the 
   center for the logarithmic map. This minimizes distortion for designs placed relative to the specified origin.

4. **Compute Log Map**: Generate a 2D parameterization U (shape (N, 2)) for all points relative to the source, 
   approximating a tangent space unfolding.

5. **Build KD-Tree**: Create a spatial index on U for efficient nearest-neighbor queries.

6. **Map Each Stroke**:
   - For each flat stroke (resampled 2D points in meter coords), treat them as coordinates in the log map space.
   - Query the KD-Tree to find the closest point indices in U.
   - Retrieve the corresponding 3D positions and normals from the point cloud.
   - (Optional) Resample the mapped 3D points along their arc-length to ensure even spacing on the surface, 
     interpolating positions and normals accordingly.

7. **Output**: Return the list of mapped strokes with 3D positions, original pixel coords (for 2D visualization), 
   normals, and G-code text.

This separation allows the core G-code parsing to remain flat and agnostic to the surface, while this module handles 
the 3D projection. Assumptions:
- The point cloud is dense enough for accurate mapping.
- Designs are not too large relative to curvature to avoid high distortion (if needed, use multiple sources or 
  external parameterization).
- Normals point outward; flip if the robot requires inward orientation.

Integration:
------------
- Call this after parsing flat paths in `make_gcode_strokes`.
- Update `Stroke` class to include a `normals: np.ndarray` field.
- For non-mapped strokes (e.g., inkdip, rest), use zero arrays as dummies.

Dependencies: numpy, potpourri3d, scipy.spatial.KDTree
"""

import numpy as np
from scipy.spatial import KDTree
import potpourri3d as pp3d

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke
from tatbot.utils.log import get_logger

log = get_logger("gen.map", "🗺️")


def map_strokes_to_surface(
    point_cloud_file: str,
    strokes: list[Stroke],
    design_origin: Pose,
    stroke_length: int = 100,
) -> list[Stroke]:
    """
    Map flat 2D strokes to 3D surface positions and normals.

    Args:
        point_cloud_file (str): Path to the point cloud file (numpy .npy format).
        strokes (list[Stroke]): List of Stroke objects with flat 2D coordinates in ee_pos (z=0).
        design_origin (Pose): The design origin pose used to select the source point for the logarithmic map.
        stroke_length (int): Number of points to resample each stroke to for even spacing.

    Returns:
        list[Stroke]: List of Stroke objects with updated 3D surface positions in ee_pos and normals added.
    """
    assert strokes, "No strokes provided"
    try:
        log.info(f"Loading pointcloud from {point_cloud_file}")
        P = pp3d.read_point_cloud(point_cloud_file)
    except Exception as e:
        raise ValueError(f"Failed to load point cloud from {point_cloud_file}: {e}")
    
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

    mapped_strokes = []

    for stroke in strokes:
        # Skip strokes that don't have position data or are inkdip strokes
        if stroke.ee_pos is None or stroke.is_inkdip:
            mapped_strokes.append(stroke)
            continue

        # Extract flat 2D coords (ignore z=0)
        pts_flat = stroke.ee_pos[:, :2]  # (N,2)

        # Query nearest indices in log map space
        dists, indices = u_tree.query(pts_flat, k=1)
        indices = indices.flatten()  # (N,)

        # Map to 3D positions and normals
        meter_coords_3d = P[indices]  # (N,3)
        normals = basisN[indices]  # (N,3)

        # Optional: Resample along 3D arc-length for even surface spacing
        # Compute cumulative 3D lengths
        seg = np.diff(meter_coords_3d, axis=0)
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
                    resampled_3d[i] = meter_coords_3d[seg_idx]
                    resampled_normals[i] = normals[seg_idx]
                else:
                    alpha = (t - t0) / (t1 - t0)
                    resampled_3d[i] = (1 - alpha) * meter_coords_3d[seg_idx] + alpha * meter_coords_3d[seg_idx + 1]
                    resampled_normals[i] = (1 - alpha) * normals[seg_idx] + alpha * normals[seg_idx + 1]
                    # Normalize the interpolated normal
                    norm_len = np.linalg.norm(resampled_normals[i])
                    if norm_len > 0:
                        resampled_normals[i] /= norm_len

            meter_coords_3d = resampled_3d
            normals = resampled_normals

        # Create a new Stroke object with the mapped 3D positions and normals
        mapped_stroke = Stroke(
            description=stroke.description,
            arm=stroke.arm,
            ee_pos=meter_coords_3d,
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
        
        mapped_strokes.append(mapped_stroke)

    return mapped_strokes