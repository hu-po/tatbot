"""
Surface Mapping

This module provides functionality to map flat 2D strokes to 3D positions on a point cloud surface 
representing the skin. It reconstructs a triangular mesh from the point cloud using Open3D's Poisson surface reconstruction, 
then computes exact geodesic paths on the mesh using the pygeodesic library. This allows "wrapping" the strokes onto 
the curved surface by projecting flat points to the mesh via XY closest points and connecting them with geodesic segments, 
preserving true shortest paths on the surface. Per-point normals are interpolated from the mesh vertex normals for 
orienting the end effector perpendicular to the surface.

The point cloud can be loaded from one or more PLY files. If multiple files are provided, they are merged directly into a single point cloud assuming they are already aligned in the same coordinate frame. Optional cleaning steps, powered by Open3D, include voxel downsampling for density reduction, statistical outlier removal for global noise 
elimination, and radius-based outlier removal for local isolates. These preprocessing steps improve the accuracy and 
efficiency of the mesh reconstruction and geodesic computations, especially for noisy or dense skin scans.
"""

import os
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import pygeodesic.geodesic as geodesic

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.utils.log import get_logger

log = get_logger("gen.map", "🗺️")


def map_strokes_to_surface(
    ply_files: str | list[str],
    strokes: StrokeList,
    design_origin: Pose,
    stroke_length: int = 100,
    clean_cloud: bool = True,
    voxel_size: float = 0.001,
    stat_nb_neighbors: int = 20,
    stat_std_ratio: float = 2.0,
    radius_nb_points: int = 16,
    radius: float = 0.005,
    poisson_depth: int = 8,  # Depth for Poisson reconstruction; higher for more detail
) -> StrokeList:
    assert strokes, "No strokes provided"
    if isinstance(ply_files, str):
        ply_files = [ply_files]

    log.info(f"Loading {len(ply_files)} point clouds...")
    pcds = []
    for file in ply_files:
        try:
            pcd = o3d.io.read_point_cloud(os.path.expanduser(file))
            pcds.append(pcd)
        except Exception as e:
            raise ValueError(f"Failed to load {file}: {e}")

    if len(pcds) > 1:
        log.info("Merging multiple point clouds (assuming pre-alignment)...")
        ref_pcd = pcds[0]  # Use first as reference
        for i in range(1, len(pcds)):
            source = pcds[i]
            # Merge directly
            ref_pcd += source
            log.debug(f"Merged cloud {i}; total points now: {len(ref_pcd.points)}")
    else:
        ref_pcd = pcds[0]

    if clean_cloud:
        log.info("Cleaning point cloud with Open3D...")
        # Downsample to reduce density and noise
        ref_pcd = ref_pcd.voxel_down_sample(voxel_size=voxel_size)
        log.debug(f"Downsampled to {len(ref_pcd.points)} points")
        # Remove global outliers based on statistical deviation
        ref_pcd, ind = ref_pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
        log.debug(f"After statistical removal: {len(ref_pcd.points)} points")
        # Remove local isolates with few neighbors
        ref_pcd, ind = ref_pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
        log.debug(f"After radius removal: {len(ref_pcd.points)} points")

    # Step 1: Reconstruct triangular mesh from cleaned point cloud
    log.info("Reconstructing mesh from point cloud using Poisson method...")
    ref_pcd.estimate_normals()  # Required for Poisson
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ref_pcd, depth=poisson_depth)
    # Optional: Remove low-density vertices for better quality
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    # Compute vertex normals for the mesh
    mesh.compute_vertex_normals()
    log.debug(f"Reconstructed mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")

    # Extract points (vertices) and faces (triangles) for pygeodesic
    points = np.asarray(mesh.vertices)
    if points.shape[1] != 3:
        raise ValueError(f"Mesh vertices must have shape (N, 3), got {points.shape}")
    faces = np.asarray(mesh.triangles)

    # Step 2: Initialize pygeodesic algorithm on the mesh
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

    # Step 3: Build KDTree on mesh vertices' XY for projecting flat 2D points to closest mesh points
    p_xy_tree = KDTree(points[:, :2])
    design_xy = design_origin.pos.xyz[:2]  # Extract XY coordinates from design origin
    dist, source_idx = p_xy_tree.query(np.array([design_xy]), k=1)
    source_idx = source_idx[0]  # Not used directly, but could be for validation

    def map_stroke(stroke: Stroke) -> Stroke:
        # inkdip or rest strokes do not need to be mapped
        if stroke.is_inkdip or stroke.is_rest:
            return stroke
        
        # Step 4: Project flat 2D stroke points to closest mesh vertices using XY projection
        pts_flat = stroke.meter_coords[:, :2]  # (N,2)
        dists, proj_indices = p_xy_tree.query(pts_flat, k=1)
        proj_indices = proj_indices.flatten()  # (N,) indices of closest mesh vertices

        # Step 5: Connect projected points with geodesic paths to follow the surface
        mapped_pts_list = []  # List to collect all points along geodesic segments
        for i in range(len(proj_indices) - 1):
            src_idx = proj_indices[i]
            tgt_idx = proj_indices[i + 1]
            # Compute geodesic distance and path between consecutive projected points
            distance, path = geoalg.geodesicDistance(src_idx, tgt_idx)
            if path is None or len(path) == 0:
                log.warning(f"Geodesic path empty between indices {src_idx} and {tgt_idx}; skipping segment")
                continue
            mapped_pts_list.append(path)  # path is (M,3) array of points along geodesic

        if not mapped_pts_list:
            log.warning("No valid geodesic paths found for stroke; returning original")
            return stroke

        # Step 6: Concatenate all geodesic segment points into a single array
        pts_mapped = np.vstack(mapped_pts_list)  # (Total_points, 3)

        # Step 7: Resample the concatenated path along 3D arc-length for even surface spacing
        seg = np.diff(pts_mapped, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.insert(np.cumsum(seg_len), 0, 0.0)
        total_len = cum_len[-1]

        if total_len > 0:
            target_lens = np.linspace(0.0, total_len, stroke_length)
            resampled_3d = np.zeros((stroke_length, 3), dtype=np.float32)

            seg_idx = 0
            for i, t in enumerate(target_lens):
                while seg_idx < len(cum_len) - 2 and t > cum_len[seg_idx + 1]:
                    seg_idx += 1

                t0, t1 = cum_len[seg_idx], cum_len[seg_idx + 1]
                if t1 == t0:  # duplicate
                    resampled_3d[i] = pts_mapped[seg_idx]
                else:
                    alpha = (t - t0) / (t1 - t0)
                    resampled_3d[i] = (1 - alpha) * pts_mapped[seg_idx] + alpha * pts_mapped[seg_idx + 1]

            pts_mapped = resampled_3d

        # Step 8: Compute normals at resampled points by finding closest mesh vertex and using its normal
        vertex_normals = np.asarray(mesh.vertex_normals)  # (V,3)
        points_tree = KDTree(points)  # KDTree on all 3D vertices
        dists, closest_indices = points_tree.query(pts_mapped, k=1)
        closest_indices = closest_indices.flatten()
        normals = vertex_normals[closest_indices]  # (stroke_length, 3)

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