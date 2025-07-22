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
    radius_nb_points: int = 10,
    radius: float = 0.01,
    poisson_depth: int = 8,  # Depth for Poisson reconstruction; higher for more detail
) -> StrokeList:
    log.info(f"Mapping {len(strokes.strokes)} stroke pairs to surface")
    assert strokes and len(strokes.strokes) > 0, "No strokes provided"
    if isinstance(ply_files, str):
        ply_files = [ply_files]

    # Load and merge point clouds
    log.info(f"Loading {len(ply_files)} point cloud(s)...")
    pcds = []
    for i, file in enumerate(ply_files):
        try:
            pcd = o3d.io.read_point_cloud(os.path.expanduser(file))
            if len(pcd.points) == 0:
                raise ValueError(f"Point cloud {file} is empty")
            log.info(f"Loaded {file}: {len(pcd.points)} points")
            pcds.append(pcd)
        except Exception as e:
            raise ValueError(f"Failed to load {file}: {e}")

    # Merge multiple point clouds if needed
    if len(pcds) > 1:
        log.info("Merging point clouds...")
        ref_pcd = pcds[0]
        for i in range(1, len(pcds)):
            ref_pcd += pcds[i]
        log.info(f"Merged to {len(ref_pcd.points)} total points")
    else:
        ref_pcd = pcds[0]

    # Clean point cloud if requested
    if clean_cloud:
        log.info("Cleaning point cloud...")
        initial_points = len(ref_pcd.points)
        
        # Downsample
        ref_pcd = ref_pcd.voxel_down_sample(voxel_size=voxel_size)
        log.info(f"Downsampled: {initial_points} → {len(ref_pcd.points)} points")
        
        # Remove statistical outliers
        ref_pcd, _ = ref_pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
        log.info(f"After statistical removal: {len(ref_pcd.points)} points")
        
        # Remove radius outliers
        ref_pcd, _ = ref_pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
        log.info(f"After radius removal: {len(ref_pcd.points)} points")

    # Reconstruct mesh from point cloud
    log.info("Reconstructing mesh...")
    ref_pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ref_pcd, depth=poisson_depth)
    
    if len(mesh.vertices) == 0:
        raise ValueError("Poisson reconstruction failed - no vertices generated")
    
    log.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # Remove low-density vertices
    densities = np.asarray(densities)
    if len(densities) > 0:
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        log.info(f"Removed {np.sum(vertices_to_remove)} low-density vertices")
    
    mesh.compute_vertex_normals()
    log.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Extract mesh data for geodesic computation
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    if points.shape[1] != 3:
        raise ValueError(f"Mesh vertices must have shape (N, 3), got {points.shape}")
    if len(faces) == 0:
        raise ValueError("Mesh has no faces - cannot compute geodesics")
    log.info(f"Mesh validation: {len(points)} vertices, {len(faces)} faces")
    if len(points) < 10:
        log.warning(f"Very small mesh: {len(points)} vertices")
    if len(faces) < 10:
        log.warning(f"Very few faces: {len(faces)} faces")

    # Initialize geodesic algorithm
    log.info("Initializing geodesic algorithm...")
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

    # Build KDTree for XY projection
    p_xy_tree = KDTree(points[:, :2])
    design_xy = design_origin.pos.xyz[:2]
    _, source_idx = p_xy_tree.query(np.array([design_xy]), k=1)
    log.info(f"Design origin mapped to vertex {source_idx[0]}")

    def map_stroke(stroke: Stroke) -> Stroke:
        # Skip inkdip or rest strokes
        if stroke.is_inkdip or stroke.is_rest:
            return stroke
        
        if len(stroke.meter_coords) < 2:
            log.warning(f"Stroke {stroke.description} has < 2 points, skipping")
            return stroke
        
        # Project flat 2D stroke points to mesh vertices
        pts_flat = stroke.meter_coords[:, :2]
        log.debug(f"Stroke {stroke.description}: {len(pts_flat)} points, XY range: {pts_flat.min(axis=0)} to {pts_flat.max(axis=0)}")
        
        _, proj_indices = p_xy_tree.query(pts_flat, k=1)
        proj_indices = proj_indices.flatten()
        log.debug(f"Projected to {len(set(proj_indices))} unique vertices out of {len(proj_indices)} points")

        # Connect projected points with geodesic paths
        mapped_pts_list = []
        log.info(f"Computing geodesic paths for {len(proj_indices)} projected points")
        
        for i in range(len(proj_indices) - 1):
            src_idx = proj_indices[i]
            tgt_idx = proj_indices[i + 1]
            
            # Skip if same vertex
            if src_idx == tgt_idx:
                log.debug(f"Skipping segment {i}: same vertex {src_idx}")
                continue
                
            # Compute geodesic path
            try:
                distance, path = geoalg.geodesicDistance(src_idx, tgt_idx)
                if path is None or len(path) == 0:
                    log.warning(f"Empty geodesic path between vertices {src_idx} and {tgt_idx}")
                    continue
                    
                mapped_pts_list.append(path)
                log.debug(f"Segment {i}: {src_idx} → {tgt_idx}, distance={distance:.4f}, path_length={len(path)}")
            except Exception as e:
                log.warning(f"Geodesic computation failed for segment {i} ({src_idx} → {tgt_idx}): {e}")
                continue

        if not mapped_pts_list:
            log.warning(f"No valid geodesic paths for stroke {stroke.description}")
            return stroke

        # Concatenate geodesic segments
        pts_mapped = np.vstack(mapped_pts_list)
        
        # Resample along 3D arc-length
        if len(pts_mapped) > 1:
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
                    if t1 == t0:  # duplicate points
                        resampled_3d[i] = pts_mapped[seg_idx]
                    else:
                        alpha = (t - t0) / (t1 - t0)
                        resampled_3d[i] = (1 - alpha) * pts_mapped[seg_idx] + alpha * pts_mapped[seg_idx + 1]

                pts_mapped = resampled_3d

        # Compute normals at resampled points
        vertex_normals = np.asarray(mesh.vertex_normals)
        points_tree = KDTree(points)
        _, closest_indices = points_tree.query(pts_mapped, k=1)
        closest_indices = closest_indices.flatten()
        normals = vertex_normals[closest_indices]

        # Create mapped stroke
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

    # Process all stroke pairs
    mapped_strokes = StrokeList(strokes=[])
    for i, (stroke_l, stroke_r) in enumerate(strokes.strokes):
        try:
            mapped_stroke_l = map_stroke(stroke_l)
            mapped_stroke_r = map_stroke(stroke_r)
            mapped_strokes.strokes.append((mapped_stroke_l, mapped_stroke_r))
        except Exception as e:
            log.error(f"Failed to map stroke pair {i+1}: {e}")
            raise
    
    log.info(f"Successfully mapped {len(mapped_strokes.strokes)} stroke pairs")
    return mapped_strokes