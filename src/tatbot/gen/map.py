"""
Surface Mapping

This module provides functionality to map flat 2D strokes to 3D positions on a mesh surface 
representing the skin. It computes exact geodesic paths on the mesh using the pygeodesic library. 
This allows "wrapping" the strokes onto the curved surface by projecting flat points to the mesh 
via XY closest points and connecting them with geodesic segments, preserving true shortest paths 
on the surface. Per-point normals are interpolated from the mesh vertex normals for orienting 
the end effector perpendicular to the surface.

The mesh is provided as vertices and faces arrays. The function expects a valid triangular mesh
with proper connectivity for accurate geodesic computations.
"""

import numpy as np
import open3d as o3d
import pygeodesic.geodesic as geodesic
from scipy.spatial import KDTree
import jaxlie

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.utils.log import get_logger

log = get_logger("gen.map", "🗺️")


def map_strokes_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    strokes: StrokeList,
    design_origin: Pose,
    stroke_length: int = 100,
) -> StrokeList:
    """
    Map flat 2D strokes to 3D positions on a mesh surface.
    
    Args:
        vertices: numpy array of vertex coordinates (N, 3)
        faces: numpy array of face indices (M, 3)
        strokes: StrokeList containing stroke pairs to map
        design_origin: Pose representing the design origin
        stroke_length: Number of points to resample each stroke to
        
    Returns:
        StrokeList with mapped strokes
    """
    log.info(f"Mapping {len(strokes.strokes)} stroke pairs to surface")
    assert strokes and len(strokes.strokes) > 0, "No strokes provided"
    
    # Validate mesh data
    if vertices.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (N, 3), got {vertices.shape}")
    if faces.shape[1] != 3:
        raise ValueError(f"Faces must have shape (M, 3), got {faces.shape}")
    if len(faces) == 0:
        raise ValueError("Mesh has no faces - cannot compute geodesics")
    
    # Check for invalid face indices
    max_vertex_idx = len(vertices) - 1
    if np.any(faces >= len(vertices)) or np.any(faces < 0):
        raise ValueError("Mesh has invalid face indices")
    
    # Create Open3D mesh for normal computation
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    log.info(f"Using mesh with {len(vertices)} vertices and {len(faces)} faces")

    # Initialize geodesic algorithm
    log.info("Initializing geodesic algorithm...")
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
    log.info("Geodesic algorithm initialized successfully")

    # Build KDTree for 3D closest point projection
    p_3d_tree = KDTree(vertices)
    
    # Create jaxlie SE3 transformation from design pose
    design_translation = jaxlie.SE3(wxyz_xyz=np.concatenate([design_origin.rot.wxyz, design_origin.pos.xyz], axis=-1))
    
    log.info(f"Design origin: pos={design_origin.pos.xyz}, quat={design_origin.rot.wxyz}")

    def map_stroke(stroke: Stroke) -> Stroke:
        # Skip inkdip or rest strokes
        if stroke.is_inkdip or stroke.is_rest:
            return stroke
        
        if len(stroke.meter_coords) < 2:
            log.warning(f"Stroke {stroke.description} has < 2 points, skipping")
            return stroke
        
        # Transform stroke points from design plane to global coordinates
        pts_flat = stroke.meter_coords[:, :2]  # 2D points in design plane
        pts_3d_design = np.column_stack([pts_flat, np.zeros(len(pts_flat))])  # Add Z=0
        
        # Apply design transformation using jaxlie SE3
        pts_3d_global = design_translation @ pts_3d_design
        
        log.info(f"Stroke {stroke.description}: {len(pts_3d_global)} points, 3D range: {pts_3d_global.min(axis=0)} to {pts_3d_global.max(axis=0)}")
        
        # Project 3D points to mesh vertices using closest point projection
        _, proj_indices = p_3d_tree.query(pts_3d_global, k=1)
        proj_indices = proj_indices.flatten()
        log.info(f"Projected to {len(set(proj_indices))} unique vertices out of {len(proj_indices)} points")

        # Connect projected points with geodesic paths
        mapped_pts_list = []
        log.info(f"Computing geodesic paths for {len(proj_indices)} projected points")
        
        for i in range(len(proj_indices) - 1):
            src_idx = proj_indices[i]
            tgt_idx = proj_indices[i + 1]
            
            # Skip if same vertex
            if src_idx == tgt_idx:
                log.info(f"Skipping segment {i}: same vertex {src_idx}")
                continue
                
            # Compute geodesic path
            try:
                distance, path = geoalg.geodesicDistance(src_idx, tgt_idx)
                if path is None or len(path) == 0:
                    log.warning(f"Empty geodesic path between vertices {src_idx} and {tgt_idx}")
                    continue
                    
                mapped_pts_list.append(path)
                log.info(f"Segment {i}: {src_idx} → {tgt_idx}, distance={distance:.4f}, path_length={len(path)}")
            except Exception as e:
                log.warning(f"Geodesic computation failed for segment {i} ({src_idx} → {tgt_idx}): {e}")
                continue

        if not mapped_pts_list:
            log.warning(f"No valid geodesic paths for stroke {stroke.description}")
            # Return original stroke with added normals field for consistency
            vertex_normals = np.asarray(mesh.vertex_normals)
            points_tree = KDTree(vertices)
            _, closest_indices = points_tree.query(stroke.meter_coords, k=1)
            closest_indices = closest_indices.flatten()
            normals = vertex_normals[closest_indices]
            
            return Stroke(
                description=stroke.description,
                arm=stroke.arm,
                meter_coords=stroke.meter_coords,
                ee_pos=stroke.ee_pos,
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
        points_tree = KDTree(vertices)
        _, closest_indices = points_tree.query(pts_mapped, k=1)
        closest_indices = closest_indices.flatten()
        normals = vertex_normals[closest_indices]

        # Create mapped stroke
        return Stroke(
            description=stroke.description,
            arm=stroke.arm,
            meter_coords=stroke.meter_coords,
            ee_pos=pts_mapped,
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
    for stroke_l, stroke_r in strokes.strokes:
        mapped_stroke_l = map_stroke(stroke_l)
        mapped_stroke_r = map_stroke(stroke_r)
        mapped_strokes.strokes.append((mapped_stroke_l, mapped_stroke_r))
    
    log.info(f"Successfully mapped {len(mapped_strokes.strokes)} stroke pairs")
    return mapped_strokes