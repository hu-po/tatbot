"""
Surface Mapping

This module provides functionality to map flat 2D strokes to 3D positions on a mesh surface 
representing the skin. It computes geodesic paths on the mesh using the pp3d (potpourri3d) library. 
This allows "wrapping" the strokes onto the curved surface by projecting flat points to the mesh 
via XY closest points and connecting them with geodesic segments, preserving true shortest paths 
on the surface. Per-point normals are interpolated from the mesh vertex normals for orienting 
the end effector perpendicular to the surface.

The mesh is provided as vertices and faces arrays. The function expects a valid triangular mesh
with proper connectivity for accurate geodesic computations.
"""

import numpy as np


# Defer heavy imports until needed
def _import_jaxlie():
    import jaxlie
    return jaxlie

def _import_open3d():
    import open3d as o3d
    return o3d
import potpourri3d as pp3d
from scipy.spatial import KDTree

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.utils.jnp_types import ensure_numpy_array
from tatbot.utils.log import get_logger

log = get_logger("gen.map", "🗺️")


def map_strokes_to_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    strokes: StrokeList,
    design_origin: Pose,
    stroke_length: int,
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
    if np.any(faces >= len(vertices)) or np.any(faces < 0):
        raise ValueError("Mesh has invalid face indices")
    
    # Create Open3D mesh for normal computation
    o3d = _import_open3d()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    log.info(f"Using mesh with {len(vertices)} vertices and {len(faces)} faces")

    # Initialize geodesic tracer
    log.info("Initializing geodesic tracer...")
    tracer = pp3d.GeodesicTracer(vertices, faces)
    log.info("Geodesic tracer initialized successfully")

    # Build KDTree for 3D closest point projection
    p_3d_tree = KDTree(vertices)
    
    # Create jaxlie SE3 transformation from design pose
    jaxlie = _import_jaxlie()
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
        # Convert JAX arrays to NumPy arrays for serialization
        pts_3d_global = ensure_numpy_array(pts_3d_global)
        
        log.info(f"Stroke {stroke.description}: {len(pts_3d_global)} points, 3D range: {pts_3d_global.min(axis=0)} to {pts_3d_global.max(axis=0)}")
        
        # Project 3D points to mesh vertices using closest point projection
        _, proj_indices = p_3d_tree.query(pts_3d_global, k=1)
        proj_indices = proj_indices.flatten()
        log.info(f"Projected to {len(set(proj_indices))} unique vertices out of {len(proj_indices)} points")

        # Trace geodesic paths from each point using original stroke direction
        mapped_pts_list = []
        log.info(f"Tracing geodesic paths for {len(pts_3d_global)} points")
        
        for i in range(len(pts_3d_global) - 1):
            src_idx = proj_indices[i]
            
            # Calculate direction from original stroke coordinates
            stroke_direction_2d = pts_flat[i + 1] - pts_flat[i]  # Direction in design plane
            stroke_direction_3d = np.array([stroke_direction_2d[0], stroke_direction_2d[1], 0.0])  # Add Z=0
            
            # Transform direction to global coordinates using design transformation
            # We need to apply the rotation part of the transformation to the direction vector
            direction_global = design_translation.rotation() @ stroke_direction_3d
            # Convert JAX array to NumPy array
            direction_global = ensure_numpy_array(direction_global)
            
            # Trace geodesic from source vertex in the stroke direction
            try:
                path = tracer.trace_geodesic_from_vertex(src_idx, direction_global)
                if path is None or len(path) == 0:
                    log.warning(f"Empty geodesic trace from vertex {src_idx} in direction {direction_global}")
                    continue
                    
                # Calculate distance for logging
                if len(path) > 1:
                    seg = np.diff(path, axis=0)
                    distance = np.sum(np.linalg.norm(seg, axis=1))
                else:
                    distance = 0.0
                    
                path_np = ensure_numpy_array(path)
                mapped_pts_list.append(path_np)
                log.info(f"Segment {i}: vertex {src_idx}, direction={direction_global}, distance={distance:.4f}, path_length={len(path_np)}")
            except Exception as e:
                log.warning(f"Geodesic tracing failed for segment {i} (vertex {src_idx}): {e}")
                continue

        if not mapped_pts_list:
            log.warning(f"No valid geodesic paths for stroke {stroke.description}")
            # Return original stroke with added normals field for consistency
            vertex_normals = np.asarray(mesh.vertex_normals)
            points_tree = KDTree(vertices)
            _, closest_indices = points_tree.query(stroke.meter_coords, k=1)
            closest_indices = closest_indices.flatten()
            normals = vertex_normals[closest_indices]
            
            # Ensure arrays are NumPy arrays
            ee_rot_np = ensure_numpy_array(stroke.ee_rot) if stroke.ee_rot is not None else None
            meter_coords_np = ensure_numpy_array(stroke.meter_coords) if stroke.meter_coords is not None else None
            pixel_coords_np = ensure_numpy_array(stroke.pixel_coords) if stroke.pixel_coords is not None else None
            ee_pos_np = ensure_numpy_array(stroke.ee_pos) if stroke.ee_pos is not None else None
            
            return Stroke(
                description=stroke.description,
                arm=stroke.arm,
                meter_coords=meter_coords_np,
                ee_pos=ee_pos_np,
                ee_rot=ee_rot_np,
                pixel_coords=pixel_coords_np,
                gcode_text=stroke.gcode_text,
                inkcap=stroke.inkcap,
                is_inkdip=stroke.is_inkdip,
                color=stroke.color,
                frame_path=stroke.frame_path,
                normals=ensure_numpy_array(normals),
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
        # Ensure all arrays are NumPy arrays for serialization
        ee_pos_np = ensure_numpy_array(pts_mapped)
        normals_np = ensure_numpy_array(normals)
        ee_rot_np = ensure_numpy_array(stroke.ee_rot) if stroke.ee_rot is not None else None
        meter_coords_np = ensure_numpy_array(stroke.meter_coords) if stroke.meter_coords is not None else None
        pixel_coords_np = ensure_numpy_array(stroke.pixel_coords) if stroke.pixel_coords is not None else None
        
        # Debug: check array types
        log.debug(f"ee_pos_np type: {type(ee_pos_np)}, dtype: {ee_pos_np.dtype}")
        log.debug(f"normals_np type: {type(normals_np)}, dtype: {normals_np.dtype}")
        log.debug(f"ee_rot_np type: {type(ee_rot_np)}")
        
        return Stroke(
            description=stroke.description,
            arm=stroke.arm,
            meter_coords=meter_coords_np,
            ee_pos=ee_pos_np,
            ee_rot=ee_rot_np,
            pixel_coords=pixel_coords_np,
            gcode_text=stroke.gcode_text,
            inkcap=stroke.inkcap,
            is_inkdip=stroke.is_inkdip,
            color=stroke.color,
            frame_path=stroke.frame_path,
            normals=normals_np,
        )

    # Process all stroke pairs
    mapped_strokes = StrokeList(strokes=[])
    for stroke_l, stroke_r in strokes.strokes:
        mapped_stroke_l = map_stroke(stroke_l)
        mapped_stroke_r = map_stroke(stroke_r)
        mapped_strokes.strokes.append((mapped_stroke_l, mapped_stroke_r))
    
    log.info(f"Successfully mapped {len(mapped_strokes.strokes)} stroke pairs")
    return mapped_strokes