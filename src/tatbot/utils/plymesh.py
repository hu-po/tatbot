"""
PLY Mesh Utilities

This module provides utilities for loading, processing, and creating meshes from PLY point cloud files.
It handles point cloud loading, merging, cleaning, and mesh reconstruction using Open3D.
Uses Poisson surface reconstruction, which creates watertight surfaces suitable for closed shapes.
Includes density-based trimming to remove low-confidence regions and Laplacian smoothing for smoother surfaces.
"""

import os

import numpy as np


# Defer heavy imports until needed
def _import_jax_numpy():
    import jax.numpy as jnp
    return jnp

def _import_jaxlie():
    import jaxlie
    return jaxlie

def _import_open3d():
    import open3d as o3d
    return o3d

from tatbot.data.pose import Pose
from tatbot.utils.log import get_logger

log = get_logger("utils.plymesh", "📦")


def create_mesh_from_ply_files(
    ply_files: str | list[str],
    clean_cloud: bool = True,
    voxel_size: float = 0.002,
    stat_nb_neighbors: int = 32,
    stat_std_ratio: float = 1.5,
    radius_nb_points: int = 16,
    radius: float = 0.008,
    zone_pose: Pose | None = None,
    zone_depth_m: float | None = None,
    zone_width_m: float | None = None,
    zone_height_m: float | None = None,
    poisson_depth: int = 7,
    density_trim_quantile: float = 0.16,
    smooth_iterations: int = 32,
    smooth_lambda: float = 0.9,
    output_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PLY files, combine point clouds, clean them, and create a mesh using Poisson surface reconstruction.
    
    Enhanced mesh cleaning includes:
    - Duplicate edge detection and removal
    - Manifold topology verification and repair
    - Comprehensive mesh validation
    
    Args:
        ply_files: Single PLY file path or list of PLY file paths
        clean_cloud: Whether to apply cleaning operations
        voxel_size: Voxel size for downsampling (smaller = more detail, larger = smoother)
        stat_nb_neighbors: Number of neighbors for statistical outlier removal
        stat_std_ratio: Standard deviation ratio for statistical outlier removal
        radius_nb_points: Minimum number of points for radius outlier removal
        radius: Radius for radius outlier removal
        zone_pose: Pose defining the center and orientation of the zone
        zone_depth_m: Depth of the zone in meters (x-axis)
        zone_width_m: Width of the zone in meters (y-axis)
        zone_height_m: Height of the zone in meters (z-axis)
        poisson_depth: Depth parameter for Poisson reconstruction (higher = more detail)
        density_trim_quantile: Quantile for trimming low-density vertices (0.0 = no trim, higher = more aggressive trim)
        smooth_iterations: Number of Laplacian smoothing iterations
        smooth_lambda: Laplacian smoothing strength (0-1; higher = stronger smoothing)
        
    Returns:
        Tuple of (points, faces) where:
        - points: numpy array of vertex coordinates (N, 3)
        - faces: numpy array of face indices (M, 3)
        
    Raises:
        ValueError: If point clouds are empty or mesh creation fails
    """
    if isinstance(ply_files, str):
        ply_files = [ply_files]

    # Load and merge point clouds
    log.info(f"Loading {len(ply_files)} point cloud(s)...")
    pcds = []
    for file in ply_files:
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

    # Apply zone-based clipping if zone parameters are provided
    if all(param is not None for param in [zone_depth_m, zone_width_m, zone_height_m, zone_pose]):
        log.info("Applying zone-based clipping...")
        initial_points = len(ref_pcd.points)
        
        # Convert point cloud to numpy array for clipping
        points = np.asarray(ref_pcd.points)
        
        # Define bounding box based on zone dimensions and zone pose
        center = zone_pose.pos.xyz
        half_depth = zone_depth_m / 2.0
        half_width = zone_width_m / 2.0
        half_height = zone_height_m / 2.0
        
        # Create SE3 transformation from zone pose
        jaxlie = _import_jaxlie()
        jnp = _import_jax_numpy()
        zone_se3 = jaxlie.SE3(wxyz_xyz=jnp.concatenate([zone_pose.rot.wxyz, zone_pose.pos.xyz], axis=-1))
        
        # Transform points to zone coordinate system
        points_jax = jnp.array(points)
        points_zone_frame = zone_se3.inverse() @ points_jax
        
        # Create bounding box mask in zone coordinate system
        x_mask = (points_zone_frame[:, 0] >= -half_depth) & (points_zone_frame[:, 0] <= half_depth)
        y_mask = (points_zone_frame[:, 1] >= -half_width) & (points_zone_frame[:, 1] <= half_width)
        z_mask = (points_zone_frame[:, 2] >= -half_height) & (points_zone_frame[:, 2] <= half_height)
        
        # Combine masks
        zone_mask = x_mask & y_mask & z_mask
        
        # Apply clipping
        ref_pcd = ref_pcd.select_by_index(np.where(zone_mask)[0])
        log.info(f"Zone clipping: {initial_points} → {len(ref_pcd.points)} points")
        log.info(f"Zone center: {center}, rotation: {zone_pose.rot.wxyz}")
        log.info(f"Zone dimensions: depth={zone_depth_m:.3f}m, width={zone_width_m:.3f}m, height={zone_height_m:.3f}m")

    # Clean point cloud if requested
    if clean_cloud:
        log.info("Cleaning point cloud...")
        initial_points = len(ref_pcd.points)
        
        # Downsample to reduce density and noise
        ref_pcd = ref_pcd.voxel_down_sample(voxel_size=voxel_size)
        log.info(f"Downsampled: {initial_points} → {len(ref_pcd.points)} points")
        
        # Remove statistical outliers to eliminate global noise
        ref_pcd, _ = ref_pcd.remove_statistical_outlier(nb_neighbors=stat_nb_neighbors, std_ratio=stat_std_ratio)
        log.info(f"After statistical removal: {len(ref_pcd.points)} points")
        
        # Remove local isolates to reduce sparse outliers
        ref_pcd, _ = ref_pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
        log.info(f"After radius removal: {len(ref_pcd.points)} points")

    # Estimate and orient normals for Poisson reconstruction
    log.info("Estimating normals...")
    ref_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    log.info("Orienting normals...")
    ref_pcd.orient_normals_consistent_tangent_plane(100)

    # Reconstruct mesh using Poisson
    log.info(f"Reconstructing mesh using Poisson (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        ref_pcd, depth=poisson_depth
    )
    
    if len(mesh.vertices) == 0:
        raise ValueError("Poisson reconstruction failed - no vertices generated")
    
    log.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # Validate initial mesh
    if len(mesh.vertices) < 3:
        raise ValueError(f"Initial mesh has too few vertices: {len(mesh.vertices)}")
    if len(mesh.triangles) < 1:
        raise ValueError(f"Initial mesh has no faces: {len(mesh.triangles)}")
    
    # Trim low-density vertices to remove blobby or extrapolated regions
    if density_trim_quantile > 0:
        log.info(f"Trimming low-density vertices (quantile={density_trim_quantile})...")
        densities_low = np.quantile(densities, density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < densities_low)
        log.info(f"After density trimming: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    log.info("Cleaning mesh (pre-smoothing)...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    log.info(f"After mesh cleaning: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Apply Laplacian smoothing for smoother surface
    if smooth_iterations > 0:
        log.info(f"Applying Laplacian smoothing ({smooth_iterations} iterations, lambda={smooth_lambda})...")
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=smooth_iterations, 
            lambda_filter=smooth_lambda
        )
        log.info(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    log.info("Cleaning mesh (post-smoothing)...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    log.info(f"After mesh cleaning: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    while not mesh.is_vertex_manifold():
        nm_verts = np.asarray(mesh.get_non_manifold_vertices())
        if len(nm_verts) == 0:
            break
        log.info(f"Removing {len(nm_verts)} non-manifold vertices")
        mesh.remove_vertices_by_index(nm_verts)
        # Re-clean
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    log.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Extract mesh data
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices after processing")
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh has no faces after processing")
        
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    
    if points.shape[1] != 3:
        raise ValueError(f"Mesh vertices must have shape (N, 3), got {points.shape}")
    if len(faces) == 0:
        raise ValueError("Mesh has no faces - cannot compute geodesics")
    
    # Ensure faces are properly formatted (should be 3 vertices per face)
    if faces.shape[1] != 3:
        raise ValueError(f"Faces must have shape (N, 3), got {faces.shape}")
    
    log.info(f"Mesh validation: {len(points)} vertices, {len(faces)} faces")
    if len(points) < 10:
        log.warning(f"Very small mesh: {len(points)} vertices")
    if len(faces) < 10:
        log.warning(f"Very few faces: {len(faces)} faces")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        log.error("Mesh contains NaN or infinite vertex coordinates")
        raise ValueError("Invalid vertex coordinates in mesh")
    
    # Check mesh connectivity
    unique_vertices = np.unique(faces)
    if len(unique_vertices) != len(points):
        log.warning(f"Mesh has {len(points)} vertices but only {len(unique_vertices)} are referenced in faces")

    # Validate mesh data before returning
    log.info(f"Points shape: {points.shape}, dtype: {points.dtype}")
    log.info(f"Faces shape: {faces.shape}, dtype: {faces.dtype}")
    
    # Check for any invalid indices in faces
    max_vertex_idx = len(points) - 1
    if np.any(faces >= len(points)) or np.any(faces < 0):
        invalid_faces = np.where((faces >= len(points)) | (faces < 0))[0]
        log.error(f"Invalid face indices found in {len(invalid_faces)} faces")
        log.error(f"Face indices range: {faces.min()} to {faces.max()}, but max vertex index is {max_vertex_idx}")
        raise ValueError("Mesh has invalid face indices")
    
    # Check for degenerate faces (triangles with repeated vertices)
    degenerate_faces = []
    for i, face in enumerate(faces):
        if len(set(face)) < 3:
            degenerate_faces.append(i)
    
    if degenerate_faces:
        log.warning(f"Found {len(degenerate_faces)} degenerate faces, removing them")
        valid_faces = np.ones(len(faces), dtype=bool)
        valid_faces[degenerate_faces] = False
        faces = faces[valid_faces]
        log.info(f"Removed degenerate faces, now have {len(faces)} faces")
    
    if output_dir is not None:
        log.info(f"Saving mesh to {output_dir}")
        mesh_obj = o3d.geometry.TriangleMesh()
        mesh_obj.vertices = o3d.utility.Vector3dVector(points)
        mesh_obj.triangles = o3d.utility.Vector3iVector(faces)
        mesh_obj.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(output_dir, "mesh.ply"), mesh_obj, write_ascii=False)
    
    return points, faces


def ply_files_from_dir(ply_dir: str) -> list[str]:
    """Find all .ply files in a directory."""
    ply_dir = os.path.expanduser(ply_dir)
    assert os.path.exists(ply_dir), f"Directory does not exist: {ply_dir}"
    assert os.path.isdir(ply_dir), f"Directory does not exist: {ply_dir}"
    ply_files = [os.path.join(ply_dir, file) for file in os.listdir(ply_dir) if file.lower().endswith('.ply')]
    ply_files.sort()
    log.info(f"Found {len(ply_files)} .ply file(s) in {ply_dir}")
    return ply_files


def save_ply(filename: str, points: np.ndarray, colors: np.ndarray | None = None):
    """
    Save a point cloud to a PLY file.
    
    Args:
        filename: Output PLY file path
        points: Point coordinates as numpy array (N, 3)
        colors: Optional color values as numpy array (N, 3) with values in [0, 255]
    """
    log.info(f"Saving point cloud to {filename}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
    
    o3d.io.write_point_cloud(filename, pcd)


def load_ply(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a point cloud from a PLY file.
    
    Args:
        filename: Input PLY file path
        
    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of point coordinates (N, 3)
        - colors: numpy array of color values (N, 3) with values in [0, 255]
    """
    log.info(f"Loading point cloud from {filename}")
    pcd = o3d.io.read_point_cloud(os.path.expanduser(filename))
    
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud {filename} is empty")
    
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64) * 255.0 if len(pcd.colors) > 0 else np.zeros((len(points), 3), dtype=np.float64)
    
    log.info(f"Loaded {len(points)} points from {filename}")
    return points, colors.astype(np.uint8)