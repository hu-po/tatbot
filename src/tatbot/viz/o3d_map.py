"""
Surface Mapping Debug Visualizer
================================

This module provides an interactive Open3D visualizer for debugging the surface mapping process.
It allows visualization of:
- Point cloud surface
- Original flat strokes (2D)
- Mapped 3D strokes on surface
- Surface normals
- Design origin position
- Interactive exploration and comparison

Usage:
    python -m tatbot.viz.o3d_map --pointcloud path/to/pointcloud.ply --scene scene_name
    python -m tatbot.viz.o3d_map --pointcloud path/to/pointcloud.npy --scene scene_name
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Optional, Tuple

from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.data.scene import Scene
from tatbot.gen.map import map_strokes_to_surface
from tatbot.gen.strokes import load_make_strokes
from tatbot.utils.log import get_logger

log = get_logger("viz.o3d_map", "🔍")


class MappingDebugVisualizer:
    """
    Interactive Open3D visualizer for debugging surface mapping process.
    """
    
    def __init__(self, point_cloud_file: str, scene_name: str, skip_mapping: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            point_cloud_file: Path to point cloud file (.npy format)
            scene_name: Name of the scene to load strokes from
            skip_mapping: If True, skip surface mapping and use original strokes
        """
        self.point_cloud_file = point_cloud_file
        self.scene_name = scene_name
        self.skip_mapping = skip_mapping
        
        # Visualization state
        self.vis = None
        self.geometries = {}
        self.current_stroke_idx = 0
        self.show_normals = True
        self.show_flat_strokes = False
        self.show_mapped_strokes = True
        self.show_origin = True
        self.normal_length = 0.005  # meters
        
        # Data
        self.point_cloud = None
        self.original_strokes = []
        self.mapped_strokes = []
        self.scene = None
        self.design_origin = None
        
        # Load data
        self._load_data()
        
    def _simple_map_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """
        Simple fallback mapping that projects strokes onto the point cloud surface
        without using potpourri3d's advanced features.
        """
        log.info("Using simple fallback mapping method")
        
        # Create a KD-tree for the point cloud (using only X,Y coordinates for 2D queries)
        from scipy.spatial import KDTree
        tree = KDTree(self.point_cloud[:, :2])  # Only use X,Y coordinates
        
        mapped_strokes = []
        
        for stroke in strokes:
            if stroke.ee_pos is None or stroke.is_inkdip:
                mapped_strokes.append(stroke)
                continue
            
            # For each 2D point, find the closest 3D point on the surface
            # We'll project the 2D coordinates to 3D by finding the closest surface point
            pts_2d = stroke.ee_pos[:, :2]  # Take only X,Y coordinates
            
            # Find closest points in the point cloud (using X,Y coordinates)
            # We need to reshape pts_2d to match the expected format
            distances, indices = tree.query(pts_2d.reshape(-1, 2), k=1)
            
            # Get the 3D positions
            mapped_positions = self.point_cloud[indices]
            
            # Create simple normals (pointing upward as fallback)
            simple_normals = np.zeros_like(mapped_positions)
            simple_normals[:, 2] = 1.0  # Point upward
            
            # Create mapped stroke
            mapped_stroke = Stroke(
                description=stroke.description,
                arm=stroke.arm,
                ee_pos=mapped_positions,
                ee_rot=stroke.ee_rot,
                dt=stroke.dt,
                pixel_coords=stroke.pixel_coords,
                gcode_text=stroke.gcode_text,
                inkcap=stroke.inkcap,
                is_inkdip=stroke.is_inkdip,
                color=stroke.color,
                frame_path=stroke.frame_path,
                normals=simple_normals,
            )
            mapped_strokes.append(mapped_stroke)
        
        return mapped_strokes

    def _load_data(self):
        """Load point cloud and stroke data."""
        log.info(f"Loading point cloud from {self.point_cloud_file}")
        
        # Check if file exists
        file_path = Path(self.point_cloud_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {self.point_cloud_file}")
        
        # Handle different file formats
        try:
            if file_path.suffix.lower() == '.ply':
                # Load PLY file using Open3D
                pcd = o3d.io.read_point_cloud(str(file_path))
                if len(pcd.points) == 0:
                    raise ValueError("PLY file contains no points")
                self.point_cloud = np.asarray(pcd.points)
                log.info(f"Loaded PLY point cloud with {len(self.point_cloud)} points")
            elif file_path.suffix.lower() == '.npy':
                # Load NumPy array
                self.point_cloud = np.load(self.point_cloud_file)
                log.info(f"Loaded NPY point cloud with shape {self.point_cloud.shape}")
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .ply, .npy")
        except Exception as e:
            raise ValueError(f"Failed to load point cloud file {self.point_cloud_file}: {e}")
        
        # Validate point cloud data
        if self.point_cloud is None or len(self.point_cloud) == 0:
            raise ValueError("Point cloud is empty or None")
        
        if self.point_cloud.shape[1] != 3:
            raise ValueError(f"Point cloud must have 3 columns (x,y,z), got shape {self.point_cloud.shape}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self.point_cloud)) or np.any(np.isinf(self.point_cloud)):
            log.warning("Point cloud contains NaN or infinite values, removing them")
            valid_mask = ~(np.isnan(self.point_cloud).any(axis=1) | np.isinf(self.point_cloud).any(axis=1))
            self.point_cloud = self.point_cloud[valid_mask]
            log.info(f"After cleaning, point cloud has {len(self.point_cloud)} points")
        
        # Check for duplicate points (which can cause issues with potpourri3d)
        unique_points, unique_indices = np.unique(self.point_cloud, axis=0, return_index=True)
        if len(unique_points) < len(self.point_cloud):
            log.warning(f"Point cloud contains {len(self.point_cloud) - len(unique_points)} duplicate points, removing them")
            self.point_cloud = unique_points
            log.info(f"After deduplication, point cloud has {len(self.point_cloud)} points")
        
        log.info(f"Loading scene: {self.scene_name}")
        self.scene = Scene.from_name(self.scene_name)
        
        # Get design origin from scene's skin configuration
        self.design_origin = self.scene.skin.design_pose
        log.info(f"Design origin from skin: {self.design_origin.pos.xyz}")
        
        log.info(f"Loading strokes from scene")
        # Use design directory if available, otherwise use a temporary directory
        design_dir = self.scene.design_dir
        if design_dir is None:
            import tempfile
            design_dir = tempfile.mkdtemp()
            log.info(f"No design directory specified, using temporary directory: {design_dir}")
        
        stroke_list, stroke_batch = load_make_strokes(self.scene, design_dir, resume=False)
        # Extract individual strokes from stroke pairs
        for stroke_pair in stroke_list.strokes:
            self.original_strokes.extend(stroke_pair)
        
        if self.skip_mapping:
            log.info("Skipping surface mapping as requested")
            self.mapped_strokes = self.original_strokes
        else:
            log.info(f"Mapping {len(self.original_strokes)} strokes to surface")
            try:
                self.mapped_strokes = map_strokes_to_surface(
                    self.point_cloud_file,
                    self.original_strokes,
                    self.design_origin,
                    stroke_length=100
                )
            except Exception as e:
                log.error(f"Failed to map strokes to surface with potpourri3d: {e}")
                log.info("Falling back to simple mapping method")
                try:
                    self.mapped_strokes = self._simple_map_strokes(self.original_strokes)
                except Exception as e2:
                    log.error(f"Simple mapping also failed: {e2}")
                    log.info("Falling back to original strokes without mapping")
                    # Create a copy of original strokes with empty normals
                    self.mapped_strokes = []
                    for stroke in self.original_strokes:
                        if stroke.ee_pos is not None:
                            # Create a copy with the same 3D positions but no normals
                            mapped_stroke = Stroke(
                                description=stroke.description,
                                arm=stroke.arm,
                                ee_pos=stroke.ee_pos,  # Keep original positions
                                ee_rot=stroke.ee_rot,
                                dt=stroke.dt,
                                pixel_coords=stroke.pixel_coords,
                                gcode_text=stroke.gcode_text,
                                inkcap=stroke.inkcap,
                                is_inkdip=stroke.is_inkdip,
                                color=stroke.color,
                                frame_path=stroke.frame_path,
                                normals=None,  # No normals available
                            )
                        else:
                            mapped_stroke = stroke
                        self.mapped_strokes.append(mapped_stroke)
        
    def _create_point_cloud_geometry(self) -> o3d.geometry.PointCloud:
        """Create Open3D point cloud geometry."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        
        # Downsample for better performance
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # Color based on height (Z coordinate)
        colors = np.zeros((len(pcd.points), 3))
        z_coords = np.asarray(pcd.points)[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        if z_max > z_min:
            normalized_z = (z_coords - z_min) / (z_max - z_min)
            colors[:, 0] = 0.5 + 0.3 * normalized_z  # Red component
            colors[:, 1] = 0.5 + 0.3 * normalized_z  # Green component
            colors[:, 2] = 0.5 + 0.3 * normalized_z  # Blue component
        else:
            colors[:] = [0.5, 0.5, 0.5]  # Gray if no height variation
            
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    
    def _create_stroke_geometry(self, stroke: Stroke, color: List[float], 
                               is_3d: bool = True) -> o3d.geometry.LineSet:
        """Create Open3D line set geometry for a stroke."""
        if stroke.ee_pos is None or len(stroke.ee_pos) < 2:
            return None
            
        # Use 3D positions or 2D positions with z=0
        if is_3d:
            points = stroke.ee_pos
        else:
            points = np.column_stack([stroke.ee_pos[:, :2], np.zeros(len(stroke.ee_pos))])
        
        # Create line segments
        lines = [[i, i+1] for i in range(len(points) - 1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
        
        return line_set
    
    def _create_normal_geometry(self, stroke: Stroke) -> o3d.geometry.LineSet:
        """Create Open3D line set geometry for stroke normals."""
        if stroke.normals is None or stroke.ee_pos is None:
            return None
            
        normal_lines = []
        normal_points = np.zeros((len(stroke.ee_pos) * 2, 3))
        normal_colors = [[0, 0, 1] for _ in range(len(stroke.ee_pos))]  # Blue
        
        for i in range(len(stroke.ee_pos)):
            start_idx = i * 2
            end_idx = start_idx + 1
            normal_points[start_idx] = stroke.ee_pos[i]
            normal_points[end_idx] = stroke.ee_pos[i] + stroke.normals[i] * self.normal_length
            normal_lines.append([start_idx, end_idx])
            
        normal_set = o3d.geometry.LineSet()
        normal_set.points = o3d.utility.Vector3dVector(normal_points)
        normal_set.lines = o3d.utility.Vector2iVector(normal_lines)
        normal_set.colors = o3d.utility.Vector3dVector(normal_colors)
        
        return normal_set
    
    def _create_origin_geometry(self) -> o3d.geometry.TriangleMesh:
        """Create Open3D geometry for design origin position."""
        origin_pos = self.design_origin.pos.xyz
        
        # Create a small sphere at origin
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(origin_pos)
        sphere.paint_uniform_color([1, 1, 0])  # Yellow
        
        return sphere
    
    def _update_geometries(self):
        """Update all geometries in the visualizer."""
        # Clear existing geometries
        for geometry in self.geometries.values():
            if geometry is not None:
                self.vis.remove_geometry(geometry, False)
        
        self.geometries.clear()
        
        # Add point cloud
        pcd = self._create_point_cloud_geometry()
        self.geometries['pointcloud'] = pcd
        self.vis.add_geometry(pcd, False)
        
        # Add origin marker
        if self.show_origin:
            origin_sphere = self._create_origin_geometry()
            self.geometries['origin'] = origin_sphere
            self.vis.add_geometry(origin_sphere, False)
        
        # Add current stroke
        if 0 <= self.current_stroke_idx < len(self.original_strokes):
            stroke = self.original_strokes[self.current_stroke_idx]
            mapped_stroke = self.mapped_strokes[self.current_stroke_idx]
            
            # Original flat stroke (red)
            if self.show_flat_strokes and stroke.ee_pos is not None:
                flat_geom = self._create_stroke_geometry(stroke, [1, 0, 0], is_3d=False)
                if flat_geom is not None:
                    self.geometries['flat_stroke'] = flat_geom
                    self.vis.add_geometry(flat_geom, False)
            
            # Mapped 3D stroke (green)
            if self.show_mapped_strokes and mapped_stroke.ee_pos is not None:
                mapped_geom = self._create_stroke_geometry(mapped_stroke, [0, 1, 0], is_3d=True)
                if mapped_geom is not None:
                    self.geometries['mapped_stroke'] = mapped_geom
                    self.vis.add_geometry(mapped_geom, False)
                
                # Normals (blue)
                if self.show_normals:
                    normal_geom = self._create_normal_geometry(mapped_stroke)
                    if normal_geom is not None:
                        self.geometries['normals'] = normal_geom
                        self.vis.add_geometry(normal_geom, False)
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _key_callback(self, vis, action, mods):
        """Handle key press events."""
        if action == o3d.visualization.KeyEvent.Action.Down:
            return False
            
        key = vis.get_key()
        
        if key == ord('N') or key == ord('n'):
            # Toggle normals
            self.show_normals = not self.show_normals
            log.info(f"Normals: {'ON' if self.show_normals else 'OFF'}")
            self._update_geometries()
            
        elif key == ord('F') or key == ord('f'):
            # Toggle flat strokes
            self.show_flat_strokes = not self.show_flat_strokes
            log.info(f"Flat strokes: {'ON' if self.show_flat_strokes else 'OFF'}")
            self._update_geometries()
            
        elif key == ord('M') or key == ord('m'):
            # Toggle mapped strokes
            self.show_mapped_strokes = not self.show_mapped_strokes
            log.info(f"Mapped strokes: {'ON' if self.show_mapped_strokes else 'OFF'}")
            self._update_geometries()
            
        elif key == ord('O') or key == ord('o'):
            # Toggle origin marker
            self.show_origin = not self.show_origin
            log.info(f"Origin marker: {'ON' if self.show_origin else 'OFF'}")
            self._update_geometries()
            
        elif key == ord('L') or key == ord('l'):
            # Increase normal length
            self.normal_length *= 1.5
            log.info(f"Normal length: {self.normal_length:.6f}")
            self._update_geometries()
            
        elif key == ord('K') or key == ord('k'):
            # Decrease normal length
            self.normal_length /= 1.5
            log.info(f"Normal length: {self.normal_length:.6f}")
            self._update_geometries()
            
        elif key == ord('RIGHT') or key == ord('D') or key == ord('d'):
            # Next stroke
            self.current_stroke_idx = (self.current_stroke_idx + 1) % len(self.original_strokes)
            log.info(f"Stroke {self.current_stroke_idx + 1}/{len(self.original_strokes)}: {self.original_strokes[self.current_stroke_idx].description}")
            self._update_geometries()
            
        elif key == ord('LEFT') or key == ord('A') or key == ord('a'):
            # Previous stroke
            self.current_stroke_idx = (self.current_stroke_idx - 1) % len(self.original_strokes)
            log.info(f"Stroke {self.current_stroke_idx + 1}/{len(self.original_strokes)}: {self.original_strokes[self.current_stroke_idx].description}")
            self._update_geometries()
            
        elif key == ord('H') or key == ord('h'):
            # Show help
            self._show_help()
            
        elif key == ord('Q') or key == ord('q'):
            # Quit
            vis.close()
            
        return True
    
    def _show_help(self):
        """Display help information."""
        help_text = """
        Mapping Debug Visualizer Controls:
        ===================================
        
        Navigation:
        - Mouse: Rotate, pan, zoom
        - A/LEFT: Previous stroke
        - D/RIGHT: Next stroke
        
        Toggle Visibility:
        - N: Toggle normals (blue arrows)
        - F: Toggle flat strokes (red, 2D)
        - M: Toggle mapped strokes (green, 3D)
        - O: Toggle origin marker (yellow sphere)
        
        Normal Display:
        - L: Increase normal length
        - K: Decrease normal length
        
        Other:
        - H: Show this help
        - Q: Quit
        
        Colors:
        - Gray: Point cloud surface
        - Yellow: Design origin
        - Red: Original flat strokes (2D)
        - Green: Mapped strokes (3D)
        - Blue: Surface normals
        """
        log.info(help_text)
    
    def run(self):
        """Run the interactive visualizer."""
        log.info("Starting Mapping Debug Visualizer")
        log.info(f"Loaded {len(self.original_strokes)} strokes")
        
        # Check if we're in a headless environment
        import os
        if not os.environ.get('DISPLAY'):
            log.info("No display detected, running in headless mode")
            self._run_headless()
            return
        
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        
        # Try to create window, but fall back to headless if it fails
        window_created = False
        try:
            self.vis.create_window(
                window_name="Surface Mapping Debugger",
                width=1200,
                height=800
            )
            window_created = True
            log.info("Interactive window created successfully")
        except Exception as e:
            log.warning(f"Failed to create interactive window: {e}")
        
        # If window creation failed, run in headless mode
        if not window_created:
            log.info("Running in headless mode - saving screenshots instead")
            self._run_headless()
            return
        
        # Set up key callback
        try:
            self.vis.register_key_callback(self._key_callback)
        except Exception as e:
            log.warning(f"Failed to register key callback: {e}")
            log.info("Running without interactive controls")
        
        # Initial geometry setup
        self._update_geometries()
        
        # Show help
        self._show_help()
        
        # Main loop
        log.info("Visualizer ready. Press H for help.")
        while self.vis.poll_events():
            self.vis.update_renderer()
        
        self.vis.destroy_window()
        log.info("Visualizer closed")
    
    def _run_headless(self):
        """Run visualizer in headless mode, saving screenshots."""
        log.info("Creating headless visualizer...")
        log.info(f"Processing {len(self.original_strokes)} strokes...")
        
        # In headless mode, just create a data summary
        log.info("Running in headless mode - creating data summary")
        self._create_data_summary()
    
    def _create_data_summary(self):
        """Create a simple text summary of the mapping data."""
        log.info("Creating data summary...")
        
        summary = []
        summary.append("=== Surface Mapping Debug Summary ===")
        summary.append(f"Point cloud: {self.point_cloud_file}")
        summary.append(f"Point cloud shape: {self.point_cloud.shape}")
        summary.append(f"Scene: {self.scene_name}")
        summary.append(f"Design origin: {self.design_origin.pos.xyz}")
        summary.append(f"Number of strokes: {len(self.original_strokes)}")
        summary.append("")
        
        for i, (stroke, mapped_stroke) in enumerate(zip(self.original_strokes, self.mapped_strokes)):
            summary.append(f"Stroke {i+1}: {stroke.description}")
            summary.append(f"  Arm: {stroke.arm}")
            summary.append(f"  Is inkdip: {stroke.is_inkdip}")
            if stroke.ee_pos is not None:
                summary.append(f"  Original points: {len(stroke.ee_pos)}")
                summary.append(f"  Original bounds: X[{stroke.ee_pos[:, 0].min():.4f}, {stroke.ee_pos[:, 0].max():.4f}], Y[{stroke.ee_pos[:, 1].min():.4f}, {stroke.ee_pos[:, 1].max():.4f}]")
            if mapped_stroke.ee_pos is not None:
                summary.append(f"  Mapped points: {len(mapped_stroke.ee_pos)}")
                summary.append(f"  Mapped bounds: X[{mapped_stroke.ee_pos[:, 0].min():.4f}, {mapped_stroke.ee_pos[:, 0].max():.4f}], Y[{mapped_stroke.ee_pos[:, 1].min():.4f}, {mapped_stroke.ee_pos[:, 1].max():.4f}], Z[{mapped_stroke.ee_pos[:, 2].min():.4f}, {mapped_stroke.ee_pos[:, 2].max():.4f}]")
            if mapped_stroke.normals is not None:
                summary.append(f"  Has normals: Yes ({len(mapped_stroke.normals)} points)")
            summary.append("")
        
        # Write summary to file
        summary_path = "mapping_debug_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary))
        
        log.info(f"Data summary saved to: {summary_path}")
        
        # Also print to console
        print('\n'.join(summary))





def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Surface Mapping Debug Visualizer")
    parser.add_argument("--pointcloud", required=True, help="Path to point cloud file (.ply or .npy)")
    parser.add_argument("--scene", required=True, help="Name of the scene to load strokes from")
    parser.add_argument("--skip-mapping", action="store_true", help="Skip surface mapping and use original strokes")
    
    args = parser.parse_args()
    
    # Validate point cloud file
    if not Path(args.pointcloud).exists():
        log.error(f"Point cloud file not found: {args.pointcloud}")
        return 1
    
    # Create and run visualizer
    try:
        visualizer = MappingDebugVisualizer(args.pointcloud, args.scene, skip_mapping=args.skip_mapping)
        visualizer.run()
    except Exception as e:
        log.error(f"Visualizer error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 