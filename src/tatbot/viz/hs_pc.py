import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import open3d as o3d
import potpourri3d as pp3d

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.hs_pc", "📊")

@dataclass
class PointcloudVizConfig(BaseVizConfig):
    """Configuration for pointcloud visualization"""
    pointcloud_path: Optional[str] = "~/tatbot/nfs/3d/fakeskin.ply"
    """Path to the pointcloud file to visualize."""
    
    # Override camera settings for pointcloud viewing
    view_camera_position: tuple[float, float, float] = (0.0, 0.0, 2.0)
    """Initial camera position for pointcloud viewing."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at for pointcloud viewing."""

@dataclass
class AppState:
    """Application state for geometric operations"""
    geodesic_source: int = -1
    extension_sources: list = field(default_factory=list)
    transport_source: int = -1
    transport_vector: list = field(default_factory=lambda: [6.0, 6.0])
    logmap_source: int = -1
    curves: list = field(default_factory=lambda: [[]])

class PointcloudViz(BaseViz):
    """Pointcloud visualization using viser, inheriting from BaseViz"""
    
    def __init__(self, config: PointcloudVizConfig):
        # Initialize base class
        super().__init__(config)
        
        # Pointcloud-specific state
        self.points = None
        self.solver = None
        self.basis_x = None
        self.basis_y = None
        self.basis_n = None
        self.pointcloud_name = ""
        self.selected_point = None
        
        # Point cloud visualization
        self.point_cloud_handle = None
        self.scalar_quantities = {}
        self.vector_quantities = {}
        
        # Initialize AppState for geometric operations
        self.state = AppState()
        
        # Setup pointcloud-specific GUI
        self._setup_pointcloud_gui()
    
    def _setup_pointcloud_gui(self):
        """Setup the pointcloud-specific GUI controls"""
        
        # Geodesic Distance Section
        with self.server.gui.add_folder("Geodesic Distance", expand_by_default=False):
            self.geodesic_source_text = self.server.gui.add_text(
                "Source", 
                initial_value="None", 
                disabled=True,
                hint="Selected source point for geodesic distance"
            )
            
            def set_geodesic_source():
                if self.selected_point is not None:
                    self.state.geodesic_source = self.selected_point
                    self.geodesic_source_text.value = f"Point {self.state.geodesic_source}"
                    log.info(f"Set geodesic source to {self.state.geodesic_source}")
                else:
                    log.warning("No point selected for geodesic source")
            
            self.server.gui.add_button("Set Source Point", set_geodesic_source)
            
            def compute_geodesic_distance():
                if self.state.geodesic_source != -1:
                    dists = self.solver.compute_distance(self.state.geodesic_source)
                    self._add_scalar_quantity("Geodesic Distance", dists)
                    log.info("Computed geodesic distance")
                else:
                    log.warning("No source selected for geodesic distance")
            
            self.server.gui.add_button("Compute Geodesic Distance", compute_geodesic_distance)
        
        # Scalar Extension Section
        with self.server.gui.add_folder("Scalar Extension", expand_by_default=False):
            self.extension_sources_text = self.server.gui.add_text(
                "Sources", 
                initial_value="[]", 
                disabled=True,
                hint="Selected source points for scalar extension"
            )
            
            def add_extension_source():
                if self.selected_point is not None:
                    self.state.extension_sources.append(self.selected_point)
                    self.extension_sources_text.value = str(self.state.extension_sources)
                    log.info(f"Added extension source {self.selected_point}")
                else:
                    log.warning("No point selected for extension source")
            
            self.server.gui.add_button("Add Source", add_extension_source)
            
            def clear_extension_sources():
                self.state.extension_sources.clear()
                self.extension_sources_text.value = "[]"
                log.info("Cleared extension sources")
            
            self.server.gui.add_button("Clear Sources", clear_extension_sources)
            
            def compute_scalar_extension():
                if len(self.state.extension_sources) >= 2:
                    points = self.state.extension_sources
                    values = np.linspace(0.0, 1.0, len(points)).tolist()
                    ext = self.solver.extend_scalar(points, values)
                    self._add_scalar_quantity("Scalar Extension", ext)
                    log.info("Computed scalar extension")
                else:
                    log.warning("Need at least 2 sources for scalar extension")
            
            self.server.gui.add_button("Compute Scalar Extension", compute_scalar_extension)
        
        # Vector Transport Section
        with self.server.gui.add_folder("Vector Transport", expand_by_default=False):
            self.transport_source_text = self.server.gui.add_text(
                "Source", 
                initial_value="None", 
                disabled=True,
                hint="Selected source point for vector transport"
            )
            
            def set_transport_source():
                if self.selected_point is not None:
                    self.state.transport_source = self.selected_point
                    self.transport_source_text.value = f"Point {self.state.transport_source}"
                    log.info(f"Set transport source to {self.state.transport_source}")
                else:
                    log.warning("No point selected for transport source")
            
            self.server.gui.add_button("Set Source", set_transport_source)
            
            self.transport_vx = self.server.gui.add_slider(
                "vx", 
                min=-10.0, 
                max=10.0, 
                initial_value=self.state.transport_vector[0]
            )
            self.transport_vy = self.server.gui.add_slider(
                "vy", 
                min=-10.0, 
                max=10.0, 
                initial_value=self.state.transport_vector[1]
            )
            
            def compute_vector_transport():
                if self.state.transport_source != -1:
                    vector = [self.transport_vx.value, self.transport_vy.value]
                    ext = self.solver.transport_tangent_vector(self.state.transport_source, vector)
                    # potpourri3d returns 2D vectors, convert to 3D using basis
                    ext3d = ext[:, 0, np.newaxis] * self.basis_x + ext[:, 1, np.newaxis] * self.basis_y
                    self._add_vector_quantity("Transported Vector", ext3d)
                    log.info("Computed vector transport")
                else:
                    log.warning("No source selected for vector transport")
            
            self.server.gui.add_button("Compute Vector Transport", compute_vector_transport)
        
        # Log Map Section
        with self.server.gui.add_folder("Log Map", expand_by_default=False):
            self.logmap_source_text = self.server.gui.add_text(
                "Source", 
                initial_value="None", 
                disabled=True,
                hint="Selected source point for log map"
            )
            
            def set_logmap_source():
                if self.selected_point is not None:
                    self.state.logmap_source = self.selected_point
                    self.logmap_source_text.value = f"Point {self.state.logmap_source}"
                    log.info(f"Set logmap source to {self.state.logmap_source}")
                else:
                    log.warning("No point selected for logmap source")
            
            self.server.gui.add_button("Set Source", set_logmap_source)
            
            def compute_log_map():
                if self.state.logmap_source != -1:
                    logmap = self.solver.compute_log_map(self.state.logmap_source)
                    # potpourri3d returns 2D vectors, convert to 3D using basis
                    logmap3d = logmap[:, 0, np.newaxis] * self.basis_x + logmap[:, 1, np.newaxis] * self.basis_y
                    self._add_vector_quantity("Log Map", logmap3d)
                    log.info("Computed log map")
                else:
                    log.warning("No source selected for log map")
            
            self.server.gui.add_button("Compute Log Map", compute_log_map)
        
        # Signed Distance Section
        with self.server.gui.add_folder("Signed Distance", expand_by_default=False):
            self.curves_text = self.server.gui.add_text(
                "Curves", 
                initial_value="[[]]", 
                disabled=True,
                hint="Selected curves for signed distance computation"
            )
            
            def add_point_to_curve():
                if self.selected_point is not None:
                    if not self.state.curves:
                        self.state.curves.append([])
                    self.state.curves[-1].append(self.selected_point)
                    self.curves_text.value = str(self.state.curves)
                    log.info(f"Added point {self.selected_point} to curve")
                else:
                    log.warning("No point selected to add to curve")
            
            self.server.gui.add_button("Add Point to Last Curve", add_point_to_curve)
            
            def add_new_curve():
                self.state.curves.append([])
                self.curves_text.value = str(self.state.curves)
                log.info("Added new curve")
            
            self.server.gui.add_button("Add New Curve", add_new_curve)
            
            def clear_curves():
                self.state.curves = [[]]
                self.curves_text.value = str(self.state.curves)
                log.info("Cleared curves")
            
            self.server.gui.add_button("Clear Curves", clear_curves)
            
            def compute_signed_distance():
                valid_curves = [c for c in self.state.curves if c]
                if valid_curves:
                    signed_dist = self.solver.compute_signed_distance(valid_curves, self.basis_n)
                    self._add_scalar_quantity("Signed Distance", signed_dist)
                    log.info("Computed signed distance")
                else:
                    log.warning("No valid curves to compute signed distance")
            
            self.server.gui.add_button("Compute Signed Distance", compute_signed_distance)
    
    def _add_scalar_quantity(self, name: str, values: np.ndarray):
        """Add a scalar quantity to the point cloud visualization"""
        if self.point_cloud_handle is None:
            return
        
        # Normalize values to 0-1 range for color mapping
        if values.max() > values.min():
            normalized_values = (values - values.min()) / (values.max() - values.min())
        else:
            normalized_values = np.zeros_like(values)
        
        # Create color map (blue to red)
        colors = np.zeros((len(values), 3))
        colors[:, 0] = normalized_values  # Red
        colors[:, 2] = 1.0 - normalized_values  # Blue
        
        # Update point cloud colors
        self.point_cloud_handle.color = colors
        self.scalar_quantities[name] = values
        log.info(f"Added scalar quantity: {name}")
    
    def _add_vector_quantity(self, name: str, vectors: np.ndarray):
        """Add a vector quantity to the visualization"""
        if self.point_cloud_handle is None:
            return
        
        # Create arrow visualization for vectors
        arrow_scale = 0.1
        arrow_positions = self.points
        arrow_directions = vectors * arrow_scale
        
        # Add arrows to the scene
        for i in range(0, len(arrow_positions), max(1, len(arrow_positions) // 100)):  # Sample every 100th point
            pos = arrow_positions[i]
            direction = arrow_directions[i]
            if np.linalg.norm(direction) > 0.001:  # Only show non-zero vectors
                self.server.scene.add_arrow(
                    name=f"/vectors/{name}_{i}",
                    origin=pos,
                    direction=direction,
                    color=(255, 255, 0),  # Yellow arrows
                    radius=0.01,
                )
        
        self.vector_quantities[name] = vectors
        log.info(f"Added vector quantity: {name}")
    
    def setup_pointcloud(self, points: np.ndarray, solver: Any, basis_x: np.ndarray, 
                        basis_y: np.ndarray, basis_n: np.ndarray, pointcloud_name: str):
        """Setup the point cloud visualization"""
        self.points = points
        self.solver = solver
        self.basis_x = basis_x
        self.basis_y = basis_y
        self.basis_n = basis_n
        self.pointcloud_name = pointcloud_name
        
        # Create point cloud visualization
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            name=f"/pointcloud/{pointcloud_name}",
            points=points,
            colors=np.ones((len(points), 3)) * 128,  # Gray default color
            point_size=0.02,
        )
        
        # Setup point selection
        @self.point_cloud_handle.on_click
        def _(event):
            # Find closest point to click
            click_pos = np.array([event.origin[0], event.origin[1], event.origin[2]])
            distances = np.linalg.norm(points - click_pos, axis=1)
            closest_idx = np.argmin(distances)
            self.selected_point = closest_idx
            log.info(f"Selected point {closest_idx}")
    
    def step(self):
        """Override step method - for pointcloud visualization we don't need continuous updates"""
        pass


def load_pointcloud(pointcloud_path: str) -> tuple[np.ndarray, Any]:
    """Load pointcloud using potpourri3d with open3d fallback"""
    if not pointcloud_path:
        raise ValueError("pointcloud_path must be provided")
    
    pointcloud_path = os.path.expanduser(pointcloud_path)
    log.info(f"Reading pointcloud from {pointcloud_path}")
    
    # Read pointcloud using potpourri3d
    try:
        P = pp3d.read_point_cloud(pointcloud_path)
        log.info(f"Loaded {len(P)} points using potpourri3d")
    except Exception as e:
        log.warning(f"Failed to read with potpourri3d: {e}, trying open3d fallback")
        # Fallback to open3d if potpourri3d fails
        pcd = o3d.io.read_point_cloud(pointcloud_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            log.warning("No points found, trying alternative PLY reading method")
            try:
                # Try reading as mesh and extracting vertices
                mesh = o3d.io.read_triangle_mesh(pointcloud_path)
                points = np.asarray(mesh.vertices)
                if len(points) == 0:
                    raise ValueError("No points found in the file")
            except Exception as e2:
                log.error(f"Failed to read pointcloud: {e2}")
                raise
        
        P = points
        log.info(f"Loaded {len(P)} points using open3d fallback")
    
    # Create heat solver using potpourri3d
    log.info("Creating heat solver...")
    solver = pp3d.PointCloudHeatSolver(P)
    
    # Get tangent frames from potpourri3d
    log.info("Computing tangent frames...")
    basis_x, basis_y, basis_n = solver.get_tangent_frames()
    
    return P, solver, basis_x, basis_y, basis_n


def create_pointcloud_viz(config: PointcloudVizConfig) -> PointcloudViz:
    """Factory function to create a pointcloud visualization"""
    return PointcloudViz(config)


def view_pointcloud(config: PointcloudVizConfig):
    """Main function to view a pointcloud with geometric processing"""
    # Load pointcloud and create solver
    points, solver, basis_x, basis_y, basis_n = load_pointcloud(config.pointcloud_path)
    
    # Create and setup pointcloud visualization
    viz = PointcloudViz(config)
    viz.setup_pointcloud(points, solver, basis_x, basis_y, basis_n, "My Pointcloud")
    
    # Run the visualization
    viz.run()


if __name__ == "__main__":
    args = setup_log_with_config(PointcloudVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    view_pointcloud(args) 