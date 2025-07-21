import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from tatbot.data.stroke import StrokeList
from tatbot.data.scene import Scene
from tatbot.data.pose import Pose, Pos, Rot
from tatbot.gen.map import map_strokes_to_surface
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.map", "🗺️")


@dataclass
class VizMapConfig(BaseVizConfig):
    ply_file: str = "/tmp/rs_000001.ply"
    """Path to the PLY file to visualize."""
    
    stroke_point_size: float = 0.003
    """Size of stroke points in the visualization (meters)."""
    stroke_point_shape: str = "rounded"
    """Shape of stroke points in the visualization."""
    
    surface_point_size: float = 0.001
    """Size of surface points in the visualization (meters)."""
    surface_point_shape: str = "rounded"
    """Shape of surface points in the visualization."""
    
    transform_control_scale: float = 0.1
    """Scale of the transform control frame for design pose visualization."""
    transform_control_opacity: float = 0.8
    """Opacity of the transform control frame for design pose visualization."""


class VizMap(BaseViz):
    def __init__(self, config: VizMapConfig):
        super().__init__(config)
        
        # Load scene and generate strokes
        self.scene = Scene.from_name(config.scene)
        self.strokes: StrokeList = make_gcode_strokes(self.scene)
        
        # Create stroke pointcloud
        self.stroke_pcd = self._create_stroke_pointcloud(self.strokes)
        
        # Add stroke pointcloud to scene
        self.stroke_pointcloud = self.server.scene.add_point_cloud(
            name="/strokes",
            points=self.stroke_pcd["points"],
            colors=self.stroke_pcd["colors"],
            point_size=config.stroke_point_size,
            point_shape=config.stroke_point_shape,
        )
        
        # Add surface pointcloud to scene
        pcd = o3d.io.read_point_cloud(config.ply_file)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        self.surface_pointcloud = self.server.scene.add_point_cloud(
            name="/surface",
            points=points,
            colors=colors,
            point_size=config.surface_point_size,
            point_shape=config.surface_point_shape,
        )
        
        log.info(f"Loaded surface pointcloud with {len(points)} points from {config.ply_file}")
    
        # Get initial design pose
        initial_pose = self.scene.skin.design_pose
        
        # Add transform control
        self.design_pose_transform = self.server.scene.add_transform_controls(
            "/design_pose",
            position=initial_pose.pos.xyz,
            wxyz=initial_pose.rot.wxyz,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        
        # Add GUI controls for design pose
        with self.server.gui.add_folder("Design Pose", expand_by_default=True):
            self.reset_pose_button = self.server.gui.add_button(
                "Reset Design Pose",
                hint="Reset design pose to original position"
            )
            self.auto_map_checkbox = self.server.gui.add_checkbox(
                "Auto-map on pose change",
                initial_value=False,
                hint="Automatically remap strokes when design pose changes"
            )
        
        # Store original pose for reset functionality
        self.original_design_pose = initial_pose
        
        # Set up callbacks
        @self.design_pose_transform.on_update
        def _(_):
            self._on_design_pose_changed()
        
        @self.reset_pose_button.on_click
        def _(_):
            self._reset_design_pose()

        # Add GUI controls
        with self.server.gui.add_folder("Mapping", expand_by_default=True):
            self.map_strokes_button = self.server.gui.add_button(
                "Map Strokes to Surface",
                hint="Apply surface mapping to strokes"
            )
            self.show_mapped_checkbox = self.server.gui.add_checkbox(
                "Show Mapped Strokes",
                initial_value=False,
                hint="Toggle visibility of mapped strokes"
            )
            self.show_original_checkbox = self.server.gui.add_checkbox(
                "Show Original Strokes", 
                initial_value=True,
                hint="Toggle visibility of original strokes"
            )
        
        # Initialize mapped strokes
        self.mapped_strokes = None
        self.mapped_pointcloud = None
        
        @self.map_strokes_button.on_click
        def _(_):
            self._map_strokes_to_surface()
        
        @self.show_mapped_checkbox.on_update
        def _(_):
            if self.mapped_pointcloud is not None:
                self.mapped_pointcloud.visible = self.show_mapped_checkbox.value
        
        @self.show_original_checkbox.on_update
        def _(_):
            self.stroke_pointcloud.visible = self.show_original_checkbox.value

    def _create_stroke_pointcloud(self, strokes: StrokeList) -> dict:
        """
        Create a colored pointcloud from stroke positions and normals.
        
        Args:
            strokes: StrokeList containing mapped strokes with ee_pos and normals
            
        Returns:
            dict: Dictionary with points, normals, and colors arrays
        """
        all_points = []
        all_normals = []
        all_colors = []
        
        for stroke_l, stroke_r in strokes.strokes:
            # Process left stroke
            if stroke_l.ee_pos is not None:
                all_points.append(stroke_l.ee_pos)
                if stroke_l.normals is not None:
                    all_normals.append(stroke_l.normals)
                else:
                    all_normals.append(np.zeros((len(stroke_l.ee_pos), 3)))
                # Make all points black
                all_colors.extend([[0.0, 0.0, 0.0]] * len(stroke_l.ee_pos))
            
            # Process right stroke
            if stroke_r.ee_pos is not None:
                all_points.append(stroke_r.ee_pos)
                if stroke_r.normals is not None:
                    all_normals.append(stroke_r.normals)
                else:
                    all_normals.append(np.zeros((len(stroke_r.ee_pos), 3)))
                # Make all points black
                all_colors.extend([[0.0, 0.0, 0.0]] * len(stroke_r.ee_pos))
        
        # Combine all points, normals, and colors
        points = np.vstack(all_points)
        normals = np.vstack(all_normals)
        colors = np.array(all_colors)
        
        log.info(f"Created stroke pointcloud with {len(points)} points")
        return {
            "points": points,
            "normals": normals,
            "colors": colors
        }

    def _on_design_pose_changed(self):
        """Handle design pose changes."""
        # Update the scene's design pose
        new_pos = self.design_pose_transform.position
        new_wxyz = self.design_pose_transform.wxyz
        
        # Create new pose using the proper constructor
        new_pose = Pose(
            pos=Pos(xyz=np.array(new_pos)),
            rot=Rot(wxyz=np.array(new_wxyz))
        )
        
        # Update scene's design pose
        self.scene.skin.design_pose = new_pose
        
        log.info(f"Design pose updated to position {new_pos}")
        
        # Auto-remap if enabled
        if self.auto_map_checkbox.value and self.mapped_strokes is not None:
            self._map_strokes_to_surface()

    def _reset_design_pose(self):
        """Reset design pose to original position."""
        original_pos = self.original_design_pose.pos.xyz
        original_wxyz = self.original_design_pose.rot.wxyz
        
        # Update transform control
        self.design_pose_transform.position = original_pos
        self.design_pose_transform.wxyz = original_wxyz
        
        # Update scene's design pose
        self.scene.skin.design_pose = self.original_design_pose
        
        log.info("Design pose reset to original position")
        
        # Auto-remap if enabled
        if self.auto_map_checkbox.value and self.mapped_strokes is not None:
            self._map_strokes_to_surface()

    def _map_strokes_to_surface(self):
        """Map strokes to the surface and visualize the result."""
        try:
            log.info("Mapping strokes to surface...")
            
            # Map strokes to surface
            mapped_strokes = map_strokes_to_surface(
                self.config.ply_file,
                self.strokes,
                self.scene.skin.design_pose,
                self.scene.stroke_length,
            )
            
            # Create pointcloud from mapped strokes
            mapped_pcd = self._create_stroke_pointcloud(mapped_strokes)
            
            # Remove existing mapped pointcloud if it exists
            if self.mapped_pointcloud is not None:
                self.server.scene.remove("/mapped_strokes")
            
            # Add new mapped pointcloud
            self.mapped_pointcloud = self.server.scene.add_point_cloud(
                name="/mapped_strokes",
                points=mapped_pcd["points"],
                colors=mapped_pcd["colors"],
                point_size=self.config.stroke_point_size,
                point_shape=self.config.stroke_point_shape,
            )
            
            # Set visibility based on checkbox
            self.mapped_pointcloud.visible = self.show_mapped_checkbox.value
            
            self.mapped_strokes = mapped_strokes
            log.info("Successfully mapped strokes to surface")
            
        except Exception as e:
            log.error(f"Failed to map strokes to surface: {e}")

    def step(self):
        """Empty step function - this visualization is static."""
        pass


if __name__ == "__main__":
    args = setup_log_with_config(VizMapConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = VizMap(args)
    viz.run() 