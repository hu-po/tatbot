import logging
from dataclasses import dataclass
import os

import numpy as np
import open3d as o3d

from tatbot.data.pose import Pose
from tatbot.data.stroke import StrokeList
from tatbot.gen.map import map_strokes_to_surface
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.map", "🗺️")


@dataclass
class VizMapConfig(BaseVizConfig):
    ply_files: tuple[str, ...] = (
        "~/tatbot/nfs/3d/hand/rs_000000.ply",
        "~/tatbot/nfs/3d/hand/rs_000001.ply",
    )
    """Path to the PLY files to visualize."""
    
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

        self.design_pose_tf = self.server.scene.add_transform_controls(
            "/design_pose",
            position=self.scene.skin.design_pose.pos.xyz,
            wxyz=self.scene.skin.design_pose.rot.wxyz,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        
        self.strokes: StrokeList = make_gcode_strokes(self.scene)
        self.stroke_pointclouds = {"l": [], "r": []}
        self.mapped_stroke_pointclouds = {"l": [], "r": []}
        for i, (stroke_l, stroke_r) in enumerate(self.strokes.strokes):
            if not stroke_l.is_inkdip and not stroke_l.is_rest:
                pointcloud = self.server.scene.add_point_cloud(
                    # By making this a child of design_pose, meter_coords are correctly transformed
                    name=f"/design_pose/stroke_l_{i:03d}",
                    points=stroke_l.meter_coords,
                    colors=np.zeros((len(stroke_l.meter_coords), 3), dtype=np.uint8),
                    point_size=config.stroke_point_size,
                    point_shape=config.stroke_point_shape,
                )
                self.stroke_pointclouds["l"].append(pointcloud)
                # mapped pointclouds are in world frame, start off as empty
                mapped_pointcloud = self.server.scene.add_point_cloud(
                    name=f"/mapped_stroke_l_{i:03d}",
                    points=np.zeros((1, 3)),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=config.stroke_point_size,
                    point_shape=config.stroke_point_shape,
                )
                self.mapped_stroke_pointclouds["l"].append(mapped_pointcloud)
            if not stroke_r.is_inkdip and not stroke_r.is_rest:
                pointcloud = self.server.scene.add_point_cloud(
                    name=f"/design_pose/stroke_r_{i:03d}",
                    points=stroke_r.meter_coords,
                    colors=np.zeros((len(stroke_r.meter_coords), 3), dtype=np.uint8),
                    point_size=config.stroke_point_size,
                    point_shape=config.stroke_point_shape,
                )
                self.stroke_pointclouds["r"].append(pointcloud)
                mapped_pointcloud = self.server.scene.add_point_cloud(
                    name=f"/mapped_stroke_r_{i:03d}",
                    points=np.zeros((1, 3)),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=config.stroke_point_size,
                    point_shape=config.stroke_point_shape,
                )
                self.mapped_stroke_pointclouds["r"].append(mapped_pointcloud)

        self.skin_pointclouds = {}
        for ply_file in config.ply_files:
            pcd = o3d.io.read_point_cloud(os.path.expanduser(ply_file))
            self.skin_pointclouds[ply_file] = self.server.scene.add_point_cloud(
                name=f"/skin/{ply_file.split('/')[-1]}",
                points=np.asarray(pcd.points),
                colors=np.asarray(pcd.colors),
                point_size=config.surface_point_size,
                point_shape=config.surface_point_shape,
            )
            log.info(f"Loaded skin pointcloud with {len(pcd.points)} points from {ply_file}")

        with self.server.gui.add_folder("Mapping", expand_by_default=True):
            self.map_strokes_button = self.server.gui.add_button(
                "Map Strokes to Skin",
                hint="Apply surface mapping to strokes"
            )
            self.show_stroke_pointclouds = self.server.gui.add_checkbox(
                "Show Strokes", 
                initial_value=True,
            )
            self.show_mapped_stroke_pointclouds = self.server.gui.add_checkbox(
                "Show Mapped Strokes",
                initial_value=False,
            )
        
        @self.map_strokes_button.on_click
        def _(_):
            try:
                log.info("Mapping strokes to surface...")
                mapped_strokes = map_strokes_to_surface(
                    self.config.ply_files,
                    self.strokes,
                    Pose.from_wxyz_xyz(self.design_pose_tf.wxyz, self.design_pose_tf.position),
                    self.scene.stroke_length,
                )
                for i, (stroke_l, stroke_r) in enumerate(mapped_strokes.strokes):
                    self.mapped_stroke_pointclouds["l"][i].points = stroke_l.meter_coords
                    self.mapped_stroke_pointclouds["r"][i].points = stroke_r.meter_coords
                log.info("Successfully mapped strokes to surface")

            except Exception as e:
                log.error(f"Failed to map strokes to surface: {e}")
        
        @self.show_mapped_stroke_pointclouds.on_update
        def _(_):
            for pointcloud in self.mapped_stroke_pointclouds["l"]:
                pointcloud.visible = self.show_mapped_stroke_pointclouds.value
            for pointcloud in self.mapped_stroke_pointclouds["r"]:
                pointcloud.visible = self.show_mapped_stroke_pointclouds.value
        
        @self.show_stroke_pointclouds.on_update
        def _(_):
            for pointcloud in self.stroke_pointclouds["l"]:
                pointcloud.visible = self.show_stroke_pointclouds.value
            for pointcloud in self.stroke_pointclouds["r"]:
                pointcloud.visible = self.show_stroke_pointclouds.value

    def step(self):
        """Empty step function - this visualization is static."""
        pass


if __name__ == "__main__":
    args = setup_log_with_config(VizMapConfig, submodules=["gen.map"])
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = VizMap(args)
    viz.run() 