import logging
from dataclasses import dataclass
import traceback

import numpy as np

from tatbot.data.pose import Pose
from tatbot.data.stroke import StrokeList
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.gen.map import map_strokes_to_mesh
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.utils.plymesh import (
    create_mesh_from_ply_files,
    load_ply,
    ply_files_from_dir,
)
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.map", "🗺️")


@dataclass
class VizMapConfig(BaseVizConfig):
    stroke_point_size: float = 0.0005
    """Size of stroke points in the visualization (meters)."""
    stroke_point_shape: str = "rounded"
    """Shape of stroke points in the visualization."""
    
    skin_ply_point_size: float = 0.0005
    """Size of skin ply points in the visualization (meters)."""
    skin_ply_point_shape: str = "rounded"
    """Shape of skin ply points in the visualization."""
    
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
        
        self.skin_zone = self.server.scene.add_box(
            name="/skin/zone",
            color=(0, 255, 0),
            dimensions=(
                self.scene.skin.zone_depth_m,
                self.scene.skin.zone_width_m,
                self.scene.skin.zone_height_m,
            ),
            position=self.scene.skin.design_pose.pos.xyz,
            wxyz=self.scene.skin.design_pose.rot.wxyz,
            opacity=0.2,
            visible=True,
        )
        
        @self.design_pose_tf.on_update
        def _(_):
            self.skin_zone.position = self.design_pose_tf.position
            self.skin_zone.wxyz = self.design_pose_tf.wxyz
        
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

        self.skin_mesh = None
        self.skin_mesh_vertices = None
        self.skin_mesh_faces = None
        self.skin_ply_files = ply_files_from_dir(self.scene.skin.plymesh_dir)
        self.enabled_skin_ply_files = {ply_file: True for ply_file in self.skin_ply_files}
        self.skin_pointclouds = {}
        for ply_file in self.skin_ply_files:
            points, colors = load_ply(ply_file)
            self.skin_pointclouds[ply_file] = self.server.scene.add_point_cloud(
                name=f"/skin/{ply_file.split('/')[-1]}",
                points=points,
                colors=colors,
                point_size=config.skin_ply_point_size,
                point_shape=config.skin_ply_point_shape,
            )
            log.info(f"Loaded skin pointcloud with {len(points)} points from {ply_file}")

        with self.server.gui.add_folder("Mapping", expand_by_default=True):
            self.build_skin_mesh_button = self.server.gui.add_button(
                "Build Skin Mesh",
                hint="Build the skin mesh from the PLY files"
            )
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
            self.show_skin_mesh = self.server.gui.add_checkbox(
                "Show Skin Mesh",
                initial_value=False,
            )
            self.show_skin_zone = self.server.gui.add_checkbox(
                "Show Skin Zone",
                initial_value=True,
            )
            with self.server.gui.add_folder("Skin PLY Files", expand_by_default=False):
                self.enabled_skin_ply_files_checkboxes = {}
                for ply_file in self.skin_ply_files:
                    self.enabled_skin_ply_files_checkboxes[ply_file] = self.server.gui.add_checkbox(
                        ply_file.split("/")[-1],
                        initial_value=True,
                    )
                    @self.enabled_skin_ply_files_checkboxes[ply_file].on_update
                    def _(_):
                        self.enabled_skin_ply_files[ply_file] = self.enabled_skin_ply_files_checkboxes[ply_file].value
                        if self.enabled_skin_ply_files[ply_file]:
                            self.skin_pointclouds[ply_file].visible = True
                        else:
                            self.skin_pointclouds[ply_file].visible = False
                
        
        @self.build_skin_mesh_button.on_click
        def _(_):
            try:
                log.info("Building skin mesh...")
                points, faces = create_mesh_from_ply_files(
                    [ply_file for ply_file in self.skin_ply_files if self.enabled_skin_ply_files[ply_file]],
                    zone_pose=Pose.from_wxyz_xyz(self.skin_zone.wxyz, self.skin_zone.position),
                    zone_depth_m=self.scene.skin.zone_depth_m,
                    zone_width_m=self.scene.skin.zone_width_m,
                    zone_height_m=self.scene.skin.zone_height_m,
                )
                self.skin_mesh_vertices = points
                self.skin_mesh_faces = faces
                self.skin_mesh = self.server.scene.add_mesh_simple(
                    name="/skin/mesh",
                    vertices=points,
                    faces=faces,
                    color=(200, 200, 200),
                    opacity=0.8,
                    material='standard',
                    flat_shading=False,
                    side='double',
                    cast_shadow=True,
                    receive_shadow=True,
                    visible=True
                )
                log.info("Successfully built and added skin mesh to scene")

            except Exception:
                log.error(f"Failed to build skin mesh: {traceback.format_exc()}")

        @self.map_strokes_button.on_click
        def _(_):
            try:
                if self.skin_mesh_vertices is None or self.skin_mesh_faces is None:
                    log.error("No mesh available. Please build the skin mesh first.")
                    return
                    
                log.info("Mapping strokes to surface...")
                mapped_strokes = map_strokes_to_mesh(
                    vertices=self.skin_mesh_vertices,
                    faces=self.skin_mesh_faces,
                    strokes=self.strokes,
                    design_origin=Pose.from_wxyz_xyz(self.design_pose_tf.wxyz, self.design_pose_tf.position),
                    stroke_length=self.scene.stroke_length,
                )
                for i, (stroke_l, stroke_r) in enumerate(mapped_strokes.strokes):
                    self.mapped_stroke_pointclouds["l"][i].points = stroke_l.ee_pos
                    self.mapped_stroke_pointclouds["r"][i].points = stroke_r.ee_pos
                log.info("Successfully mapped strokes to surface")

            except Exception:
                log.error(f"Failed to map strokes to surface: {traceback.format_exc()}")
        
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
        
        @self.show_skin_mesh.on_update
        def _(_):
            if self.skin_mesh is not None:
                self.skin_mesh.visible = self.show_skin_mesh.value

        @self.show_skin_zone.on_update
        def _(_):
            self.skin_zone.visible = self.show_skin_zone.value

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