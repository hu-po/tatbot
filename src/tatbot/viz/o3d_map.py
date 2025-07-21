from dataclasses import dataclass
import logging

import numpy as np
import open3d as o3d

from tatbot.data.stroke import StrokeList
from tatbot.data.scene import Scene
from tatbot.gen.map import map_strokes_to_surface
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("viz.o3d_map", "🔍")

@dataclass
class O3DMapVizConfig:
    debug: bool = False
    """Enable debug logging."""

    ply_file: str = "/tmp/rs_000001.ply"
    """Path to the PLY file to visualize."""

    scene: str = "default"
    """Name of the scene (Scene)."""


def visualize_mapping(config: O3DMapVizConfig):
    pcd = o3d.io.read_point_cloud(config.ply_file)
    scene = Scene.from_name(config.scene)
    strokes: StrokeList = make_gcode_strokes(scene)
    strokes = map_strokes_to_surface(
        config.ply_file,
        strokes,
        scene.skin.design_pose,
        scene.stroke_length,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
            window_name="Minimal PLY Viewer",
            width=800,
            height=600
        )
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    args = setup_log_with_config(O3DMapVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    visualize_mapping(args)