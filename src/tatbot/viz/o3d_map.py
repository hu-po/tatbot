""" DISPLAY=:1 uv run python src/tatbot/viz/o3d_map.py """
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
    
    window_height: int = 800
    """Height of the window."""
    window_width: int = 1200
    """Width of the window."""
    window_bg_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    """Background color of the window."""

    scene: str = "default"
    """Name of the scene (Scene)."""


def create_stroke_pointcloud(strokes: StrokeList) -> o3d.geometry.PointCloud:
    """
    Create a colored pointcloud from stroke positions and normals.
    
    Args:
        strokes: StrokeList containing mapped strokes with ee_pos and normals
        
    Returns:
        o3d.geometry.PointCloud: Colored pointcloud with positions and normals
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
    
    # Create pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    log.info(f"Created stroke pointcloud with {len(points)} points")
    return pcd


def visualize_mapping(config: O3DMapVizConfig):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
            window_name="Stroke Mapping Visualization",
            width=config.window_width,
            height=config.window_height
        )
    vis.get_render_option().background_color = np.asarray(config.window_bg_color)

    pcd = o3d.io.read_point_cloud(config.ply_file)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    
    scene = Scene.from_name(config.scene)
    strokes: StrokeList = make_gcode_strokes(scene)
    stroke_pcd = create_stroke_pointcloud(strokes)
    # strokes = map_strokes_to_surface(
    #     config.ply_file,
    #     strokes,
    #     scene.skin.design_pose,
    #     scene.stroke_length,
    # )
    vis.add_geometry(stroke_pcd)
    vis.get_render_option().point_size = 3.0
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    args = setup_log_with_config(O3DMapVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    visualize_mapping(args)