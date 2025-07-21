import logging
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
import jaxlie
import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs
from viser import PointCloudHandle

import open3d as o3d
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.rs_pc", "📹")


@dataclass
class RsPcVizConfig(BaseVizConfig):
    debug: bool = False
    """Enable debug logging."""
    save_dir: str = ""
    """If set, save pointclouds as PLY files to this directory."""

@dataclass
class RealSenseConfig:
    serial_number: str = ""
    """Serial number of the RealSense camera device."""
    point_size: float = 0.001
    """Size of points in the point cloud visualization."""
    decimation: int = 6
    """Decimation filter magnitude for depth frames (integer >= 1)."""
    clipping: Tuple[float, float] = (0.03, 0.8)
    """Clipping range for depth in meters (min, max)."""
    timeout_ms: int = 8000
    """Timeout for the RealSense camera in milliseconds."""

class RealSenseCamera:
    def __init__(self, config: RealSenseConfig):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.config.resolve(pipeline_wrapper)
        self.config.enable_device(config.serial_number)
        # TODO: tune the stream parameters for optimal high quality skin detection
        self.config.enable_stream(rs.stream.depth) #, rs.format.z16, config.fps)
        self.config.enable_stream(rs.stream.color) #, rs.format.rgb8, config.fps)
        self.pipeline.start(self.config)
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.decimation = config.decimation
        self.clipping = config.clipping
        self.timeout_ms = config.timeout_ms
        
    @property
    def fov(self) -> float:
        """Vertical FOV in radians."""
        return 2 * np.arctan2(self.intrinsics.height / 2, self.intrinsics.fy)

    @property
    def aspect(self) -> float:
        """Aspect ratio of the camera."""
        return self.intrinsics.width / self.intrinsics.height

    def make_observation(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        point_cloud = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, self.decimation)
        frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
        depth_min, depth_max = self.clipping
        depth_frame = rs.threshold_filter(depth_min, depth_max).process(depth_frame)
        color_frame = frames.get_color_frame()
        point_cloud.map_to(color_frame)
        points = point_cloud.calculate(depth_frame)
        texture_uv = (
            np.asanyarray(points.get_texture_coordinates())
            .view(np.float32)
            .reshape((-1, 2))
        )
        color_image = np.asanyarray(color_frame.get_data())
        color_h, color_w, _ = color_image.shape
        texture_uv = texture_uv.clip(0.0, 1.0)
        positions = np.asanyarray(points.get_vertices()).view(np.float32)
        positions = positions.reshape((-1, 3))
        colors = color_image[
            (texture_uv[:, 1] * (color_h - 1.0)).astype(np.int32),
            (texture_uv[:, 0] * (color_w - 1.0)).astype(np.int32),
            :,
        ]
        return color_image, positions, colors

class RsPcViz(BaseViz):
    def __init__(self, config: RsPcVizConfig):
        super().__init__(config)

        self.realsense_cams: Dict[str, RealSenseCamera] = {}
        self.realsense_pointclouds: Dict[str, PointCloudHandle] = {}
        self.save_dir = config.save_dir
        self.frame_idx = 0
        
        # Add GUI controls for point cloud saving
        with self.server.gui.add_folder("Point Cloud Recording", expand_by_default=False):
            self.save_checkbox = self.server.gui.add_checkbox(
                "Save Point Clouds", 
                initial_value=False,
                hint="Toggle saving point clouds as PLY files"
            )
            self.save_dir_textbox = self.server.gui.add_text(
                "Save Directory", 
                initial_value=self.save_dir or "~/tatbot/pointclouds",
                hint="Directory to save point cloud PLY files"
            )
            self.frames_saved_textbox = self.server.gui.add_text(
                "Frames Saved", 
                initial_value="0", 
                disabled=True,
                hint="Number of frames saved so far"
            )
        
        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
        for i, realsense in enumerate(self.scene.cams.realsenses):
            _config = RealSenseConfig(serial_number=realsense.serial_number)
            log.info(f"Initializing {realsense.name} config: {_config}")
            self.realsense_cams[realsense.name] = RealSenseCamera(_config)
            self.realsense_pointclouds[realsense.name] = self.server.scene.add_point_cloud(
                f"/pointcloud_{realsense.name}",
                points=np.zeros((1, 3)),
                colors=np.zeros((1, 3), dtype=np.uint8),
                point_size=_config.point_size,
            )

    def save_ply(self, filename, points, colors):
        # points: (N, 3), colors: (N, 3) uint8
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Open3D expects colors in [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        o3d.io.write_point_cloud(filename, pcd)

    def step(self):
        for realsense in self.scene.cams.realsenses:
            realsense_start_time = time.time()
            rgb, positions, colors = self.realsense_cams[realsense.name].make_observation()
            self.realsense_frustrums[realsense.name].image = rgb
            positions_world = jaxlie.SE3(wxyz_xyz=jnp.concatenate([
                self.realsense_frustrums[realsense.name].wxyz,
                self.realsense_frustrums[realsense.name].position,
            ], axis=-1)) @ positions
            self.realsense_pointclouds[realsense.name].points = np.array(positions_world)
            self.realsense_pointclouds[realsense.name].colors = np.array(colors)
            
            # Check if saving is enabled via GUI checkbox
            if self.save_checkbox.value:
                # Use the directory from the textbox
                save_dir = self.save_dir_textbox.value.strip()
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    fname = f"{save_dir}/pc_{realsense.name}_{self.global_step:06d}.ply"
                    self.save_ply(fname, np.array(positions_world), np.array(colors))
                    self.frame_idx += 1
                    # Update the frames saved counter
                    self.frames_saved_textbox.value = str(self.frame_idx)
            
            log.info(f"{realsense.name} took {time.time() - realsense_start_time} seconds")


if __name__ == "__main__":
    args = setup_log_with_config(RsPcVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = RsPcViz(args)
    viz.run()