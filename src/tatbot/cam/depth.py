import os
from typing import Tuple

import jax.numpy as jnp
import jaxlie
import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

from tatbot.data.pose import Pose
from tatbot.utils.log import get_logger
from tatbot.utils.plymesh import save_ply

log = get_logger("cam.depth", "📹")

class DepthCamera:
    def __init__(self,
            serial_number: str,
            initial_pose: Pose,
            decimation: int = 1, # Decimation filter magnitude for depth frames (integer >= 1)
            clipping: Tuple[float, float] = (0.01, 0.3), # Clipping range for depth in meters (min, max).
            timeout_ms: int = 8000, # Timeout for the RealSense camera in milliseconds.
            save_prefix: str = "rs_",
            save_dir: str = "/tmp",
        ):
        self.pose: Pose = initial_pose
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.config.resolve(pipeline_wrapper)
        self.config.enable_device(serial_number)
        # TODO: tune the stream parameters for optimal high quality skin detection
        self.config.enable_stream(rs.stream.depth) #, rs.format.z16, config.fps)
        self.config.enable_stream(rs.stream.color) #, rs.format.rgb8, config.fps)
        self.pipeline.start(self.config)
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.decimation: int = decimation
        self.clipping: Tuple[float, float] = clipping
        self.timeout_ms: int = timeout_ms
        self.save_dir: str = os.path.expanduser(save_dir)
        self.save_prefix: str = save_prefix
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.frame_idx: int = 0 # counter keeps track of number of saved frames

    def get_pointcloud(self, save: bool = False) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        log.info("Capturing pointcloud...")
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
        # TODO: https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md#holes-filling-filter
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
        positions_world = jaxlie.SE3(wxyz_xyz=jnp.concatenate([self.pose.rot.wxyz, self.pose.pos.xyz], axis=-1)) @ positions
        
        if save:
            output_path = os.path.join(self.save_dir, f"{self.save_prefix}{self.frame_idx:06d}.ply")
            save_ply(output_path, positions_world, colors)
            self.frame_idx += 1
        return color_image, positions_world, colors
