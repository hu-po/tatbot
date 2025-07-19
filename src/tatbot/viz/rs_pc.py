import logging
import os
from dataclasses import dataclass
from typing import Tuple
import time

import numpy as np
import pyrealsense2 as rs

from tatbot.bot.urdf import get_link_indices, get_link_poses
from tatbot.data.pose import ArmPose, Pose
from tatbot.gen.ik import ik
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.rs_pc", "�")


@dataclass
class RsPcVizConfig(BaseVizConfig):
    debug: bool = False
    """Enable debug logging."""

    realsense_a: RealSenseConfig = RealSenseConfig(serial_number="230422273017", link_name="right/camera_depth_optical_frame")
    """Configuration for RealSense Camera A (attached to right arm)."""
    realsense_b: RealSenseConfig = RealSenseConfig(serial_number="218622278376", link_name="cam_high_depth_optical_frame")
    """Configuration for RealSense Camera B (overhead)."""

@dataclass
class RealSenseConfig:
    fps: int = 5
    """Frames per second for the RealSense camera."""
    point_size: float = 0.001
    """Size of points in the point cloud visualization."""
    serial_number: str = ""
    """Serial number of the RealSense camera device."""
    link_name: str = ""
    """Name of the camera link in the robot URDF."""
    decimation: int = 6
    """Decimation filter magnitude for depth frames (integer >= 1)."""
    clipping: Tuple[float, float] = (0.03, 0.8)
    """Clipping range for depth in meters (min, max)."""

class RealSenseCamera:
    def __init__(self, config: RealSenseConfig):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.config.resolve(pipeline_wrapper)
        self.config.enable_device(config.serial_number)
        self.config.enable_stream(rs.stream.depth, rs.format.z16, config.fps)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8, config.fps)
        self.pipeline.start(self.config)
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.decimation = config.decimation
        self.clipping = config.clipping
        
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
        frames = self.pipeline.wait_for_frames()
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
        for realsense_camera in self.scene.cams.realsenses:
            self.realsense_cams[realsense_camera.name] = RealSenseCamera(
                RealSenseConfig(
                    fps=realsense_camera.fps,
                    point_size=realsense_camera.point_size,
                    serial_number=realsense_camera.serial_number,
                    link_name=realsense_camera.link_name,
                    frustrum_scale=realsense_camera.frustrum_scale,
                )
            )
            self.realsense_pointclouds[realsense_camera.name] = self.server.scene.add_point_cloud(
                f"/pointcloud_{realsense_camera.name}",
                points=np.zeros((1, 3)),
                colors=np.zeros((1, 3), dtype=np.uint8),
                point_size=realsense_camera.point_size,
            )

    def step(self):
        for rs_name in self.scene.cams.realsenses:
            realsense_start_time = time.time()
            rs_config = self.scene.cams.realsenses[rs_name]
            rs_camera = RealSenseCamera(rs_config)
            rgb, positions, colors = rs_camera.make_observation()
            self.realsense_frustrums[rs_name].image = rgb
            
            camera_link_idx_a = self.robot.links.names.index(config.realsense_a.link_name)
            joint_array = np.concatenate([joint_pos_current.left, joint_pos_current.right])
            camera_pose_a = self.robot.forward_kinematics(joint_array)[camera_link_idx_a]
            realsense_a_frustrum.wxyz = camera_pose_a[:4]
            realsense_a_frustrum.position = camera_pose_a[-3:]
            camera_transform_a = jaxlie.SE3(camera_pose_a)
            positions_world_a = camera_transform_a @ positions_a
            pointcloud_a.points = np.array(positions_world_a)
            pointcloud_a.colors = np.array(colors_a)
            
            log.info(f"🔍 {rs_name} took {time.time() - realsense_start_time} seconds")


        # ORIGINAL CODE BELOW - REFERENCE ONLY
        realsense_start_time = time.time()
        rgb_a, positions_a, colors_a = self.realsense_a.make_observation()
        self.realsense_a_frustrum.image = rgb_a
        rgb_b, positions_b, colors_b = self.realsense_b.make_observation()
        realsense_b_frustrum.image = rgb_b
        camera_link_idx_a = robot.links.names.index(config.realsense_a.link_name)
        joint_array = np.concatenate([joint_pos_current.left, joint_pos_current.right])
        camera_pose_a = robot.forward_kinematics(joint_array)[camera_link_idx_a]
        realsense_a_frustrum.wxyz = camera_pose_a[:4]
        realsense_a_frustrum.position = camera_pose_a[-3:]
        realsense_b_frustrum.wxyz = camera_pose_b_static[:4]
        realsense_b_frustrum.position = camera_pose_b_static[-3:]
        camera_transform_a = jaxlie.SE3(camera_pose_a)
        camera_transform_b = jaxlie.SE3(camera_pose_b_static)
        positions_world_a = camera_transform_a @ positions_a
        positions_world_b = camera_transform_b @ positions_b
        pointcloud_a.points = np.array(positions_world_a)
        pointcloud_a.colors = np.array(colors_a)
        pointcloud_b.points = np.array(positions_world_b)
        pointcloud_b.colors = np.array(colors_b)
        realsense_elapsed_time = time.time() - realsense_start_time
        realsense_duration_ms.value = realsense_elapsed_time * 1000


if __name__ == "__main__":
    args = setup_log_with_config(RsPcVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = RsPcViz(args)
    viz.run()
