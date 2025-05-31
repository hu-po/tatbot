# INFO: This file controls tatbot: a robotic tattoo machine.
# INFO: This python file requires dependencies defined in the pyproject.toml file.
# INFO: This file is a python script indended to be run directly with optional cli args.
# INFO: This file will attempt to use a GPU if available.
# INFO: When editing, do not remove any TODOs in this file.
# INFO: When editing, do not add any additional comments to the code.
# INFO: When editing, use log to add minimal but essential debug and info messages.
# INFO: When setting float values in configs, use 3 decimal places
# INFO: Use emojis tastefully.

from dataclasses import dataclass, field
import logging
import os
import time
from typing import List, Tuple

import cv2
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jaxtyping import Array, Float, Int
import numpy as np
import numpy.typing as npt
import PIL.Image
import pyrealsense2 as rs
import pyroki as pk
import tyro
import trimesh
import trossen_arm
import viser
from viser.extras import ViserUrdf
import yourdfpy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclass
class CLIArgs:
    debug: bool = False
    """Enables debug mode: allows for moving objects in the scene, enables debug logging."""
    robot: bool = False
    """Enables the real robot (if False then only sim)."""
    realsense: bool = False
    """Enables the RealSense cameras."""
    device_name: str = "cuda:0"
    """Name of the JAX device to use (i.e. 'gpu', 'cpu')."""

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"]
    wxyz: Float[Array, "4"]

@jdc.pytree_dataclass
class JointPos:
    left: Float[Array, "8"]
    right: Float[Array, "8"]

@jdc.pytree_dataclass
class PixelTarget:
    pose: Pose

@jdc.pytree_dataclass
class PixelBatch:
    center_pose: Pose
    """Center pose of the batch patch."""
    radius_m: float
    """Radius of the patch in meters."""
    pixel_targets: List[PixelTarget]
    """List of pixel targets within this batch."""

@jdc.pytree_dataclass
class IKConfig:
    pos_weight: float = 50.0
    """Weight for the position part of the IK cost function."""
    ori_weight: float = 10.0
    """Weight for the orientation part of the IK cost function."""
    limit_weight: float = 100.0
    """Weight for the joint limit part of the IK cost function."""
    lambda_initial: float = 1.0
    """Initial lambda value for the IK trust region solver."""

@jdc.pytree_dataclass
class InkCap:
    palette_pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Pose of the inkcap (relative to palette frame)."""
    dip_depth_m: float = 0.005
    """Depth of the inkcap when dipping (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0) # black
    """RGB color of the ink in the inkcap."""

@jdc.pytree_dataclass
class RealSenseConfig:
    fps: int = 5
    """Frames per second for the RealSense camera."""
    point_size: float = 0.001
    """Size of points in the point cloud visualization."""
    serial_number: str = ""
    """Serial number of the RealSense camera device."""
    link_name: str = ""
    """Name of the camera link in the robot URDF."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    env_map_background: bool = True
    """Whether to show the environment map as background."""

@dataclass
class TatbotConfig:
    urdf_path: str = os.path.expanduser("~/tatbot-urdf/tatbot.urdf")
    """Local path to the URDF file for the robot (https://github.com/hu-po/tatbot-urdf)."""
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    """Arm model for the robot."""
    ip_address_l: str = "192.168.1.2"
    """IP address of the left robot arm."""
    ip_address_r: str = "192.168.1.3"
    """IP address of the right robot arm."""
    end_effector_model_l: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_leader
    """End effector model for the left robot arm."""
    end_effector_model_r: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    """End effector model for the right robot arm."""
    joint_pos_sleep: JointPos = JointPos(left=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), right=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    """Sleep: robot is folded up, motors can be released (radians)."""
    joint_pos_work: JointPos = JointPos(
        left=jnp.array([0.431, 1.120, 0.270, 0.241, 0.360, 0.241, 0.022, 0.044]),
        right=jnp.array([-0.147, 1.107, 0.526, -0.416, -0.963, -1.062, 0.022, 0.022])
    )
    """Work: robot is ready to work (radians)."""
    joint_pos_calib: JointPos = JointPos(
        left=jnp.array([0.530, 1.503, 1.167, -1.165, 0.027, 0.166, 0.022, 0.044]),
        right=jnp.array([0.144, 1.354, 0.796, -0.986, 0.021, -0.209, 0.022, 0.022])
    )
    """Calibration: left arm is at workspace (42, 28) and right arm is at workspace (2, 28) (radians)."""
    set_all_position_goal_time_slow: float = 3.0
    """Goal time in seconds when the goal positions should be reached (slow)."""
    set_all_position_goal_time_fast: float = 0.1
    """Goal time in seconds when the goal positions should be reached (fast)."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the links to be controlled."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""
    num_pixels_per_ink_dip: int = 60
    """Number of pixels to draw before dipping the pen in the ink cup again."""
    min_fps: float = 1.0
    """Minimum frames per second to maintain. If 0 or negative, no minimum framerate is enforced."""
    ik_target_pose_l: Pose = Pose(pos=jnp.array([0.243, 0.127, 0.070]), wxyz=jnp.array([0.960, 0.000, 0.279, 0.000]))
    """Initial pose of the grabbable transform IK target for left robot (relative to root frame)."""
    ik_target_pose_r: Pose = Pose(pos=jnp.array([0.253, -0.105, 0.111]), wxyz=jnp.array([0.821, -0.190, 0.173, 0.505]))
    """Initial pose of the grabbable transform IK target for right robot (relative to root frame)."""
    states: List[str] = field(default_factory=lambda: ["MANUAL", "WORK", "STANDOFF", "POKE", "DIP", "PAUSED"])
    """Possible states of the robot.
      > MANUAL: Manual control mode, robot follows ik targets.
      > WORK: Robot is ready to work.
      > STANDOFF: Robot is in standoff position above a pixel target.
      > POKE: Robot is poking the pixel target.
      > DIP: Robot is dipping the pen in the inkcap.
      > PAUSED: Robot is paused at current IK target position.
    """
    initial_state: str = "PAUSED"
    """Initial state of the robot."""
    design_pose: Pose = Pose(pos=jnp.array([0.313, 0.074, 0.065]), wxyz=jnp.array([1.000, 0.000, 0.000, 0.000]))
    """Pose of the design (relative to root frame)."""
    image_path: str = os.path.expanduser("~/tatbot/assets/designs/flower.png")
    """Local path to the tattoo design PNG image."""
    image_threshold: int = 127
    """Threshold for B/W image. Pixels less than or equal to this value are targets. [0, 255]"""
    max_draw_pixels: int = 0
    """Maximum number of target pixels to process. If 0 or less, process all."""
    image_width_px: int = 256
    """Width to resize the input image to before processing (pixels)."""
    image_height_px: int = 256
    """Height to resize the input image to before processing (pixels)."""
    image_width_m: float = 0.04
    """Width of the area on the skin where the image will be projected (meters)."""
    image_height_m: float = 0.04
    """Height of the area on the skin where the image will be projected (meters)."""
    point_size: float = 0.001
    """Size of points in the point cloud visualization (meters)."""
    point_color: Tuple[int, int, int] = (0, 0, 0) # black
    """Color for the points in the point cloud (RGB tuple)."""
    batch_radius_m: float = 0.004
    """Radius in meters for each batch of pixels to process."""
    point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""
    standoff_offset_m: Float[Array, "3"] = field(default_factory=lambda: jnp.array([0.01, 0.0, 0.0]))
    """Offset vector for the standoff position (meters)."""
    poke_offset_m: Float[Array, "3"] = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    """Offset vector for the poke position (meters)."""
    palette_init_pose: Pose = Pose(pos=jnp.array([0.281, 0.156, 0.032]), wxyz=jnp.array([0.971, 0.006, -0.024, 0.240]))
    """Pose of the palette (relative to root frame)."""
    palette_mesh_path: str = os.path.expanduser("~/tatbot/assets/3d/inkpalette-lowpoly/inkpalette-lowpoly.obj")
    """Path to the .obj file for the palette mesh."""
    inkcaps: Tuple[InkCap, ...] = (
        InkCap(
            palette_pose=Pose(pos=jnp.array([0.005, 0.014, 0.000]), wxyz=jnp.array([1.000, 0.000, 0.000, 0.000])),
            color=(0, 0, 0) # black
        ),
        InkCap(
            palette_pose=Pose(pos=jnp.array([-0.011, 0.015, 0.001]), wxyz=jnp.array([1.000, 0.000, 0.000, 0.000])),
            color=(255, 0, 0) # red
        ),
        InkCap(
            palette_pose=Pose(pos=jnp.array([-0.020, -0.005, 0.000]), wxyz=jnp.array([1.000, 0.000, 0.000, 0.000])),
            color=(0, 255, 0) # green
        ),
        InkCap(
            palette_pose=Pose(pos=jnp.array([-0.026, 0.015, 0.005]), wxyz=jnp.array([0.999, 0.000, 0.000, 0.013])),
            color=(0, 0, 255) # blue
        ),
    )
    skin_init_pose: Pose = Pose(pos=jnp.array([0.303, 0.071, 0.044]), wxyz=jnp.array([0.701, 0.115, -0.698, 0.097]))
    """Pose of the skin (relative to root frame)."""
    skin_mesh_path: str = os.path.expanduser("~/tatbot/assets/3d/fakeskin-lowpoly/fakeskin-lowpoly.obj")
    """Path to the .obj file for the skin mesh."""
    workspace_init_pose: Pose = Pose(pos=jnp.array([0.287, 0.049, 0.022]), wxyz=jnp.array([-0.115, 0.000, 0.000, 0.993]))
    """Pose of the workspace origin (relative to root frame)."""
    workspace_mesh_path: str = os.path.expanduser("~/tatbot/assets/3d/mat-lowpoly/mat-lowpoly.obj")
    """Path to the .obj file for the workspace mat mesh."""
    view_camera_position: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    realsense_a: RealSenseConfig = RealSenseConfig(serial_number="230422273017", link_name="right/camera_depth_optical_frame")
    """Configuration for RealSense Camera A (attached to right arm)."""
    realsense_b: RealSenseConfig = RealSenseConfig(serial_number="218622278376", link_name="cam_high_depth_optical_frame")
    """Configuration for RealSense Camera B (overhead)."""
    # CLI overrides
    enable_robot: bool = False
    """Override for arg.robot."""
    enable_realsense: bool = False
    """Override for arg.realsense."""
    debug_mode: bool = False
    """Override for arg.debug."""

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
        
    def get_points(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        point_cloud = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 3)        
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
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
        N = positions.shape[0]
        assert positions.shape == (N, 3)
        assert positions.dtype == np.float32
        assert colors.shape == (N, 3)
        assert colors.dtype == np.uint8
        return positions, colors


@jdc.jit
def transform_targets(
    design_pose: Pose,
    targets: Float[Array, "B 3"],
    offsets: Float[Array, "B 3"],
) -> Float[Array, "B 3"]:
    design_to_root = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(design_pose.wxyz),
        design_pose.pos
    )
    return jax.vmap(lambda pos, offset: design_to_root @ pos + offset)(targets, offsets)

@jdc.jit
def ik(
    robot: pk.Robot,
    target_link_indices: Int[Array, "B"],
    target_wxyz: Float[Array, "B 4"],
    target_position: Float[Array, "B 3"],
    config: IKConfig,
) -> Float[Array, "B 16"]:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_indices,
            pos_weight=config.pos_weight,
            ori_weight=config.ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            jnp.array([config.limit_weight] * robot.joints.num_joints),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky", # TODO: is this the best?
            trust_region=jaxls.TrustRegionConfig(lambda_initial=config.lambda_initial),
        )
    )
    return sol[joint_var]

def main(config: TatbotConfig):
    if config.debug_mode:
        log.info("🐛 Debug mode enabled.")
        log.setLevel(logging.DEBUG)

    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = config.view_camera_position
        client.camera.look_at = config.view_camera_look_at

    log.info("🖼️ Loading design...")
    img_pil = PIL.Image.open(config.image_path)
    original_width, original_height = img_pil.size
    if original_width > config.image_width_px or original_height > config.image_height_px:
        img_pil = img_pil.resize((config.image_width_px, config.image_height_px), PIL.Image.LANCZOS)
    img_pil = img_pil.convert("L")
    img_np = np.array(img_pil)
    img_width_px, img_height_px = img_pil.size
    # TODO: use opencv dependency to do fancier contours/thresholding
    thresholded_pixels = img_np <= config.image_threshold
    pixel_to_meter_x = config.image_width_m / img_width_px
    pixel_to_meter_y = config.image_height_m / img_height_px
    
    log.info("🔢 Creating pixel batches...")
    num_targets: int = img_height_px * img_width_px
    target_pointcloud_positions: np.ndarray = np.zeros((num_targets, 3))
    target_pointcloud_colors: np.ndarray = np.zeros((num_targets, 3))
    batch_radius_px_x = int(config.batch_radius_m / pixel_to_meter_x)
    batch_radius_px_y = int(config.batch_radius_m / pixel_to_meter_y)
    batches: List[PixelBatch] = []
    for center_y in range(batch_radius_px_y, img_height_px, batch_radius_px_y * 2):
        for center_x in range(batch_radius_px_x, img_width_px, batch_radius_px_x * 2):
            batch_pixels: List[PixelTarget] = []
            for y in range(max(0, center_y - batch_radius_px_y), min(img_height_px, center_y + batch_radius_px_y)):
                for x in range(max(0, center_x - batch_radius_px_x), min(img_width_px, center_x + batch_radius_px_x)):
                    if thresholded_pixels[y, x]:
                        meter_x = (x - img_width_px/2) * pixel_to_meter_x
                        meter_y = (y - img_height_px/2) * pixel_to_meter_y
                        pixel_target = PixelTarget(
                            pose=Pose( # in design frame
                                pos=jnp.array([meter_x, meter_y, 0.0]),
                                wxyz=config.design_pose.wxyz
                            )
                        )
                        batch_pixels.append(pixel_target)
                        target_pointcloud_positions[y * img_width_px + x] = jnp.array([meter_x, meter_y, 0.0])
                        target_pointcloud_colors[y * img_width_px + x] = config.point_color
            # Only create batch if it contains targets
            if batch_pixels:
                batch = PixelBatch(
                    center_pose=Pose(
                        pos=jnp.array([(center_x - img_width_px/2) * pixel_to_meter_x, (center_y - img_height_px/2) * pixel_to_meter_y, 0.0]),
                        wxyz=config.design_pose.wxyz
                    ),
                    radius_m=config.batch_radius_m,
                    pixel_targets=batch_pixels
                )
                batches.append(batch)

    num_batches: int = len(batches)
    log.info(f"🔢 Design has {num_targets} targets in {num_batches} batches.")

    log.info("🖼️ Adding design...")
    if config.debug_mode:
        design_tf = server.scene.add_transform_controls(
            name="/design",
            position=config.design_pose.pos,
            wxyz=config.design_pose.wxyz,
            scale=0.05,
            opacity=0.2,
        )
    else:
        design_tf = server.scene.add_frame(
            name="/design",
            position=config.design_pose.pos,
            wxyz=config.design_pose.wxyz,
            show_axes=False,
        )
    design_pose = Pose(pos=design_tf.position, wxyz=design_tf.wxyz)
    server.scene.add_point_cloud(
        name="/design/targets",
        points=target_pointcloud_positions,
        colors=target_pointcloud_colors,
        point_size=config.point_size,
        point_shape=config.point_shape,
    )

    log.info("🔲 Adding workspace...")
    if config.debug_mode:
        workspace_tf = server.scene.add_transform_controls(
            "/workspace",
            position=config.workspace_init_pose.pos,
            wxyz=config.workspace_init_pose.wxyz,
            scale=0.2,
            opacity=0.2,
        )
    else:
        workspace_tf = server.scene.add_frame(
            "/workspace",
            position=config.workspace_init_pose.pos,
            wxyz=config.workspace_init_pose.wxyz,
            show_axes=False,
        )
    server.scene.add_mesh_trimesh(
        name="/workspace/mesh",
        mesh=trimesh.load(config.workspace_mesh_path),
    )

    log.info("🎨 Adding palette...")
    if config.debug_mode:
        palette_tf = server.scene.add_transform_controls(
            "/palette",
            position=config.palette_init_pose.pos,
            wxyz=config.palette_init_pose.wxyz,
            scale=0.2,
            opacity=0.2,
        )
    else:
        palette_tf = server.scene.add_frame(
            "/palette",
            position=config.palette_init_pose.pos,
            wxyz=config.palette_init_pose.wxyz,
            show_axes=False,
        )
    server.scene.add_mesh_trimesh(
        name="/palette/mesh",
        mesh=trimesh.load(config.palette_mesh_path),
    )
    inkcap_tfs: List[viser.TransformControls] = []
    for i, inkcap in enumerate(config.inkcaps):
        log.info(f"🎨 Adding inkcap {i}...")
        if config.debug_mode:
            inkcap_tf = server.scene.add_transform_controls(
                f"/palette/inkcap_{i}",
                position=inkcap.palette_pose.pos,
                wxyz=inkcap.palette_pose.wxyz,
                scale=0.2,
                opacity=0.2,
            )
        else:
            inkcap_tf = server.scene.add_frame(
                f"/palette/inkcap_{i}",
                position=inkcap.palette_pose.pos,
                wxyz=inkcap.palette_pose.wxyz,
                show_axes=False,
            )
        inkcap_tfs.append(inkcap_tf)

    log.info("💪 Adding skin...")
    if config.debug_mode:
        skin_tf = server.scene.add_transform_controls(
            "/skin",
            position=config.skin_init_pose.pos,
            wxyz=config.skin_init_pose.wxyz,
            scale=0.2,
            opacity=0.2,
        )
    else:
        skin_tf = server.scene.add_frame(
            "/skin",
            position=config.skin_init_pose.pos,
            wxyz=config.skin_init_pose.wxyz,
            show_axes=False,
        )
    server.scene.add_mesh_trimesh(
        name="/skin/mesh",
        mesh=trimesh.load(config.skin_mesh_path),
    )

    log.info("🎯 Adding ik targets...")
    ik_target_positions: Float[Array, "B 3"] = jnp.zeros((num_batches, 3))
    ik_target_l = server.scene.add_transform_controls(
        "/ik_target_l",
        position=config.ik_target_pose_l.pos,
        wxyz=config.ik_target_pose_l.wxyz,
        scale=0.1,
        opacity=0.5,
        visible=True if config.debug_mode else False,
    )
    ik_target_r = server.scene.add_transform_controls(
        "/ik_target_r",
        position=config.ik_target_pose_r.pos,
        wxyz=config.ik_target_pose_r.wxyz,
        scale=0.1,
        opacity=0.5,
        visible=True if config.debug_mode else False,
    )

    log.info("🦾 Adding robots...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(config.urdf_path)
    robot: pk.Robot = pk.Robot.from_urdf(urdf)
    joint_pos_current: JointPos = config.joint_pos_sleep
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")

    log.info("🖥️  Adding gui...")
    with server.gui.add_folder("Session"):
        batch_index = server.gui.add_slider(
            "Batch Index",
            min=0,
            max=num_batches - 1,
            step=1,
            initial_value=0,
        )
        target_index = server.gui.add_slider(
            "Target Index",
            min=0,
            max=len(batches[0].pixel_targets) - 1,
            step=1,
            initial_value=0,
        )
        design_image = server.gui.add_image(
            image=img_np,
            label=config.image_path,
            format="png",
            order="rgb",
            visible=True,
        )
    with server.gui.add_folder("Timing"):
        ik_duration_ms = server.gui.add_number("ik (ms)", 0.001, disabled=True)
        realsense_duration_ms = server.gui.add_number("realsense (ms)", 0.001, disabled=True)
        move_duration_ms = server.gui.add_number("robot move (ms)", 0.001, disabled=True)
        step_duration_ms = server.gui.add_number("step (ms)", 0.001, disabled=True)
    with server.gui.add_folder("Robot"):
        state_handle = server.gui.add_dropdown(
            "State", options=config.states,
            initial_value=config.initial_state
        )
        sleep_button = server.gui.add_button("Go to Sleep")

        @sleep_button.on_click
        def _(_):
            log.debug("😴 Moving left robot to sleep pose...")
            state_handle.value = "PAUSED"
            move_robot(config.joint_pos_sleep)

    try:
        if config.enable_realsense:
            log.info("📷 Initializing RealSense cameras...")
            realsense_a = RealSenseCamera(config.realsense_a)
            realsense_b = RealSenseCamera(config.realsense_b)
            pointcloud_a = server.scene.add_point_cloud(
                f"/realsense/a",
                points=np.zeros((1, 3)),
                colors=np.zeros((1, 3), dtype=np.uint8),
                point_size=config.realsense_a.point_size,
            )
            pointcloud_b = server.scene.add_point_cloud(
                f"/realsense/b",
                points=np.zeros((1, 3)),
                colors=np.zeros((1, 3), dtype=np.uint8),
                point_size=config.realsense_b.point_size,
            )
            camera_link_idx_b = robot.links.names.index(config.realsense_b.link_name)
            home_joint_array = np.concatenate([config.joint_pos_work.left, config.joint_pos_work.right])
            camera_pose_b_static = robot.forward_kinematics(home_joint_array)[camera_link_idx_b]
        if config.enable_robot:
            log.info("🤖 Initializing robot drivers...")

            def init_robot(clear_error: bool = False) -> Tuple[trossen_arm.TrossenArmDriver, trossen_arm.TrossenArmDriver]:
                driver_l = trossen_arm.TrossenArmDriver()
                driver_r = trossen_arm.TrossenArmDriver()
                driver_l.configure(
                    config.arm_model,
                    config.end_effector_model_l,
                    config.ip_address_l,
                    clear_error,
                )
                driver_r.configure(
                    config.arm_model,
                    config.end_effector_model_r,
                    config.ip_address_r,
                    clear_error,
                )
                driver_l.set_all_modes(trossen_arm.Mode.position)
                driver_r.set_all_modes(trossen_arm.Mode.position)
                return driver_l, driver_r
            
            driver_l, driver_r = init_robot()

        def move_robot(joint_pos: JointPos, goal_time: float = config.set_all_position_goal_time_slow):
            log.debug(f"🤖 Moving robot to: {joint_pos}")
            urdf_vis.update_cfg(np.concatenate([joint_pos.left, joint_pos.right]))
            if config.enable_robot:
                driver_l.set_all_positions(
                    trossen_arm.VectorDouble(joint_pos.left[:7].tolist()),
                    goal_time=goal_time,
                    blocking=True,
                )
                driver_r.set_all_positions(
                    trossen_arm.VectorDouble(joint_pos.right[:7].tolist()),
                    goal_time=goal_time,
                    blocking=True,
                )
            else:
                time.sleep(goal_time)
        
        log.info("🤖 Moving robots to sleep pose...")
        move_robot(config.joint_pos_sleep)
        if config.debug_mode:
            log.info("🤖 Moving robots to calibration pose...")
            move_robot(config.joint_pos_calib)
            joint_pos_current = config.joint_pos_calib
            state_handle.value = "PAUSED"
        else:
            log.info("🤖 Moving robots to home pose...")
            move_robot(config.joint_pos_work)
            joint_pos_current = config.joint_pos_work
            state_handle.value = "MANUAL"

        while True:
            step_start_time = time.time()
            log.debug(f"State: {state_handle.value}")
            
            if state_handle.value == "WORK":
                log.info(f"🔢 Current batch: {batch_index.value}")
                if batch_index.value >= num_batches:
                    log.info("Completed all batches, looping back to start.")
                    batch_index.value = 0
                    target_index.value = 0
                current_batch: PixelBatch = batches[batch_index.value]
                log.debug(f"🖥️ Updating GUI with batch {batch_index.value}")
                target_index.max = len(current_batch.pixel_targets) - 1
                img_viz = cv2.cvtColor(img_np.copy(), cv2.COLOR_GRAY2RGB)
                center_x_viz = int((current_batch.center_pose.pos[0] / pixel_to_meter_x) + img_width_px/2)
                center_y_viz = int((current_batch.center_pose.pos[1] / pixel_to_meter_y) + img_height_px/2)
                cv2.ellipse(img_viz, (center_x_viz, center_y_viz), (batch_radius_px_x, batch_radius_px_y), 0, 0, 360, (0, 255, 0), 1)
                for target in current_batch.pixel_targets:
                    x_viz = int((target.pose.pos[0] / pixel_to_meter_x) + img_width_px / 2)
                    y_viz = int((target.pose.pos[1] / pixel_to_meter_y) + img_height_px / 2)
                    cv2.circle(img_viz, (x_viz, y_viz), 1, (255, 0, 0), -1)
                design_image.image = img_viz
                log.debug(f"🧮 Calculating standoff and target positions...")
                ik_target_positions = transform_targets(
                    design_pose,
                    # concatenate standoff and target positions
                    jnp.concatenate([
                        jnp.array([current_batch.center_pose.pos]),
                        jnp.array([target.pose.pos for target in current_batch.pixel_targets])
                    ]),
                    jnp.concatenate([
                        jnp.array([config.standoff_offset_m]),
                        jnp.array([config.poke_offset_m] * len(current_batch.pixel_targets))
                    ])
                )
                log.debug(f"🎯 Setting IK target to standoff position...")
                ik_target_l.position = ik_target_positions[0]
                state_handle.value = "STANDOFF"
            elif state_handle.value == "STANDOFF":
                log.info(f"🔢 Current target: {target_index.value}")
                log.debug(f"🎯 Setting IK target to target position...")
                ik_target_l.position = ik_target_positions[target_index.value]
                state_handle.value = "POKE"
            elif state_handle.value == "POKE": # robot has already performed poke
                target_index.value += 1
                if target_index.value >= len(current_batch.pixel_targets):
                    log.debug(f" Completed all targets in batch, moving to next batch...")
                    batch_index.value += 1
                    state_handle.value = "WORK"
                else:
                    log.debug(f"🎯 Setting IK target to standoff position...")
                    ik_target_l.position = ik_target_positions[0]
                    state_handle.value = "STANDOFF"
            elif state_handle.value == "PAUSED":
                log.debug(" Paused")
                pass

            if config.enable_realsense:
                log.debug("📷 Updating point clouds...")
                realsense_start_time = time.time()
                positions_a, colors_a = realsense_a.get_points()
                positions_b, colors_b = realsense_b.get_points()
                camera_link_idx_a = robot.links.names.index(config.realsense_a.link_name)
                joint_array = np.concatenate([joint_pos_current.left, joint_pos_current.right])
                camera_pose_a = robot.forward_kinematics(joint_array)[camera_link_idx_a]
                camera_pose_b = camera_pose_b_static
                camera_transform_a = jaxlie.SE3(camera_pose_a)
                camera_transform_b = jaxlie.SE3(camera_pose_b)
                positions_world_a = camera_transform_a @ positions_a
                positions_world_b = camera_transform_b @ positions_b
                pointcloud_a.points = np.array(positions_world_a)
                pointcloud_a.colors = np.array(colors_a)
                pointcloud_b.points = np.array(positions_world_b)
                pointcloud_b.colors = np.array(colors_b)
                realsense_elapsed_time = time.time() - realsense_start_time
                realsense_duration_ms.value = realsense_elapsed_time * 1000

            if state_handle.value in ["MANUAL", "WORK", "STANDOFF", "POKE"]:
                log.debug("🔍 Solving IK...")
                ik_start_time = time.time()
                log.debug(f"🎯 Left arm IK target - pos: {ik_target_l.position}, wxyz: {ik_target_l.wxyz}")
                log.debug(f"🎯 Right arm IK target - pos: {ik_target_r.position}, wxyz: {ik_target_r.wxyz}")
                target_link_indices = jnp.array([
                    robot.links.names.index(config.target_links_name[0]),
                    robot.links.names.index(config.target_links_name[1])
                ])
                solution = ik(
                    robot=robot,
                    target_link_indices=target_link_indices,
                    target_wxyz=jnp.array([ik_target_l.wxyz, ik_target_r.wxyz]),
                    target_position=jnp.array([ik_target_l.position, ik_target_r.position]),
                    config=config.ik_config,
                )
                joint_pos_current = JointPos(
                    left=np.array(solution[:8]),
                    right=np.array(solution[8:])
                )
                log.debug(f"🎯 Left arm joints: {joint_pos_current.left}")
                log.debug(f"🎯 Right arm joints: {joint_pos_current.right}")
                ik_elapsed_time = time.time() - ik_start_time
                ik_duration_ms.value = ik_elapsed_time * 1000

                log.debug("🤖 Moving robots...")
                robot_move_start_time = time.time()
                move_robot(joint_pos_current, goal_time=config.set_all_position_goal_time_fast)
                robot_move_elapsed_time = time.time() - robot_move_start_time
                move_duration_ms.value = robot_move_elapsed_time * 1000

            log.debug(f"🖼️ Design - pos: {design_tf.position}, wxyz: {design_tf.wxyz}")
            log.debug(f"🔲 Workspace - pos: {workspace_tf.position}, wxyz: {workspace_tf.wxyz}")
            log.debug(f"🎨 Palette - pos: {palette_tf.position}, wxyz: {palette_tf.wxyz}")
            for inkcap_tf in inkcap_tfs:
                log.debug(f"🎨 Inkcap {inkcap_tf.name} - pos: {inkcap_tf.position}, wxyz: {inkcap_tf.wxyz}")
            log.debug(f"💪 Skin - pos: {skin_tf.position}, wxyz: {skin_tf.wxyz}")
            step_elapsed_time = time.time() - step_start_time
            step_duration_ms.value = step_elapsed_time * 1000

    except Exception as e:
        log.error(f"Error: {e}")
        if config.enable_robot:
            log.info("🦾 Getting robot error information ...")
            error_info_l = driver_l.get_error_information()
            error_info_r = driver_r.get_error_information()
            if error_info_l:
                log.error(f"Left arm error: {error_info_l}")
            if error_info_r:
                log.error(f"Right arm error: {error_info_r}")
        raise e
    
    finally:
        log.info("🏁 Shutting down...")
        if config.enable_realsense:
            log.info("📷 Shutting down cameras...")
            realsense_a.pipeline.stop()
            realsense_b.pipeline.stop()
        if config.enable_robot:
            log.info("🦾 Shutting down robots...")
            driver_l, driver_r = init_robot(clear_error=True)
            move_robot(config.joint_pos_sleep)
            log.info("🧹 Idling robot motors")
            driver_l.set_all_modes(trossen_arm.Mode.idle)
            driver_r.set_all_modes(trossen_arm.Mode.idle)

        log.info("🏁 Script complete.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    jax.config.update('jax_platform_name', args.device_name)
    log.info(f"🎮 Using JAX device: {args.device_name}")
    config = TatbotConfig()
    config.enable_robot = args.robot
    config.enable_realsense = args.realsense
    config.debug_mode = args.debug
    main(config=config)