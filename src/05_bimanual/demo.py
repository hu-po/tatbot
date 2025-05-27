# INFO: This file controls a tatbot robotic tattoo machine.
# INFO: This python file requires dependencies in the pyproject.toml file.
# INFO: This file is a python script indended to be run directly with optional cli args.
# INFO: This file will attempt to use a GPU if available.
# INFO: When editing, do not remove any TODOs in this file.
# INFO: When editing, do not add any additional comments to the code.
# INFO: When editing, use log to add minimal but essential debug and info messages.
# INFO: Use emojis tastefully.

from dataclasses import dataclass
import logging
import os
import random
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import jaxtyping
from jaxtyping import Array, Float, Int
import numpy as np
import PIL.Image
import pyroki as pk
import trossen_arm
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
import yourdfpy
import tyro

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclass
class CLIArgs:
    debug: bool = False
    """Enables debug logging."""
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
class RobotConfig:
    urdf_path: str = "/home/oop/tatbot-urdf/tatbot.urdf"
    """Local path to the URDF file for the robot (https://github.com/hu-po/tatbot-urdf)."""
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    """Arm model for the robot."""
    ip_address_l: str = "192.168.1.3"
    """IP address of the left robot arm."""
    ip_address_r: str = "192.168.1.4"
    """IP address of the right robot arm."""
    end_effector_model_l: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    """End effector model for the left robot arm."""
    end_effector_model_r: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    """End effector model for the right robot arm."""
    joint_pos_sleep: JointPos = JointPos(left=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), right=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    """Sleep: robot is folded up, motors can be released (radians)."""
    joint_pos_home: JointPos = JointPos(
        left=jnp.array([-0.03240784, 2.1669774, 1.8036014, -1.2129925, 0.00018064, -0.03240734, 0.022, 0.0]),
        right=jnp.array([-0.06382597, 1.2545787, 0.78800493, -1.0274638, -0.00490101, -0.06363778, 0.022, 0.022])
    )
    """Home: robot is ready to work (radians)."""
    set_all_position_goal_time: float = 1.0
    """Goal time in seconds when the goal positions should be reached."""
    set_all_position_blocking: bool = False
    """Whether to block until the goal positions are reached."""
    clear_error_state: bool = True
    """Whether to clear the error state of the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the links to be controlled."""
    # gripper_open_width: float = 0.04
    # """Width of the gripper when open (meters)."""
    # gripper_grip_timeout: float = 1.0
    # """Timeout for effort-based gripping (seconds)."""
    # gripper_grip_effort: float = -20.0
    # """Maximum force for effort-based gripping (newtons)."""
    ik_pos_weight: float = 50.0
    """Weight for the position part of the IK cost function."""
    ik_ori_weight: float = 10.0
    """Weight for the orientation part of the IK cost function."""
    ik_limit_weight: float = 100.0
    """Weight for the joint limit part of the IK cost function."""
    ik_lambda_initial: float = 1.0
    """Initial lambda value for the IK trust region solver."""

@dataclass
class SessionConfig:
    enable_robot: bool = False
    """Whether to enable the real robot."""
    num_pixels_per_ink_dip: int = 60
    """Number of pixels to draw before dipping the pen in the ink cup again."""
    use_ik_target: bool = True
    """Whether to use an IK target for the robot."""
    ik_target_pose_l: Pose = Pose(pos=jnp.array([0.2, -0.0571740852, 0.0]), wxyz=jnp.array([-0.00277338172, 0.0, 0.994983532, 0.0]))
    """Initial pose of the grabbable transform IK target for left robot (relative to root frame)."""
    ik_target_pose_r: Pose = Pose(pos=jnp.array([0.2568429, -0.30759474, 0.00116006]), wxyz=jnp.array([0.714142855, 0.0, 0.686137207, 0.0]))
    """Initial pose of the grabbable transform IK target for right robot (relative to root frame)."""

@dataclass
class DesignConfig:
    pose: Pose = Pose(pos=jnp.array([0.21333221, 0.00441298, -0.02088978]), wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Pose of the design (relative to root frame)."""
    image_path: str = "/home/oop/tatbot/assets/designs/circle.png"
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
    point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""

@dataclass
class TattooPenConfig:
    standoff_depth_m: float = 0.01
    """Depth of the standoff: when the pen is above the pixel target size, but before it begins the stroke (meters)."""
    stroke_depth_m: float = 0.008
    """Length of pen stroke when drawing a pixel (meters)."""

@dataclass
class InkCapConfig:
    init_pose: Pose = Pose(pos=jnp.array([0.16813426, 0.03403597, -0.01519414]), wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Pose of the inkcap (relative to root frame)."""
    diameter_m: float = 0.018
    """Diameter of the inkcap (meters)."""
    height_m: float = 0.01
    """Height of the inkcap (meters)."""
    dip_depth_m: float = 0.005
    """Depth of the inkcap when dipping (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0) # black
    """RGB color of the ink in the inkcap."""

@dataclass
class SkinConfig:
    init_pose: Pose = Pose(pos=jnp.array([0.21561891, 0.0046067, -0.02167064]), wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Pose of the skin (relative to root frame)."""
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Normal vector of the skin surface (pointing outwards from the surface)."""
    width_m: float = 0.12
    """Width of the area on the skin where the image will be projected (meters)."""
    height_m: float = 0.09
    """Height of the area on the skin where the image will be projected (meters)."""
    thickness: float = 0.001
    """Thickness of the visualized skin plane box (meters)."""
    color: Tuple[int, int, int] = (220, 180, 150) # pink
    """RGB color for the skin plane (e.g., a skin-like tone)."""

@dataclass
class WorkspaceConfig:
    init_pose: Pose = Pose(pos=jnp.array([0.08088932, 0.1035288, -0.05307121]), wxyz=jnp.array([1.0, 0.0, 0.0, 0.0]))
    """Pose of the workspace origin (relative to root frame)."""
    mat_center_offset: Pose = Pose(pos=jnp.array([0.14, -0.21, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Offset of the workspace mat center from the initial pose above"""
    mat_width_m: float = 0.28
    """Width of the workspace mat (meters)."""
    mat_height_m: float = 0.42
    """Height of the workspace mat (meters)."""
    mat_thickness: float = 0.001
    """Thickness of the workspace mat (meters)."""
    mat_color: Tuple[int, int, int] = (0, 0, 0) # black
    """RGB color for the workspace mat."""

@jdc.pytree_dataclass
class PixelTarget:
    pose: Pose

@jdc.jit
def ik(
    robot: pk.Robot,
    target_link_indices: Int[Array, "2"],
    target_wxyz: Float[Array, "2 4"],
    target_position: Float[Array, "2 3"],
    pos_weight: float,
    ori_weight: float,
    limit_weight: float,
    lambda_initial: float,
) -> Float[Array, "16"]:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_indices,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            jnp.array([limit_weight] * robot.joints.num_joints),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky", # TODO: is this the best?
            trust_region=jaxls.TrustRegionConfig(lambda_initial=lambda_initial),
        )
    )
    return sol[joint_var]

def main(
    robot_config: RobotConfig,
    session_config: SessionConfig,
    workspace_config: WorkspaceConfig,
    skin_config: SkinConfig,
    design_config: DesignConfig,
    inkcap_config: InkCapConfig,
    pen_config: TattooPenConfig,
):
    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    ik_timing_handle = server.gui.add_number("ik (ms)", 0.001, disabled=True)
    if session_config.enable_robot:
        robot_move_timing_handle = server.gui.add_number("robot move (ms)", 0.001, disabled=True)
    render_timing_handle = server.gui.add_number("render (ms)", 0.001, disabled=True)
    step_timing_handle = server.gui.add_number("step (ms)", 0.001, disabled=True)

    log.info("🖼️ Loading design...")
    img_pil = PIL.Image.open(design_config.image_path)
    original_width, original_height = img_pil.size
    if original_width > design_config.image_width_px or original_height > design_config.image_height_px:
        img_pil = img_pil.resize((design_config.image_width_px, design_config.image_height_px), PIL.Image.LANCZOS)
    img_pil = img_pil.convert("L")
    img_np = np.array(img_pil)
    img_width_px, img_height_px = img_pil.size
    img_viz = server.gui.add_image(
        image=img_np,
        label=design_config.image_path,
        format="png",
        order="rgb",
        visible=True,
    )
    thresholded_pixels = img_np <= design_config.image_threshold
    pixel_targets: List[Pose] = []
    pixel_to_meter_x = design_config.image_width_m / img_width_px
    pixel_to_meter_y = design_config.image_height_m / img_height_px
    for y in range(img_height_px):
        for x in range(img_width_px):
            if thresholded_pixels[y, x]:
                meter_x = (x - img_width_px/2) * pixel_to_meter_x
                meter_y = (y - img_height_px/2) * pixel_to_meter_y
                pixel_target = Pose(
                    pos=jnp.array([meter_x, meter_y, 0.0]),
                    wxyz=design_config.pose.wxyz
                )
                pixel_targets.append(pixel_target)
    num_targets: int = len(pixel_targets)
    log.info(f"🎨 Created {num_targets} pixel targets.")
    positions = np.array([pt.pos for pt in pixel_targets])
    design_tf = server.scene.add_transform_controls(
        name="/design",
        position=design_config.pose.pos,
        wxyz=design_config.pose.wxyz,
        scale=0.05,
        opacity=0.2,
    )
    server.scene.add_point_cloud(
        name="/design/pixel_targets",
        points=positions,
        colors=np.array([design_config.point_color] * len(positions)),
        point_size=design_config.point_size,
        point_shape=design_config.point_shape,
    )
    with server.gui.add_folder("Session Progress"):
        progress_bar = server.gui.add_progress_bar(0.0)
        target_slider = server.gui.add_slider(
            "Target Index",
            min=0,
            max=num_targets - 1,
            step=1,
            initial_value=0,
        )
    current_target_index: int = 0

    log.info("🔲 Adding workspace...")
    workspace_tf = server.scene.add_transform_controls(
        "/workspace",
        position=workspace_config.init_pose.pos,
        wxyz=workspace_config.init_pose.wxyz,
        scale=0.05,
        opacity=0.2,
    )
    server.scene.add_box(
        name="/workspace/mat",
        position=workspace_config.mat_center_offset.pos,
        wxyz=workspace_config.mat_center_offset.wxyz,
        dimensions=(workspace_config.mat_width_m, workspace_config.mat_height_m, workspace_config.mat_thickness),
        color=workspace_config.mat_color
    )

    log.info("🎨 Adding inkcap...")
    inkcap_tf = server.scene.add_transform_controls(
        "/inkcap",
        position=inkcap_config.init_pose.pos,
        wxyz=inkcap_config.init_pose.wxyz,
        scale=0.05,
        opacity=0.2,
    )
    server.scene.add_box(
        name="/inkcap/box",
        dimensions=(inkcap_config.diameter_m, inkcap_config.diameter_m, inkcap_config.height_m),
        color=inkcap_config.color
    )

    log.info("💪 Adding skin...")
    skin_tf = server.scene.add_transform_controls(
        "/skin",
        position=skin_config.init_pose.pos,
        wxyz=skin_config.init_pose.wxyz,
        scale=0.05,
        opacity=0.2,
    )
    server.scene.add_box(
        name="/skin/box",
        dimensions=(skin_config.width_m, skin_config.height_m, skin_config.thickness),
        color=skin_config.color
    )

    log.info("🦾 Adding robots...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(robot_config.urdf_path)
    robot: pk.Robot = pk.Robot.from_urdf(urdf)
    joint_pos_current: JointPos = robot_config.joint_pos_sleep
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")
    
    def move_robot(joint_pos: JointPos):
        log.debug(f"🤖 Moving robot to: {joint_pos}")
        urdf_vis.update_cfg(np.concatenate([joint_pos.left, joint_pos.right]))
        if session_config.enable_robot:
            driver_l.set_all_positions(
                trossen_arm.VectorDouble(joint_pos.left[:7].tolist()),
                goal_time=robot_config.set_all_position_goal_time,
                blocking=True,
            )
            driver_r.set_all_positions(
                trossen_arm.VectorDouble(joint_pos.right[8:-2].tolist()),
                goal_time=robot_config.set_all_position_goal_time,
                blocking=True,
            )

    with server.gui.add_folder("Robot Control"):
        sleep_button = server.gui.add_button("Sleep")
        use_ik = server.gui.add_checkbox("enable ik", initial_value=True)

        @sleep_button.on_click
        def _(_):
            log.debug("😴 Moving left robot to sleep pose...")
            use_ik.value = False
            move_robot(robot_config.joint_pos_sleep)

    if session_config.use_ik_target:
        ik_target_l = server.scene.add_transform_controls(
            "/ik_target_l",
            position=session_config.ik_target_pose_l.pos,
            wxyz=session_config.ik_target_pose_l.wxyz,
            scale=0.1,
            opacity=0.5,
        )
        ik_target_r = server.scene.add_transform_controls(
            "/ik_target_r",
            position=session_config.ik_target_pose_r.pos,
            wxyz=session_config.ik_target_pose_r.wxyz,
            scale=0.1,
            opacity=0.5,
        )

    try:
        if session_config.enable_robot:
            log.info("🤖 Initializing robot drivers...")
            driver_l = trossen_arm.TrossenArmDriver()
            driver_r = trossen_arm.TrossenArmDriver()
            driver_l.configure(
                robot_config.arm_model,
                robot_config.end_effector_model_l,
                robot_config.ip_address_l,
                robot_config.clear_error_state
            )
            driver_r.configure(
                robot_config.arm_model,
                robot_config.end_effector_model_r,
                robot_config.ip_address_r,
                robot_config.clear_error_state
            )
            driver_l.set_all_modes(trossen_arm.Mode.position)
            driver_r.set_all_modes(trossen_arm.Mode.position)
        
        log.info("🤖 Moving robots to sleep pose...")
        move_robot(robot_config.joint_pos_sleep)
        log.info("🤖 Moving robots to home pose...")
        move_robot(robot_config.joint_pos_home)

        while True:
            step_start_time = time.time()
            
            log.debug(f"🎯 Calculating target {current_target_index}...")
            current_target_index = target_slider.value
            progress_bar.value = float(current_target_index) / (num_targets - 1)
            current_target = pixel_targets[current_target_index]
            design_to_root = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(design_tf.wxyz),
                design_tf.position
            )
            pixel_pos_root = design_to_root @ current_target.pos
            ik_target_l.position = pixel_pos_root

            log.debug("🔍 Solving IK...")
            ik_start_time = time.time()
            if use_ik.value:
                log.debug(f"🎯 Left arm IK target - pos: {ik_target_l.position}, wxyz: {ik_target_l.wxyz}")
                log.debug(f"🎯 Right arm IK target - pos: {ik_target_r.position}, wxyz: {ik_target_r.wxyz}")
                target_link_indices = jnp.array([
                    robot.links.names.index(robot_config.target_links_name[0]),
                    robot.links.names.index(robot_config.target_links_name[1])
                ])
                solution = ik(
                    robot=robot,
                    target_link_indices=target_link_indices,
                    target_wxyz=jnp.array([ik_target_l.wxyz, ik_target_r.wxyz]),
                    target_position=jnp.array([ik_target_l.position, ik_target_r.position]),
                    pos_weight=robot_config.ik_pos_weight,
                    ori_weight=robot_config.ik_ori_weight,
                    limit_weight=robot_config.ik_limit_weight,
                    lambda_initial=robot_config.ik_lambda_initial,
                )
                
                # Split solution into left and right arm joint positions
                joint_pos_current = JointPos(
                    left=np.array(solution[:8]),  # First 8 joints for left arm
                    right=np.array(solution[8:])  # Last 8 joints for right arm
                )
                log.debug(f"🎯 IK solution shape: {solution.shape}")
                log.debug(f"🎯 Left arm joints: {joint_pos_current.left}")
                log.debug(f"🎯 Right arm joints: {joint_pos_current.right}")
            ik_elapsed_time = time.time() - ik_start_time

            log.debug("🤖 Moving robots...")
            robot_move_start_time = time.time()
            move_robot(joint_pos_current)
            robot_move_elapsed_time = time.time() - robot_move_start_time

            log.debug(f"🔲 Workspace - pos: {workspace_tf.position}, wxyz: {workspace_tf.wxyz}")
            log.debug(f"🎨 Inkcap - pos: {inkcap_tf.position}, wxyz: {inkcap_tf.wxyz}")
            log.debug(f"🖼️ Design - pos: {design_tf.position}, wxyz: {design_tf.wxyz}")
            log.debug(f"💪 Skin - pos: {skin_tf.position}, wxyz: {skin_tf.wxyz}")
            ik_timing_handle.value = ik_elapsed_time * 1000
            if session_config.enable_robot:
                robot_move_timing_handle.value = robot_move_elapsed_time * 1000
            step_elapsed_time = time.time() - step_start_time
            step_timing_handle.value = step_elapsed_time * 1000

    # except Exception as e:
    #     log.error(f"❌ Error: {e}")
    
    finally:
        if session_config.enable_robot:
            log.info("🦾 Shutting down robots...")
            driver_l.cleanup()
            driver_l.configure(
                robot_config.arm_model,
                robot_config.end_effector_model_l,
                robot_config.ip_address_l,
                robot_config.clear_error_state
            )
            driver_r.cleanup()
            driver_r.configure(
                robot_config.arm_model,
                robot_config.end_effector_model_r,
                robot_config.ip_address_r,
                robot_config.clear_error_state
            )
            driver_l.set_all_modes(trossen_arm.Mode.position)
            driver_r.set_all_modes(trossen_arm.Mode.position)
            move_robot(robot_config.joint_pos_sleep)
            log.info("🧹 Idling robot motors")
            driver_l.set_all_modes(trossen_arm.Mode.idle)
            driver_r.set_all_modes(trossen_arm.Mode.idle)
            
        log.info("🏁 Script complete.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    if args.debug:
        log.info("🐛 Debug mode enabled.")
        log.setLevel(logging.DEBUG)
    jax.config.update('jax_platform_name', args.device_name)
    log.info(f"🎮 Using JAX device: {args.device_name}")
    # TODO: cli args to override values
    # TODO: wrap entire script for hyperparameter tuning
    main(
        robot_config=RobotConfig(),
        design_config=DesignConfig(),
        session_config=SessionConfig(),
        workspace_config=WorkspaceConfig(),
        inkcap_config=InkCapConfig(),
        pen_config=TattooPenConfig(),
        skin_config=SkinConfig(),
    )