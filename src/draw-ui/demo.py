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
from jaxtyping import Array, Float
import numpy as np
import PIL.Image
import pyroki as pk
import trossen_arm
import viser
import viser.transforms as vtf
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

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"]
    wxyz: Float[Array, "4"]

@dataclass
class RobotConfig:
    pose: Pose = Pose(pos=jnp.array([0.0, -0.22, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the design (relative to root frame)."""
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    """Arm model for the robot."""
    ip_address: str = "192.168.1.3"
    """IP address of the robot."""
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    """End effector model for the robot."""
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """7D joint radians for the sleep pose (robot is folded up, motors can be released)."""
    joint_pos_home: tuple[float, ...] = (0.0, 1.05, 0.5, -1.06, 0.0, 0.0, 0.0, 0.0)
    """7D joint radians for the home pose (robot is active, staring down at workspace)."""
    set_all_position_goal_time: float = 1.0
    """Goal time in seconds when the goal positions should be reached."""
    set_all_position_blocking: bool = False
    """Whether to block until the goal positions are reached."""
    clear_error_state: bool = True
    """Whether to clear the error state of the robot."""
    urdf_path: str = "/home/oop/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
    """Local path to the URDF file for the robot."""
    target_link_name: str = "ee_gripper_link"
    """Name of the link to be controlled."""
    gripper_open_width: float = 0.04
    """Width of the gripper when open (meters)."""
    gripper_grip_timeout: float = 1.0
    """Timeout for effort-based gripping (seconds)."""
    gripper_grip_effort: float = -20.0
    """Maximum force for effort-based gripping (newtons)."""
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
    ik_target_pose_l: Pose = Pose(pos=jnp.array([0.2, 0.0, 0.0]), wxyz=jnp.array([0.7, 0.0, 0.7, 0.0]))
    """Initial pose of the grabbable transform IK target for left robot (relative to root frame)."""
    ik_target_pose_r: Pose = Pose(pos=jnp.array([0.2, 0.0, 0.0]), wxyz=jnp.array([0.7, 0.0, 0.7, 0.0]))
    """Initial pose of the grabbable transform IK target for right robot (relative to root frame)."""

@dataclass
class DesignConfig:
    pose: Pose = Pose(pos=jnp.array([0.15, -0.25, 0.08]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the design (relative to workspace origin)."""
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
    pose: Pose = Pose(pos=jnp.array([0.15, -0.25, 0.06]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the tattoo pen (relative to workspace origin)."""
    diameter_m: float = 0.025
    """Diameter of the tattoo pen (meters)."""
    height_m: float = 0.12
    """Height of the tattoo pen (meters)."""
    gripper_grip_width: float = 0.032
    """Width of the gripper before using effort-based gripping (meters)."""
    standoff_depth_m: float = 0.01
    """Depth of the standoff: when the pen is above the pixel target size, but before it begins the stroke (meters)."""
    stroke_depth_m: float = 0.008
    """Length of pen stroke when drawing a pixel (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0)
    """RGB color of the pen."""

@dataclass
class InkCapConfig:
    pose: Pose = Pose(pos=jnp.array([0.1, -0.31, 0.05]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the inkcap (relative to workspace origin)."""
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
    pose: Pose = Pose(pos=jnp.array([0.145, -0.36, 0.04]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the skin (relative to workspace origin)."""
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
    origin: Pose = Pose(pos=jnp.array([0.1, -0.10, -0.1]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Pose of the workspace origin (relative to root)."""
    center_offset: Pose = Pose(pos=jnp.array([0.14, -0.21, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0]))
    """Offset of the workspace center from the origin (relative to workspace origin)."""
    width_m: float = 0.28
    """Width of the workspace (meters)."""
    height_m: float = 0.42
    """Height of the workspace (meters)."""
    thickness: float = 0.001
    """Thickness of the workspace (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0) # black
    """RGB color for the workspace."""

@jdc.pytree_dataclass
class PixelTarget:
    pos: Float[Array, "3"]
    norm: Float[Array, "3"]
    standoff_depth_m: float
    stroke_depth_m: float

@jdc.jit
def ik(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    pos_weight: float,
    ori_weight: float,
    limit_weight: float,
    lambda_initial: float,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=limit_weight,
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
    robot_l_config: RobotConfig,
    robot_r_config: RobotConfig,
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

    # Add sleep position buttons
    with server.gui.add_folder("Robot Control"):
        sleep_left_button = server.gui.add_button("Sleep Left Arm")
        sleep_right_button = server.gui.add_button("Sleep Right Arm")
        use_ik_left = server.gui.add_checkbox("Use IK Left Arm", initial_value=True)
        use_ik_right = server.gui.add_checkbox("Use IK Right Arm", initial_value=True)

        @sleep_left_button.on_click
        def _(_):
            log.debug("😴 Moving left robot to sleep pose...")
            urdf_vis_l.update_cfg(robot_joint_pos_sleep_l)
            if session_config.enable_robot:
                driver_l.set_all_positions(
                    trossen_arm.VectorDouble(list(robot_l_config.joint_pos_sleep)),
                    goal_time=robot_l_config.set_all_position_goal_time,
                    blocking=True,
                )

        @sleep_right_button.on_click
        def _(_):
            log.debug("😴 Moving right robot to sleep pose...")
            urdf_vis_r.update_cfg(robot_joint_pos_sleep_r)
            if session_config.enable_robot:
                driver_r.set_all_positions(
                    trossen_arm.VectorDouble(list(robot_r_config.joint_pos_sleep)),
                    goal_time=robot_r_config.set_all_position_goal_time,
                    blocking=True,
                )

    log.info("🦾 Adding robots...")
    urdf_l : yourdfpy.URDF = yourdfpy.URDF.load(robot_l_config.urdf_path)
    urdf_r : yourdfpy.URDF = yourdfpy.URDF.load(robot_r_config.urdf_path)
    robot_l: pk.Robot = pk.Robot.from_urdf(urdf_l)
    robot_r: pk.Robot = pk.Robot.from_urdf(urdf_r)
    robot_joint_pos_sleep_l = np.array(list(robot_l_config.joint_pos_sleep))
    robot_joint_pos_sleep_r = np.array(list(robot_r_config.joint_pos_sleep))
    server.scene.add_frame(
        "/robot_l",
        position=robot_l_config.pose.pos,
        wxyz=robot_l_config.pose.wxyz,
        show_axes=False if log.getEffectiveLevel() > logging.DEBUG else True,
    )
    server.scene.add_frame(
        "/robot_r",
        position=robot_r_config.pose.pos,
        wxyz=robot_r_config.pose.wxyz,
        show_axes=False if log.getEffectiveLevel() > logging.DEBUG else True,
    )
    urdf_vis_l = ViserUrdf(server, urdf_l, root_node_name="/robot_l/base")
    urdf_vis_r = ViserUrdf(server, urdf_r, root_node_name="/robot_r/base")
    urdf_vis_l.update_cfg(robot_joint_pos_sleep_l)
    urdf_vis_r.update_cfg(robot_joint_pos_sleep_r)

    if session_config.use_ik_target:
        ik_target_l = server.scene.add_transform_controls(
            "/robot_l/ik_target",
            position=session_config.ik_target_pose_l.pos,
            wxyz=session_config.ik_target_pose_l.wxyz,
            scale=0.1,
        )
        ik_target_r = server.scene.add_transform_controls(
            "/robot_r/ik_target",
            position=session_config.ik_target_pose_r.pos,
            wxyz=session_config.ik_target_pose_r.wxyz,
            scale=0.1,
        )

    log.info("🔲 Adding workspace...")
    workspace_transform = server.scene.add_frame(
        "/workspace",
        position=workspace_config.origin.pos,
        wxyz=workspace_config.origin.wxyz,
        show_axes=False if log.getEffectiveLevel() > logging.DEBUG else True,
    )
    workspace_viz = server.scene.add_box(
        name="/workspace/mat",
        position=workspace_config.center_offset.pos,
        wxyz=workspace_config.center_offset.wxyz,
        dimensions=(workspace_config.width_m, workspace_config.height_m, workspace_config.thickness),
        color=workspace_config.color
    )

    log.info("🎨 Adding inkcap...")
    inkcap_viz = server.scene.add_box(
        name="/workspace/inkcap",
        position=inkcap_config.pose.pos,
        wxyz=inkcap_config.pose.wxyz,
        dimensions=(inkcap_config.diameter_m, inkcap_config.diameter_m, inkcap_config.height_m),
        color=inkcap_config.color
    )

    log.info("🖋️ Adding pen...")
    pen_viz = server.scene.add_box(
        name="/workspace/pen",
        position=pen_config.pose.pos,
        wxyz=pen_config.pose.wxyz,
        dimensions=(pen_config.diameter_m, pen_config.diameter_m, pen_config.height_m),
        color=pen_config.color
    )

    log.info("💪 Adding skin...")
    skin_viz = server.scene.add_box(
        name="/workspace/skin",
        position=skin_config.pose.pos,
        wxyz=skin_config.pose.wxyz,
        dimensions=(skin_config.width_m, skin_config.height_m, skin_config.thickness),
        color=skin_config.color
    )

    log.info("🖼️ Loading design...")
    img_pil = PIL.Image.open(design_config.image_path)
    img_pil = img_pil.resize((design_config.image_width_px, design_config.image_height_px), PIL.Image.LANCZOS)
    img_pil = img_pil.convert("L")
    img_np = np.array(img_pil)
    img_viz = server.gui.add_image(
        image=img_np,
        label=design_config.image_path,
        format="png",
        order="rgb",
        visible=True,
    )
    thresholded_pixels = img_np <= design_config.image_threshold
    pixel_targets: List[PixelTarget] = []
    pixel_to_meter_x = design_config.image_width_m / design_config.image_width_px
    pixel_to_meter_y = design_config.image_height_m / design_config.image_height_px
    for y in range(design_config.image_height_px):
        for x in range(design_config.image_width_px):
            if thresholded_pixels[y, x]:
                meter_x = (x - design_config.image_width_px/2) * pixel_to_meter_x
                meter_y = (y - design_config.image_height_px/2) * pixel_to_meter_y
                pixel_target = PixelTarget(
                    pos=jnp.array([meter_x, meter_y, 0.0]),
                    norm=jnp.array([0.0, 0.0, 1.0]),
                    standoff_depth_m=pen_config.standoff_depth_m,
                    stroke_depth_m=pen_config.stroke_depth_m,
                )
                pixel_targets.append(pixel_target)
    log.info(f"🎨 Created {len(pixel_targets)} pixel targets.")
    positions = np.array([pt.pos for pt in pixel_targets])
    design_frame = server.scene.add_transform_controls(
        name="/design",
        position=design_config.pose.pos,
        wxyz=design_config.pose.wxyz,
        scale=0.1,
    )
    server.scene.add_point_cloud(
        name="/design/pixel_targets",
        points=positions,
        colors=np.array([design_config.point_color] * len(positions)),
        point_size=design_config.point_size,
        point_shape=design_config.point_shape,
    )

    if session_config.enable_robot:
        log.info("🤖 Initializing robot drivers...")
        driver_l = trossen_arm.TrossenArmDriver()
        driver_r = trossen_arm.TrossenArmDriver()
        driver_l.configure(
            robot_l_config.arm_model,
            robot_l_config.end_effector_model,
            robot_l_config.ip_address,
            robot_l_config.clear_error_state
        )
        driver_l.set_all_modes(trossen_arm.Mode.position)
        driver_r.configure(
            robot_r_config.arm_model,
            robot_r_config.end_effector_model,
            robot_r_config.ip_address,
            robot_r_config.clear_error_state
        )
        driver_r.set_all_modes(trossen_arm.Mode.position)

        log.info("😴 Going to sleep pose at startup.")
        driver_l.set_all_positions(
            trossen_arm.VectorDouble(list(robot_l_config.joint_pos_sleep)),
            goal_time=robot_l_config.set_all_position_goal_time,
            blocking=True,
        )
        driver_r.set_all_positions(
            trossen_arm.VectorDouble(list(robot_r_config.joint_pos_sleep)),
            goal_time=robot_r_config.set_all_position_goal_time,
            blocking=True,
        )
        urdf_vis_l.update_cfg(robot_joint_pos_sleep_l)
        urdf_vis_r.update_cfg(robot_joint_pos_sleep_r)

    try:
        while True:
            step_start_time = time.time()
            
            log.debug("🔍 Solving IK...")
            ik_start_time = time.time()
            if use_ik_left.value:
                solution_l : jax.Array = ik(
                    robot=robot_l,
                    target_link_index=jnp.array(robot_l.links.names.index(robot_l_config.target_link_name)),
                    target_wxyz=jnp.array(ik_target_l.wxyz),
                    target_position=jnp.array(ik_target_l.position),
                    pos_weight=robot_l_config.ik_pos_weight,
                    ori_weight=robot_l_config.ik_ori_weight,
                    limit_weight=robot_l_config.ik_limit_weight,
                    lambda_initial=robot_l_config.ik_lambda_initial,
                )
            else:
                solution_l = jnp.array(urdf_vis_l.cfg)

            if use_ik_right.value:
                solution_r : jax.Array = ik(
                    robot=robot_r,
                    target_link_index=jnp.array(robot_r.links.names.index(robot_r_config.target_link_name)),
                    target_wxyz=jnp.array(ik_target_r.wxyz),
                    target_position=jnp.array(ik_target_r.position),
                    pos_weight=robot_r_config.ik_pos_weight,
                    ori_weight=robot_r_config.ik_ori_weight,
                    limit_weight=robot_r_config.ik_limit_weight,
                    lambda_initial=robot_r_config.ik_lambda_initial,
                )
            else:
                solution_r = jnp.array(urdf_vis_r.cfg)
            ik_elapsed_time = time.time() - ik_start_time

            if session_config.enable_robot:
                log.debug("🤖 Moving robots...")
                robot_move_start_time = time.time()
                driver_l.set_all_positions(
                    trossen_arm.VectorDouble(np.array(solution_l[:-1]).tolist()),
                    goal_time=robot_l_config.set_all_position_goal_time,
                    blocking=robot_l_config.set_all_position_blocking,
                )
                driver_r.set_all_positions(
                    trossen_arm.VectorDouble(np.array(solution_r[:-1]).tolist()),
                    goal_time=robot_r_config.set_all_position_goal_time,
                    blocking=robot_r_config.set_all_position_blocking,
                )
                robot_move_elapsed_time = time.time() - robot_move_start_time

            render_start_time = time.time()
            log.debug("🎬 Rendering scene...")
            log.debug(f"🎯 IK Target L - pos: {ik_target_l.position}, wxyz: {ik_target_l.wxyz}")
            log.debug(f"🎯 IK Target R - pos: {ik_target_r.position}, wxyz: {ik_target_r.wxyz}")
            log.debug(f"🔲 Workspace - pos: {workspace_transform.position}, wxyz: {workspace_transform.wxyz}")
            log.debug(f"🖼️ Design - pos: {design_frame.position}, wxyz: {design_frame.wxyz}")
            log.debug(f"🎨 Inkcap - pos: {inkcap_viz.position}, wxyz: {inkcap_viz.wxyz}")
            log.debug(f"🖋️ Pen - pos: {pen_viz.position}, wxyz: {pen_viz.wxyz}")
            log.debug(f"💪 Skin - pos: {skin_viz.position}, wxyz: {skin_viz.wxyz}")
            urdf_vis_l.update_cfg(np.array(solution_l))
            urdf_vis_r.update_cfg(np.array(solution_r))
            render_elapsed_time = time.time() - render_start_time
            render_timing_handle.value = render_elapsed_time * 1000
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
                robot_l_config.arm_model,
                robot_l_config.end_effector_model,
                robot_l_config.ip_address,
                robot_l_config.clear_error_state
            )
            driver_l.set_all_modes(trossen_arm.Mode.position)
            log.info("😴 Returning left robot to sleep pose.")
            driver_l.set_all_positions(trossen_arm.VectorDouble(list(robot_l_config.joint_pos_sleep)))
            log.info("🧹 Idling left robot motors")
            driver_l.set_all_modes(trossen_arm.Mode.idle)
            driver_r.cleanup()
            driver_r.configure(
                robot_r_config.arm_model,
                robot_r_config.end_effector_model,
                robot_r_config.ip_address,
                robot_r_config.clear_error_state
            )
            driver_r.set_all_modes(trossen_arm.Mode.position)
            log.info("😴 Returning right robot to sleep pose.")
            driver_r.set_all_positions(trossen_arm.VectorDouble(list(robot_r_config.joint_pos_sleep)))
            log.info("🧹 Idling right robot motors")
            driver_r.set_all_modes(trossen_arm.Mode.idle)
            
        log.info("🏁 Script complete.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    if args.debug:
        log.info("🐛 Debug mode enabled.")
        log.setLevel(logging.DEBUG)
    # TODO: cli args to override values
    # TODO: wrap entire script for hyperparameter tuning
    main(
        robot_l_config=RobotConfig(
            pose=Pose(pos=jnp.array([-0.15, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 0.0])),
            ip_address="192.168.1.3",
            end_effector_model=trossen_arm.StandardEndEffector.wxai_v0_base,
            joint_pos_sleep=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_pos_home=(0.0, 1.05, 0.5, -1.06, 0.0, 0.0, 0.0, 0.0),
        ),
        robot_r_config=RobotConfig(), # right arm uses the default config values
        design_config=DesignConfig(),
        session_config=SessionConfig(),
        workspace_config=WorkspaceConfig(),
        inkcap_config=InkCapConfig(),
        pen_config=TattooPenConfig(),
        skin_config=SkinConfig(),
    )