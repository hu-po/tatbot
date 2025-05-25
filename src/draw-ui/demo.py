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

@dataclass
class RobotConfig:
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    """Arm model for the robot."""
    ip_address: str = "192.168.1.3"
    """IP address of the robot."""
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    """End effector model for the robot."""
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """7D joint radians for the sleep pose (robot is folded up, motors can be released)."""
    joint_pos_home: tuple[float, ...] = (0.0, 1.05, 0.5, -1.06, 0.0, 0.0, 0.0)
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
    use_ik_target: bool = True
    """Whether to use an IK target for the robot."""
    num_pixels_per_ink_dip: int = 60
    """Number of pixels to draw before dipping the pen in the ink cup again."""

@jdc.pytree_dataclass
class Pose:
    pos: Float[Array, "3"]
    wxyz: Float[Array, "4"]

@dataclass
class DesignConfig:
    pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the design."""
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
    splat_length: float = 0.0000001
    """Length of the splat along its main oriented axis (meters)."""
    splat_thickness: float = 0.0000001
    """Thickness of the splat for its other two axes (meters)."""
    splat_color: Tuple[int, int, int] = (0, 0, 0)
    """Color for the splats (RGB tuple)."""

@dataclass
class TattooPenConfig:
    pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the tattoo pen."""
    diameter_m: float = 0.008
    """Diameter of the tattoo pen (meters)."""
    height_m: float = 0.01
    """Height of the tattoo pen (meters)."""
    gripper_grip_width: float = 0.032
    """Width of the gripper before using effort-based gripping (meters)."""
    pen_height_delta: float = 0.136
    """Distance from pen tip to end effector tip (meters)."""
    pen_stroke_length: float = 0.008
    """Length of pen stroke when drawing a pixel (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0)
    """RGB color of the pen."""
    # TODO: holder as seperate object?
    holder_pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the tattoo pen holder."""
    holder_width_m: float = 0.032
    """Width of the pen holder (meters)."""
    holder_height_m: float = 0.01
    """Height of the pen holder (meters)."""
    holder_color: Tuple[int, int, int] = (0, 0, 0)
    """RGB color of the pen holder."""

@dataclass
class InkCapConfig:
    pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the inkcap."""
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
    pose: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the skin."""
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Normal vector of the skin surface (pointing outwards from the surface)."""
    width_m: float = 0.09
    """Width of the area on the skin where the image will be projected (meters)."""
    height_m: float = 0.12
    """Height of the area on the skin where the image will be projected (meters)."""
    thickness: float = 0.001
    """Thickness of the visualized skin plane box (meters)."""
    color: Tuple[int, int, int] = (220, 180, 150) # pink
    """RGB color for the skin plane (e.g., a skin-like tone)."""

@dataclass
class WorkspaceConfig:
    origin: Pose = Pose(pos=jnp.array([0.0, 0.0, 0.0]), wxyz=jnp.array([0.0, 0.0, 0.0, 1.0]))
    """Pose of the workspace origin."""
    width_m: float = 0.42
    """Width of the workspace (meters)."""
    height_m: float = 0.28
    """Height of the workspace (meters)."""
    thickness: float = 0.001
    """Thickness of the workspace (meters)."""
    color: Tuple[int, int, int] = (0, 0, 0) # black
    """RGB color for the workspace."""

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

    log.info("🔲 Adding workspace...")
    workspace_viz = server.scene.add_box(
        name="/workspace",
        position=workspace_config.origin.pos,
        wxyz=workspace_config.origin.wxyz,
        dimensions=(workspace_config.width_m, workspace_config.height_m, workspace_config.thickness),
        color=workspace_config.color
    )
    
    log.info("🦾 Adding robot...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(robot_config.urdf_path)
    robot: pk.Robot = pk.Robot.from_urdf(urdf)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    if session_config.use_ik_target:
        ik_target = server.scene.add_transform_controls(
            "/ik_target", scale=0.2, position=(0.30, 0.0, 0.30), wxyz=(0, 0, 0, 0)
        )    

    log.info("🎨 Adding inkcap...")
    inkcap_viz = server.scene.add_box(
        name="/inkcap",
        position=inkcap_config.pose.pos,
        wxyz=inkcap_config.pose.wxyz,
        dimensions=(inkcap_config.diameter_m, inkcap_config.diameter_m, inkcap_config.height_m),
        color=inkcap_config.color
    )
    
    log.info("🖋️ Adding pen...")
    pen_viz = server.scene.add_box(
        name="/pen",
        position=pen_config.pose.pos,
        wxyz=pen_config.pose.wxyz,
        dimensions=(pen_config.diameter_m, pen_config.diameter_m, pen_config.height_m),
        color=pen_config.color
    )
    pen_holder_viz = server.scene.add_box(
        name="/pen_holder",
        position=pen_config.holder_pose.pos,
        wxyz=pen_config.holder_pose.wxyz,
        dimensions=(pen_config.holder_width_m, pen_config.holder_width_m, pen_config.holder_height_m),
        color=pen_config.holder_color
    )

    log.info("💪 Adding skin...")
    skin_viz = server.scene.add_box(
        name="/skin",
        position=skin_config.pose.pos,
        wxyz=skin_config.pose.wxyz,
        dimensions=(skin_config.width_m, skin_config.height_m, skin_config.thickness),
        color=skin_config.color
    )

    log.info("🖼️ Loading design...")
    img = PIL.Image.open(design_config.image_path)
    img = img.resize((design_config.image_width_px, design_config.image_height_px), PIL.Image.LANCZOS)
    img = img.convert("L")
    img_viz = server.gui.add_image(
        image=np.array(img),
        label=design_config.image_path,
        format="jpeg",
        jpeg_quality=90,
        order=None,
        visible=True,
    )
    # TODO: REPLACE THIS CODE HERE BELOW
    # arr = jnp.array(np.array(img))
    # h_px, w_px = arr.shape
    # target_mask = arr <= design_config.image_threshold
    # target_rows, target_cols = jnp.where(target_mask)
    # norm_u = (target_cols / w_px) - 0.5
    # norm_v = 0.5 - (target_rows / h_px)
    # target_coords_normalized = list(zip(norm_u.tolist(), norm_v.tolist()))
    # # TODO: superpixel design instead of random sampling
    # if design_config.max_draw_pixels > 0 and len(target_coords_normalized) > design_config.max_draw_pixels:
    #     target_coords_normalized = random.sample(target_coords_normalized, design_config.max_draw_pixels)
    # num_targets = len(targets)
    # positions_np = jnp.array([target.position for target in targets])
    # orientations_np = jnp.array([target.orientation.wxyz for target in targets])
    # colors_np = jnp.full((num_targets, 3), config.splat_color, dtype=jnp.uint8)
    # scales_np = jnp.full((num_targets, 3), (config.splat_thickness, config.splat_thickness, config.splat_length), dtype=jnp.float32)
    # covariances_np = jnp.array([
    #     target.orientation.as_matrix() @ jnp.diag([config.splat_thickness, config.splat_thickness, config.splat_length]) @ target.orientation.as_matrix().T
    #     for target in targets
    # ], dtype=jnp.float32)
    # design_viz = server.scene.add_gaussian_splats(
    #     name="/design",
    #     centers=jnp.array([target.pos for target in design.targets]),
    #     covariances=covariances_np,
    #     rgbs=colors_np,
    #     opacities=opacities_np
    # )

    if session_config.enable_robot:
        log.info("🤖 Initializing robot driver...")
        driver = trossen_arm.TrossenArmDriver()
        driver.configure(
            robot_config.arm_model,
            robot_config.end_effector_model,
            robot_config.ip_address,
            robot_config.clear_error_state
        )
        driver.set_all_modes(trossen_arm.Mode.position)
        log.info("😴 Going to sleep pose at startup.")
        driver.set_all_positions(
            trossen_arm.VectorDouble(list(robot_config.joint_pos_sleep)),
            goal_time=robot_config.set_all_position_goal_time,
            blocking=True,
        )

    try:
        while True:
            step_start_time = time.time()
            
            log.debug("🔍 Solving IK...")
            ik_start_time = time.time()
            solution : jax.Array = ik(
                robot=robot,
                # TODO: probably slow to create these datatypes every step
                target_link_index=jnp.array(robot.links.names.index(robot_config.target_link_name)),
                target_wxyz=jnp.array(ik_target.wxyz),
                target_position=jnp.array(ik_target.position),
                pos_weight=robot_config.ik_pos_weight,
                ori_weight=robot_config.ik_ori_weight,
                limit_weight=robot_config.ik_limit_weight,
                lambda_initial=robot_config.ik_lambda_initial,
            )
            ik_elapsed_time = time.time() - ik_start_time
            if session_config.enable_robot:
                log.debug("🤖 Moving robot...")
                robot_move_start_time = time.time()
                driver.set_all_positions(
                    trossen_arm.VectorDouble(np.array(solution[:-1]).tolist()),
                    goal_time=robot_config.set_all_position_goal_time,
                    blocking=robot_config.set_all_position_blocking,
                )
                robot_move_elapsed_time = time.time() - robot_move_start_time

            render_start_time = time.time()
            log.debug("🎬 Rendering scene...")
            urdf_vis.update_cfg(np.array(solution))
            render_elapsed_time = time.time() - render_start_time

            step_elapsed_time = time.time() - step_start_time
            step_timing_handle.value = step_elapsed_time * 1000
            ik_timing_handle.value = ik_elapsed_time * 1000
            if session_config.enable_robot:
                robot_move_timing_handle.value = robot_move_elapsed_time * 1000
            render_timing_handle.value = render_elapsed_time * 1000

    # except Exception as e:
    #     log.error(f"❌ Error: {e}")
    
    finally:
        if session_config.enable_robot:
            log.info("🦾 Shutting down robot...")
            driver.cleanup()
            driver.configure( # TODO: is this needed? should the driver object be reinitialized?
                robot_config.arm_model,
                robot_config.end_effector_model,
                robot_config.ip_address,
                robot_config.clear_error_state
            )
            driver.set_all_modes(trossen_arm.Mode.position)
            log.info("😴 Returning to sleep pose.")
            driver.set_all_positions(trossen_arm.VectorDouble(list(robot_config.joint_pos_sleep)))
            log.info("🧹 Idling motors")
            driver.set_all_modes(trossen_arm.Mode.idle)
        log.info("🏁 Script complete.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    if args.debug:
        log.info("🐛 Debug mode enabled.")
        log.setLevel(logging.DEBUG)
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