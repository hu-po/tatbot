from dataclasses import dataclass
import logging
import os
import random
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
from PIL import Image
import viser
import viser.transforms as vtf
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
import trossen_arm
from viser.extras import ViserUrdf
import yourdfpy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S', 
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class RobotConfig:
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    ip_address: str = "192.168.1.3"
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """7d joint radians: sleep pose,robot is folded up, motors can be released."""
    set_all_position_goal_time: float = 1.0
    """goal time in s when the goal positions should be reached"""
    set_all_position_blocking: bool = False
    """whether to block until the goal positions are reached"""

@dataclass
class VizConfig:
    image_path: str = "engmfyh5p9rma0cpz319px91gg.png"
    """Path to the input PNG image."""
    skin_center_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of the image on the skin patch in world coordinates (meters)."""
    skin_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Normal vector of the skin surface (pointing outwards from the surface)."""
    skin_width_m: float = 0.09
    """Width of the area on the skin where the image will be projected (meters)."""
    skin_height_m: float = 0.12
    """Height of the area on the skin where the image will be projected (meters)."""
    image_threshold: int = 127
    """[0, 255] threshold for B/W image. Pixels <= threshold are targets."""
    max_draw_pixels: int = 0
    """Maximum number of target pixels to process. If 0 or less, process all."""
    invert_image: bool = False
    """If True, pixels > threshold become targets (e.g., for white lines on black bg)."""
    show_skin_plane: bool = True
    """Whether to visualize the skin plane itself."""
    skin_plane_thickness: float = 0.001
    """Thickness of the visualized skin plane box (meters)."""
    skin_plane_color: Tuple[int, int, int] = (220, 180, 150)
    """RGB color for the skin plane, e.g., a skin-like tone."""
    splat_length: float = 0.0000001
    """Length of the splat along its main oriented axis (meters)"""
    splat_thickness: float = 0.0000001
    """Thickness of the splat for its other two axes (meters)"""
    splat_color: Tuple[int, int, int] = (0, 0, 0)
    """Color for the splats"""
    image_width: int = 256
    """Width to resize the input image to before processing (pixels)."""
    image_height: int = 256
    """Height to resize the input image to before processing (pixels)."""

@dataclass
class PixelTarget:
    position: np.ndarray
    orientation: vtf.SO3

class ProcessedImageData(NamedTuple):
    targets: List[PixelTarget]
    skin_frame_origin: np.ndarray
    skin_frame_T1: np.ndarray
    skin_frame_T2: np.ndarray
    skin_frame_N: np.ndarray

@dataclass
class State:
    visuals: Dict[str, Any]
    processed_data: ProcessedImageData
    skin_control: Any = None

@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
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
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: np.ndarray. Target orientation.
        target_position: np.ndarray. Target position.

    Returns:
        cfg: np.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return np.array(cfg)

def robot(config: RobotConfig):
    """Main function for basic IK."""

    # Load URDF from file
    urdf_path : str = "/home/oop/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf"
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(urdf_path)
    target_link_name : str = "ee_gripper_link"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.30, 0.0, 0.30), wxyz=(0, 0, 0, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    # Initialize Robot
    driver = trossen_arm.TrossenArmDriver()
    log.info("🚀 Initializing driver...")
    driver.configure(
        config.arm_model,
        config.end_effector_model,
        config.ip_address,
        True # whether to clear the error state of the robot
    )
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(trossen_arm.VectorDouble(list(config.joint_pos_sleep)))
    try:
        while True:
            # Solve IK.
            start_time = time.time()
            solution : np.ndarray = solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=np.array(ik_target.position),
                target_wxyz=np.array(ik_target.wxyz),
            )

            # Set robot to solution
            driver.set_all_positions(
                trossen_arm.VectorDouble(solution[:-1]),
                goal_time=config.set_all_position_goal_time,
                blocking=config.set_all_position_blocking,
            )

            # Update timing handle.
            elapsed_time = time.time() - start_time
            timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

            # Update visualizer.
            urdf_vis.update_cfg(solution)
    except Exception as e:
        log.error(f"❌ Error: {e}")
    
    finally:
        driver.cleanup()
        driver.configure(
            config.arm_model,
            config.end_effector_model,
            config.ip_address,
            True # whether to clear the error state of the robot
        )
        log.info("😴 Returning to sleep pose.")
        driver.set_all_modes(trossen_arm.Mode.position)
        driver.set_all_positions(trossen_arm.VectorDouble(list(config.joint_pos_sleep)))
        log.info("🧹 Idling motors")
        driver.set_all_modes(trossen_arm.Mode.idle)
        log.info("🏁 Script complete.")

def process_skin_and_targets(config: VizConfig, target_rows, target_cols, skin_center_position, T1, T2, N):
    if target_rows.size == 0:
        return ProcessedImageData([], np.array(skin_center_position), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]))
    h_px, w_px = target_rows.max() + 1, target_cols.max() + 1
    norm_u = (target_cols / w_px) - 0.5
    norm_v = 0.5 - (target_rows / h_px)
    target_coords_normalized = list(zip(norm_u.tolist(), norm_v.tolist()))
    if config.max_draw_pixels > 0 and len(target_coords_normalized) > config.max_draw_pixels:
        target_coords_normalized = random.sample(target_coords_normalized, config.max_draw_pixels)
    pixel_targets_list: List[PixelTarget] = []
    skin_origin_pos = np.array(skin_center_position)
    for u, v in target_coords_normalized:
        pos_offset = u * config.skin_width_m * T1 + v * config.skin_height_m * T2
        pixel_world_pos = skin_origin_pos + pos_offset
        rotation_matrix = np.stack([T1, T2, N], axis=1)
        so3_orientation = vtf.SO3.from_matrix(rotation_matrix)
        pixel_targets_list.append(PixelTarget(position=pixel_world_pos, orientation=so3_orientation))
    return ProcessedImageData(
        targets=pixel_targets_list,
        skin_frame_origin=skin_origin_pos,
        skin_frame_T1=T1,
        skin_frame_T2=T2,
        skin_frame_N=N
    )

def update_scene(server, config, state):
    for handle in state.visuals.values():
        try:
            handle.remove()
        except Exception:
            pass
    state.visuals.clear()
    processed_data = state.processed_data
    targets = processed_data.targets
    skin_origin = processed_data.skin_frame_origin
    T1, T2, N = processed_data.skin_frame_T1, processed_data.skin_frame_T2, processed_data.skin_frame_N
    skin_rot_matrix = np.stack([T1, T2, N], axis=1)
    skin_so3 = vtf.SO3.from_matrix(skin_rot_matrix)
    with server.atomic():
        if config.show_skin_plane:
            state.visuals['skin_box'] = server.scene.add_box(
                name="/skin_patch",
                wxyz=skin_so3.wxyz,
                position=skin_origin,
                dimensions=(config.skin_width_m, config.skin_height_m, config.skin_plane_thickness),
                color=config.skin_plane_color
            )
            state.visuals['skin_frame'] = server.scene.add_frame(
                name="/skin_coordinate_frame",
                position=skin_origin,
                wxyz=skin_so3.wxyz,
                axes_length=min(config.skin_width_m, config.skin_height_m) * 0.6,
                axes_radius=config.skin_plane_thickness * 2.5
            )
        if targets:
            num_targets = len(targets)
            positions_np = np.array([target.position for target in targets])
            orientations_np = np.array([target.orientation.wxyz for target in targets])
            colors_np = np.full((num_targets, 3), config.splat_color, dtype=np.uint8)
            scales_np = np.full((num_targets, 3), (config.splat_thickness, config.splat_thickness, config.splat_length), dtype=np.float32)
            covariances_np = np.array([
                target.orientation.as_matrix() @ np.diag([config.splat_thickness, config.splat_thickness, config.splat_length]) @ target.orientation.as_matrix().T
                for target in targets
            ], dtype=np.float32)
            opacities_np = np.full((num_targets, 1), 1.0, dtype=np.float32)
            state.visuals['splats'] = server.scene.add_gaussian_splats(
                name="/pixel_targets/oriented_gaussians",
                centers=positions_np,
                covariances=covariances_np,
                rgbs=colors_np,
                opacities=opacities_np
            )
    return skin_origin, skin_so3

def update_from_gizmo(server, config, state):
    config.skin_center_position = tuple(state.skin_control.position)
    rot_matrix = vtf.SO3(state.skin_control.wxyz).as_matrix()
    T1 = rot_matrix[:, 0]
    T2 = rot_matrix[:, 1]
    N = rot_matrix[:, 2]
    config.skin_normal = tuple(N.tolist())
    state.processed_data = process_skin_and_targets(
        config,
        state.target_rows,
        state.target_cols,
        config.skin_center_position,
        T1, T2, N
    )
    update_scene(server, config, state)

def main(config: VizConfig):
    try:
        img = Image.open(config.image_path)
    except Exception:
        return
    img = img.resize((config.image_width, config.image_height), Image.LANCZOS)
    img = img.convert("L")
    arr = np.array(img)
    h_px, w_px = arr.shape
    if config.invert_image:
        target_mask = arr > config.image_threshold
    else:
        target_mask = arr <= config.image_threshold
    target_rows, target_cols = np.where(target_mask)
    # Initial axes from config
    N0 = np.array(config.skin_normal, dtype=float)
    if np.linalg.norm(N0) < 1e-9:
        N0 = np.array([0.0, 0.0, 1.0])
    else:
        N0 /= np.linalg.norm(N0)
    world_Z = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(N0, world_Z)) > 0.999:
        T1_0 = np.cross(np.array([0.0, 1.0, 0.0]), N0)
        if np.linalg.norm(T1_0) < 1e-5:
            T1_0 = np.cross(np.array([1.0, 0.0, 0.0]), N0)
    else:
        T1_0 = np.cross(world_Z, N0)
    T1_0 /= np.linalg.norm(T1_0)
    T2_0 = np.cross(N0, T1_0)
    T2_0 /= np.linalg.norm(T2_0)
    processed_data = process_skin_and_targets(config, target_rows, target_cols, config.skin_center_position, T1_0, T2_0, N0)
    server = viser.ViserServer()
    state = State(visuals={}, processed_data=processed_data)
    state.target_rows = target_rows
    state.target_cols = target_cols
    skin_origin, skin_so3 = update_scene(server, config, state)
    state.skin_control = server.scene.add_transform_controls(
        name="/skin_patch_control",
        position=skin_origin,
        wxyz=skin_so3.wxyz,
        scale=0.15,
    )
    state.skin_control.on_update(lambda _: update_from_gizmo(server, config, state))
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main(VizConfig())