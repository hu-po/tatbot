from dataclasses import dataclass, field
import time
import os
import functools

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jaxtyping import Array, Float, Int
import pyroki as pk
import yourdfpy

from _log import get_logger

log = get_logger('_ik')
log.info(f"🧠 JAX devices: {jax.devices()}")

@dataclass
class IKRobotConfig:
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_link_names: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the ee links in the URDF for left and right ik solving."""
    rest_pose: Float[Array, "16"] = field(default_factory=lambda: jnp.array([
        # left arm
        0.42, 1.01, 0.49, 0.60, 0.0, 0.0, 0.0, 0.0,
        # right arm
        0.58, 0.05, 0.93, 1.56, 0.10, 0.0, 0.04, 0.04]))
    """Rest pose for the robot."""

@jdc.pytree_dataclass
class IKConfig:
    pos_weight: float = 50.0
    """Weight for the position part of the IK cost function."""
    ori_weight: float = 10.0
    """Weight for the orientation part of the IK cost function."""
    rest_weight: float = 1.0
    """Weight for the rest pose cost function."""
    limit_weight: float = 100.0
    """Weight for the limit cost function."""
    lambda_initial: float = 1.0
    """Initial lambda value for the IK trust region solver."""

@jdc.jit
def ik(
    robot: pk.Robot,
    config: IKConfig,
    target_link_indices: Int[Array, "n"], # n=2 for bimanual
    target_wxyz: Float[Array, "n 4"],
    target_position: Float[Array, "n 3"],
    rest_pose: Float[Array, "16"],
) -> Float[Array, "16"]:
    log.debug(f"🧮 performing ik on batch of size {target_wxyz.shape[0]}")
    start_time = time.time()
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
        # pk.costs.rest_cost(
        #     joint_var,
        #     rest_pose,
        #     config.rest_weight,
        # ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky", # TODO: try other solvers
            trust_region=jaxls.TrustRegionConfig(lambda_initial=config.lambda_initial),
        )
    )
    _solution = sol[joint_var]
    log.debug(f"🧮 ik solution: {_solution}")
    log.debug(f"🧮 ik time: {time.time() - start_time:.2f}s")
    return _solution

@functools.lru_cache(maxsize=2)
def load_robot(urdf_path: str, target_links_name: tuple[str, str]) -> tuple[pk.Robot, Int[Array, "2"]]:
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    target_link_indices = jnp.array([
        robot.links.names.index(target_links_name[0]),
        robot.links.names.index(target_links_name[1]),
    ])
    return robot, target_link_indices

def batch_ik(
    target_wxyz: Float[Array, "b n 4"],
    target_pos: Float[Array, "b n 3"],
    ik_config: IKConfig = IKConfig(),
    robot_config: IKRobotConfig = IKRobotConfig(),
) -> Float[Array, "b 16"]:
    log.debug(f"🧮🤖 Loading PyRoKi robot from URDF at {robot_config.urdf_path}...")
    start_time = time.time()
    robot, target_link_indices = load_robot(robot_config.urdf_path, robot_config.target_link_names)
    log.debug(f"🧮 load robot time: {time.time() - start_time:.4f}s")
    log.debug(f"🧮 performing batch ik on batch of size {target_pos.shape[0]}")
    start_time = time.time()
    _ik_vmap = jax.vmap(
        lambda wxyz, pos, rest: ik(robot, ik_config, target_link_indices, wxyz, pos, rest),
        in_axes=(0, 0, None),
    )
    solutions = _ik_vmap(target_wxyz, target_pos, robot_config.rest_pose)
    log.debug(f"🧮 batch ik time: {time.time() - start_time:.4f}s")
    return solutions
