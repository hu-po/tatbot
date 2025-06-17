import time
import os

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

@jdc.jit
def ik(
    robot: pk.Robot,
    target_link_indices: Int[Array, "n"], # n is number of targets (2 for bimanual)
    target_wxyz: Float[Array, "n 4"],
    target_position: Float[Array, "n 3"],
    config: IKConfig,
) -> Float[Array, "n 16"]:
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
            # TODO: limit weights should be higher for finger joints on left arm
            jnp.array([config.limit_weight] * robot.joints.num_joints),
        ),
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


def batch_ik(
    target_link_indices: Int[Array, "n"], # n is number of targets (2 for bimanual)
    target_wxyz: Float[Array, "n 4"],
    target_positions: Float[Array, "b n 3"], # b is batch size
    config: IKConfig,
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf"),
) -> Float[Array, "b n 16"]:
    log.info(f"🧮🤖 Making PyRoKi robot from URDF at {urdf_path}...")
    urdf: yourdfpy.URDF = yourdfpy.URDF.load(urdf_path)
    robot: pk.Robot = pk.Robot.from_urdf(urdf)
    log.info(f"🧮 performing batch ik on batch of size {target_positions.shape[0]}")
    start_time = time.time()
    _ik_vmap = jax.vmap(lambda pos: ik(robot, target_link_indices, target_wxyz, pos, config), in_axes=0)
    solutions = _ik_vmap(target_positions)
    log.info(f"🧮 batch ik time: {time.time() - start_time:.2f}s")
    return solutions
