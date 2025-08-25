import time
from typing import Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import pyroki as pk
from jaxtyping import Array, Float, Int

from tatbot.bot.urdf import get_link_indices, load_robot
from tatbot.utils.log import get_logger

log = get_logger("gen.ik", "🧮")


@jdc.pytree_dataclass
class IKConfig:
    pos_weight: float = 32.0
    """Weight for the position part of the IK cost function."""
    ori_weight: float = 1.0
    """Weight for the orientation part of the IK cost function."""
    rest_weight: float = 0.1
    """Weight for the rest pose cost function."""
    limit_weight: float = 32.0
    """Weight for the limit cost function."""
    lambda_initial: float = 1.0
    """Initial lambda value for the IK trust region solver."""
    max_iterations: int = 32
    """Maximum iterations for IK solver."""

def _ik(
    robot: pk.Robot,
    target_link_indices: Int[Array, "n"],  # n=2 for bimanual
    target_wxyz: Float[Array, "n 4"],
    target_position: Float[Array, "n 3"],
    rest_pose: Float[Array, "14"],
    config: IKConfig,
) -> Float[Array, "14"]:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position),
            target_link_indices,
            pos_weight=config.pos_weight,
            ori_weight=config.ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=config.limit_weight,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose,
            weight=config.rest_weight,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(rest_pose)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=config.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=config.max_iterations, early_termination=False),
        )
    )
    return sol[joint_var].astype(jnp.float32)


# Create JIT-compiled version with static config
_ik_jit = jax.jit(_ik, static_argnums=(5,))

def ik(
    robot: pk.Robot,
    target_link_indices: Int[Array, "n"],  # n=2 for bimanual
    target_wxyz: Float[Array, "n 4"],
    target_position: Float[Array, "n 3"],
    rest_pose: Float[Array, "14"],
    config: Optional[IKConfig] = None,
) -> Float[Array, "14"]:
    if config is None:
        config = IKConfig()
    
    # Ensure inputs are JAX arrays with proper types
    target_link_indices = jnp.array(target_link_indices, dtype=jnp.int32)
    target_wxyz = jnp.array(target_wxyz, dtype=jnp.float32)
    target_position = jnp.array(target_position, dtype=jnp.float32)
    rest_pose = jnp.array(rest_pose, dtype=jnp.float32)
    
    log.debug(f"performing ik on {target_link_indices.shape[0]} targets")
    start_time = time.time()
    
    solution = _ik_jit(
        robot,
        target_link_indices,
        target_wxyz,
        target_position,
        rest_pose,
        config,
    )
    
    log.debug(f"ik solution: {solution}")
    log.debug(f"ik time: {time.time() - start_time:.2f}s")
    return solution


def batch_ik(
    target_wxyz: Float[Array, "b n 4"],
    target_pos: Float[Array, "b n 3"],
    joints: Float[Array, "14"],
    urdf_path: str,
    link_names: tuple[str, ...],
    ik_config: Optional[IKConfig] = None,
) -> Float[Array, "b 14"]:
    if ik_config is None:
        ik_config = IKConfig()
    _, robot = load_robot(urdf_path)
    link_indices, _ = get_link_indices(urdf_path, link_names)
    log.debug(f"performing batch ik on batch of size {target_pos.shape[0]}")
    start_time = time.time()
    _ik_vmap = jax.vmap(
        lambda wxyz, pos, joints: ik(robot, link_indices, wxyz, pos, joints, ik_config),
        in_axes=(0, 0, None),
    )
    solutions = _ik_vmap(target_wxyz, target_pos, joints)
    log.debug(f"batch ik time: {time.time() - start_time:.4f}s")
    return solutions.astype(jnp.float32)