from dataclasses import dataclass
import logging
import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
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
log.setLevel(logging.INFO)

@dataclass
class IKConfig:
    arm_model: trossen_arm.Model = trossen_arm.Model.wxai_v0
    ip_address: str = "192.168.1.3"
    end_effector_model: trossen_arm.StandardEndEffector = trossen_arm.StandardEndEffector.wxai_v0_follower
    joint_pos_sleep: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """7d joint radians: sleep pose,robot is folded up, motors can be released."""

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


def main(config: IKConfig):
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
            driver.set_all_positions(trossen_arm.VectorDouble(solution[:-1]))

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


if __name__ == "__main__":
    main(IKConfig())
