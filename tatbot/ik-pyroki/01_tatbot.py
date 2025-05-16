"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yourdfpy

import pyroki_snippets as pks


def main():
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
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 0, 1, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution : np.ndarray = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(solution)


if __name__ == "__main__":
    main()
