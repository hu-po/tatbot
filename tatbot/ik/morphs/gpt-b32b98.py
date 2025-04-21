import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),         # Shape: (num_envs * 6)
    alpha: float,                              # Step size
    num_envs: int,
    dof: int,
    max_delta: float,                          # Maximum allowed update per joint (clamping)
    delta_q: wp.array(dtype=wp.float32)        # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        sum_val = 0.0
        for j in range(6):  # 6D error
            sum_val += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        update_val = -alpha * sum_val
        # Clamp the update to avoid overly large changes
        if update_val > max_delta:
            update_val = max_delta
        elif update_val < -max_delta:
            update_val = -max_delta
        delta_q[tid] = update_val

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph with Finite Difference Jacobian and Clamped Joint Updates.

    This morph implements an IK solver by numerically approximating the end-effector's 
    Jacobian using the central finite difference method. It then uses a Jacobian Transpose
    approach to iteratively adjust the joint angles. An incremental improvement in this version
    is the inclusion of a maximum step size clamp (max_delta) to prevent unstable large updates,
    and optional logging to monitor the error norm.
    """

    def _update_config(self):
        """
        Updates configuration for the Finite Difference Jacobian Morph.

        Sets the step size, disables gradients for joint angles, and adds custom parameters:
        - eps: Finite difference perturbation size.
        - max_delta: Maximum allowed change in a joint per iteration (for stability).
        - log_info: Flag to enable logging of the error norm.
        """
        self.config.step_size = 0.5                   # Step size for joint angle updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,       # Epsilon for finite difference computation
            "max_delta": 1.0,  # Maximum joint update magnitude for clamping
            "log_info": False  # Set to True to enable per-iteration error logging
        }

    def _step(self):
        """
        Performs one iteration of the IK solver using a Finite Difference Jacobian.

        The Jacobian is approximated by perturbing each joint angle (positive and negative)
        and using the central difference formula. The joint update is then computed via the
        Jacobian Transpose multiplied by the error, scaled by the step size. The update is
        clamped to avoid overly large changes. Optionally, the error norm is logged.
        """
        # Get current end-effector error (flattened 6D error for all environments)
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()
        # Optional logging of error norm
        if self.config.config_extras.get("log_info", False):
            error_norm = np.linalg.norm(ee_error_flat_initial)
            log.info("IK error norm: {:.6f}".format(error_norm))
            
        # Retrieve current joint angles as a numpy array
        q0 = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]

        # Prepare a container for the Jacobian: shape (num_envs, 6, dof)
        num_envs = self.num_envs
        dof = self.dof
        jacobians = np.zeros((num_envs, 6, dof), dtype=np.float32)

        # Compute the Jacobian using central differences
        for e in range(num_envs):
            for i in range(dof):
                # Perturb joint i positively for environment e
                q_plus = q0.copy()
                q_plus[e * dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Perturb joint i negatively for environment e
                q_minus = q0.copy()
                q_minus[e * dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Central difference approximation for column i of the Jacobian
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        # Restore the original joint angles
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer the estimated Jacobian to the GPU
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Initialize the joint update (delta_q) array on the GPU
        delta_q_wp = wp.zeros(num_envs * dof, dtype=wp.float32, device=self.config.device)

        # Launch the kernel to compute delta_q using the Jacobian Transpose method,
        # with clamping via the max_delta parameter.
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=num_envs * dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, num_envs, dof, self.config.config_extras["max_delta"]],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Update joint angles by adding the computed delta_q
        wp.launch(
            add_delta_q_kernel,
            dim=num_envs * dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )