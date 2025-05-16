import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),       # Shape: (num_envs * 6)
    alpha: float,                            # Step size
    damping: float,                          # Damping factor for numerical stability
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)      # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        # Compute (J^T * error)_i = sum_j J_ji * error_j
        sum_val = 0.0
        for j in range(6):  # 6D error
            sum_val += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        # Apply damping to prevent large updates and improve stability
        delta_q[tid] = -alpha * sum_val / (1.0 + damping)

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
    Improved Inverse Kinematics Morph using Finite Difference Jacobian with Adaptive Damping.

    This morph builds upon the original IK solver by introducing an adaptive damping mechanism.
    The damping factor now adjusts dynamically based on the magnitude of the end-effector error.
    For larger errors, damping is increased to stabilize updates, while for smaller errors,
    damping is reduced to allow finer adjustments. This aims to improve convergence speed
    and stability across a wider range of scenarios.
    """

    def _update_config(self):
        """
        Updates the configuration for the Finite Difference Jacobian Morph with adaptive damping.

        Sets the step size for joint updates, the epsilon value for finite difference
        calculations, and introduces parameters for adaptive damping. Gradients are disabled
        for joint angles as the Jacobian is computed numerically.
        """
        self.config.step_size = 0.6  # Slightly increased step size for faster convergence
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,          # Epsilon value for finite difference calculation
            "base_damping": 0.05, # Base damping factor
            "max_damping": 0.2,   # Maximum damping factor for large errors
            "error_threshold": 0.5 # Error threshold to scale damping
        }

    def _step(self):
        """
        Performs one IK step using the Finite Difference Jacobian method with adaptive damping.

        Approximates the Jacobian matrix J using central difference by perturbing each joint angle.
        Computes the end-effector error magnitude to adjust the damping factor dynamically.
        Updates joint angles using delta_q = -step_size * J^T * error / (1 + damping),
        where damping adapts to the error magnitude for improved stability and convergence.
        """
        # Get initial error and joint angles
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()
        q0 = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]
        base_damping = self.config.config_extras["base_damping"]
        max_damping = self.config.config_extras["max_damping"]
        error_threshold = self.config.config_extras["error_threshold"]

        # Compute error magnitude per environment for adaptive damping
        error_mags = np.zeros(self.num_envs, dtype=np.float32)
        for e in range(self.num_envs):
            error_vec = ee_error_flat_initial[e * 6:(e + 1) * 6]
            error_mags[e] = np.linalg.norm(error_vec)
        
        # Compute adaptive damping based on error magnitude
        avg_error_mag = np.mean(error_mags)
        damping = base_damping + (max_damping - base_damping) * min(avg_error_mag / error_threshold, 1.0)
        log.debug(f"Adaptive damping: {damping:.4f} (error mag: {avg_error_mag:.4f})")

        # Initialize Jacobian on GPU
        jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Compute Jacobian using finite differences
        for e in range(self.num_envs):
            for i in range(self.dof):
                # Perturb positive
                q_plus = q0.copy()
                q_plus[e * self.dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Perturb negative
                q_minus = q0.copy()
                q_minus[e * self.dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Compute column of Jacobian using central difference
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        # Restore original joint angles
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer Jacobian to GPU
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Initialize delta_q on GPU
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint updates using Jacobian transpose method on GPU with adaptive damping
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, damping, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Update joint angles on GPU
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )