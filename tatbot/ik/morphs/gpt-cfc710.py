import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),         # Shape: (num_envs * 6)
    alpha: float,                              # Step size
    damping: float,                            # Damping factor for numerical stability
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)        # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        s = 0.0
        for j in range(6):  # 6D error
            s += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        # Damping added to denominator for stability.
        delta_q[tid] = -alpha * s / (1.0 + damping)

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
    Inverse Kinematics Morph using Finite Difference Jacobian with Damping and Adaptive Step Size.

    This morph implements an IK solver that approximates the 6D (position + orientation)
    end-effector Jacobian using numerical finite differences. It perturbs each joint angle
    slightly, observes the change in the end-effector pose error, and uses central differences
    to estimate the Jacobian matrix. Joint angles are then updated iteratively to minimize
    the pose error. An adaptive update scheme is added: if the update increases the error,
    the update is reverted and the step size is reduced, which adds robustness.
    """

    def _update_config(self):
        """
        Updates the configuration for the IK Morph. Sets the step size for joint updates,
        the epsilon value for finite difference calculation, a damping factor for numerical
        stability, and related extra configuration parameters.
        """
        self.config.step_size = 0.5  # Base step size for joint angle updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,         # Epsilon value for finite difference calculation
            "damping": 0.1,      # Damping factor to improve numerical stability
            "min_step_size": 1e-3  # Minimum allowed step size to prevent excessive reduction
        }

    def _step(self):
        """
        Performs one iteration of IK update using a finite-difference approximation of the Jacobian.
        After computing the joint update delta_q as:
            delta_q = -step_size * J^T * error / (1 + damping)
        the update is tentatively applied to the joint angles. The new error is then computed.
        If the update does not lower the average error across environments, the update is rolled back
        and the step size is decreased for stability. This adaptive scheme enhances robustness
        without changing the underlying algorithm.
        """
        # Retrieve parameters and backup current joint angles
        eps = self.config.config_extras["eps"]
        damping = self.config.config_extras["damping"]
        min_step_size = self.config.config_extras["min_step_size"]
        step_size = self.config.step_size

        # Store initial joint values and error
        q0 = self.model.joint_q.numpy().copy()
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()

        # Compute and record the average error norm over environments
        old_error_norms = []
        for e in range(self.num_envs):
            err_vec = ee_error_flat_initial[e*6:(e+1)*6]
            old_error_norms.append(np.linalg.norm(err_vec))
        avg_old_error = np.mean(old_error_norms)

        # Allocate a NumPy Jacobian for the parallel environments
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Compute the Jacobian using central finite differences for each environment and joint.
        for e in range(self.num_envs):
            for i in range(self.dof):
                # Perturb positively
                q_plus = q0.copy()
                q_plus[e * self.dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Perturb negatively
                q_minus = q0.copy()
                q_minus[e * self.dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Compute the central difference for the i-th joint
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        # Restore the joint angles to the original configuration
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer the computed Jacobian to GPU memory
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Allocate GPU array for the joint update delta_q
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Launch GPU kernel to compute delta_q using Jacobian-transpose
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, step_size, damping, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Save current joint angles before applying update
        q_before_update = self.model.joint_q.numpy().copy()

        # Apply computed update to joint angles on GPU
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )
        wp.synchronize()  # Ensure kernel execution is complete

        # Compute the new end-effector error after applying the tentative update
        ee_error_flat_new_wp = self.compute_ee_error()
        ee_error_flat_new = ee_error_flat_new_wp.numpy().copy()

        # Compute the new average error norm across environments
        new_error_norms = []
        for e in range(self.num_envs):
            err_vec = ee_error_flat_new[e*6:(e+1)*6]
            new_error_norms.append(np.linalg.norm(err_vec))
        avg_new_error = np.mean(new_error_norms)

        # Adaptive step size: if the update did not improve the average error, then revert update
        # and reduce the step size to encourage smaller, more stable updates.
        if avg_new_error >= avg_old_error:
            log.info("No improvement detected (old error: %.6f, new error: %.6f). Reverting update and reducing step_size.", 
                     avg_old_error, avg_new_error)
            # Revert joint angles to the pre-update state
            self.model.joint_q.assign(wp.array(q_before_update, dtype=wp.float32, device=self.config.device))
            # Reduce the step size, ensuring it does not drop below a minimum threshold.
            new_step_size = max(step_size * 0.5, min_step_size)
            log.info("Updating step_size from %.6f to %.6f", step_size, new_step_size)
            self.config.step_size = new_step_size
        else:
            log.info("IK step successful (old avg error: %.6f, new avg error: %.6f).", avg_old_error, avg_new_error)