import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),      # Shape: (num_envs * 6)
    alpha: float,                           # Step size
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)     # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof

    if env_idx < num_envs:
        # Compute (J^T * error)_i = sum_j J_ji * error_j
        sum = 0.0
        for j in range(6):  # 6D error
            sum += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        delta_q[tid] = -alpha * sum

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
    Inverse Kinematics Morph using Finite Difference Jacobian (Forward Difference).

    This morph implements an IK solver that approximates the 6D (position + orientation)
    end-effector Jacobian using numerical *forward* differences. It perturbs each joint angle
    slightly in the positive direction, observes the change in the end-effector pose error
    (computed by the base class), and uses this information to estimate the Jacobian matrix.
    The joint angles are then updated iteratively using the Jacobian transpose method
    to minimize the pose error.

    This version uses forward difference instead of central difference to improve
    computational efficiency by reducing the number of required forward kinematics
    evaluations by approximately half.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the Finite Difference Jacobian Morph.

        Sets the step size for joint updates, the epsilon value for finite difference
        calculations, and ensures gradients are disabled for joint angles, as this
        method computes the Jacobian numerically.
        """
        self.config.step_size = 0.5  # Step size for joint angle updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,  # Epsilon value for finite difference calculation
        }

    def _step(self):
        """
        Performs one IK step using the Forward Finite Difference Jacobian method.

        It approximates the Jacobian matrix J by perturbing each joint angle q_i
        by +epsilon, computing the resulting end-effector pose error, and using
        the forward difference formula: J[:, i] = (error(q + eps) - error(q)) / eps.

        Once the Jacobian is approximated, it computes the change in joint angles delta_q
        using the formula: delta_q = -step_size * J^T * error.

        The computed delta_q is then added to the current joint angles.
        """
        # Get initial error and joint angles
        ee_error_flat_wp = self.compute_ee_error() # This computes error(q)
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy() # error(q) as numpy array
        q0 = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]

        # Initialize Jacobian on CPU (will be transferred to GPU later)
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Pre-allocate q_perturbed array for efficiency
        q_perturbed = q0.copy()

        # Compute Jacobian using forward differences
        for e in range(self.num_envs):
            # Get the initial error for this specific environment
            f0_e = ee_error_flat_initial[e * 6:(e + 1) * 6]

            for i in range(self.dof):
                joint_index = e * self.dof + i

                # Restore original joint angle before perturbation
                q_perturbed[joint_index] = q0[joint_index]

                # Perturb positive
                q_perturbed[joint_index] += eps
                self.model.joint_q.assign(wp.array(q_perturbed, dtype=wp.float32, device=self.config.device))
                f_plus_e = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy() # error(q + eps_i)

                # Restore original joint angle after perturbation to avoid accumulation
                q_perturbed[joint_index] = q0[joint_index]

                # Compute column of Jacobian using forward difference
                jacobians[e, :, i] = (f_plus_e - f0_e) / eps

        # Restore original joint angles for all envs before applying update
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer Jacobian to GPU
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Initialize delta_q on GPU
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint updates using Jacobian transpose method on GPU
        # Use the initial error (error(q)) for the update calculation
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, self.num_envs, self.dof],
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