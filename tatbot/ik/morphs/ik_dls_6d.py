import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def assign_jacobian_slice_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    q_grad: wp.array(dtype=wp.float32),     # Shape: (num_envs * dof)
    dim_idx: int,                           # Which dimension (0-5)
    num_envs: int,
    dof: int
):
    tid = wp.tid()  # Thread per environment
    env_idx = tid // dof
    dof_idx = tid % dof
    if env_idx < num_envs:
        jacobians[env_idx, dim_idx, dof_idx] = q_grad[tid]

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    out_joint_q[tid] = joint_q[tid] - delta_q[tid]

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Damped Least Squares (DLS) Jacobian.

    This morph implements the Damped Least Squares IK method. It computes the
    full 6D Geometric Jacobian (J) using Warp's autodifferentiation (`wp.Tape`)
    similar to ik_geom_6d.

    Instead of the simple transpose, it uses the DLS update rule:
      delta_q = step_size * J^T * (J * J^T + lambda^2 * I)^(-1) * error_6d
    where lambda is a damping factor.

    This method is generally more robust near singularities than the pseudoinverse
    or simple transpose methods.

    Note: The matrix inversion/solve step `(J*J^T + lambda^2*I)^(-1) * error`
          is currently performed using NumPy on the CPU per step, which may
          impact performance due to GPU-CPU data transfers.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the DLS Jacobian Morph.

        Sets the step size, enables gradients for autodiff, and adds the
        DLS damping factor lambda to config_extras.
        """
        self.config.step_size = 1.0  # Step size (learning rate) for joint angle updates
        self.config.joint_q_requires_grad = True
        self.config.config_extras = {
            "lambda": 0.1,  # Damping factor lambda
        }

    def _step(self):
        """
        Performs one IK step using the Damped Least Squares method.

        1. Computes the 6D pose error using compute_ee_error.
        2. Computes the 6D Jacobian using wp.Tape and GPU kernels.
        3. Transfers J and error to CPU for DLS solve.
        4. Calculates the DLS update using NumPy linear algebra:
           - Computes JJT = J @ J.T
           - Forms the damped matrix A = JJT + lambda^2 * I
           - Solves the system A * x = e for x
           - Computes delta_q = step_size * J.T @ x
        5. Updates the joint angles on GPU.
        """
        lambda_val = self.config.config_extras["lambda"]
        lambda_sq = lambda_val * lambda_val
        step_size = self.config.step_size

        # Initialize Jacobian on GPU
        jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
        tape = wp.Tape()

        # Record computation graph for EE error
        with tape:
            ee_error_flat_wp = self.compute_ee_error()

        # Compute Jacobian rows via backpropagation
        for o in range(6):  # Iterate through 6 error dimensions
            # Create a gradient vector with 1.0 for the current dimension
            select_gradient = np.zeros(6, dtype=np.float32)
            select_gradient[o] = 1.0
            e_grad = wp.array(np.tile(select_gradient, self.num_envs), dtype=wp.float32, device=self.config.device)

            # Backpropagate
            tape.backward(grads={ee_error_flat_wp: e_grad})
            q_grad = tape.gradients[self.model.joint_q]

            if q_grad is not None:
                # Assign this row to the Jacobian matrix on GPU
                wp.launch(
                    assign_jacobian_slice_kernel,
                    dim=self.num_envs * self.dof,
                    inputs=[jacobians_wp, q_grad, o, self.num_envs, self.dof],
                    device=self.config.device
                )
            else:
                log.warning(f"No gradients found for dimension {o}")

            # Reset gradients for next dimension
            tape.zero()

        # Transfer data to CPU for DLS solve
        jacobians_np = jacobians_wp.numpy()  # Shape (num_envs, 6, dof)
        error_np = ee_error_flat_wp.numpy().reshape(self.num_envs, 6, 1)  # Shape (num_envs, 6, 1)

        # Perform DLS solve on CPU
        delta_q_np = np.zeros((self.num_envs, self.dof, 1), dtype=np.float32)
        identity_6 = np.identity(6, dtype=np.float32)

        # Process each environment individually
        for e in range(self.num_envs):
            J = jacobians_np[e]  # Shape (6, dof)
            J_T = J.T           # Shape (dof, 6)
            err = error_np[e]   # Shape (6, 1)

            JJT = J @ J_T                             # Shape (6, 6)
            A = JJT + lambda_sq * identity_6          # Shape (6, 6)

            try:
                # Solve (JJT + lambda^2*I) * x = error
                solve_result = np.linalg.solve(A, err)  # Shape (6, 1)
                # Compute delta_q = step_size * J^T * solve_result
                delta_q_np[e] = step_size * (J_T @ solve_result)  # Shape (dof, 1)
            except np.linalg.LinAlgError:
                log.warning(f"DLS solve failed for environment {e}, skipping update")
                continue

        # Transfer delta_q back to GPU and update joint angles
        delta_q_wp = wp.array(delta_q_np.flatten(), dtype=wp.float32, device=self.config.device)

        # Update joint angles on GPU
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )