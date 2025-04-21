import warp as wp
import numpy as np
import logging

from tatbot.ik.morph import BaseMorph

log = logging.getLogger(__name__)

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
        delta_q[tid] = -alpha * sum  # Note: negative step for gradient descent

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
    Inverse Kinematics Morph using Autodiff Geometric Jacobian (6D Pose).

    This morph implements an IK solver that uses Warp's automatic differentiation
    feature (`wp.Tape`) to compute the full 6D Geometric Jacobian relating joint
    angle changes to the end-effector's pose (position and orientation) change.

    The Jacobian is computed by backpropagating a unit gradient through the 6D pose
    error calculation (handled by `compute_ee_error` in the base class) for each
    of the 6 error dimensions (3 position, 3 orientation).

    Joint angles are updated iteratively using the Jacobian transpose method:
    delta_q = -step_size * J^T * error_6d.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the 6D Autodiff Jacobian Morph.

        Sets the step size for joint updates and ensures gradients are enabled
        for joint angles, as this method relies on automatic differentiation.
        """
        self.config.step_size = 0.1
        self.config.joint_q_requires_grad = True

    def _step(self):
        """
        Performs one IK step using the 6D Autodiff Geometric Jacobian method.

        Uses wp.Tape to record the computation of the 6D end-effector pose error.
        For each of the 6 error dimensions, backpropagates a unit gradient to compute
        the corresponding row of the Jacobian. Updates joint angles using the
        Jacobian transpose method on the GPU.
        """
        # Initialize Jacobian on GPU
        jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
        tape = wp.Tape()

        # Record computation graph for EE error
        with tape:
            # Get current 6D EE error (includes position and orientation)
            # This tensor MUST be computed within the tape to be differentiable
            ee_error_flat_wp = self.compute_ee_error()

        # Compute Jacobian rows via backpropagation
        for o in range(6):  # Iterate through 6 error dimensions
            # Create a gradient vector with 1.0 for the current dimension
            select_gradient = np.zeros(6, dtype=np.float32)
            select_gradient[o] = 1.0  # Index 0-2=Pos, 3-5=Ori
            e_grad = wp.array(np.tile(select_gradient, self.num_envs), dtype=wp.float32, device=self.config.device)

            # Backpropagate using the error tensor recorded by the tape
            tape.backward(grads={ee_error_flat_wp: e_grad})
            q_grad = tape.gradients[self.model.joint_q]  # Gradient of error[o] w.r.t. joint angles

            # Assign this row to the 6xDOF Jacobian matrix on GPU
            if q_grad is not None:  # Check if gradient exists
                wp.launch(
                    assign_jacobian_slice_kernel,
                    dim=self.num_envs * self.dof,
                    inputs=[jacobians_wp, q_grad, o, self.num_envs, self.dof],
                    device=self.config.device
                )
            else:
                # Handle cases where gradient might be None if a joint doesn't affect this dimension
                log.warning(f"Gradient for dim {o} is None, Jacobian row will be zero.")

            # Reset gradients for next dimension
            tape.zero()

        # Initialize delta_q on GPU
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint updates using Jacobian transpose method on GPU
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