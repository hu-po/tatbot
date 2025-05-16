import warp as wp
import numpy as np
import logging as log

from tatbot.ik.morph import BaseMorph

@wp.kernel
def clip_gradients_kernel(
    grad_q: wp.array(dtype=wp.float32),
    max_norm: float
):
    """Clips gradients element-wise if their norm exceeds max_norm."""
    tid = wp.tid()
    grad_val = grad_q[tid]
    grad_norm = wp.sqrt(grad_val * grad_val)
    if grad_norm > max_norm:
        grad_q[tid] = (grad_val / grad_norm) * max_norm

@wp.kernel
def adam_update_kernel(
    q: wp.array(dtype=wp.float32),
    grad_q: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    t: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    q_out: wp.array(dtype=wp.float32),
    m_out: wp.array(dtype=wp.float32),
    v_out: wp.array(dtype=wp.float32)
):
    """Performs the Adam optimization update step element-wise for each joint."""
    tid = wp.tid()

    # Compute bias-corrected learning rate for this step
    # Avoid division by zero for t=0 (although t starts at 1 in practice)
    lr_t = learning_rate
    if t > 0.0:
        lr_t = learning_rate * wp.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)

    # Update biased first moment estimate
    m_new = beta1 * m[tid] + (1.0 - beta1) * grad_q[tid]

    # Update biased second raw moment estimate
    v_new = beta2 * v[tid] + (1.0 - beta2) * (grad_q[tid] * grad_q[tid])

    # Compute update step
    update = lr_t * m_new / (wp.sqrt(v_new) + epsilon)

    # Apply update (gradient descent)
    q_out[tid] = q[tid] - update
    m_out[tid] = m_new
    v_out[tid] = v_new

@wp.kernel
def elementwise_square_kernel(
    input_arr: wp.array(dtype=wp.float32),
    output_arr: wp.array(dtype=wp.float32)
):
    """Calculates output[i] = input[i] * input[i]"""
    tid = wp.tid()
    output_arr[tid] = input_arr[tid] * input_arr[tid]

@wp.kernel
def sum_reduce_kernel(
    input_arr: wp.array(dtype=wp.float32),
    output_arr: wp.array(dtype=wp.float32)
):
    """Reduces an array to a single scalar sum using atomic add."""
    tid = wp.tid()
    wp.atomic_add(output_arr, 0, input_arr[tid])

class Morph(BaseMorph):
    def __init__(self, config):
        """Initializes the Adam morph, including optimizer state."""
        super().__init__(config)  # Call BaseMorph init first

        # Initialize Adam state variables (persistent across steps)
        # Ensure they are on the same device as the model
        q_shape = self.model.joint_q.shape
        self.adam_m = wp.zeros(q_shape, dtype=wp.float32, device=self.config.device)
        self.adam_v = wp.zeros(q_shape, dtype=wp.float32, device=self.config.device)
        self.adam_t = 0  # Timestep counter

    def _update_config(self):
        """
        Updates the configuration specific to the Adam Optimization Morph.
        """
        self.config.step_size = 1e-2  # Base learning rate
        self.config.joint_q_requires_grad = True
        self.config.config_extras = {
            "beta1": 0.9,      # Exponential decay rate for first moment
            "beta2": 0.999,    # Exponential decay rate for second moment
            "epsilon": 1e-8,   # Small constant for numerical stability
            "max_grad_norm": 1.0,  # Maximum gradient norm for clipping
        }

    def _step(self):
        """
        Performs one IK step using the Adam optimization algorithm.

        1. Computes the 6D pose error using compute_ee_error.
        2. Computes squared error and reduces to scalar loss on GPU.
        3. Computes gradient of scalar loss w.r.t. joint angles.
        4. Updates joint angles using Adam optimizer on GPU.
        """
        beta1 = self.config.config_extras["beta1"]
        beta2 = self.config.config_extras["beta2"]
        epsilon = self.config.config_extras["epsilon"]
        learning_rate = self.config.step_size

        self.adam_t += 1  # Increment timestep counter

        tape = wp.Tape()

        # Record computation graph
        with tape:
            # Get current EE error
            ee_error_flat_wp = self.compute_ee_error()

            # Compute squared error
            squared_error = wp.zeros_like(ee_error_flat_wp)
            wp.launch(
                elementwise_square_kernel,
                dim=ee_error_flat_wp.shape[0],
                inputs=[ee_error_flat_wp],
                outputs=[squared_error],
                device=self.config.device
            )

            # Reduce to scalar loss
            scalar_loss = wp.zeros(1, dtype=wp.float32, device=self.config.device, requires_grad=True)
            wp.launch(
                sum_reduce_kernel,
                dim=squared_error.shape[0],
                inputs=[squared_error],
                outputs=[scalar_loss],
                device=self.config.device
            )

        # Compute gradients
        try:
            tape.backward(loss=scalar_loss)
            grad_q = tape.gradients[self.model.joint_q]
            if grad_q is None:
                log.warning("No gradients found for joint_q")
                return
            else:
                # Apply gradient clipping
                max_grad_norm = self.config.config_extras["max_grad_norm"]
                wp.launch(
                    clip_gradients_kernel,
                    dim=grad_q.shape[0],
                    inputs=[grad_q, max_grad_norm],
                    device=self.config.device
                )
        except Exception as e:
            log.error(f"Error in backward pass: {e}", exc_info=True)
            return

        # Create output array for updated joint angles
        q_new = wp.zeros_like(self.model.joint_q)

        # Launch Adam update kernel
        wp.launch(
            adam_update_kernel,
            dim=self.model.joint_q.shape[0],  # Launch one thread per joint angle
            inputs=[
                self.model.joint_q,
                grad_q,
                self.adam_m,
                self.adam_v,
                float(self.adam_t),
                learning_rate,
                beta1,
                beta2,
                epsilon,
            ],
            outputs=[
                q_new,       # Updated joint angles
                self.adam_m, # Updated first moment (in-place)
                self.adam_v, # Updated second moment (in-place)
            ],
            device=self.config.device
        )

        # Update joint angles
        self.model.joint_q.assign(q_new) 