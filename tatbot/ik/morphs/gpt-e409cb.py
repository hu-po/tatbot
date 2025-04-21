import warp as wp
import numpy as np

from tatbot.ik.morph import BaseMorph

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Sampling-based Jacobian Estimation with Gauss-Newton update.

    This novel IK solver estimates the local Jacobian via random sampling in joint space.
    For each environment, a set of small random perturbations are applied to the current joint configuration.
    The resulting changes in the 6D end‐effector error are collected, and a least-squares regression is used 
    to estimate the local Jacobian. A damped Gauss–Newton update is then computed by solving
      (J·Jᵀ + λI) y = error_base    and    Δq = -Jᵀ y,
    and the joint angles are updated accordingly. This approach is distinct from the standard finite-difference 
    or Jacobian-transpose methods.

    Note: This algorithm samples multiple perturbations per environment. Although the sampling loop is conducted on 
    the CPU in Python, most heavy lifting (error computation and joint updates) leverages Warp's GPU-accelerated kernels.
    """

    def _update_config(self):
        """
        Configures the Morph for the Sampling-based Gauss-Newton IK.
        
        Parameters:
          step_size: Not used directly, but can be interpreted as an overall scaling factor.
          joint_q_requires_grad: Set to False since we use sampling-based estimation.
          config_extras: Contains parameters for the random perturbation scale 'sigma',
                         the number of samples per environment 'num_samples', and the damping factor 'damping'.
        """
        self.config.step_size = 1.0  # scaling factor (unused directly in update computation)
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "sigma": 1e-2,      # Standard deviation for random perturbations
            "num_samples": 10,  # Number of random samples per environment to estimate the local Jacobian
            "damping": 0.1      # Damping factor for the Gauss–Newton update (λ)
        }

    def _step(self):
        """
        Performs one IK step using a sampling-based estimation of the Jacobian and a Gauss–Newton update.

        For each environment:
          1. The current end-effector error vector (6D) is computed.
          2. A set of random joint-space perturbations (drawn from a Gaussian with standard deviation sigma)
             is generated.
          3. For each perturbation, the joint configuration is altered for that environment,
             and the resulting change in end-effector error is recorded.
          4. Using least-squares regression, a local Jacobian matrix J (6 x dof) is estimated so that
             Δ_error ≈ J · perturbation.
          5. The damped Gauss–Newton update is computed by solving:
                 (J·Jᵀ + damping*I)*y = error  and  Δq = -Jᵀ·y.
          6. The joint configuration is updated by adding Δq.
        Finally, the updated joint configurations for all environments are assigned back to the model.
        """
        # Read configuration extras
        sigma      = self.config.config_extras["sigma"]
        num_samples = self.config.config_extras["num_samples"]
        damping    = self.config.config_extras["damping"]

        # Retrieve the current joint angles as a numpy array from the Warp GPU array.
        q_all = self.model.joint_q.numpy().copy()  # shape: (num_envs * dof,)
        num_envs = self.num_envs
        dof = self.dof

        # Compute the baseline 6D error for all environments.
        error_all = self.compute_ee_error().numpy()  # shape: (num_envs * 6,)
        # We'll update a local copy of the joint angles for each environment.
        q_updated = q_all.copy()

        # For each environment, estimate the local Jacobian and compute update.
        for e in range(num_envs):
            # Extract the baseline joint configuration and EE error for environment e.
            base_idx_q = e * dof
            base_idx_err = e * 6
            q_e = q_all[base_idx_q:base_idx_q + dof].copy()          # (dof,)
            error_base = error_all[base_idx_err:base_idx_err + 6].copy()  # (6,)
            
            # Prepare containers for samples.
            # D will store joint-space perturbations: shape (num_samples, dof)
            # Delta will store the corresponding change in error: shape (num_samples, 6)
            D = np.random.randn(num_samples, dof).astype(np.float32) * sigma
            Delta = np.zeros((num_samples, 6), dtype=np.float32)
            
            # For each random perturbation sample, compute the change in error.
            for i in range(num_samples):
                # Form the perturbed joint configuration for environment e.
                q_sample = q_e + D[i]
                # Create a temporary copy of all joint angles and replace this environment's configuration.
                q_temp = q_updated.copy()
                q_temp[base_idx_q:base_idx_q + dof] = q_sample
                # Assign the temporary joint configurations to the model on the proper device.
                self.model.joint_q.assign(wp.array(q_temp, dtype=wp.float32, device=self.config.device))
                # Compute the new end-effector error.
                error_temp = self.compute_ee_error().numpy()
                error_sample = error_temp[base_idx_err:base_idx_err + 6].copy()
                # Record the error difference from the baseline.
                Delta[i] = error_sample - error_base
            
            # Estimate the local Jacobian J (shape: 6 x dof) via least squares.
            # The assumed local linear model is: Delta ≈ D @ (J^T), i.e., for each sample i:
            #     Delta[i] ~ J @ (D[i])  where D[i] is a column vector.
            #
            # We solve for J^T using numpy.linalg.lstsq:
            #     D * (J^T) ≈ Delta
            # Thus:
            #     J_transpose_est, _, _, _ = np.linalg.lstsq(D, Delta, rcond=None)
            # And then J = (J_transpose_est).T.
            J_transpose_est, _, _, _ = np.linalg.lstsq(D, Delta, rcond=None)
            J_est = J_transpose_est.T  # shape: (6, dof)
            
            # Now, compute the damped Gauss-Newton update.
            # Solve the 6x6 linear system: (J * Jᵀ + damping * I) y = error_base.
            JJt = J_est @ J_est.T  # shape: (6, 6)
            # Add damping to the diagonal.
            A = JJt + damping * np.eye(6, dtype=np.float32)
            # Solve for y: y = A^(-1) * error_base.
            # If A is singular, np.linalg.solve may raise an error; in practice damping should help.
            try:
                y = np.linalg.solve(A, error_base)  # shape: (6,)
            except np.linalg.LinAlgError:
                # In case of numerical issues, use a small update.
                y = np.zeros(6, dtype=np.float32)
            # Compute the joint update: Δq = -Jᵀ * y.
            delta_q = - J_est.T @ y  # shape: (dof,)
            
            # Update the joint configuration for environment e.
            q_new = q_e + delta_q
            q_updated[base_idx_q:base_idx_q + dof] = q_new

        # After processing all environments, assign the updated joint angles back to the model.
        self.model.joint_q.assign(wp.array(q_updated, dtype=wp.float32, device=self.config.device))
    
# End of Morph class implementation.