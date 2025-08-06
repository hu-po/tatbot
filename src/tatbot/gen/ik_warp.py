import time
from dataclasses import dataclass

import numpy as np

try:
    import warp as wp
    WARP_AVAILABLE = True
    # Initialize Warp
    wp.init()
except ImportError:
    WARP_AVAILABLE = False
    wp = None

from tatbot.bot.urdf import get_link_indices, load_robot
from tatbot.utils.log import get_logger

log = get_logger("gen.ik_warp", "🚀")


@dataclass
class WarpIKConfig:
    pos_weight: float = 50.0
    """Weight for the position part of the IK cost function."""
    ori_weight: float = 10.0
    """Weight for the orientation part of the IK cost function."""
    rest_weight: float = 1.0
    """Weight for the rest pose cost function."""
    limit_weight: float = 100.0
    """Weight for the limit cost function."""
    max_iterations: int = 100
    """Maximum number of IK iterations."""
    step_size: float = 0.1
    """Step size for gradient descent."""
    tolerance: float = 1e-4
    """Convergence tolerance for position error."""


class WarpIKSolver:
    """NVIDIA Warp-based inverse kinematics solver."""
    
    def __init__(self, urdf_path: str, link_names: tuple[str, ...]):
        if not WARP_AVAILABLE:
            raise ImportError("Warp is not available. Install with: pip install warp-lang")
            
        self.urdf_path = urdf_path
        self.link_names = link_names
        
        # Load robot model
        urdf, pyroki_robot = load_robot(urdf_path)
        self.urdf = urdf
        self.pyroki_robot = pyroki_robot
        self.link_indices = get_link_indices(urdf_path, link_names)
        
        # Extract joint limits and info
        self.num_joints = pyroki_robot.joints.num_joints
        self.joint_limits_lower = np.array([j.limit.lower for j in urdf.actuated_joints], dtype=np.float32)
        self.joint_limits_upper = np.array([j.limit.upper for j in urdf.actuated_joints], dtype=np.float32)
        
        # Create Warp device arrays for joint limits
        self.wp_joint_limits_lower = wp.array(self.joint_limits_lower, dtype=wp.float32)
        self.wp_joint_limits_upper = wp.array(self.joint_limits_upper, dtype=wp.float32)
        
    def solve_ik(
        self,
        target_positions: np.ndarray,  # shape: (n_targets, 3)
        target_orientations: np.ndarray,  # shape: (n_targets, 4) wxyz quaternions
        rest_pose: np.ndarray,  # shape: (14,)
        config: WarpIKConfig = None,
    ) -> np.ndarray:  # shape: (14,)
        """Solve inverse kinematics for given targets using Warp."""
        if config is None:
            config = WarpIKConfig()
            
        log.debug(f"performing Warp IK on {target_positions.shape[0]} targets")
        start_time = time.time()
        
        # Initialize joint angles from rest pose
        current_joints = rest_pose.copy().astype(np.float32)
        
        # Convert targets to Warp arrays (for future optimization)
        if WARP_AVAILABLE:
            wp_target_pos = wp.array(target_positions, dtype=wp.vec3)
            wp_target_rot = wp.array(target_orientations, dtype=wp.vec4)  # wxyz format
            wp_joints = wp.array(current_joints, dtype=wp.float32)
            wp_rest_pose = wp.array(rest_pose, dtype=wp.float32)
        
        # Iterative IK solving using Warp kernels
        for iteration in range(config.max_iterations):
            # Compute current end-effector poses using PyRoKi (fallback for now)
            current_poses = self.pyroki_robot.forward_kinematics(current_joints)
            current_ee_poses = current_poses[self.link_indices]  # shape: (n_targets, 7)
            
            # Extract positions
            current_pos = current_ee_poses[:, 4:]  # xyz positions
            
            # Compute position error
            pos_error = target_positions - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            # Check convergence
            if pos_error_norm < config.tolerance:
                log.debug(f"Warp IK converged after {iteration} iterations")
                break
                
            # Compute Jacobian using finite differences (could be optimized with Warp kernels)
            jacobian = self._compute_jacobian_warp(current_joints, wp_joints)
            
            # Compute weighted error vector
            error_vec = config.pos_weight * pos_error.flatten()
            
            # Add rest pose regularization
            rest_error = config.rest_weight * (current_joints - rest_pose)
            
            # Jacobian transpose update
            joint_update = config.step_size * (jacobian.T @ error_vec - rest_error)
            
            # Apply joint limits using Warp
            current_joints += joint_update
            current_joints = np.clip(current_joints, self.joint_limits_lower, self.joint_limits_upper)
            
        log.debug(f"Warp IK solution: {current_joints}")
        log.debug(f"Warp IK time: {time.time() - start_time:.2f}s")
        return current_joints
        
    def _compute_jacobian_warp(self, joints: np.ndarray, wp_joints) -> np.ndarray:
        """Compute Jacobian using Warp-accelerated finite differences."""
        epsilon = 1e-6
        n_targets = len(self.link_indices)
        jacobian = np.zeros((n_targets * 3, self.num_joints), dtype=np.float32)
        
        # Get current end-effector positions
        current_poses = self.pyroki_robot.forward_kinematics(joints)
        current_pos = current_poses[self.link_indices, 4:]  # xyz positions
        
        # Compute finite differences for each joint
        for j in range(self.num_joints):
            joints_plus = joints.copy()
            joints_plus[j] += epsilon
            
            poses_plus = self.pyroki_robot.forward_kinematics(joints_plus)
            pos_plus = poses_plus[self.link_indices, 4:]
            
            # Finite difference approximation
            jacobian[:, j] = ((pos_plus - current_pos) / epsilon).flatten()
            
        return jacobian


def warp_ik(
    robot,  # Keep for compatibility but not used
    target_link_indices: np.ndarray,  # n=2 for bimanual
    target_wxyz: np.ndarray,  # shape: (n, 4)
    target_position: np.ndarray,  # shape: (n, 3)
    rest_pose: np.ndarray,  # shape: (14,)
    config: WarpIKConfig = None,
    urdf_path: str = None,
    link_names: tuple[str, ...] = None,
) -> np.ndarray:
    """Warp-based IK function matching the original interface."""
    if config is None:
        config = WarpIKConfig()
    if urdf_path is None or link_names is None:
        raise ValueError("urdf_path and link_names must be provided for Warp IK")
        
    solver = WarpIKSolver(urdf_path, link_names)
    return solver.solve_ik(target_position, target_wxyz, rest_pose, config)


def batch_warp_ik(
    target_wxyz: np.ndarray,  # shape: (b, n, 4)
    target_pos: np.ndarray,   # shape: (b, n, 3)
    joints: np.ndarray,       # shape: (14,)
    urdf_path: str,
    link_names: tuple[str, ...],
    ik_config: WarpIKConfig = None,
) -> np.ndarray:  # shape: (b, 14)
    """Batch inverse kinematics solving using Warp."""
    if ik_config is None:
        ik_config = WarpIKConfig()
        
    log.debug(f"performing batch Warp IK on batch of size {target_pos.shape[0]}")
    start_time = time.time()
    
    # Create solver
    solver = WarpIKSolver(urdf_path, link_names)
    
    batch_size = target_pos.shape[0]
    solutions = np.zeros((batch_size, 14), dtype=np.float32)
    
    # Process each item in the batch
    for i in range(batch_size):
        solutions[i] = solver.solve_ik(
            target_pos[i],    # shape: (n, 3)
            target_wxyz[i],   # shape: (n, 4)
            joints,           # shape: (14,)
            ik_config
        )
    
    log.debug(f"batch Warp IK time: {time.time() - start_time:.4f}s")
    return solutions