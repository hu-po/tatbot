from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import importlib
import json
import logging
import math
import os
import sys
import time
import psutil
import numpy as np
import wandb
import warp as wp
import warp.sim
import warp.sim.render

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class MorphConfig:
    seed: int = 42 # random seed
    morph: str = 'base' # unique identifier for the morph (when testing use "blank" morph)
    backend: str = os.environ.get("BACKEND") # compute backend variant
    root_dir: str = os.environ.get("TATBOT_ROOT") # root directory of the warp-ik project
    assets_dir: str = f"{root_dir}/assets" # assets directory for the morphs
    output_dir: str = f"{root_dir}/output" # output directory for the morphs
    morph_dir: str = f"{root_dir}/tatbot/ik/morphs" # directory for the morphs
    morph_output_dir: str = f"{output_dir}/{morph}" # output directory for this unique morph
    targets_path: str  = f"{assets_dir}/targets/zorya-128x128.npy"   # path to N×7 .npy file of target poses
    urdf_path: str = f"{assets_dir}/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # path to the urdf file
    usd_output_path: str = f"{output_dir}/ik-{morph}.usd" # path to the usd file to save the model
    device: str = os.environ.get("DEVICE", None) # nvidia device to run the simulation on
    headless: bool = False # turns off rendering
    num_envs: int = os.environ.get("NUM_ENVS", 4) # number of parallel environments
    num_rollouts: int = 2 # number of rollouts to perform
    train_iters: int = 64 # number of training iterations per rollout
    track: bool = False # turns on tracking with wandb
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "hug")
    wandb_project: str = os.environ.get("WANDB_PROJECT", "warp-ik")
    created_on: str = datetime.now().strftime("%Y%m%d%H%M%S")
    start_time: float = 0.0 # start time for the simulation
    fps: int = 60 # frames per second
    step_size: float = 1.0 # step size in q space for updates
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # offset from the ee_gripper_link to the end effector
    gizmo_radius: float = 0.005 # radius of the gizmo (used for arrow base radius)
    gizmo_length: float = 0.05 # total length of the gizmo arrow
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0) # color of the x gizmo for the ee
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0) # color of the y gizmo for the ee
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0) # color of the z gizmo for the ee
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5) # color of the x gizmo for the target
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5) # color of the y gizmo for the target
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0) # color of the z gizmo for the target
    arm_spacing_xz: float = 1.0 # spacing between arms in the x-z plane
    arm_height: float = 0.0 # height of the arm off the floor
    target_offset: tuple[float, float, float] = (0.3, 0, 0.1) # offset of the target from the base (in robot coordinates)
    target_spawn_pos_noise: float = 0.1 # maximum position noise to add when spawning targets
    target_spawn_rot_noise: float = np.pi/8  # maximum rotation angle noise to add when spawning targets
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054),    # base
        (0.0, 3.14),        # shoulder
        (0.0, 2.356),       # elbow
        (-1.57, 1.57),      # wrist 1
        (-1.57, 1.57),      # wrist 2
        (-3.14, 3.14),      # wrist 3
        (0.0, 0.044),       # right finger
        (0.0, 0.044),       # left finger
    ]) # joint limits for arm
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5), # quarter turn about x-axis
        # ((0.0, 0.0, 1.0), -math.pi * 0.5), # quarter turn about z-axis
    ]) # list of axis angle rotations for initial arm orientation offset
    qpos_home: list[float] = field(default_factory=lambda: [
        0,          # base joint
        np.pi/12,   # shoulder joint
        np.pi/12,   # elbow joint
        0,          # wrist 1 joint
        0,          # wrist 2 joint
        0,          # wrist 3 joint
        0,          # right finger joint
        0           # left finger joint
    ]) # home position for the arm joints
    q_angle_shuffle: list[float] = field(default_factory=lambda: [
        np.pi/2,    # base joint noise range
        np.pi/4,    # shoulder joint noise range
        np.pi/4,    # elbow joint noise range
        np.pi/4,    # wrist 1 joint noise range
        np.pi/4,    # wrist 2 joint noise range
        np.pi/4,    # wrist 3 joint noise range
        0.01,       # right finger joint noise range
        0.01        # left finger joint noise range
    ]) # amount of random noise to add to the arm joint angles during initialization
    config_extras: dict = field(default_factory=dict) # additional morph-specific config values
    body_q_requires_grad: bool = True # whether to require gradients for the body positions
    joint_q_requires_grad: bool = True # whether to require gradients for the joint angles
    joint_attach_ke: float = 1600.0 # stiffness of the joint attachment
    joint_attach_kd: float = 20.0 # damping of the joint attachment

class MorphState(Enum):
    NOT_RUN_YET = auto()
    ALREADY_RAN = auto()
    ERRORED_OUT = auto()

@dataclass(order=True)
class ActiveMorph:
    score: float
    name: str
    state: MorphState = MorphState.NOT_RUN_YET

@wp.func
def quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    )

@wp.func
def quat_conjugate(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])

@wp.func
def quat_orientation_error(target: wp.quat, current: wp.quat) -> wp.vec3:
    """Computes 3D orientation error vector from target and current quaternions."""
    q_err = quat_mul(target, quat_conjugate(current))
    # Ensure scalar part is non-negative for consistency
    qw = wp.where(q_err[3] < 0.0, -q_err[3], q_err[3])
    qx = wp.where(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.where(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.where(q_err[3] < 0.0, -q_err[2], q_err[2])
    # Return axis * angle (scaled by 2), which is approx 2 * axis * sin(angle/2)
    # Magnitude is related to the angle error
    return wp.vec3(qx, qy, qz) * 2.0

@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3),
    target_ori: wp.array(dtype=wp.quat),
    error_out: wp.array(dtype=wp.float32)  # Flattened array (num_envs*6)
):
    tid = wp.tid()
    # Get EE link transform [px,py,pz, qx,qy,qz,qw, scale]
    t_flat = body_q[tid * num_links + ee_link_index]
    t_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    t_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])

    # Calculate current EE tip position
    current_pos = wp.transform_point(wp.transform(t_pos, t_ori), ee_link_offset)
    # Current EE orientation is just t_ori
    current_ori = t_ori

    # Calculate errors
    pos_err = target_pos[tid] - current_pos
    ori_err = quat_orientation_error(target_ori[tid], current_ori)

    # Write to output array
    base = tid * 6
    error_out[base + 0] = pos_err.x
    error_out[base + 1] = pos_err.y
    error_out[base + 2] = pos_err.z
    error_out[base + 3] = ori_err.x
    error_out[base + 4] = ori_err.y
    error_out[base + 5] = ori_err.z

@wp.kernel
def clip_joints_kernel(
    joint_q: wp.array(dtype=wp.float32), # Input/Output array
    joint_limits_min: wp.array(dtype=wp.float32), # Min limits per DOF type
    joint_limits_max: wp.array(dtype=wp.float32), # Max limits per DOF type
    num_envs: int,
    dof: int
):
    tid = wp.tid() # Unique index for each joint DOF across all envs

    # Calculate which DOF this thread corresponds to
    joint_idx = tid % dof

    # Clamp the value
    current_val = joint_q[tid]
    min_val = joint_limits_min[joint_idx]
    max_val = joint_limits_max[joint_idx]
    joint_q[tid] = wp.clamp(current_val, min_val, max_val)

@wp.kernel
def update_targets_kernel(
    target_origins: wp.array(dtype=wp.vec3),
    initial_base_orientation: wp.quat, # The common base rotation
    pos_noise_scale: float,
    rot_noise_scale: float,
    seed: int, # Seed for RNG state
    # Outputs:
    out_target_pos: wp.array(dtype=wp.vec3),
    out_target_ori: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Environment index

    # Initialize random state for this thread
    state = wp.rand_init(seed, tid)

    # Positional Noise
    noise_x = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    noise_y = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    noise_z = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    pos_noise_vec = wp.vec3(noise_x, noise_y, noise_z)

    out_target_pos[tid] = target_origins[tid] + pos_noise_vec

    # Rotational Noise (Axis-Angle)
    axis_x = wp.randf(state, -1.0, 1.0)
    axis_y = wp.randf(state, -1.0, 1.0)
    axis_z = wp.randf(state, -1.0, 1.0)
    noise_axis = wp.normalize(wp.vec3(axis_x, axis_y, axis_z))
    noise_angle = wp.randf(state, -rot_noise_scale, rot_noise_scale)

    random_rot = wp.quat_from_axis_angle(noise_axis, noise_angle)

    # Apply random rotation relative to the initial base orientation
    out_target_ori[tid] = quat_mul(initial_base_orientation, random_rot)

@wp.kernel
def calculate_error_magnitudes_kernel(
    flat_errors: wp.array(dtype=wp.float32), # Input: num_envs * 6
    num_envs: int,
    # Outputs:
    out_pos_mag: wp.array(dtype=wp.float32),
    out_ori_mag: wp.array(dtype=wp.float32)
):
    tid = wp.tid() # Environment index
    base = tid * 6

    # Extract pos and ori error vectors
    pos_err = wp.vec3(flat_errors[base + 0], flat_errors[base + 1], flat_errors[base + 2])
    ori_err = wp.vec3(flat_errors[base + 3], flat_errors[base + 4], flat_errors[base + 5])

    # Calculate and store magnitudes
    out_pos_mag[tid] = wp.length(pos_err)
    out_ori_mag[tid] = wp.length(ori_err)

@wp.kernel
def calculate_gizmo_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    targets_pos: wp.array(dtype=wp.vec3),
    targets_ori: wp.array(dtype=wp.quat),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    num_envs: int,
    rot_x_axis_q: wp.quat, # Precomputed axis rotations
    rot_y_axis_q: wp.quat,
    rot_z_axis_q: wp.quat,
    cone_half_height: float,
    # Outputs (flat array, order: TgtX,TgtY,TgtZ, EEX,EEY,EEZ per env)
    out_gizmo_pos: wp.array(dtype=wp.vec3),
    out_gizmo_rot: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Environment index

    # --- Target Gizmos ---
    target_pos = targets_pos[tid]
    target_ori = targets_ori[tid]

    target_rot_x = quat_mul(target_ori, rot_x_axis_q)
    target_rot_y = quat_mul(target_ori, rot_y_axis_q)
    target_rot_z = quat_mul(target_ori, rot_z_axis_q)

    offset_vec = wp.vec3(0.0, cone_half_height, 0.0)
    offset_x = wp.quat_rotate(target_rot_x, offset_vec)
    offset_y = wp.quat_rotate(target_rot_y, offset_vec)
    offset_z = wp.quat_rotate(target_rot_z, offset_vec)

    base_idx = tid * 6
    out_gizmo_pos[base_idx + 0] = target_pos - offset_x
    out_gizmo_rot[base_idx + 0] = target_rot_x
    out_gizmo_pos[base_idx + 1] = target_pos - offset_y
    out_gizmo_rot[base_idx + 1] = target_rot_y
    out_gizmo_pos[base_idx + 2] = target_pos - offset_z
    out_gizmo_rot[base_idx + 2] = target_rot_z

    # --- End-Effector Gizmos ---
    t_flat = body_q[tid * num_links + ee_link_index]
    ee_link_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    ee_link_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])
    ee_tip_pos = wp.transform_point(wp.transform(ee_link_pos, ee_link_ori), ee_link_offset)

    ee_rot_x = quat_mul(ee_link_ori, rot_x_axis_q)
    ee_rot_y = quat_mul(ee_link_ori, rot_y_axis_q)
    ee_rot_z = quat_mul(ee_link_ori, rot_z_axis_q)

    ee_offset_x = wp.quat_rotate(ee_rot_x, offset_vec)
    ee_offset_y = wp.quat_rotate(ee_rot_y, offset_vec)
    ee_offset_z = wp.quat_rotate(ee_rot_z, offset_vec)

    out_gizmo_pos[base_idx + 3] = ee_tip_pos - ee_offset_x
    out_gizmo_rot[base_idx + 3] = ee_rot_x
    out_gizmo_pos[base_idx + 4] = ee_tip_pos - ee_offset_y
    out_gizmo_rot[base_idx + 4] = ee_rot_y
    out_gizmo_pos[base_idx + 5] = ee_tip_pos - ee_offset_z
    out_gizmo_rot[base_idx + 5] = ee_rot_z

class BaseMorph:

    def _update_config(self):
        raise NotImplementedError("Morphs must implement their own update_config function")

    def __init__(self, config: MorphConfig):

        self.config = config
        self._update_config()
        log.debug(f"config:{json.dumps(self.config.__dict__, indent=4)}")

        self.config.morph_output_dir = os.path.join(self.config.output_dir, self.config.morph)
        os.makedirs(config.morph_output_dir, exist_ok=True)
        log.debug(f"morph specific output_dir: {config.morph_output_dir}")

        config_filepath = os.path.join(config.morph_output_dir, "config.json")
        with open(config_filepath, 'w') as f:
            # Convert fields that might not be JSON serializable (like numpy arrays in defaults)
            config_dict = {}
            for key, value in self.config.__dict__.items():
                if isinstance(value, np.ndarray):
                    config_dict[key] = value.tolist() # Convert numpy arrays to lists
                elif isinstance(value, (wp.vec3, wp.quat, wp.transform)):
                     config_dict[key] = list(value) # Convert warp types if needed
                else:
                    config_dict[key] = value
            json.dump(config_dict, f, indent=4)

        self.wandb_run = None
        if self.config.track:
            if wandb is None:
                log.error("WandB tracking enabled but wandb library not found. pip install wandb")
                self.config.track = False # Disable tracking if library missing
            else:
                try:
                    wandb.login()
                    self.wandb_run = wandb.init(
                        entity=self.config.wandb_entity,
                        project=self.config.wandb_project,
                        name=f"{self.config.backend}/{self.config.morph}",
                        config=config_dict,
                        tags=[f"family:{self.config.morph.split('-')[0]}"]  # Strip UUID suffix after dash for family tag
                    )
                    # Update config with the *final* values used
                    self.wandb_run.config.update({
                        "num_envs": self.config.num_envs,
                        "device": self.config.device,
                        "backend": self.config.backend,
                        "morph": self.config.morph
                    }, allow_val_change=True)

                    # Save config file artifact
                    if os.path.exists(config_filepath):
                        self.wandb_run.save(config_filepath)

                    # Save morph code artifact
                    morph_code_path = os.path.join(self.config.morph_dir, f"{self.config.morph}.py")
                    if os.path.exists(morph_code_path):
                        self.wandb_run.save(morph_code_path)
                    log.info(f"WandB tracking enabled: {self.wandb_run.get_url()}")
                except Exception as e:
                    log.error(f"Failed to initialize WandB: {e}. Disabling tracking.")
                    self.config.track = False
                    if self.wandb_run:
                        wandb.finish(exit_code=1)
                        self.wandb_run = None

        # -------------------------------------------------------------
        # LOAD FIXED TARGETS (if supplied) *BEFORE* we build anything
        # -------------------------------------------------------------
        if self.config.targets_path:
            tgt_path = os.path.expanduser(self.config.targets_path)
            if not os.path.isfile(tgt_path):
                raise FileNotFoundError(f"targets_path not found: {tgt_path}")
            tgt_np = np.load(tgt_path).astype(np.float32)
            if tgt_np.ndim != 2 or tgt_np.shape[1] != 7:
                raise ValueError("targets .npy must be shape (N,7)")

            # Over‑ride env count to the number of rows in the file
            self.config.num_envs = len(tgt_np)

            # Split into pos / quat  ------------------------------------------------
            self._fixed_target_pos_np = tgt_np[:, :3].copy()
            self._fixed_target_ori_np = tgt_np[:, 3:].copy()

            # Disable rollout noise so update_targets_kernel becomes a no‑op
            self.config.target_spawn_pos_noise = 0.0
            self.config.target_spawn_rot_noise = 0.0

        self.rng = np.random.default_rng(self.config.seed)
        self.num_envs = self.config.num_envs
        self.render_time = self.config.start_time
        self.fps = self.config.fps
        self.frame_dt = 1.0 / self.fps

        # Parse URDF and build model.
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.expanduser(self.config.urdf_path),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False
        )
        builder = wp.sim.ModelBuilder()
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = np.array(self.config.joint_limits, dtype=np.float32)
        
        # Create Warp arrays for joint limits
        self.joint_limits_min_wp = wp.array(self.joint_limits[:, 0], dtype=wp.float32, device=self.config.device)
        self.joint_limits_max_wp = wp.array(self.joint_limits[:, 1], dtype=wp.float32, device=self.config.device)
        
        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")

        # Locate ee_gripper link.
        self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
        self.ee_link_index = -1
        for i, name in enumerate(articulation_builder.body_name): # Iterate through bodies to find link index
             # The joint child gives the *joint* index, not the *body* index directly connected.
             # We need the index of the body specified as the joint's child.
             # Let's find the body name associated with the joint child index.
             # This seems overly complex, let's assume the joint name convention is reliable for now.
             # TODO: Verify robust EE link index finding if URDFs change structure
             if name == "ee_gripper_link": # Assuming the link name matches the convention often used with joints
                 self.ee_link_index = i
                 break
             # Fallback using joint info (less direct)
             # joint_idx = -1
             # for j_idx, j_name in enumerate(articulation_builder.joint_name):
             #     if j_name == "ee_gripper": # Find the joint named 'ee_gripper'
             #         joint_idx = j_idx
             #         break
             # if joint_idx != -1:
             #      self.ee_link_index = articulation_builder.joint_child[joint_idx] # This is the child *joint* index, need body
                 # Need mapping from joint child index to body index...

        if self.ee_link_index == -1:
             # Let's try finding the joint first and getting its child body index
             found_joint_idx = -1
             for j_idx, j_name in enumerate(articulation_builder.joint_name):
                  if j_name == "ee_gripper":
                       found_joint_idx = j_idx
                       break
             if found_joint_idx != -1:
                  # joint_child gives the index of the child *link/body*
                  self.ee_link_index = articulation_builder.joint_child[found_joint_idx]
                  log.info(f"Found ee_gripper joint, using child link index: {self.ee_link_index} (Name: {articulation_builder.body_name[self.ee_link_index]})")
             else:
                  raise ValueError("Could not find ee_gripper joint or link in URDF")


        # Compute initial arm orientation.
        _initial_arm_orientation_wp = wp.quat(0.,0.,0.,1.) # Start with identity
        for axis, angle in self.config.arm_rot_offset:
            rot_quat_wp = wp.quat_from_axis_angle(wp.vec3(axis[0], axis[1], axis[2]), angle)
            _initial_arm_orientation_wp = quat_mul(rot_quat_wp, _initial_arm_orientation_wp) # Apply rotation
        self.initial_arm_orientation = _initial_arm_orientation_wp # Store as Warp quat
        log.debug(f"Initial arm orientation: {list(self.initial_arm_orientation)}")


        # --- Target origin computation and initial joint angles ---
        target_origins = [] # List to store wp.vec3 origins temporarily
        self.arm_spacing_xz = self.config.arm_spacing_xz
        self.arm_height = self.config.arm_height
        self.num_rows = int(math.sqrt(self.num_envs)) if self.num_envs > 0 else 0
        if self.num_rows * self.num_rows != self.num_envs:
             log.warning(f"num_envs ({self.num_envs}) is not a perfect square. Grid layout might be uneven.")
             self.num_rows = int(np.ceil(math.sqrt(self.num_envs))) # Adjust for layout

        log.info(f"Spawning {self.num_envs} arms in a grid approximating {self.num_rows}x{self.num_rows}")

        initial_joint_q = [] # Store initial joint angles for all envs

        for e in range(self.num_envs):
            row = e // self.num_rows
            col = e % self.num_rows
            x = col * self.arm_spacing_xz
            z = row * self.arm_spacing_xz
            base_translation = wp.vec3(x, self.arm_height, z)
            base_quat = self.initial_arm_orientation # Use the final computed Warp orientation

            # Define the target offset in robot coordinates and transform to world
            target_offset_local = wp.vec3(self.config.target_offset)
            target_world = wp.transform_point(wp.transform(base_translation, base_quat), target_offset_local)
            target_origins.append(target_world)

            # Add arm instance to the main builder
            builder.add_builder(articulation_builder, xform=wp.transform(base_translation, self.initial_arm_orientation))

            # Set initial joint angles for this environment
            env_joint_q = []
            num_joints_in_arm = len(self.config.qpos_home)
            if num_joints_in_arm != self.dof:
                 log.warning(f"Mismatch: len(qpos_home)={num_joints_in_arm} != dof={self.dof}. Check config.")
                 # Adjust loop range defensively
                 num_joints_to_set = min(num_joints_in_arm, self.dof, len(self.config.q_angle_shuffle), len(self.joint_limits))
            else:
                 num_joints_to_set = self.dof

            for i in range(num_joints_to_set):
                value = self.config.qpos_home[i] + self.rng.uniform(-self.config.q_angle_shuffle[i], self.config.q_angle_shuffle[i])
                # Clip value based on joint limits before adding
                clipped_value = np.clip(value, self.joint_limits[i, 0], self.joint_limits[i, 1])
                env_joint_q.append(clipped_value)

            # Handle potential DOF mismatch if warning occurred
            if len(env_joint_q) < self.dof:
                 env_joint_q.extend([0.0] * (self.dof - len(env_joint_q))) # Pad with zeros

            initial_joint_q.extend(env_joint_q)

        # Convert target origins list to Warp array
        self.target_origin_wp = wp.array(target_origins, dtype=wp.vec3, device=self.config.device)
        # Keep numpy version if needed for compatibility
        self.target_origin_np = self.target_origin_wp.numpy()

        # *** CORRECTED: Initial target orientation using np.tile first ***
        initial_arm_orientation_np = np.array(list(self.initial_arm_orientation), dtype=np.float32)
        initial_target_ori_np = np.tile(initial_arm_orientation_np, (self.num_envs, 1))
        # Create the persistent Warp array from the tiled NumPy array
        self.target_ori_wp = wp.array(initial_target_ori_np, dtype=wp.quat, device=self.config.device, requires_grad=False)
        log.debug(f"Initial target orientations (first env): {initial_target_ori_np[0]}")

        # Finalize model.
        self.model = builder.finalize(device=self.config.device) # Specify device
        self.model.ground = False
        self.model.joint_q.assign(wp.array(initial_joint_q, dtype=wp.float32, device=self.config.device))
        self.model.body_q.requires_grad = self.config.body_q_requires_grad
        self.model.joint_q.requires_grad = self.config.joint_q_requires_grad
        self.model.joint_attach_ke = self.config.joint_attach_ke
        self.model.joint_attach_kd = self.config.joint_attach_kd

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # --- Renderer Setup ---
        self.renderer = None
        if not self.config.headless:
            try:
                self.usd_output_path = os.path.join(self.config.morph_output_dir, "recording.usd")
                self.renderer = wp.sim.render.SimRenderer(self.model, self.usd_output_path)
                log.info(f"Initialized renderer, outputting to {self.usd_output_path}")
                # Log USD path to wandb if tracking
                if self.wandb_run:
                    self.wandb_run.summary['usd_output_path'] = self.usd_output_path
                # Initialize gizmo transform arrays and precomputed rotations
                self.gizmo_pos_wp = wp.zeros(self.num_envs * 6, dtype=wp.vec3, device=self.config.device)
                self.gizmo_rot_wp = wp.zeros(self.num_envs * 6, dtype=wp.quat, device=self.config.device)
                self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
                self.rot_y_axis_q_wp = wp.quat_identity()
                self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)

            except Exception as e:
                log.error(f"Failed to initialize renderer: {e}. Running headless.")
                self.config.headless = True # Force headless if renderer fails
        # --- End Renderer Setup ---

        # --- Simulation State ---
        # For now, keep it here for potential broader use.
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True, device=self.config.device)
        # ee_error stores the flattened 6D error [err_px, py, pz, ox, oy, oz] * num_envs
        self.ee_error = wp.zeros(self.num_envs * 6, dtype=wp.float32, requires_grad=True, device=self.config.device)
        # Add arrays for error magnitudes
        self.ee_pos_err_mag_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.config.device)
        self.ee_ori_err_mag_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.config.device)
        
        # Get initial state from model
        self.state = self.model.state(requires_grad=True) # Get state AFTER setting initial joint_q

        # -------------------------------------------------------------
        # If fixed targets were loaded, offset each local target by the environment grid location
        # -------------------------------------------------------------
        if self.config.targets_path:
            # Compute base translations for each environment
            base_translations = np.zeros((self.num_envs, 3), dtype=np.float32)
            for e in range(self.num_envs):
                row = e // self.num_rows
                col = e % self.num_rows
                base_translations[e, 0] = col * self.arm_spacing_xz
                base_translations[e, 1] = self.arm_height
                base_translations[e, 2] = row * self.arm_spacing_xz

            # Add grid spacing to each local target position
            world_targets = self._fixed_target_pos_np + base_translations
            self.targets = world_targets
            self.targets_wp = wp.array(self.targets, dtype=wp.vec3, device=self.config.device)
            self.target_ori = self._fixed_target_ori_np
            self.target_ori_wp = wp.array(self.target_ori, dtype=wp.quat, device=self.config.device)
            # Keep target_origin_np for compatibility if needed
            self.target_origin_np = self.targets.copy()
        else:
            # Original behaviour: start with target_origin_wp and allow noise
            self.targets_wp = wp.clone(self.target_origin_wp)
            self.target_origin_np = self.target_origin_wp.numpy()
            self.targets = self.target_origin_np.copy()
            self.target_ori = initial_target_ori_np.copy()

        # Profiler dictionary
        self.profiler = {}
        # --- End Simulation State ---


    def compute_ee_error(self) -> wp.array:
        """Computes the 6D end-effector error and returns the flattened Warp array."""
        with wp.ScopedTimer("eval_fk", print=False, active=True, dict=self.profiler, color="blue"):
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
            # Note: self.state is updated in-place by eval_fk
        with wp.ScopedTimer("error_kernel", print=False, active=True, dict=self.profiler, color="green"):
            # Fixed arrays are already on the device; avoid realloc when possible
            if self.config.targets_path:
                targets_pos_wp = self.targets_wp
                targets_ori_wp = self.target_ori_wp
            else:
                targets_pos_wp = wp.array(self.targets, dtype=wp.vec3, device=self.config.device)
                targets_ori_wp = wp.array(self.target_ori, dtype=wp.quat, device=self.config.device)
            wp.launch(
                compute_ee_error_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q, # body_q is updated by eval_fk
                    self.num_links,
                    self.ee_link_index,
                    self.ee_link_offset,
                    targets_pos_wp,
                    targets_ori_wp,
                    # Removed precomputed current_ori
                ],
                outputs=[self.ee_error], # ee_error is updated in-place
                device=self.config.device
            )
        return self.ee_error # Return the updated Warp array


    def _step(self):
        """Placeholder for the morph-specific IK update logic."""
        raise NotImplementedError("Morphs must implement their own _step function")


    def step(self):
        """
        Performs one simulation step, including error calculation and the morph-specific update.
        Also handles profiling of the morph's _step implementation.
        """
        # Profile the morph-specific implementation
        with wp.ScopedTimer("_step", print=False, active=True, dict=self.profiler, color="red"):
            self._step() # Execute the morph's core logic

        # Clipping joint angles after the step to respect limits (on GPU)
        with wp.ScopedTimer("clip_joints", print=False, active=True, dict=self.profiler):
            wp.launch(
                kernel=clip_joints_kernel,
                dim=self.num_envs * self.dof, # Launch one thread per DOF instance
                inputs=[
                    self.model.joint_q,
                    self.joint_limits_min_wp,
                    self.joint_limits_max_wp,
                    self.num_envs,
                    self.dof
                ],
                device=self.config.device
            )


    def render_gizmos(self):
        if self.renderer is None:
            return
        # Ensure FK is up-to-date for rendering the current state
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        radius = self.config.gizmo_radius
        cone_height = self.config.gizmo_length
        cone_half_height = cone_height / 2.0

        # Calculate all gizmo transforms on GPU
        with wp.ScopedTimer("gizmo_kernel", print=False, active=True, dict=self.profiler):
            wp.launch(
                kernel=calculate_gizmo_transforms_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q,
                    self.targets_wp,
                    self.target_ori_wp,
                    self.num_links,
                    self.ee_link_index,
                    self.ee_link_offset,
                    self.num_envs,
                    self.rot_x_axis_q_wp,
                    self.rot_y_axis_q_wp,
                    self.rot_z_axis_q_wp,
                    cone_half_height
                ],
                outputs=[self.gizmo_pos_wp, self.gizmo_rot_wp],
                device=self.config.device
            )

        # Transfer results to CPU
        gizmo_pos_np = self.gizmo_pos_wp.numpy()
        gizmo_rot_np = self.gizmo_rot_wp.numpy()

        # Render gizmos using precomputed transforms
        for e in range(self.num_envs):
            base_idx = e * 6
            # Target Gizmos
            self.renderer.render_cone(name=f"target_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 0]), rot=tuple(gizmo_rot_np[base_idx + 0]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_target)
            self.renderer.render_cone(name=f"target_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 1]), rot=tuple(gizmo_rot_np[base_idx + 1]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_target)
            self.renderer.render_cone(name=f"target_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 2]), rot=tuple(gizmo_rot_np[base_idx + 2]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_target)

            # End-Effector Gizmos
            self.renderer.render_cone(name=f"ee_pos_x_{e}", pos=tuple(gizmo_pos_np[base_idx + 3]), rot=tuple(gizmo_rot_np[base_idx + 3]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_ee)
            self.renderer.render_cone(name=f"ee_pos_y_{e}", pos=tuple(gizmo_pos_np[base_idx + 4]), rot=tuple(gizmo_rot_np[base_idx + 4]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_ee)
            self.renderer.render_cone(name=f"ee_pos_z_{e}", pos=tuple(gizmo_pos_np[base_idx + 5]), rot=tuple(gizmo_rot_np[base_idx + 5]), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_ee)

    def render_error_bars(self):
        # TODO: implement error bar rendering if desired
        pass

    def render(self):
        if self.renderer is None:
            return
        # Timer for rendering
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler, color="purple"):
            self.renderer.begin_frame(self.render_time)
            # State should be up-to-date from FK calls before rendering
            self.renderer.render(self.state)
            self.render_gizmos()
            self.render_error_bars()
            self.renderer.end_frame()
        self.render_time += self.frame_dt # Increment render time *after* frame


def run_morph(config: MorphConfig) -> dict:
    """Runs the simulation for a given morph configuration."""
    morph_path = os.path.join(config.morph_dir, f"{config.morph}.py")
    log.info(f"Running morph {config.morph} from {morph_path}")

    try:
        # Dynamically import the specified morph class
        spec = importlib.util.spec_from_file_location(f"morphs.{config.morph}", morph_path)
        if spec is None:
            raise ImportError(f"Could not find spec for morph: {config.morph} at {morph_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"morphs.{config.morph}"] = module # Add to sys.modules for potential relative imports within morph
        spec.loader.exec_module(module)
        MorphClass = module.Morph
    except Exception as e:
        log.exception(f"Failed to load morph '{config.morph}': {e}")
        return {"error": f"Failed to load morph: {e}"}

    # Initialize Warp
    try:
        wp.init()
    except Exception as e:
        log.exception(f"Failed to initialize Warp: {e}")
        return {"error": f"Warp initialization failed: {e}"}

    log.info(f"Warp device: {wp.get_device().name} (CUDA: {wp.get_device().is_cuda})")

    # Instantiate the morph (handles WandB init inside)
    try:
        morph = MorphClass(config)
    except Exception as e:
        log.exception(f"Failed to initialize morph '{config.morph}': {e}")
        # Clean up wandb run if init failed after wandb.init
        if config.track and wandb and wandb.run:
            wandb.finish(exit_code=1)
        return {"error": f"Morph initialization failed: {e}"}


    # --- Simulation Loop ---
    log.info("Starting simulation loop...")
    results = {} # To store final results
    global_step = 0

    try:
        with wp.ScopedDevice(config.device):
            for i in range(config.num_rollouts):
                log.info(f"--- Starting Rollout {i+1}/{config.num_rollouts} ---")

                # --- Update Targets for New Rollout ---
                # ------------------------------------------------------------------
                # Skip the random‑noise target update if we are using fixed poses
                # ------------------------------------------------------------------
                if not morph.config.targets_path:
                    with wp.ScopedTimer("target_update", print=False, active=True, dict=morph.profiler):
                        wp.launch(
                            kernel=update_targets_kernel,
                            dim=morph.num_envs,
                            inputs=[
                                wp.array(morph.target_origin_np, dtype=wp.vec3, device=morph.config.device),
                                morph.initial_arm_orientation,
                                config.target_spawn_pos_noise,
                                config.target_spawn_rot_noise,
                                config.seed + i
                            ],
                            outputs=[morph.targets_wp, morph.target_ori_wp],
                            device=morph.config.device
                        )
                        morph.targets     = morph.targets_wp.numpy()
                        morph.target_ori  = morph.target_ori_wp.numpy()

                # --- Training Iterations within Rollout ---
                for j in range(config.train_iters):
                    # 1. Calculate Error *before* the step for logging
                    ee_error_flat_wp = morph.compute_ee_error()

                    # Calculate error magnitudes on GPU
                    with wp.ScopedTimer("error_mag_kernel", print=False, active=True, dict=morph.profiler):
                        wp.launch(
                            kernel=calculate_error_magnitudes_kernel,
                            dim=morph.num_envs,
                            inputs=[ee_error_flat_wp, morph.num_envs],
                            outputs=[morph.ee_pos_err_mag_wp, morph.ee_ori_err_mag_wp],
                            device=morph.config.device
                        )

                    # Transfer error magnitudes to CPU for logging
                    pos_err_mag = morph.ee_pos_err_mag_wp.numpy()
                    ori_err_mag = morph.ee_ori_err_mag_wp.numpy()

                    # 2. Perform the actual IK step
                    morph.step() # This internally profiles _step

                    # 3. Log metrics to WandB (if enabled)
                    if morph.wandb_run:
                        log_data = {
                            "rollout": i,
                            "train_iter": j,
                            "error/pos/mean": np.mean(pos_err_mag),
                            "error/pos/min": np.min(pos_err_mag),
                            "error/pos/max": np.max(pos_err_mag),
                            "error/ori/mean": np.mean(ori_err_mag),
                            "error/ori/min": np.min(ori_err_mag),
                            "error/ori/max": np.max(ori_err_mag),
                        }
                        # Get the last _step time from the profiler's list
                        if '_step' in morph.profiler and morph.profiler['_step']:
                             log_data["perf/_step_time_ms"] = morph.profiler['_step'][-1]
                        # Log using global_step as the x-axis
                        morph.wandb_run.log(log_data, step=global_step)

                    # 4. Render the frame (optional)
                    morph.render()

                    # 5. Increment global step counter
                    global_step += 1

                log.info(f"--- Finished Rollout {i+1} ---")
                log.info(f"  Final Mean Pos Error: {np.mean(pos_err_mag):.5f}")
                log.info(f"  Final Mean Ori Error: {np.mean(ori_err_mag):.5f}")


        # --- End of Simulation ---
        log.info("Simulation loop completed.")

        # Save the USD rendering
        if not config.headless and morph.renderer is not None:
            try:
                with wp.ScopedTimer("usd_save", print=False, active=True, dict=morph.profiler):
                     morph.renderer.save()
                     log.info(f"USD recording saved to {morph.usd_output_path}")
            except Exception as e:
                log.error(f"Failed to save USD file: {e}")

        # --- Final Results & Profiling Summary ---
        log.info("--- Performance Profile Summary ---")
        total_steps = config.num_rollouts * config.train_iters
        summary_results = {}

        # Calculate average step time and throughput
        avg_step_time_ms = 0.0
        for key, times in morph.profiler.items():
            if not times: continue # Skip empty timers
            avg_time_ms = np.mean(times)
            std_time_ms = np.std(times)
            total_time_s = np.sum(times) / 1000.0
            log.info(f"  {key}: Avg: {avg_time_ms:.3f} ms, Std: {std_time_ms:.3f} ms, Total: {total_time_s:.3f} s")
            summary_results[f"perf/{key}_avg_ms"] = avg_time_ms
            summary_results[f"perf/{key}_std_ms"] = std_time_ms
            summary_results[f"perf/{key}_total_s"] = total_time_s

            # Store average _step time for throughput calculation
            if key == '_step':
                avg_step_time_ms = avg_time_ms

        # Calculate throughput metrics
        if avg_step_time_ms > 0:
            env_steps_per_sec = (1000.0 / avg_step_time_ms) * morph.num_envs
            total_steps_per_sec = (1000.0 / avg_step_time_ms)
            log.info(f"  Throughput:")
            log.info(f"    -> Env*Steps/sec: {env_steps_per_sec:.2f}")
            log.info(f"    -> Steps/sec: {total_steps_per_sec:.2f}")
            summary_results["perf/env_steps_per_sec"] = env_steps_per_sec
            summary_results["perf/steps_per_sec"] = total_steps_per_sec
        else:
            log.warning("Average step time is zero, cannot calculate throughput metrics")
            summary_results["perf/env_steps_per_sec"] = 0.0
            summary_results["perf/steps_per_sec"] = 0.0

        # Calculate accuracy score (higher is better)
        try:
            # Get the final error metrics
            final_pos_error = np.mean(pos_err_mag)
            final_ori_error = np.mean(ori_err_mag)
            # Adding 1 to denominator prevents division by zero and keeps score positive
            denominator = 1.0 + final_pos_error + final_ori_error
            score = 1.0 / denominator if denominator > 1e-6 else 0.0
        except (NameError, ValueError) as e:
            log.warning(f"Could not calculate accuracy score: {e}")
            score = 0.0

        # Store final metrics in results
        results.update({
            "final_pos_error_mean": float(np.mean(pos_err_mag)),
            "final_pos_error_min": float(np.min(pos_err_mag)),
            "final_pos_error_max": float(np.max(pos_err_mag)),
            "final_ori_error_mean": float(np.mean(ori_err_mag)),
            "final_ori_error_min": float(np.min(ori_err_mag)),
            "final_ori_error_max": float(np.max(ori_err_mag)),
            "score": float(score),
            # Add key config parameters
            "config_num_envs": morph.config.num_envs,
            "config_device": str(morph.config.device),
            "config_backend": str(morph.config.backend),
            "config_morph": str(morph.config.morph),
            "total_steps": total_steps,
            "total_env_steps": total_steps * morph.config.num_envs
        })

        # Update wandb summary with all metrics
        if morph.wandb_run:
            wandb_summary_data = {
                "final/pos_error_mean": results["final_pos_error_mean"],
                "final/pos_error_min": results["final_pos_error_min"],
                "final/pos_error_max": results["final_pos_error_max"],
                "final/ori_error_mean": results["final_ori_error_mean"],
                "final/ori_error_min": results["final_ori_error_min"],
                "final/ori_error_max": results["final_ori_error_max"],
                "final/score": results["score"],
                "final/total_steps": results["total_steps"],
                "final/total_env_steps": results["total_env_steps"]
            }
            wandb_summary_data.update(summary_results)
            morph.wandb_run.summary.update(wandb_summary_data)

        # Save results dictionary to file
        results_filepath = os.path.join(config.morph_output_dir, "results.json")
        try:
            with open(results_filepath, 'w') as f:
                # Convert numpy types to native python types for JSON serialization
                serializable_results = {}
                for k, v in results.items():
                    if isinstance(v, (np.float32, np.float64)):
                        serializable_results[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        serializable_results[k] = int(v)
                    elif isinstance(v, np.ndarray):
                        serializable_results[k] = v.tolist()
                    else:
                        serializable_results[k] = v
                json.dump(serializable_results, f, indent=4)
            log.info(f"Results saved to {results_filepath}")
            # Save to wandb as artifact
            if morph.wandb_run:
                results_artifact = wandb.Artifact(f"results_{morph.wandb_run.id}", type="results")
                results_artifact.add_file(results_filepath)
                morph.wandb_run.log_artifact(results_artifact)
        except Exception as e:
            log.error(f"Failed to save results JSON: {e}")

    except Exception as e:
        log.exception(f"Simulation loop failed: {e}")
        results["error"] = f"Simulation loop failed: {e}"
        # Ensure wandb run is finished with error status if it exists
        if config.track and wandb and wandb.run:
             wandb.finish(exit_code=1)
             morph.wandb_run = None # Prevent finishing again in finally
        return results # Return error state
    finally:
        # Ensure wandb run is finished cleanly if simulation completes or fails after init
        if config.track and morph and morph.wandb_run:
            log.info("Finishing WandB run...")
            wandb.finish()

    log.info(f"Simulation complete for morph {config.morph}!")
    log.info(f"Performed {total_steps} total training steps across {config.num_rollouts} rollouts.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    parser.add_argument("--track", action='store_true', help="Turn on tracking with wandb.")
    parser.add_argument("--morph", type=str, default=MorphConfig.morph, help="Unique identifier for the morph.")
    parser.add_argument("--seed", type=int, default=MorphConfig.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default=MorphConfig.device, help="Override default Warp device.")
    parser.add_argument("--num_envs", type=int, default=MorphConfig.num_envs, help="Number of environments to simulate.")
    parser.add_argument("--num_rollouts", type=int, default=MorphConfig.num_rollouts, help="Number of rollouts to perform.")
    parser.add_argument("--train_iters", type=int, default=MorphConfig.train_iters, help="Training iterations per rollout.")
    parser.add_argument("--targets", type=str, default=MorphConfig.targets_path, help="Path to N×7 .npy of fixed 6‑D targets.")
    args = parser.parse_known_args()[0]
    config = MorphConfig(
        morph=args.morph,
        device=args.device,
        headless=args.headless,
        track=args.track,
        seed=args.seed,
        num_envs=args.num_envs,
        num_rollouts=args.num_rollouts,
        train_iters=args.train_iters,
        targets_path=args.targets,
    )
    run_morph(config)