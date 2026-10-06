"""WidowX AI agent with registry-resolved wrist views and seven-axis control.

Action contract matches the real follower: 7 absolute joint positions
(joint_0..joint_5 + left_carriage_joint) at the env's control frequency. The
upstream ``widowxai`` agent splits arm/gripper into two controller groups and
lists the carriage in both; one flat group here keeps the action layout
identical to ``lerobot_robot_tatbot``.

Geometry comes from :mod:`tatbot_sim.urdf`; camera poses come from the canonical
robot's optical frames. Historical two-view geometry is explicitly selected.
"""

from functools import cached_property

import numpy as np
import sapien
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosVelControllerConfig
from mani_skill.agents.robots.widowxai.widowxai import WidowXAI
from mani_skill.sensors.camera import CameraConfig
from tatbot_contracts.observations import FOLLOWER_JOINTS
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat
from wrist_cameras import fixed_chain

from tatbot_sim.repo import repo_root
from tatbot_sim.resolved import ResolvedConfig
from tatbot_sim.urdf import build_tatbot_urdf


class TatbotWXAI(WidowXAI):
    uid = "tatbot_wxai"
    def __init__(self, *args, config: ResolvedConfig, camera_seed: int | None = None, **kwargs):
        self.config = config
        self.sensor_profile = config.sensor_profile
        self.camera_descriptions = config.cameras
        self.urdf_path = build_tatbot_urdf(config=config)
        self.camera_dr = config.dr.camera
        self.camera_seed = config.seed_for('camera') if camera_seed is None else camera_seed
        self._camera_rng = np.random.default_rng(self.camera_seed)
        self.camera_mounts = {}
        self.keyframes = {'rest': Keyframe(
            qpos=np.array([*config.staged_pose[:6], config.carriage_rest_m]),
            pose=sapien.Pose(),
        )}
        super().__init__(*args, **kwargs)

    # 7-dim action: 6 arm joints + gripper carriage, matching the real follower.
    joint_names = list(FOLLOWER_JOINTS)
    # The needle tip is the tool that touches skin, so it is the TCP for both
    # IK targeting and ink deposition.
    ee_link_name = "tattoo_needle"

    @property
    def _controller_configs(self):
        # The carriage is position-held at rest like every other joint: the
        # real follower runs it in position mode, and the tool no longer
        # stops the fingers (nothing is gripped), so there is no reason for
        # it to float. Same stiffness as the arm.
        pd_joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            lower=None,
            upper=None,
            stiffness=[self.arm_stiffness] * 7,
            damping=[self.arm_damping] * 7,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.joint_names,
            lower=None,
            upper=None,
            stiffness=[self.arm_stiffness] * 7,
            damping=[self.arm_damping] * 7,
            force_limit=self.arm_force_limit,
            normalize_action=False,
            # The production worker validates command velocities against its
            # carried limits. This engine adapter must not rescale or clip the
            # accepted feed-forward references to a second set of limits.
            vel_lower=-float('inf'),
            vel_upper=float('inf'),
        )
        return {"pd_joint_pos": {"arm": pd_joint_pos},
                "pd_joint_pos_vel": {"arm": pd_joint_pos_vel}}

    def _jittered_mount_pose(self) -> sapien.Pose:
        """Independent, reproducible mount nuisance for this world."""
        position_m = self.camera_dr.mount_jitter_mm / 1000.0
        rotation_rad = float(np.radians(self.camera_dr.mount_jitter_deg))
        dp = self._camera_rng.uniform(-position_m, position_m, 3)
        rpy = self._camera_rng.uniform(-rotation_rad, rotation_rad, 3)
        from transforms3d.euler import euler2quat
        return sapien.Pose(p=dp.tolist(), q=euler2quat(*rpy).tolist())

    @cached_property
    def _sensor_configs(self):
        from pathlib import Path

        configs = []
        # SAPIEN camera axes are forward/left/up; the registry uses optical
        # right/down/forward axes. Keep that conversion inside this backend.
        optical_from_sapien = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        for camera in self.camera_descriptions:
            if camera.arm != 'right':
                raise ValueError('follower environment cannot mount another arm camera')
            if self.sensor_profile == 'legacy-two-view':
                origins = fixed_chain(repo_root(), 'link_6', camera.optical_frame.removeprefix('right/'),
                                      urdf=Path(self.urdf_path))
            else:
                origins = fixed_chain(repo_root(), 'right/link_6', camera.optical_frame)
            transform = np.eye(4)
            for xyz, rpy in origins:
                local = np.eye(4)
                local[:3, :3] = euler2mat(*rpy)
                local[:3, 3] = xyz
                transform = transform @ local
            pose = sapien.Pose(p=transform[:3, 3], q=mat2quat(transform[:3, :3] @ optical_from_sapien))
            fx, fy, cx, cy = camera.intrinsic
            pose = pose * self._jittered_mount_pose()
            self.camera_mounts[camera.role] = {"position_m": pose.p.tolist(), "quaternion_wxyz": pose.q.tolist()}
            configs.append(CameraConfig(
                uid=camera.role, pose=pose,
                width=camera.width, height=camera.height,
                intrinsic=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                near=0.01, far=100, mount=self.robot.links_map['link_6'],
            ))
        return configs
