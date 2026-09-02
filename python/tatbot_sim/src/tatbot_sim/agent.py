"""WidowX AI agent variant for tatbot: tattoo pen, two wrist D405s, 7-dim joint control.

Action contract matches the real follower: 7 absolute joint positions
(joint_0..joint_5 + left_carriage_joint) at the env's control frequency. The
upstream ``widowxai`` agent splits arm/gripper into two controller groups and
lists the carriage in both; one flat group here keeps the action layout
identical to ``lerobot_robot_tatbot``.

Geometry comes from :mod:`tatbot_sim.urdf`, which grafts the tattoo pen and the
lower D405 onto the stock ManiSkill asset. Both camera chains now use the real
rig's measured transforms rather than guesses, and the TCP is the needle tip —
so ink deposition and IK both key off the tool that actually touches skin.
"""

import numpy as np
import sapien
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.widowxai.widowxai import WidowXAI
from mani_skill.sensors.camera import CameraConfig

from tatbot_sim.tools import carriage_rest_m, staged_pose
from tatbot_sim.urdf import build_tatbot_urdf

# The carriage is the tool's contact axis (2026-08-30): the tool sits in a
# mount on the left finger carriage, and the carriage moves it along its own
# axis. The real follower holds the carriage in position mode at its rest
# (the selected profile's carriage_rest_m) and only
# the safety layer ever moves it, so the recorded action AND state are both
# the rest value. The sim commands the same: one number, from the same file.
CARRIAGE_REST = carriage_rest_m()
# Kept as aliases for the two names the rest of the sim used to import.
PEN_GRIP = CARRIAGE_REST
GRIP_REST = CARRIAGE_REST


@register_agent(asset_download_ids=["widowxai"])
class TatbotWXAI(WidowXAI):
    uid = "tatbot_wxai"
    urdf_path = build_tatbot_urdf()

    CAM_WIDTH = 640
    CAM_HEIGHT = 480
    # Mounting tolerance, set by the env from the DR config before each
    # scene build: real brackets seat within a millimetre and a degree, so
    # every build draws a fresh small offset per camera. Zero disables.
    CAM_JITTER_POS_M = 0.0
    CAM_JITTER_ROT_RAD = 0.0
    # D405 colour stream is ~70 deg horizontal / ~55 deg vertical; CameraConfig
    # fov is vertical.
    CAM_FOV = 0.96

    # 7-dim action: 6 arm joints + gripper carriage, matching the real follower.
    joint_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "left_carriage_joint",
    ]
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
        return {"pd_joint_pos": {"arm": pd_joint_pos}}

    def _jittered_mount_pose(self) -> sapien.Pose:
        """Small independent draw per camera per scene build (unseeded, like
        lighting: build-time nuisance, not part of the episode seed)."""
        if self.CAM_JITTER_POS_M <= 0 and self.CAM_JITTER_ROT_RAD <= 0:
            return sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        rng = np.random.default_rng()
        dp = rng.uniform(-self.CAM_JITTER_POS_M, self.CAM_JITTER_POS_M, 3)
        rpy = rng.uniform(-self.CAM_JITTER_ROT_RAD, self.CAM_JITTER_ROT_RAD, 3)
        from transforms3d.euler import euler2quat
        return sapien.Pose(p=dp.tolist(), q=euler2quat(*rpy).tolist())

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_upper",
                pose=self._jittered_mount_pose(),
                width=self.CAM_WIDTH,
                height=self.CAM_HEIGHT,
                fov=self.CAM_FOV,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            ),
            CameraConfig(
                uid="wrist_lower",
                pose=self._jittered_mount_pose(),
                width=self.CAM_WIDTH,
                height=self.CAM_HEIGHT,
                fov=self.CAM_FOV,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_lower_link"],
            ),
        ]

    # Fallback pose only. Because the surface height is randomized per episode,
    # the generator solves the real starting pose with IK after reset and sets
    # the arm there — see generate.py. The rest keyframe is the real arm's
    # selected staged pose with the carriage at rest; the derived URDF has one carriage
    # (the right finger is physically removed), so qpos is 7.
    keyframes = {
        "rest": Keyframe(
            qpos=np.array(staged_pose()[:6] + [CARRIAGE_REST]),
            pose=sapien.Pose(),
        )
    }
