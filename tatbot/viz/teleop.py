import logging
from dataclasses import dataclass
import os

import numpy as np

from tatbot.gen.ik import ik
from tatbot.bot.urdf import get_link_indices, get_link_poses
from tatbot.viz.base import BaseViz, BaseVizConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.data.pose import Pose, ArmPose

log = get_logger('viz.base', '🖥️')

@dataclass
class TeleopVizConfig(BaseVizConfig):
    debug: bool = False
    """Enable debug logging."""

    transform_control_scale: float = 0.2
    """Scale of the transform control frames for visualization."""
    transform_control_opacity: float = 0.2
    """Opacity of the transform control frames for visualization."""

class TeleopViz(BaseViz):
    def __init__(self, config: TeleopVizConfig):
        super().__init__(config)

        log.info("🎯 Adding ee ik targets...")
        self.ee_link_indices = get_link_indices(self.scene.urdf.path, self.scene.urdf.ee_link_names)
        link_poses = get_link_poses(self.scene.urdf.path, self.scene.urdf.ee_link_names, self.scene.ready_pos_full)
        self.ee_l_pose: Pose = link_poses[self.scene.urdf.ee_link_names[0]]
        self.ee_r_pose: Pose = link_poses[self.scene.urdf.ee_link_names[1]]
        self.ik_target_l = self.server.scene.add_transform_controls(
            "/ik_target_l",
            position=self.ee_l_pose.pos.xyz,
            wxyz=self.ee_l_pose.rot.wxyz,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        self.ik_target_r = self.server.scene.add_transform_controls(
            "/ik_target_r",
            position=self.ee_r_pose.pos.xyz,
            wxyz=self.ee_r_pose.rot.wxyz,
            scale=config.transform_control_scale,
            opacity=config.transform_control_opacity,
        )
        self.arm_l_ik_toggle: bool = True
        self.arm_r_ik_toggle: bool = True
        with self.server.gui.add_folder("Teleop"):
            arm_l_button_group = self.server.gui.add_button_group(
                "left", ("⏸️", "▶️", "💾")
            )
            arm_l_pose_text = self.server.gui.add_text(
                "left pose name",
                initial_value="foo_left_pose",
            )
            arm_r_button_group = self.server.gui.add_button_group(
                "right", ("⏸️", "▶️", "💾")
            )
            arm_r_pose_text = self.server.gui.add_text(
                "right pose name",
                initial_value="foo_right_pose",
            )

        @arm_l_button_group.on_click
        def _(_):
            if arm_l_button_group.value == "⏸️":
                log.debug("⏸️ Pause left arm")
                self.arm_l_ik_toggle = False
            elif arm_l_button_group.value == "▶️":
                log.debug("▶️ Play left arm")
                self.arm_l_ik_toggle = True
            elif arm_l_button_group.value == "💾":
                log.debug("💾 Save left arm")
                self.save_pose("left", self.joints[:8], arm_l_pose_text.value)

        @arm_r_button_group.on_click
        def _(_):
            if arm_r_button_group.value == "⏸️":
                log.debug("⏸️ Pause right arm")
                self.arm_r_ik_toggle = False
            elif arm_r_button_group.value == "▶️":
                log.debug("▶️ Play right arm")
                self.arm_r_ik_toggle = True
            elif arm_r_button_group.value == "💾":
                log.debug("💾 Save right arm")
                self.save_pose("right", self.joints[8:], arm_r_pose_text.value)

    def save_pose(self, arm: str, joints: np.ndarray, name: str):
        pose_path = os.path.join(ArmPose.get_yaml_dir(), arm, f"{name}.yaml")
        log.debug(f"💾 Saving {arm} arm pose to {pose_path}")
        ArmPose(joints=joints).to_yaml(pose_path)

    def step(self):
        if self.arm_l_ik_toggle:
            self.ee_l_pose.pos.xyz = self.ik_target_l.position.copy()
            self.ee_l_pose.rot.wxyz = self.ik_target_l.wxyz.copy()
        if self.arm_r_ik_toggle:
            self.ee_r_pose.pos.xyz = self.ik_target_r.position.copy()
            self.ee_r_pose.rot.wxyz = self.ik_target_r.wxyz.copy()
        log.debug(f"🎯 ee_l_pose: {self.ee_l_pose}")
        log.debug(f"🎯 ee_r_pose: {self.ee_r_pose}")
        solution = ik(
            self.robot,
            self.ee_link_indices,
            np.array([self.ee_l_pose.rot.wxyz, self.ee_r_pose.rot.wxyz]),
            np.array([self.ee_l_pose.pos.xyz, self.ee_r_pose.pos.xyz]),
            self.scene.ready_pos_full,
        )
        log.debug(f"🎯 left joints: {solution[:8]}")
        log.debug(f"🎯 right joints: {solution[8:]}")
        self.joints = np.asarray(solution, dtype=np.float32)

if __name__ == "__main__":
    args = setup_log_with_config(TeleopVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = TeleopViz(args)
    viz.run()