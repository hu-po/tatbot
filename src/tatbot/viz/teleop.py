import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tatbot.bot.urdf import get_link_indices, get_link_poses
from tatbot.data.pose import ArmPose, Pose
from tatbot.gen.ik import ik
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger("viz.teleop", "🎮")


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
        link_poses = get_link_poses(
            self.scene.urdf.path, self.scene.urdf.ee_link_names, self.scene.ready_pos_full.joints
        )
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
            arm_l_button_group = self.server.gui.add_button_group("left", ("⏸️", "▶️", "💾"))
            arm_l_pose_text = self.server.gui.add_text(
                "left pose name",
                initial_value="foo_left_pose",
            )
            arm_r_button_group = self.server.gui.add_button_group("right", ("⏸️", "▶️", "💾"))
            arm_r_pose_text = self.server.gui.add_text(
                "right pose name",
                initial_value="foo_right_pose",
            )
            
            # Add pose preset buttons
            with self.server.gui.add_folder("Left Arm Poses"):
                left_ready_button = self.server.gui.add_button("Go to Ready")
                left_sleep_button = self.server.gui.add_button("Go to Sleep")
                left_calibrator_button = self.server.gui.add_button("Go to Calibrator")
                left_save_offset_button = self.server.gui.add_button("Save EE Offset")
                
            with self.server.gui.add_folder("Right Arm Poses"):
                right_ready_button = self.server.gui.add_button("Go to Ready")
                right_sleep_button = self.server.gui.add_button("Go to Sleep")
                right_calibrator_button = self.server.gui.add_button("Go to Calibrator")
                right_save_offset_button = self.server.gui.add_button("Save EE Offset")

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
                self.save_pose("left", self.joints[:7], arm_l_pose_text.value)

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
                self.save_pose("right", self.joints[7:], arm_r_pose_text.value)

        # Left arm pose button handlers
        @left_ready_button.on_click
        def _(_):
            log.debug("🎯 Moving left arm to ready pose")
            self.move_arm_to_pose("left", self.scene.ready_pos_l)

        @left_sleep_button.on_click
        def _(_):
            log.debug("🎯 Moving left arm to sleep pose")
            self.move_arm_to_pose("left", self.scene.sleep_pos_l)

        # Right arm pose button handlers
        @right_ready_button.on_click
        def _(_):
            log.debug("🎯 Moving right arm to ready pose")
            self.move_arm_to_pose("right", self.scene.ready_pos_r)

        @right_sleep_button.on_click
        def _(_):
            log.debug("🎯 Moving right arm to sleep pose")
            self.move_arm_to_pose("right", self.scene.sleep_pos_r)

        # Calibrator button handlers
        @left_calibrator_button.on_click
        def _(_):
            log.debug("🎯 Moving left arm to calibrator position")
            self.move_arm_to_calibrator("left")

        @right_calibrator_button.on_click
        def _(_):
            log.debug("🎯 Moving right arm to calibrator position")
            self.move_arm_to_calibrator("right")

        # Save offset button handlers
        @left_save_offset_button.on_click
        def _(_):
            log.debug("💾 Saving left arm EE offset")
            self.save_ee_offset("left")

        @right_save_offset_button.on_click
        def _(_):
            log.debug("💾 Saving right arm EE offset")
            self.save_ee_offset("right")

    def save_pose(self, arm: str, joints: np.ndarray, name: str):
        pose_path = os.path.join(ArmPose.get_yaml_dir(), arm, f"{name}.yaml")
        log.debug(f"💾 Saving {arm} arm pose to {pose_path}")
        ArmPose(joints=joints).to_yaml(pose_path)

    def move_arm_to_pose(self, arm: str, pose: ArmPose):
        """Move a specific arm to the given pose and update IK targets."""
        # Create a new joints array instead of modifying slices
        new_joints = np.array(self.joints, dtype=np.float32)
        
        if arm == "left":
            # Update left arm joints (first 7 joints)
            new_joints[:7] = pose.joints
            self.joints = new_joints
            # Calculate new end-effector pose for the left arm
            link_poses = get_link_poses(
                self.scene.urdf.path, (self.scene.urdf.ee_link_names[0],), self.joints
            )
            self.ee_l_pose = link_poses[self.scene.urdf.ee_link_names[0]]
            # Update the IK target transform control
            self.ik_target_l.position = self.ee_l_pose.pos.xyz
            self.ik_target_l.wxyz = self.ee_l_pose.rot.wxyz
        elif arm == "right":
            # Update right arm joints (last 7 joints)
            new_joints[7:] = pose.joints
            self.joints = new_joints
            # Calculate new end-effector pose for the right arm
            link_poses = get_link_poses(
                self.scene.urdf.path, (self.scene.urdf.ee_link_names[1],), self.joints
            )
            self.ee_r_pose = link_poses[self.scene.urdf.ee_link_names[1]]
            # Update the IK target transform control
            self.ik_target_r.position = self.ee_r_pose.pos.xyz
            self.ik_target_r.wxyz = self.ee_r_pose.rot.wxyz

    def move_arm_to_calibrator(self, arm: str):
        """Move a specific arm to the calibrator position using IK (without ee_offsets)."""
        # Target position is the calibrator position
        target_pos = self.scene.calibrator_pos.xyz.copy()
        
        if arm == "left":
            # Use left arm end-effector rotation
            target_rot = self.scene.arms.ee_rot_l.wxyz.copy()
            # Set up IK targets - left arm goes to calibrator, right arm stays where it is
            target_positions = np.array([target_pos, self.ee_r_pose.pos.xyz])
            target_rotations = np.array([target_rot, self.ee_r_pose.rot.wxyz])
        elif arm == "right":
            # Use right arm end-effector rotation  
            target_rot = self.scene.arms.ee_rot_r.wxyz.copy()
            # Set up IK targets - left arm stays where it is, right arm goes to calibrator
            target_positions = np.array([self.ee_l_pose.pos.xyz, target_pos])
            target_rotations = np.array([self.ee_l_pose.rot.wxyz, target_rot])
        else:
            log.error(f"Unknown arm: {arm}")
            return
            
        # Run IK to get joint positions
        solution = ik(
            self.robot,
            self.ee_link_indices,
            target_rotations,
            target_positions,
            self.scene.ready_pos_full.joints,
        )
        
        # Update joints (ensure writable)
        self.joints = np.array(solution, dtype=np.float32)
        
        # Update the pose objects and IK targets for the moved arm
        if arm == "left":
            self.ee_l_pose.pos.xyz = target_pos
            self.ee_l_pose.rot.wxyz = target_rot
            self.ik_target_l.position = target_pos
            self.ik_target_l.wxyz = target_rot
        elif arm == "right":
            self.ee_r_pose.pos.xyz = target_pos
            self.ee_r_pose.rot.wxyz = target_rot
            self.ik_target_r.position = target_pos
            self.ik_target_r.wxyz = target_rot

    def save_ee_offset(self, arm: str):
        """Calculate and save the EE offset for the specified arm to default.yaml."""
        # Calculate the offset as the difference between current IK target and calibrator position
        if arm == "left":
            current_pos = self.ik_target_l.position
        elif arm == "right":
            current_pos = self.ik_target_r.position
        else:
            log.error(f"Unknown arm: {arm}")
            return
            
        # Calculate offset
        offset = current_pos - self.scene.calibrator_pos.xyz
        log.info(f"Calculated {arm} arm EE offset: {offset}")
        
        # Load current default.yaml
        arms_config_path = Path(__file__).parent.parent.parent / "conf" / "arms" / "default.yaml"
        
        try:
            with open(arms_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update the appropriate offset
            if arm == "left":
                config['ee_offset_l']['xyz'] = offset.tolist()
            elif arm == "right":
                config['ee_offset_r']['xyz'] = offset.tolist()
            
            # Save back to file
            with open(arms_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            log.info(f"💾 Saved {arm} arm EE offset {offset} to {arms_config_path}")
            
        except Exception as e:
            log.error(f"Failed to save EE offset: {e}")

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
            self.scene.ready_pos_full.joints,
        )
        log.debug(f"🎯 left joints: {solution[:7]}")
        log.debug(f"🎯 right joints: {solution[7:]}")
        self.joints = np.array(solution, dtype=np.float32)


if __name__ == "__main__":
    args = setup_log_with_config(TeleopVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    viz = TeleopViz(args)
    viz.run()
