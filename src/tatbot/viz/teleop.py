import logging
import os
from dataclasses import dataclass
from pathlib import Path

import jaxlie
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
        self.ee_link_indices, _ = get_link_indices(self.scene.urdf.path, self.scene.urdf.ee_link_names)
        link_poses = get_link_poses(
            self.scene.urdf.path, self.scene.urdf.ee_link_names, self.scene.ready_pos_full.joints
        )
        self.ee_l_pose: Pose = link_poses[self.scene.urdf.ee_link_names[0]]
        self.ee_r_pose: Pose = link_poses[self.scene.urdf.ee_link_names[1]]
        
        # Add lasercross transform control
        self.lasercross_tf = self.server.scene.add_transform_controls(
            "/lasercross",
            position=self.scene.lasercross_pose.pos.xyz,
            wxyz=self.scene.lasercross_pose.rot.wxyz,
            scale=config.transform_control_scale * 0.5,  # Make it slightly smaller than EE controls
            opacity=config.transform_control_opacity,
        )
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
            
            # Add lasercross controls
            with self.server.gui.add_folder("Lasercross"):
                lasercross_move_button = self.server.gui.add_button("Move Arms to Lasercross")
                lasercross_save_button = self.server.gui.add_button("Save Lasercross to URDF")
                lasercross_text = self.server.gui.add_text(
                    "lasercross",
                    initial_value=self.get_lasercross_text(),
                    disabled=True,
                )
            
            # Add emergency stop for robot safety
            if config.enable_robot:
                with self.server.gui.add_folder("🚨 Safety Controls"):
                    emergency_stop_button = self.server.gui.add_button("🛑 EMERGENCY STOP", color="red")

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
        
        # Lasercross update handler
        @self.lasercross_tf.on_update
        def _(_):
            lasercross_text.value = self.get_lasercross_text()
        
        # Lasercross move handler
        @lasercross_move_button.on_click
        def _(_):
            log.debug("🎯 Moving arms to lasercross position")
            self.move_arms_to_lasercross()
        
        # Lasercross save handler
        @lasercross_save_button.on_click
        def _(_):
            log.debug("💾 Saving lasercross to URDF")
            self.save_lasercross_to_urdf()
        
        # Emergency stop handler for robot safety
        if config.enable_robot:
            @emergency_stop_button.on_click
            def _(_):
                log.warning("🛑 EMERGENCY STOP activated!")
                self.emergency_stop()

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

        # Re-enable and show both IK targets when returning to ready/sleep poses
        self.ik_target_l.visible = True
        self.ik_target_r.visible = True

    def move_arm_to_calibrator(self, arm: str):
        """Move a specific arm to the calibrator position using IK (without ee_offsets)."""
        # Target position is the calibrator position
        target_pos = self.scene.calibrator_pose.pos.xyz.copy()
        
        # Get calibrator transformation for orientation calculations
        calibrator_tf = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(self.scene.calibrator_pose.rot.wxyz), 
            self.scene.calibrator_pose.pos.xyz
        )
        
        if arm == "left":
            # Use calibrator-relative left arm end-effector rotation
            base_ee_rot_l = jaxlie.SO3(self.scene.arms.ee_rot_l.wxyz)
            calibrator_relative_ee_rot_l = calibrator_tf.rotation() @ base_ee_rot_l
            target_rot = calibrator_relative_ee_rot_l.wxyz.copy()
            # Set up IK targets - left arm goes to calibrator, right arm stays where it is
            target_positions = np.array([target_pos, self.ee_r_pose.pos.xyz])
            target_rotations = np.array([target_rot, self.ee_r_pose.rot.wxyz])
        elif arm == "right":
            # Use calibrator-relative right arm end-effector rotation  
            base_ee_rot_r = jaxlie.SO3(self.scene.arms.ee_rot_r.wxyz)
            calibrator_relative_ee_rot_r = calibrator_tf.rotation() @ base_ee_rot_r
            target_rot = calibrator_relative_ee_rot_r.wxyz.copy()
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
            # Ensure active arm control is visible and disable opposite arm control
            self.ik_target_l.visible = True
            self.ik_target_r.visible = False
            self.arm_r_ik_toggle = False
        elif arm == "right":
            self.ee_r_pose.pos.xyz = target_pos
            self.ee_r_pose.rot.wxyz = target_rot
            self.ik_target_r.position = target_pos
            self.ik_target_r.wxyz = target_rot
            # Ensure active arm control is visible and disable opposite arm control
            self.ik_target_r.visible = True
            self.ik_target_l.visible = False
            self.arm_l_ik_toggle = False

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
        offset = current_pos - self.scene.calibrator_pose.pos.xyz
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

    def move_arms_to_lasercross(self):
        """Move both arms to lasercross position with +/- Y offset similar to align.py."""
        # Calculate half length from lasercross_len_mm (convert mm to meters and divide by 2)
        lasercross_halflen_m = self.scene.arms.lasercross_len_mm / 2000
        
        # Get the current lasercross position from the transform control
        lasercross_pos = self.lasercross_tf.position.copy()
        
        # Calculate target positions: left arm gets +X offset, right arm gets -X offset
        # Transform the X-axis offsets according to lasercross orientation
        lasercross_rotation = jaxlie.SO3(self.lasercross_tf.wxyz)
        
        # Define offsets in lasercross local frame (X-axis)
        left_offset_local = np.array([lasercross_halflen_m, 0, 0])
        right_offset_local = np.array([-lasercross_halflen_m, 0, 0])
        
        # Transform offsets to world frame using lasercross rotation
        left_offset_world = lasercross_rotation @ left_offset_local
        right_offset_world = lasercross_rotation @ right_offset_local
        
        # Apply transformed offsets to lasercross position
        left_target_pos = lasercross_pos + left_offset_world
        right_target_pos = lasercross_pos + right_offset_world
        
        # Get lasercross transformation for orientation calculations
        lasercross_tf_obj = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(self.lasercross_tf.wxyz), 
            self.lasercross_tf.position
        )
        
        # Use lasercross-relative end-effector rotations
        base_ee_rot_l = jaxlie.SO3(self.scene.arms.ee_rot_l.wxyz)
        base_ee_rot_r = jaxlie.SO3(self.scene.arms.ee_rot_r.wxyz)
        
        lasercross_relative_ee_rot_l = lasercross_tf_obj.rotation() @ base_ee_rot_l
        lasercross_relative_ee_rot_r = lasercross_tf_obj.rotation() @ base_ee_rot_r
        
        left_target_rot = lasercross_relative_ee_rot_l.wxyz.copy()
        right_target_rot = lasercross_relative_ee_rot_r.wxyz.copy()
        
        # Set up IK targets for both arms
        target_positions = np.array([left_target_pos, right_target_pos])
        target_rotations = np.array([left_target_rot, right_target_rot])
        
        # Run IK to get joint positions
        solution = ik(
            self.robot,
            self.ee_link_indices,
            target_rotations,
            target_positions,
            self.scene.ready_pos_full.joints,
        )
        
        # Update joints
        self.joints = np.array(solution, dtype=np.float32)
        
        # Update the pose objects and IK targets for both arms
        self.ee_l_pose.pos.xyz = left_target_pos
        self.ee_l_pose.rot.wxyz = left_target_rot
        self.ik_target_l.position = left_target_pos
        self.ik_target_l.wxyz = left_target_rot
        
        self.ee_r_pose.pos.xyz = right_target_pos
        self.ee_r_pose.rot.wxyz = right_target_rot
        self.ik_target_r.position = right_target_pos
        self.ik_target_r.wxyz = right_target_rot
        
        # Turn off IK target visibility and disable IK updates
        self.ik_target_l.visible = False
        self.ik_target_r.visible = False
        self.arm_l_ik_toggle = False
        self.arm_r_ik_toggle = False
        
        log.info(f"🎯 Moved arms to lasercross: left at {left_target_pos}, right at {right_target_pos}")

    def emergency_stop(self):
        """Emergency stop - immediately halt robot arms and disable IK updates."""
        log.warning("🛑 Emergency stop activated - disabling arms and IK updates")
        
        # Stop both arms immediately
        self.arm_l_ik_toggle = False
        self.arm_r_ik_toggle = False
        
        # If robot hardware is connected, send stop commands
        if hasattr(self, 'arm_l') and self.arm_l is not None:
            try:
                log.warning("🛑 Stopping left arm")
                # Stop arm movement immediately
                current_joints_l = self.arm_l.get_joint_positions()
                self.arm_l.set_joint_positions(current_joints_l)
            except Exception as e:
                log.error(f"Failed to stop left arm: {e}")
        
        if hasattr(self, 'arm_r') and self.arm_r is not None:
            try:
                log.warning("🛑 Stopping right arm")
                # Stop arm movement immediately  
                current_joints_r = self.arm_r.get_joint_positions()
                self.arm_r.set_joint_positions(current_joints_r)
            except Exception as e:
                log.error(f"Failed to stop right arm: {e}")
        
        log.warning("🛑 Emergency stop complete - manually re-enable arms using ▶️ buttons if safe")

    def get_lasercross_text(self) -> str:
        """Get the current lasercross pose as formatted text."""
        pos = self.lasercross_tf.position
        rot = self.lasercross_tf.wxyz
        return f"xyz: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\nwxyz: [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}, {rot[3]:.3f}]"
    
    def save_lasercross_to_urdf(self):
        """Save the current lasercross pose back to the URDF file."""
        # Get current position and rotation from the transform control
        pos = self.lasercross_tf.position
        wxyz = self.lasercross_tf.wxyz
        
        # Convert quaternion to RPY using jaxlie
        quat = jaxlie.SO3(wxyz=wxyz)
        roll, pitch, yaw = quat.as_rpy_radians()
        
        # Read the URDF file as text to preserve formatting
        urdf_path = Path(__file__).parent.parent.parent.parent / "urdf" / "tatbot.urdf"
        urdf_content = urdf_path.read_text()
        
        # Find the lasercross_joint section and locate its origin line
        lines = urdf_content.split('\n')
        in_lasercross_joint = False
        found_origin = False
        
        for i, line in enumerate(lines):
            if 'name="lasercross_joint"' in line:
                in_lasercross_joint = True
                continue
            
            if in_lasercross_joint and '</joint>' in line:
                in_lasercross_joint = False
                break
                
            if in_lasercross_joint and '<origin' in line and 'rpy=' in line and 'xyz=' in line:
                # Found the origin line - replace it while preserving indentation
                indent = line[:len(line) - len(line.lstrip())]
                new_origin = f'{indent}<origin rpy="{roll:.6f} {pitch:.6f} {yaw:.6f}" xyz="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" />'
                lines[i] = new_origin
                found_origin = True
                break
        
        if found_origin:
            new_content = '\n'.join(lines)
            urdf_path.write_text(new_content)
            log.info(f"💾 Saved lasercross to URDF: xyz=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], rpy=[{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        else:
            log.error("Could not find lasercross_joint origin in URDF file")


if __name__ == "__main__":
    args = setup_log_with_config(TeleopVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args, log)
    viz = TeleopViz(args)
    viz.run()
