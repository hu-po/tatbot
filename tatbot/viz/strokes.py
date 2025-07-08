import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.align import make_align_strokes
from tatbot.gen.svg import make_svg_strokes
from tatbot.gen.strokebatch import strokebatch_from_strokes
from tatbot.utils.colors import COLORS
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.viz.base import BaseViz, BaseVizConfig

log = get_logger('viz.plan', '🖥️')

@dataclass
class VizStrokesConfig(BaseVizConfig):
    design_pointcloud_point_size: float = 0.001
    """Size of points in the point cloud visualization (meters)."""
    design_pointcloud_point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""

    path_highlight_radius: int = 3
    """Radius of the path highlight in pixels."""
    pose_highlight_radius: int = 6
    """Radius of the pose highlight in pixels."""

class VizStrokes(BaseViz):
    def __init__(self, config: VizStrokesConfig):
        super().__init__(config)

        if self.scene.design_dir_path is not None:
            self.strokelist: StrokeList = make_svg_strokes(self.scene)
        else:
            self.strokelist: StrokeList = make_align_strokes(self.scene)
        self.strokebatch: StrokeBatch = strokebatch_from_strokes(
            strokelist=self.strokelist,
            stroke_length=self.scene.stroke_length,
            joints=self.scene.ready_pos_full,
            urdf_path=self.scene.urdf.path,
            link_names=self.scene.urdf.ee_link_names,
            design_pose=self.scene.skin.design_pose,
            needle_hover_offset=self.scene.needle_hover_offset,
            needle_offset_l=self.scene.needle_offset_l,
            needle_offset_r=self.scene.needle_offset_r,
        )
        self.num_strokes = len(self.strokelist.strokes)
        self.stroke_idx = 0
        self.pose_idx = 0

        with self.server.gui.add_folder("Session"):
            def _format_seconds(secs):
                secs = int(secs)
                h = secs // 3600
                m = (secs % 3600) // 60
                s = secs % 60
                return f"{h:02}h{m:02}m{s:02}s"

            self.time_label = self.server.gui.add_text(
                label="time",
                initial_value=f"{_format_seconds(0)} / {_format_seconds(0)}",
                disabled=True,
            )

            def _update_time_label():
                current_time = 0.0
                for i in range(self.stroke_idx):
                    current_time += self.strokebatch.dt[i, :].sum().item()
                current_time += self.strokebatch.dt[self.stroke_idx, :self.pose_idx+1].sum().item()
                total_time = self.strokebatch.dt.sum().item()
                self.time_label.value = f"{_format_seconds(current_time)} / {_format_seconds(total_time)}"

            self.stroke_idx_slider = self.server.gui.add_slider(
                "stroke",
                min=0,
                max=self.num_strokes - 1,
                step=1,
                initial_value=0,
            )
            self.stroke_description_l = self.server.gui.add_text(
                label="left arm description",
                initial_value=self.strokelist.strokes[0][0].description,
                disabled=True,
            )
            self.stroke_description_r = self.server.gui.add_text(
                label="right arm description",
                initial_value=self.strokelist.strokes[0][1].description,
                disabled=True,
            )
            self.pose_idx_slider = self.server.gui.add_slider(
                "pose",
                min=0,
                max=self.scene.stroke_length - 1,
                step=1,
                initial_value=0,
            )
        

        @self.stroke_idx_slider.on_update
        def _(_):
            self.stroke_idx = self.stroke_idx_slider.value
            self.pose_idx_slider.max = self.scene.stroke_length - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max
            _update_time_label()
            self.stroke_description_l.value = self.strokelist.strokes[self.stroke_idx][0].description
            self.stroke_description_r.value = self.strokelist.strokes[self.stroke_idx][1].description

        @self.pose_idx_slider.on_update
        def _(_):
            self.pose_idx = self.pose_idx_slider.value
            _update_time_label()

        _update_time_label()

        if self.scene.design_img_path is not None:
            self.design_img_np = cv2.imread(self.scene.design_img_path)
            self.design_img_gui = self.server.gui.add_image(image=self.design_img_np, format="png")
            self.frame_img_gui = self.server.gui.add_image(image=self.frame_img_np, format="png")
            log.debug(f"🖼️ Adding 2D image (design frame)...")
            self.server.scene.add_image(
                name="/design",
                image=self.design_img_np,
                wxyz=self.skin.design_pose.rot.wxyz,
                position=self.skin.design_pose.pos.xyz,
                render_width=self.scene.skin.image_width_m,
                render_height=self.scene.skin.image_height_m,
            )

        log.debug("Adding pointcloud")
        points_l = [] # pointcloud for left arm stroke
        points_r = [] # pointcloud for right arm stroke
        for i in range(self.strokebatch.ee_pos_l.shape[0]):
            for j in range(self.strokebatch.ee_pos_l.shape[1]):
                points_l.append(self.strokebatch.ee_pos_l[i, j])
                points_r.append(self.strokebatch.ee_pos_r[i, j])
        points_l = np.stack(points_l, axis=0)
        points_r = np.stack(points_r, axis=0)
        self.point_colors_stroke_l = np.tile(np.array(COLORS["blue"], dtype=np.uint8), (points_l.shape[0], 1))
        self.point_colors_stroke_r = np.tile(np.array(COLORS["purple"], dtype=np.uint8), (points_r.shape[0], 1))

        self.pointcloud_path_l = self.server.scene.add_point_cloud(
            name="/points_stroke_l",
            points=points_l,
            colors=self.point_colors_stroke_l,
            point_size=self.config.design_pointcloud_point_size,
            point_shape=self.config.design_pointcloud_point_shape,
        )
        self.pointcloud_path_r = self.server.scene.add_point_cloud(
            name="/points_stroke_r",
            points=points_r,
            colors=self.point_colors_stroke_r,
            point_size=self.config.design_pointcloud_point_size,
            point_shape=self.config.design_pointcloud_point_shape,
        )

        # robot is reset to rest pose at start of every stroke
        self.robot_at_rest: bool = True

    def step(self):
        if self.robot_at_rest:
            log.debug("Robot at rest, skipping step...")
            self.robot_at_rest = False
            return
        if self.pose_idx >= self.scene.stroke_length:
            self.stroke_idx += 1
            self.pose_idx = 0
            log.debug(f"Moving to next stroke {self.stroke_idx}")
        if self.stroke_idx >= self.num_strokes:
            log.debug(f"Looping back to stroke 0")
            self.stroke_idx = 0
            self.pose_idx = 0
        self.stroke_idx_slider.value = self.stroke_idx
        self.pose_idx_slider.value = self.pose_idx
        log.debug(f"Visualizing stroke {self.stroke_idx} pose {self.pose_idx}")

        log.debug("Updating pointclouds")
        for stroke in self.strokelist.strokes[self.stroke_idx]:
            if stroke.arm == "left":
                points_color_l = self.point_colors_stroke_l.copy()
                points_color_l[self.stroke_idx * self.scene.stroke_length:self.stroke_idx * self.scene.stroke_length + self.pose_idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_l[self.stroke_idx * self.scene.stroke_length + self.pose_idx] = np.array(COLORS["blue"], dtype=np.uint8)
                self.pointcloud_path_l.colors = points_color_l
            else:
                points_color_r = self.point_colors_stroke_r.copy()
                points_color_r[self.stroke_idx * self.scene.stroke_length:self.stroke_idx * self.scene.stroke_length + self.pose_idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_r[self.stroke_idx * self.scene.stroke_length + self.pose_idx] = np.array(COLORS["purple"], dtype=np.uint8)
                self.pointcloud_path_r.colors = points_color_r
        if self.scene.design_img_path is not None:
            log.debug("Updating design image")
            image_np = self.design_img_np.copy()
            for stroke in self.strokelist.strokes[self.stroke_idx]:
                if stroke.pixel_coords is not None and not stroke.is_inkdip:
                    # Highlight entire path in red
                    for pw, ph in stroke.pixel_coords:
                        cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["red"], -1)
                    # Highlight path up until current pose in orange
                    for pw, ph in stroke.pixel_coords[:self.pose_idx]:
                        cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["orange"], -1)
                    color = COLORS["blue"] if stroke.arm == "left" else COLORS["purple"]
                    # Highlight current pose
                    px, py = stroke.pixel_coords[self.pose_idx - 2] # -2 because of hover poses at start and end
                    cv2.circle(image_np, (int(px), int(py)), self.config.pose_highlight_radius, color, -1)
                    # Add L or R text
                    text = "L" if stroke.arm == "left" else "R"
                    text_pos = (int(px) - 5, int(py) - self.config.pose_highlight_radius - 5)
                    cv2.putText(image_np, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if stroke.is_inkdip:
                    # Text indicating arm that is inkdipping
                    text = f"{stroke.arm} inkdip {stroke.inkcap}"
                    text_pos = (10, 10)
                    cv2.putText(image_np, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, COLORS["black"], 1)
            self.design_img_gui.image = image_np

        if self.pose_idx == 0:
            log.debug("Sending robot to rest pose")
            self.robot_at_rest = True
            self.joints = self.scene.ready_pos_full
            self.step_sleep = self.scene.arms.goal_time_slow
        else:
            self.joints = np.asarray(self.strokebatch.joints[self.stroke_idx, self.pose_idx], dtype=np.float64).flatten()
            self.step_sleep = float(self.strokebatch.dt[self.stroke_idx, self.pose_idx].item())
        self.pose_idx += 1

if __name__ == "__main__":
    args = setup_log_with_config(VizStrokesConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = VizStrokes(args)
    viz.run()