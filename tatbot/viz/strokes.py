import logging
from dataclasses import dataclass
import time

import cv2
import numpy as np

from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.align import make_align_strokes
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.gen.batch import strokebatch_from_strokes
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
            self.strokelist: StrokeList = make_gcode_strokes(self.scene)
        else:
            self.strokelist: StrokeList = make_align_strokes(self.scene)
        self.strokebatch: StrokeBatch = strokebatch_from_strokes(self.scene, self.strokelist)
        self.num_strokes = len(self.strokelist.strokes)
        self.stroke_idx = 0
        self.pose_idx = 0

        with self.server.gui.add_folder("Session"):
            self.step_sleep = 1.0 / 30.0 # 30 fps
            self.speed_slider = self.server.gui.add_slider(
                "speed",
                min=0.1,
                max=100.0,
                step=0.1,
                initial_value=self.config.speed,
            )
            self.step_button_group = self.server.gui.add_button_group("", ("⏸️", "▶️"))
            self.step_button_group.value = "▶️"
            self.pause: bool = False

        @self.step_button_group.on_click
        def _(_):
            if self.step_button_group.value == "⏸️":
                log.debug("⏸️ Pause")
                self.pause = True
            elif self.step_button_group.value == "▶️":
                log.debug("▶️ Play")
                self.pause = False

        with self.server.gui.add_folder("Strokes"):
            self.stroke_idx_slider = self.server.gui.add_slider(
                "stroke",
                min=0,
                max=self.num_strokes - 1,
                step=1,
                initial_value=0,
            )
            self.stroke_description_l = self.server.gui.add_text(
                label="desc_l",
                initial_value=self.strokelist.strokes[0][0].description,
                multiline=True,
                disabled=True,
            )
            self.stroke_description_r = self.server.gui.add_text(
                label="desc_r",
                initial_value=self.strokelist.strokes[0][1].description,
                multiline=True,
                disabled=True,
            )
            self.pose_idx_slider = self.server.gui.add_slider(
                "pose",
                min=0,
                max=self.scene.stroke_length - 1,
                step=1,
                initial_value=0,
            )
            self.offset_idx_slider = self.server.gui.add_slider(
                "offset",
                min=0,
                max=self.scene.offset_num - 1,
                step=1,
                initial_value=0,
            )
        

        @self.stroke_idx_slider.on_update
        def _(_):
            self.stroke_idx = self.stroke_idx_slider.value
            self.pose_idx_slider.max = self.scene.stroke_length - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max
            self.stroke_description_l.value = self.strokelist.strokes[self.stroke_idx][0].description
            self.stroke_description_r.value = self.strokelist.strokes[self.stroke_idx][1].description

        @self.pose_idx_slider.on_update
        def _(_):
            self.pose_idx = self.pose_idx_slider.value

        if self.scene.design_img_path is not None:
            self.design_img_np = cv2.imread(self.scene.design_img_path)
            self.design_img_gui = self.server.gui.add_image(image=self.design_img_np, format="png")
            self.frame_img_gui = self.server.gui.add_image(image=self.design_img_np, format="png")
            # # 2d image plane of design image in 3d space
            # self.server.scene.add_image(
            #     name="/design",
            #     image=self.design_img_np,
            #     wxyz=self.scene.skin.design_pose.rot.wxyz,
            #     position=self.scene.skin.design_pose.pos.xyz,
            #     render_width=self.scene.skin.image_width_m,
            #     render_height=self.scene.skin.image_height_m,
            # )

        log.debug("Adding pointcloud")
        points_l = [] # pointcloud for left arm stroke
        points_r = [] # pointcloud for right arm stroke
        for i in range(self.strokebatch.ee_pos_l.shape[0]):
            for j in range(self.strokebatch.ee_pos_l.shape[1]):
                for k in range(self.scene.offset_num):
                    points_l.append(self.strokebatch.ee_pos_l[i, j, k])
                    points_r.append(self.strokebatch.ee_pos_r[i, j, k])
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
        if self.pause:
            return
        time.sleep(self.step_sleep / self.speed_slider.value)
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
                base = self.stroke_idx * self.scene.stroke_length * self.scene.offset_num
                idx  = base + self.pose_idx * self.scene.offset_num + self.offset_idx_slider.value
                points_color_l[base : idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_l[idx]            = np.array(COLORS["blue"], dtype=np.uint8)
                self.pointcloud_path_l.colors = points_color_l
            else:
                points_color_r = self.point_colors_stroke_r.copy()
                base = self.stroke_idx * self.scene.stroke_length * self.scene.offset_num
                idx  = base + self.pose_idx * self.scene.offset_num + self.offset_idx_slider.value
                points_color_r[base : idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_r[idx]            = np.array(COLORS["purple"], dtype=np.uint8)
                self.pointcloud_path_r.colors = points_color_r
        if self.scene.design_img_path is not None:
            log.debug("Updating design image")
            for stroke in self.strokelist.strokes[self.stroke_idx]:
                if stroke.frame_path is not None:
                    self.frame_img_gui.image = cv2.imread(stroke.frame_path)

        if self.pose_idx == 0:
            log.debug("Sending robot to rest pose")
            self.robot_at_rest = True
            self.joints = self.scene.ready_pos_full
            self.step_sleep = self.scene.arms.goal_time_slow
        else:
            self.joints = (
                np.asarray(
                    self.strokebatch.joints[self.stroke_idx, self.pose_idx, self.offset_idx_slider.value],
                    dtype=np.float64,
                ).flatten()
            )
            self.step_sleep = float(
                self.strokebatch.dt[self.stroke_idx, self.pose_idx, self.offset_idx_slider.value].item()
            )
        self.pose_idx += 1

if __name__ == "__main__":
    args = setup_log_with_config(VizStrokesConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = VizStrokes(args)
    viz.run()