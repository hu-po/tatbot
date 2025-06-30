import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from tatbot.data.plan import Plan
from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.viz.base import BaseViz, BaseVizConfig
from tatbot.utils.ui_colors import COLORS
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('viz.plan', '🖥️')

@dataclass
class VizPlanConfig(BaseVizConfig):
    plan_dir: str = "~/tatbot/output/plans/yawning_cat"
    """Directory containing plan."""

    design_pointcloud_point_size: float = 0.001
    """Size of points in the point cloud visualization (meters)."""
    design_pointcloud_point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""

    path_highlight_radius: int = 3
    """Radius of the path highlight in pixels."""
    pose_highlight_radius: int = 6
    """Radius of the pose highlight in pixels."""

class VizPlan(BaseViz):
    def __init__(self, config: VizPlanConfig):
        super().__init__(config)
        plan_dir = os.path.expanduser(config.plan_dir)  # Expand ~ to home directory
        self.plan: Plan = Plan.from_yaml(os.path.join(plan_dir, "plan.yaml"))
        self.strokebatch: StrokeBatch = StrokeBatch.load(os.path.join(plan_dir, "strokebatch.safetensors"))
        self.strokes: StrokeList = StrokeList.from_yaml(os.path.join(plan_dir, "strokes.yaml"))

        self.num_strokes = len(self.strokes.strokes)
        self.stroke_idx = 0
        self.pose_idx = 0

        with self.server.gui.add_folder("Plan"):
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

            self.path_idx_slider = self.server.gui.add_slider(
                "stroke",
                min=0,
                max=self.num_strokes - 1,
                step=1,
                initial_value=0,
            )
            self.path_desc_label_l = self.server.gui.add_text(
                label="left arm description",
                initial_value=self.strokes.strokes[0][0].description,
                disabled=True,
            )
            self.path_desc_label_r = self.server.gui.add_text(
                label="right arm description",
                initial_value=self.strokes.strokes[0][1].description,
                disabled=True,
            )
            self.pose_idx_slider = self.server.gui.add_slider(
                "pose",
                min=0,
                max=self.plan.path_length - 1,
                step=1,
                initial_value=0,
            )
        

        @self.path_idx_slider.on_update
        def _(_):
            self.stroke_idx = self.path_idx_slider.value
            self.pose_idx_slider.max = self.plan.path_length - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max
            _update_time_label()
            self.path_desc_label_l.value = self.strokes.strokes[self.stroke_idx][0].description
            self.path_desc_label_r.value = self.strokes.strokes[self.stroke_idx][1].description

        @self.pose_idx_slider.on_update
        def _(_):
            self.pose_idx = self.pose_idx_slider.value
            _update_time_label()

        _update_time_label()

        log.debug(f"️🖼️ Adding GUI image: design.png")
        self.image_np = cv2.imread(os.path.join(plan_dir, "design.png"))
        self.image = self.server.gui.add_image(image=self.image_np, format="png")

        log.debug(f"🖼️ Adding GUI image: pathviz.png")
        image = self.image_np.copy()
        for strokes in self.strokes.strokes:
            for stroke in strokes:
                if stroke.pixel_coords is not None:
                    path_indices = np.linspace(0, 255, len(stroke.pixel_coords), dtype=np.uint8)
                    colormap = cv2.applyColorMap(path_indices.reshape(-1, 1), cv2.COLORMAP_JET)
                    for path_idx in range(len(stroke.pixel_coords) - 1):
                        p1 = tuple(map(int, stroke.pixel_coords[path_idx]))
                        p2 = tuple(map(int, stroke.pixel_coords[path_idx + 1]))
                        color = colormap[path_idx][0].tolist()
                        cv2.line(image, p1, p2, color, 2)
        self.pathviz_np = image
        out_path = os.path.join(plan_dir, "pathviz.png")
        cv2.imwrite(out_path, image)
        self.pathviz = self.server.gui.add_image(image=self.pathviz_np, format="png")

        log.debug(f"🖼️ Adding 2D image (design frame)...")
        self.server.scene.add_image(
            name="/design",
            image=self.image_np,
            wxyz=self.skin.design_pose.rot.wxyz,
            position=self.skin.design_pose.pos.xyz,
            render_width=self.plan.image_width_m,
            render_height=self.plan.image_height_m,
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

    def step(self):
        if self.robot_at_rest:
            log.debug("Robot at rest, skipping step...")
            self.robot_at_rest = False
            return
        if self.pose_idx >= self.plan.path_length:
            self.stroke_idx += 1
            self.pose_idx = 0
            log.debug(f"Moving to next stroke {self.stroke_idx}")
        if self.stroke_idx >= self.num_strokes:
            log.debug(f"Looping back to stroke 0")
            self.stroke_idx = 0
            self.pose_idx = 0
        self.path_idx_slider.value = self.stroke_idx
        self.pose_idx_slider.value = self.pose_idx
        log.debug(f"Visualizing stroke {self.stroke_idx} pose {self.pose_idx}")

        log.debug("Updating Image and Pointclouds")
        image_np = self.image_np.copy()
        for stroke in self.strokes.strokes[self.stroke_idx]:
            if stroke.arm == "left":
                points_color_l = self.point_colors_stroke_l.copy()
                points_color_l[self.stroke_idx * self.plan.path_length:self.stroke_idx * self.plan.path_length + self.pose_idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_l[self.stroke_idx * self.plan.path_length + self.pose_idx] = np.array(COLORS["blue"], dtype=np.uint8)
                self.pointcloud_path_l.colors = points_color_l
            else:
                points_color_r = self.point_colors_stroke_r.copy()
                points_color_r[self.stroke_idx * self.plan.path_length:self.stroke_idx * self.plan.path_length + self.pose_idx + 1] = np.array(COLORS["orange"], dtype=np.uint8)
                points_color_r[self.stroke_idx * self.plan.path_length + self.pose_idx] = np.array(COLORS["purple"], dtype=np.uint8)
                self.pointcloud_path_r.colors = points_color_r
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
        self.image.image = image_np

        if self.pose_idx == 0:
            log.debug("Sending robot to rest pose")
            self.robot_at_rest = True
            self.joints = self.rest_pose.copy()
            self.step_sleep = self.plan.path_dt_slow
        else:
            self.joints = np.asarray(self.strokebatch.joints[self.stroke_idx, self.pose_idx], dtype=np.float64).flatten()
            self.step_sleep = float(self.strokebatch.dt[self.stroke_idx, self.pose_idx].item())
        self.pose_idx += 1

if __name__ == "__main__":
    args = setup_log_with_config(VizPlanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = VizPlan(args)
    viz.run()