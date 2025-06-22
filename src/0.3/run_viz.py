from dataclasses import dataclass
import logging
import os
import time

import cv2
import numpy as np
import PIL
import viser
from viser.extras import ViserUrdf
import yourdfpy

from _log import get_logger, COLORS, print_config, setup_log_with_config
from _path import PixelPath, PathBatch
from _plan import Plan

log = get_logger('run_viz')

@dataclass
class VizConfig:
    debug: bool = False
    """Enable debug logging."""

    plan_dir: str = os.path.expanduser("~/tatbot/output/plans/bench")
    """Directory containing plan."""

    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    point_size: float = 0.001
    """Size of points in the point cloud visualization (meters)."""
    point_shape: str = "rounded"
    """Shape of points in the point cloud visualization."""

    path_highlight_radius: int = 3
    """Radius of the path highlight in pixels."""
    pose_highlight_radius: int = 6
    """Radius of the pose highlight in pixels."""

    speed: float = 1.0
    """Speed multipler for visualization."""

class Viz:
    def __init__(self, config: VizConfig):
        self.config = config

        log.info("🖥️ Starting viser server...")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        self.path_idx = 0
        self.pose_idx = 0
        self.plan = Plan.from_yaml(config.plan_dir)
        self.num_paths = len(self.plan.path_descriptions)
        self.pixelpaths = self.plan.load_pixelpaths()
        self.pathbatch = self.plan.load_pathbatch()
        self.path_lengths = [int(np.sum(self.pathbatch.mask[i])) for i in range(self.num_paths)]

        # Define the path description getter before using it in the GUI
        def _get_path_description(idx):
            key = f"path_{idx:03d}"
            return self.plan.path_descriptions.get(key, "")
        self._get_path_description = _get_path_description

        # Helper to map PathBatch pose_idx to PixelPath index
        def _pose_to_pixel_idx(path_idx: int, pose_idx: int) -> int | None:
            pixels = self.pixelpaths[path_idx].pixels
            if not pixels:
                return None  # ink-dip: nothing to draw
            if pose_idx == 0 or pose_idx == self.path_lengths[path_idx] - 1:
                return None  # hover poses have no pixel
            return pose_idx - 1  # shift to account for first hover
        self._pose_to_pixel_idx = _pose_to_pixel_idx

        with self.server.gui.add_folder("Plan"):
            self.time_label = self.server.gui.add_text(
                label="time",
                initial_value=" 00h00m00s / 00h00m00s",
                disabled=True,
            )
            self.path_idx_slider = self.server.gui.add_slider(
                "path",
                min=0,
                max=self.num_paths - 1,
                step=1,
                initial_value=0,
            )
            self.path_desc_label = self.server.gui.add_text(
                label="description",
                initial_value=self._get_path_description(0),
                disabled=True,
            )
            self.pose_idx_slider = self.server.gui.add_slider(
                "pose",
                min=0,
                max=self.path_lengths[0] - 1,
                step=1,
                initial_value=0,
            )
            self.speed_slider = self.server.gui.add_slider(
                "speed",
                min=0.1,
                max=100.0,
                step=0.1,
                initial_value=self.config.speed,
            )
        
        def _format_seconds(secs):
            secs = int(secs)
            h = secs // 3600
            m = (secs % 3600) // 60
            s = secs % 60
            return f"{h:02}h{m:02}m{s:02}s"

        def _update_time_label():
            current_time = 0.0
            for i in range(self.path_idx):
                current_time += self.pathbatch.dt[i, :self.path_lengths[i]].sum().item()
            current_time += self.pathbatch.dt[self.path_idx, :self.pose_idx+1].sum().item()
            total_time = self.pathbatch.dt.sum().item()
            self.time_label.value = f"{_format_seconds(current_time)} / {_format_seconds(total_time)}"

        @self.path_idx_slider.on_update
        def _(_):
            self.path_idx = self.path_idx_slider.value
            self.pose_idx_slider.max = self.path_lengths[self.path_idx] - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max
            _update_time_label()
            self.path_desc_label.value = self._get_path_description(self.path_idx)

        @self.pose_idx_slider.on_update
        def _(_):
            self.pose_idx = self.pose_idx_slider.value
            _update_time_label()

        _update_time_label()

        log.debug(f"🖥️🤖 Adding URDF to viser from {config.urdf_path}...")
        self.urdf = ViserUrdf(self.server, yourdfpy.URDF.load(config.urdf_path), root_node_name="/root")

        log.debug(f"️🖥️🖼️ Adding images from {config.plan_dir}...")
        self.image_np = self.plan.load_image_np()
        self.image = self.server.gui.add_image(image=self.image_np, format="png")
        self.pathviz_np = make_pathviz_image(self.plan)
        self.pathviz = self.server.gui.add_image(image=self.pathviz_np, format="png")

        log.debug(f"🖥️🖼️ Adding pointclouds...")
        points_hover = []
        points_path = []
        self.path_point_ranges = []  # List of (start_idx, end_idx) for each path in points_path
        path_point_idx = 0
        for i in range(self.pathbatch.ee_pos_l.shape[0]):
            path_start = path_point_idx
            for j in range(self.pathbatch.ee_pos_l.shape[1]):
                if self.pathbatch.mask[i, j]:
                    if j == 0 or j == self.path_lengths[i] - 1:
                        points_hover.append(self.pathbatch.ee_pos_l[i, j])
                    else:
                        points_path.append(self.pathbatch.ee_pos_l[i, j])
                        path_point_idx += 1
            path_end = path_point_idx
            self.path_point_ranges.append((path_start, path_end))
        points_hover = np.stack(points_hover, axis=0)
        points_path = np.stack(points_path, axis=0)
        point_colors_hover = np.tile(np.array(COLORS["orange"], dtype=np.uint8), (points_hover.shape[0], 1))
        point_colors_path = np.tile(np.array(COLORS["black"], dtype=np.uint8), (points_path.shape[0], 1))
        self.pointcloud_hover = self.server.scene.add_point_cloud(
            name="/points_hover",
            points=points_hover,
            colors=point_colors_hover,
            point_size=self.config.point_size,
            point_shape=self.config.point_shape,
        )
        self.pointcloud_path = self.server.scene.add_point_cloud(
            name="/points_path",
            points=points_path,
            colors=point_colors_path,
            point_size=self.config.point_size,
            point_shape=self.config.point_shape,
        )

        log.debug(f"🖥️🖼️ Adding inkcaps...")
        for cap_name, cap in self.plan.inkpalette.inkcaps.items():
            pos = tuple(np.array(self.plan.ee_inkpalette_pos) + np.array(cap.palette_pos))
            radius = cap.diameter_m / 2
            color = COLORS.get(cap.color.lower(), (0, 0, 0))
            self.server.scene.add_icosphere(
                name=f"/inkcaps/{cap_name}",
                radius=radius,
                color=color,
                position=pos,
                subdivisions=4,
                opacity=0.4,
                visible=True,
            )

    def run(self):
        while True:
            if self.pose_idx >= self.path_lengths[self.path_idx]:
                self.path_idx += 1
                log.debug(f"🖥️ Moving to next path {self.path_idx}")
                self.pose_idx = 0
            if self.path_idx >= self.num_paths:
                log.debug(f"🖥️ Looping back to path 0")
                self.path_idx = 0
                self.pose_idx = 0
            self.path_idx_slider.value = self.path_idx
            self.pose_idx_slider.value = self.pose_idx
            self.pose_idx_slider.max = self.path_lengths[self.path_idx] - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max
            log.debug(f"🖥️🤖 Visualizing path {self.path_idx} pose {self.pose_idx}")

            log.debug(f"🖥️🖼️ Updating Viser image...")
            image_np = self.image_np.copy()
            # Draw the entire path in red (all drawing pixels, not hover)
            pixels = self.pixelpaths[self.path_idx].pixels
            if pixels:
                for pw, ph in pixels:
                    cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["red"], -1)
                # Highlight path up until current pose in green
                pix_idx = self._pose_to_pixel_idx(self.path_idx, self.pose_idx)
                if pix_idx is not None and pix_idx > 0:
                    for pw, ph in pixels[:pix_idx]:
                        cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["green"], -1)
                # Highlight current pose in magenta
                if pix_idx is not None and 0 <= pix_idx < len(pixels):
                    px, py = pixels[pix_idx]
                    cv2.circle(image_np, (int(px), int(py)), self.config.pose_highlight_radius, COLORS["magenta"], -1)
            self.image.image = image_np

            log.debug(f"🖥️🤖 Updating Viser robot...")
            joints_np = np.asarray(self.pathbatch.joints[self.path_idx, self.pose_idx], dtype=np.float64).flatten()
            self.urdf.update_cfg(joints_np)

            log.debug(f"🖥️🤖 Updating Viser pointclouds...")
            # Highlight entire path in red, path up until current pose in green, current pose in magenta
            path_start, path_end = self.path_point_ranges[self.path_idx]
            new_colors = np.tile(np.array(COLORS["black"], dtype=np.uint8), (self.pointcloud_path.points.shape[0], 1))
            # Highlight current path in red
            new_colors[path_start:path_end] = np.array(COLORS["red"], dtype=np.uint8)
            # Compute pixel index for current pose
            pix_idx = self._pose_to_pixel_idx(self.path_idx, self.pose_idx)
            # Highlight up to current pose in green (excluding endpoints)
            if pix_idx is not None and pix_idx > 0:
                new_colors[path_start:path_start + pix_idx] = np.array(COLORS["green"], dtype=np.uint8)
            # Highlight current pose in magenta (if not endpoint)
            if pix_idx is not None and 0 <= pix_idx < (path_end - path_start):
                new_colors[path_start + pix_idx] = np.array(COLORS["magenta"], dtype=np.uint8)
            self.pointcloud_path.colors = new_colors

            dt_val = self.pathbatch.dt[self.path_idx, self.pose_idx]
            if hasattr(dt_val, "item"):
                dt_val = dt_val.item()
            else:
                dt_val = float(dt_val)
            time.sleep(dt_val / self.speed_slider.value)
            self.pose_idx += 1

def make_pathviz_image(plan: Plan) -> np.ndarray:
    """Creates an image with overlayed paths from a plan."""
    pixelpaths = plan.load_pixelpaths()
    image = plan.load_image_np()
    for path in pixelpaths:
        path_indices = np.linspace(0, 255, len(path.pixels), dtype=np.uint8)
        colormap = cv2.applyColorMap(path_indices.reshape(-1, 1), cv2.COLORMAP_JET)
        for path_idx in range(len(path.pixels) - 1):
            p1 = tuple(map(int, path.pixels[path_idx]))
            p2 = tuple(map(int, path.pixels[path_idx + 1]))
            color = colormap[path_idx][0].tolist()
            cv2.line(image, p1, p2, color, 2)
    out_path = os.path.join(plan.dirpath, "pathviz.png")
    cv2.imwrite(out_path, image)
    return image

if __name__ == "__main__":
    args = setup_log_with_config(VizConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = Viz(args)
    viz.run()