from dataclasses import dataclass
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
        with self.server.gui.add_folder("Plan"):
            self.path_idx_slider = self.server.gui.add_slider(
                "path",
                min=0,
                max=self.num_paths - 1,
                step=1,
                initial_value=0,
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
        
        @self.path_idx_slider.on_update
        def _(_):
            self.path_idx = self.path_idx_slider.value
            self.pose_idx_slider.max = self.path_lengths[self.path_idx] - 1
            if self.pose_idx_slider.value > self.pose_idx_slider.max:
                self.pose_idx_slider.value = self.pose_idx_slider.max

        @self.pose_idx_slider.on_update
        def _(_):
            self.pose_idx = self.pose_idx_slider.value

        log.debug(f"🖥️🤖 Adding URDF to viser from {config.urdf_path}...")
        self.urdf = ViserUrdf(self.server, yourdfpy.URDF.load(config.urdf_path), root_node_name="/root")

        log.debug(f"️🖥️🖼️ Adding images from {config.plan_dir}...")
        self.image_np = self.plan.image_np(config.plan_dir)
        self.image = self.server.gui.add_image(label=self.plan.name, image=self.image_np, format="png")
        self.pathlen_np = make_pathlen_image(self.plan, self.pathbatch)
        self.pathlen = self.server.gui.add_image(image=self.pathlen_np, format="png")
        self.pathviz_np = make_pathviz_image(self.plan, self.pathbatch)
        self.pathviz = self.server.gui.add_image(image=self.pathviz_np, format="png")

        log.debug(f"🖥️🖼️ Adding pointcloud...")
        points = []
        for i in range(self.pathbatch.ee_pos_l.shape[0]):
            for j in range(self.pathbatch.ee_pos_l.shape[1]):
                if self.pathbatch.mask[i, j]:
                    points.append(self.pathbatch.ee_pos_l[i, j])
        points = np.stack(points, axis=0)
        point_colors = np.tile(np.array(COLORS["black"], dtype=np.uint8), (points.shape[0], 1))
        self.pointcloud = self.server.scene.add_point_cloud(
            name="/points",
            points=points,
            colors=point_colors,
            point_size=self.config.point_size,
            point_shape=self.config.point_shape,
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
            self.path_idx_slider.value = self.path_idx
            self.pose_idx_slider.value = self.pose_idx
            log.debug(f"🖥️🤖 Visualizing path {self.path_idx} pose {self.pose_idx}")
            self.update_image(self.path_idx, self.pose_idx)
            self.update_robot(self.pathbatch.joints[self.path_idx, self.pose_idx])
            dt_val = self.pathbatch.dt[self.path_idx, self.pose_idx]
            if hasattr(dt_val, "item"):
                dt_val = dt_val.item()
            else:
                dt_val = float(dt_val)
            time.sleep(dt_val / self.speed_slider.value)
            self.pose_idx += 1

    def update_robot(self, joints: np.ndarray):
        log.debug(f"🖥️🤖 Updating Viser robot...")
        joints_np = np.asarray(joints, dtype=np.float64).flatten()
        self.urdf.update_cfg(joints_np)

    def update_image(self, path_idx: int, pose_idx: int):
        log.debug(f"🖥️🖼️ Updating Viser image...")
        image_np = self.image_np.copy()
        valid_len = self.path_lengths[path_idx]
        # highlight entire path in red (only valid points)
        for pw, ph in self.pixelpaths[path_idx].pixels[:valid_len]:
            cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["red"], -1)
        # highlight path up until current pose in green
        for pw, ph in self.pixelpaths[path_idx].pixels[:min(pose_idx, valid_len)]:
            cv2.circle(image_np, (int(pw), int(ph)), self.config.path_highlight_radius, COLORS["green"], -1)
        # highlight current pose in magenta
        if valid_len > 0 and pose_idx < valid_len:
            px, py = self.pixelpaths[path_idx].pixels[pose_idx]
            cv2.circle(image_np, (int(px), int(py)), self.config.pose_highlight_radius, COLORS["magenta"], -1)
        self.image.image = image_np

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

def make_pathlen_image(plan: Plan) -> np.ndarray:
    """Creates a three-part image: stats, histogram, and pathviz colored by path length."""
    pixelpaths = plan.load_pixelpaths()
    stats = plan.load_pathstats()
    stats_lines = [
        f"num paths: {stats['count']}",
        f"min pathlen: {stats['min_px']:.2f} px ({stats['min_m']:.4f} m)",
        f"avg pathlen: {stats['mean_px']:.2f} px ({stats['mean_m']:.4f} m)",
        f"max pathlen: {stats['max_px']:.2f} px ({stats['max_m']:.4f} m)",
        f"sum pathlen: {stats['sum_px']:.2f} px ({stats['sum_m']:.4f} m)",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    name_font_scale = 0.6
    name_font_thickness = 2
    stats_font_scale = 0.32
    stats_font_thickness = 1
    (name_width, name_height), _ = cv2.getTextSize(plan.name, font, name_font_scale, name_font_thickness)
    stats_height = 15 * len(stats_lines) + 8
    name_stats_gap = 20
    total_stats_height = name_height + name_stats_gap + stats_height
    stats_img = np.full((total_stats_height, plan.image_width_px, 3), 255, dtype=np.uint8)
    name_x = (plan.image_width_px - name_width) // 2
    name_y = name_height + 2
    cv2.putText(stats_img, plan.name, (name_x, name_y), font, name_font_scale, (0,0,0), name_font_thickness, lineType=cv2.LINE_AA)
    y = name_y + name_stats_gap
    for line in stats_lines:
        cv2.putText(stats_img, line, (10, y), font, stats_font_scale, (0,0,0), stats_font_thickness, lineType=cv2.LINE_AA)
        y += 15
    # Path lengths for histogram and coloring
    path_lengths = [
        sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(path.pixels[:-1], path.pixels[1:]))
        if len(path.pixels) > 1 else 0.0
        for path in pixelpaths
    ]
    path_lengths = np.array(path_lengths)
    if len(path_lengths) > 0 and np.max(path_lengths) > 0:
        norm_lengths = (path_lengths - np.min(path_lengths)) / (np.ptp(path_lengths) if np.ptp(path_lengths) > 0 else 1)
    else:
        norm_lengths = np.zeros_like(path_lengths)
    hist_height = 64
    n_bins = 24
    colorbar_height = 12
    label_height = 8
    indicator_height = 32
    colorbar_margin = 12
    hist_width = plan.image_width_px
    total_hist_height = hist_height + colorbar_margin + colorbar_height + colorbar_margin + indicator_height + label_height
    hist_img = np.full((total_hist_height, hist_width, 3), 255, dtype=np.uint8)
    if len(path_lengths) > 0:
        hist, bin_edges = np.histogram(path_lengths, bins=n_bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        max_hist = np.max(hist) if np.max(hist) > 0 else 1
        bar_width = max(1, hist_width // n_bins)
        font = cv2.FONT_HERSHEY_SIMPLEX
        colorbar_top = hist_height + colorbar_margin
        for i in range(n_bins):
            color_val = int(255 * (bin_centers[i] - np.min(path_lengths)) / (np.ptp(path_lengths) if np.ptp(path_lengths) > 0 else 1))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
            x0 = i * bar_width
            x1 = (i + 1) * bar_width
            y1 = colorbar_top - 1
            bar_height = int(hist[i] / max_hist * (hist_height - 10))
            y0 = y1 - bar_height if bar_height > 0 else y1
            cv2.rectangle(hist_img, (x0, y0), (x1 - 2, y1), color, -1)
        colorbar_bottom = colorbar_top + colorbar_height
        for x in range(hist_width):
            color_val = int(255 * x / (hist_width - 1))
            color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
            hist_img[colorbar_top:colorbar_bottom, x] = color
        min_val = np.min(path_lengths)
        max_val = np.max(path_lengths)
        vals = [min_val, min_val + (max_val - min_val) / 3, min_val + 2 * (max_val - min_val) / 3, max_val]
        margin = 24
        positions = [margin, hist_width // 3, 2 * hist_width // 3, hist_width - 1 - margin]
        indicator_y = colorbar_bottom + 2
        indicator_font_scale = 0.32
        indicator_font_thickness = 1
        for v, x in zip(vals, positions):
            text = f"{int(round(v))}"
            (text_width, text_height), _ = cv2.getTextSize(text, font, indicator_font_scale, indicator_font_thickness)
            text_x = x - text_width // 2
            text_y = indicator_y + text_height
            cv2.putText(hist_img, text, (text_x, text_y), font, indicator_font_scale, (0,0,0), indicator_font_thickness)
        axis_label = "path length"
        font_scale_label = 0.38
        (label_width, label_height_text), _ = cv2.getTextSize(axis_label, font, font_scale_label, stats_font_thickness)
        label_x = (hist_width - label_width) // 2
        whitespace_between_numbers_and_label = 6
        label_y = indicator_y + text_height + whitespace_between_numbers_and_label + label_height_text - 6
        cv2.putText(hist_img, axis_label, (label_x, label_y), font, font_scale_label, (0,0,0), stats_font_thickness)
    path_viz_np = np.full((plan.image_height_px, plan.image_width_px, 3), 255, dtype=np.uint8)
    path_viz_np = plan.load_image_np()
    for idx, path in enumerate(pixelpaths):
        if len(path.pixels) < 2:
            continue
        color_val = int(255 * norm_lengths[idx])
        color = tuple(int(c) for c in cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0])
        for path_idx in range(len(path.pixels) - 1):
            p1 = tuple(map(int, path.pixels[path_idx]))
            p2 = tuple(map(int, path.pixels[path_idx + 1]))
            cv2.line(path_viz_np, p1, p2, color, 2)
    sep_thick = 2
    sep_color = (0,0,0)
    white_space = 10
    stats_pad = np.full((white_space, plan.image_width_px, 3), 255, dtype=np.uint8)
    stats_sep = np.full((sep_thick, plan.image_width_px, 3), 0, dtype=np.uint8)
    hist_pad = np.full((white_space, plan.image_width_px, 3), 255, dtype=np.uint8)
    hist_sep = np.full((sep_thick, plan.image_width_px, 3), 0, dtype=np.uint8)
    content_img = np.vstack([
        stats_img,
        stats_pad,
        stats_sep,
        hist_img,
        hist_sep,
        path_viz_np
    ])
    border_px = 16
    border_color = (255, 255, 255)
    bordered_img = np.full(
        (content_img.shape[0] + 2 * border_px, content_img.shape[1] + 2 * border_px, 3),
        border_color,
        dtype=np.uint8
    )
    bordered_img[border_px:border_px + content_img.shape[0], border_px:border_px + content_img.shape[1]] = content_img
    out_path = os.path.join(plan.dirpath, "pathlen.png")
    cv2.imwrite(out_path, bordered_img)
    return bordered_img

if __name__ == "__main__":
    args = setup_log_with_config(VizConfig)
    print_config(args)
    viz = Viz(args)
    viz.run()