from dataclasses import dataclass
import os

import cv2
import numpy as np
import PIL
import viser
from viser.extras import ViserUrdf
import yourdfpy

from _log import get_logger, COLORS
from _plan import Plan

log = get_logger('_viz')

@dataclass
class VizConfig:
    plan_dir: str = os.path.expanduser("~/tatbot/output/plans/calibration")
    """Directory containing plan."""

    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

    path_highlight_radius: int = 3
    """Radius of the path highlight in pixels."""
    pose_highlight_radius: int = 6
    """Radius of the pose highlight in pixels."""

class Viz:
    def __init__(self, config: VizConfig):
        self.config = config
        self.plan = Plan.from_yaml(config.plan_dir)
        self.paths_np = self.plan.paths_np(config.plan_dir)

        log.info("🖥️ Starting viser server...")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        log.debug(f"🖥️🤖 Adding URDF to viser from {config.urdf_path}...")
        self.urdf = ViserUrdf(self.server, yourdfpy.URDF.load(config.urdf_path), root_node_name="/root")

        log.debug(f"️🖥️🖼️ Adding image from {config.plan_dir}...")
        self.image_np = self.plan.image_np(config.plan_dir)
        self.image = self.server.gui.add_image(label=self.plan.name, image=self.image_np, format="png")

    def update_robot(self, joints: np.ndarray):
        log.debug(f"🖥️🤖 Updating Viser robot...")
        self.urdf.update_cfg(joints)

    def update_image(self, path_idx: int, pose_idx: int):
        log.debug(f"🖥️🖼️ Updating Viser image...")
        image_np = self.image_np.copy()
        for _, pw, ph in self.paths_np[path_idx].pixel_coords[:, :]:
            # highlight entire path in red
            cv2.circle(image_np, (int(pw), int(ph)), self.path_highlight_radius, COLORS["red"], -1)
        for _, pw, ph in self.paths_np[path_idx].pixel_coords[:pose_idx, :]:
            # highlight path up until current pose in green
            cv2.circle(image_np, (int(pw), int(ph)), self.path_highlight_radius, COLORS["green"], -1)
        # highlight current pose in magenta
        cv2.circle(image_np, (int(self.paths_np[path_idx].pixel_coords[pose_idx, 0]), int(self.paths_np[path_idx].pixel_coords[pose_idx, 1])), self.pose_highlight_radius, COLORS["magenta"], -1)
        self.image.image = image_np