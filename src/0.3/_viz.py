from dataclasses import dataclass
import os

import numpy as np
import PIL
import viser
from viser.extras import ViserUrdf
import yaml
import yourdfpy

from _log import get_logger
from _plan import PLAN_IMAGE_FILENAME, PLAN_METADATA_FILENAME, Plan

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

        log.info(f"🖥️🤖 Adding URDF to viser from {config.urdf_path}...")
        self.urdf = ViserUrdf(self.server, yourdfpy.URDF.load(config.urdf_path), root_node_name="/root")

        plan_metadata_path = os.path.join(config.plan_dir, PLAN_METADATA_FILENAME)
        log.info(f"🖥️⚙️ Loading plan metadata from {plan_metadata_path}...")
        with open(plan_metadata_path, "r") as f:
            self.plan: Plan = yaml.safe_load(f)

        image_path = os.path.join(config.plan_dir, PLAN_IMAGE_FILENAME)
        log.info(f"️🖥️🖼️ Loading plan image from {image_path}...")
        img_np = np.array(PIL.Image.open(image_path).convert("RGB"))
        self.plan_image = self.server.gui.add_image(
            label=self.plan.name,
            image=img_np,
            format="png",
        )

    def update_robot(self, joints: np.ndarray):
        self.urdf.update_cfg(joints)

