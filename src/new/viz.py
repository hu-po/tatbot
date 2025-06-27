from dataclasses import dataclass
import logging
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from _bot import BotConfig, load_robot, get_link_indices
from _ink import InkConfig
from _log import COLORS, get_logger, setup_log_with_config, print_config

log = get_logger('viz')

@dataclass
class BaseVizConfig:
    debug: bool = False
    """Enable debug logging."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

    speed: float = 1.0
    """Speed multipler for visualization."""

class BaseViz:
    def __init__(self, config: BaseVizConfig):
        self.config = config
        self.bot_config = BotConfig()
        self.ink_config = InkConfig()

        log.info("🖥️ Starting viser server...")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        self.step_sleep = 1.0 / 30.0 # 30 fps
        self.speed_slider = self.server.gui.add_slider(
            "speed",
            min=0.1,
            max=100.0,
            step=0.1,
            initial_value=self.config.speed,
        )

    def add_robot(self, config: BotConfig):
        log.debug(f"🖥️ Adding robot to viser from URDF at {config.urdf_path}...")
        _urdf, self.robot = load_robot(config.urdf_path)
        self.ee_link_indices = get_link_indices(config.target_link_names, config.urdf_path)
        self.urdf = ViserUrdf(self.server, _urdf, root_node_name="/root")
        self.joints = config.rest_pose.copy()
        self.robot_at_rest: bool = True
    
    def add_inkpalette(self, config: InkConfig):
        log.debug(f"🖥️ Adding inkpalette to viser...")
        for cap_name, cap in config.inkcaps.items():
            pos = tuple(np.array(config.inkpalette_pos))
            radius = cap.diameter_m / 2
            color = COLORS.get(cap.color.lower(), (0, 0, 0))
            self.server.scene.add_icosphere(
                name=f"/inkcaps/{cap_name}",
                radius=radius,
                color=color,
                position=pos,
                subdivisions=4,
                visible=True,
            )

    def step(self):
        log.info("🖥️ Empty step function, implement in subclass...")
        pass

    def run(self):
        self.add_robot(self.bot_config)
        self.add_inkpalette(self.ink_config)
        while True:
            start_time = time.time()
            if self.urdf is not None:
                log.debug(f"🖥️ Updating viser robot...")
                self.urdf.update_cfg(self.joints)
            self.step()
            log.debug(f"🖥️ step time: {time.time() - start_time:.4f}s")
            time.sleep(self.step_sleep / self.speed_slider.value)

if __name__ == "__main__":
    args = setup_log_with_config(BaseVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = BaseViz(args)
    viz.run()