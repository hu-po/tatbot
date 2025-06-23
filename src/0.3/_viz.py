from dataclasses import dataclass
import logging
import time

import viser
from viser.extras import ViserUrdf

from _bot import BotConfig, load_robot
from _log import get_logger, setup_log_with_config, print_config

log = get_logger('_viz')

@dataclass
class BaseVizConfig:
    debug: bool = False
    """Enable debug logging."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

class BaseViz:
    def __init__(self, config: BaseVizConfig, bot_config: BotConfig = BotConfig()):
        self.config = config

        log.info("🖥️ Starting viser server...")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        log.debug(f"🖥️ Adding robot to viser from URDF at {bot_config.urdf_path}...")
        _urdf, self.robot, self.ee_link_indices = load_robot(bot_config.urdf_path, bot_config.target_link_names)
        self.urdf = ViserUrdf(self.server, _urdf, root_node_name="/root")
        self.joints = bot_config.rest_pose.copy()

    def step(self):
        log.info("🖥️ Empty step function, implement in subclass...")
        pass

    def run(self):
        while True:
            start_time = time.time()
            log.debug(f"🖥️ Updating viser robot...")
            self.urdf.update_cfg(self.joints)
            self.step()
            log.debug(f"🖥️ step time: {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    args = setup_log_with_config(BaseVizConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    viz = BaseViz(args)
    while True:
        time.sleep(0.1)