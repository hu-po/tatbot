import logging
import time
from dataclasses import dataclass

import viser
from viser.extras import ViserUrdf

from tatbot.data.scene import Scene
from tatbot.bot.urdf import get_link_poses, load_robot
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('viz.base', '🖥️')

@dataclass
class BaseVizConfig:
    debug: bool = False
    """Enable debug logging."""

    scene_name: str = "default"
    """Name of the scene (Scene)."""

    env_map_hdri: str = "forest"
    """HDRI for the environment map."""
    view_camera_position: tuple[float, float, float] = (0.3, 0.3, 0.3)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""

    realsense_frustrum_scale: float = 0.02
    """Scale of the realsense camera frustrums used for visualization."""
    realsense_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the realsense camera frustrums used for visualization."""

    camera_frustrum_scale: float = 0.04
    """Scale of the ip camera frustrum used for visualization."""
    camera_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the ip camera frustrum used for visualization."""

    speed: float = 1.0
    """Speed multipler for visualization."""

class BaseViz:
    def __init__(self, config: BaseVizConfig):
        self.config = config
        self.scene: Scene = Scene.from_name(config.scene_name)

        log.info("Starting viser server")
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

        log.debug("Adding robot to viser from URDF")
        _urdf, self.robot = load_robot(self.scene.urdf.path)
        self.viser_urdf = ViserUrdf(self.server, _urdf, root_node_name="/root")
        self.joints = self.scene.home_pos_full.copy()
    
        log.debug("Adding inkpalette to viser")
        link_poses = get_link_poses(self.scene.urdf.path, self.scene.urdf.ink_link_names, self.scene.home_pos_full)
        for inkcap in self.scene.inks.inkcaps:
            self.server.scene.add_icosphere(
                name=f"/inkcaps/{inkcap.name}",
                radius=inkcap.diameter_m / 2,
                color=inkcap.ink.rgb,
                position=tuple(link_poses[inkcap.name].pos.xyz),
                subdivisions=4,
                visible=True,
            )

        log.info("Adding camera frustrums ...")
        link_poses = get_link_poses(self.scene.urdf.path, self.scene.urdf.cam_link_names, self.scene.home_pos_full)
        self.realsense_frustrums = {}
        for realsense in self.scene.cams.realsenses:
            self.realsense_frustrums[realsense.name] = self.server.scene.add_camera_frustum(
                f"/realsense/{realsense.name}",
                fov=realsense.fov,
                aspect=realsense.aspect,
                scale=config.realsense_frustrum_scale,
                color=config.realsense_frustrum_color,
                position=link_poses[realsense.urdf_link_name].pos.xyz,
                wxyz=link_poses[realsense.urdf_link_name].rot.wxyz,
            )
        self.ipcameras_frustrums = {}
        for ipcamera in self.scene.cams.ipcameras:
            self.ipcameras_frustrums[ipcamera.name] = self.server.scene.add_camera_frustum(
                f"/ipcamera/{ipcamera.name}",
                fov=ipcamera.fov,
                aspect=ipcamera.aspect,
                scale=config.camera_frustrum_scale,
                color=config.camera_frustrum_color,
                position=link_poses[ipcamera.urdf_link_name].pos.xyz,
                wxyz=link_poses[ipcamera.urdf_link_name].rot.wxyz,
            )

    def step(self):
        log.info("Empty step function, implement in subclass")
        pass

    def run(self):
        while True:
            start_time = time.time()
            if self.viser_urdf is not None:
                log.debug("Updating viser robot")
                self.viser_urdf.update_cfg(self.joints)
            self.step()
            log.debug(f"Step time: {time.time() - start_time:.4f}s")
            time.sleep(self.step_sleep / self.speed_slider.value)

if __name__ == "__main__":
    args = setup_log_with_config(BaseVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = BaseViz(args)
    viz.run()