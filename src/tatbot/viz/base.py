import logging
import time
from dataclasses import dataclass
from itertools import chain

import numpy as np
import viser
from viser.extras import ViserUrdf

from tatbot.bot.urdf import get_link_poses, load_robot
from tatbot.data.scene import Scene
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("viz.base", "🖼️")


@dataclass
class BaseVizConfig:
    debug: bool = False
    """Enable debug logging."""

    scene: str = "default"
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
    realsense_point_size: float = 0.001
    """Size of points in point cloud visualizations."""

    camera_frustrum_scale: float = 0.04
    """Scale of the ip camera frustrum used for visualization."""
    camera_frustrum_color: tuple[int, int, int] = (200, 200, 200)
    """Color of the ip camera frustrum used for visualization."""

    speed: float = 1.0
    """Speed multipler for visualization."""

    enable_robot: bool = False
    """Enable the real robot (instead of only the simulated one)."""
    enable_depth: bool = False
    """Enable the depth visualization."""


class BaseViz:
    def __init__(self, config: BaseVizConfig):
        self.config = config
        self.scene: Scene = Scene.from_name(config.scene)
        self.global_step = 0

        log.info("Starting viser server")
        self.server: viser.ViserServer = viser.ViserServer()
        self.server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = config.view_camera_position
            client.camera.look_at = config.view_camera_look_at

        log.debug("Adding robot to viser from URDF")
        _urdf, self.robot = load_robot(self.scene.urdf.path)
        self.viser_urdf = ViserUrdf(self.server, _urdf, root_node_name="/root")
        self.joints = self.scene.ready_pos_full.copy()

        with self.server.gui.add_folder("Joints", expand_by_default=False):
            with self.server.gui.add_folder("Left", expand_by_default=False):
                self.left_joint_textboxes = []
                for i in range(8):
                    tb = self.server.gui.add_text(
                        f"{i + 1}", initial_value=str(self.joints[i]), disabled=True
                    )
                    self.left_joint_textboxes.append(tb)
            with self.server.gui.add_folder("Right", expand_by_default=False):
                self.right_joint_textboxes = []
                for i in range(8):
                    tb = self.server.gui.add_text(
                        f"{i + 1}", initial_value=str(self.joints[i + 8]), disabled=True
                    )
                    self.right_joint_textboxes.append(tb)

        with self.server.gui.add_folder("System", expand_by_default=False):
            self.global_step_textbox = self.server.gui.add_text(
                "Global Step", 
                initial_value="0", 
                disabled=True,
                hint="Current global step counter"
            )
        if config.enable_depth:
            log.debug('Using pointcloud cameras')
            from tatbot.cam.depth import DepthCamera

            self.depth_cameras = {}
            with self.server.gui.add_folder("Point Cloud Recording", expand_by_default=False):
                self.save_checkbox = self.server.gui.add_checkbox(
                    "Save Point Clouds", 
                    initial_value=False,
                    hint="Toggle saving point clouds as PLY files"
                )

        self.arm_l = None
        self.arm_r = None
        self.to_trossen_vector = None
        if config.enable_robot:
            log.debug("Using real robot")
            from tatbot.bot.trossen import driver_from_arms, trossen_arm

            self.arm_l, self.arm_r = driver_from_arms(self.scene.arms)
            self.to_trossen_vector = lambda x: trossen_arm.VectorDouble(x)

        log.debug("Adding inkcaps to viser")
        for inkcap in chain(self.scene.inkcaps_l.values(), self.scene.inkcaps_r.values()):
            self.server.scene.add_icosphere(
                name=f"/inkcaps/{inkcap.name}",
                radius=inkcap.diameter_m / 2,
                color=inkcap.ink.rgb,
                position=tuple(inkcap.pose.pos.xyz),
                opacity=0.5,
                subdivisions=4,
                visible=True,
            )

        log.info("Adding camera frustrums ...")
        link_poses = get_link_poses(
            self.scene.urdf.path, self.scene.urdf.cam_link_names, self.scene.ready_pos_full
        )
        _camera_counter: int = 0
        self.realsense_frustrums = {}
        self.realsense_pointclouds = {}
        for realsense in self.scene.cams.realsenses:
            self.realsense_frustrums[realsense.name] = self.server.scene.add_camera_frustum(
                f"/realsense/{realsense.name}",
                fov=realsense.intrinsics.fov,
                aspect=realsense.intrinsics.aspect,
                scale=config.realsense_frustrum_scale,
                color=config.realsense_frustrum_color,
                position=link_poses[self.scene.urdf.cam_link_names[_camera_counter]].pos.xyz,
                wxyz=link_poses[self.scene.urdf.cam_link_names[_camera_counter]].rot.wxyz,
            )
            if config.enable_depth:
                self.depth_cameras[realsense.name] = DepthCamera(realsense.serial_number)
                self.realsense_pointclouds[realsense.name] = self.server.scene.add_point_cloud(
                    f"/realsense/{realsense.name}/pointcloud",
                    points=np.zeros((1, 3)),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=config.realsense_point_size,
                )
            _camera_counter += 1
        self.ipcameras_frustrums = {}
        for ipcamera in self.scene.cams.ipcameras:
            self.ipcameras_frustrums[ipcamera.name] = self.server.scene.add_camera_frustum(
                f"/ipcamera/{ipcamera.name}",
                fov=ipcamera.intrinsics.fov,
                aspect=ipcamera.intrinsics.aspect,
                scale=config.camera_frustrum_scale,
                color=config.camera_frustrum_color,
                position=link_poses[self.scene.urdf.cam_link_names[_camera_counter]].pos.xyz,
                wxyz=link_poses[self.scene.urdf.cam_link_names[_camera_counter]].rot.wxyz,
            )
            _camera_counter += 1
        log.info(f"Added {_camera_counter} cameras")

        log.info("Adding skin zone to viser")
        self.skin_zone = self.server.scene.add_box(
            name="/skin/zone",
            color=(0, 255, 0),
            dimensions=(
                self.scene.skin.zone_depth_m,
                self.scene.skin.zone_width_m,
                self.scene.skin.zone_height_m,
            ),
            position=self.scene.skin.design_pose.pos.xyz,
            wxyz=self.scene.skin.design_pose.rot.wxyz,
            opacity=0.2,
            visible=True,
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
            if self.arm_l is not None:
                arm_l_joints = self.joints[:7]
                log.debug(f"Setting real left arm positions: {arm_l_joints}")
                self.arm_l.set_all_positions(self.to_trossen_vector(arm_l_joints), blocking=True)
            if self.arm_r is not None:
                arm_r_joints = self.joints[8:-1]
                log.debug(f"Setting real right arm positions: {arm_r_joints}")
                self.arm_r.set_all_positions(self.to_trossen_vector(arm_r_joints), blocking=True)
            link_poses = get_link_poses(self.scene.urdf.path, self.scene.urdf.cam_link_names, self.joints)
            for i, realsense in enumerate(self.scene.cams.realsenses):
                link_name = self.scene.urdf.cam_link_names[i]
                self.realsense_frustrums[realsense.name].position = link_poses[link_name].pos.xyz
                self.realsense_frustrums[realsense.name].wxyz = link_poses[link_name].rot.wxyz
                if self.config.enable_depth:
                    realsense_start_time = time.time()
                    self.depth_cameras[realsense.name].pose = link_poses[link_name]
                    image, positions, colors = self.depth_cameras[realsense.name].make_observation(save=self.save_checkbox.value)
                    self.realsense_frustrums[realsense.name].image = image
                    self.realsense_pointclouds[realsense.name].points = np.array(positions)
                    self.realsense_pointclouds[realsense.name].colors = np.array(colors)
                    log.info(f"{realsense.name} took {time.time() - realsense_start_time} seconds")
            self.step()
            for i, tb in enumerate(self.left_joint_textboxes):
                tb.value = str(self.joints[i])
            for i, tb in enumerate(self.right_joint_textboxes):
                tb.value = str(self.joints[i + 8])
            self.global_step += 1
            self.global_step_textbox.value = str(self.global_step)
            log.debug(f"Step time: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    args = setup_log_with_config(BaseVizConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    viz = BaseViz(args)
    viz.run()
