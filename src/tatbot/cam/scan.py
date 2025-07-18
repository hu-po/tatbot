import logging
import os
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from PIL import Image

from tatbot.cam.extrinsics import get_extrinsics
from tatbot.data.cams import Cams
from tatbot.data.scene import Scene
from tatbot.utils.log import (
    TIME_FORMAT,
    get_logger,
    print_config,
    setup_log_with_config,
)

log = get_logger("cam.scan", "📡")


@dataclass
class ScanConfig:
    debug: bool = False
    """Enable debug logging."""

    scene: str = "default"
    """Name of the scene config to use (Scene)."""

    output_dir: str = "~/tatbot/nfs/scans"
    """Directory to save the dataset."""


def scan(config: ScanConfig) -> str:
    scene: Scene = Scene.from_name(config.scene)

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"🗃️ Creating output directory at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    scan_name = f"{scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
    scan_dir = f"{output_dir}/{scan_name}"
    log.info(f"🗃️ Creating scan directory at {scan_dir}...")
    os.makedirs(scan_dir, exist_ok=True)

    # copy the scene yaml to the scan directory
    scene_path = os.path.join(scan_dir, "scene.yaml")
    scene.to_yaml(scene_path)

    lerobot_camera_configs: dict[str, Union[RealSenseCameraConfig, OpenCVCameraConfig]] = {}
    for cam in scene.cams.realsenses:
        lerobot_camera_configs[cam.name] = RealSenseCameraConfig(
            fps=cam.fps,
            width=cam.width,
            height=cam.height,
            serial_number_or_name=cam.serial_number,
        )
    for cam in scene.cams.ipcameras:
        lerobot_camera_configs[cam.name] = OpenCVCameraConfig(
            fps=cam.fps,
            width=cam.width,
            height=cam.height,
            ip=cam.ip,
            username=cam.username,
            password=os.environ.get(cam.password, None),
            rtsp_port=cam.rtsp_port,
        )
    lerobot_cameras = make_cameras_from_configs(lerobot_camera_configs)
    for cam in lerobot_cameras.values():
        try:
            cam.connect()
            log.info(f"✅ Connected to {cam}")
        except Exception as e:
            log.warning(f"❌Error connecting to {cam}:\n{e}")
            continue

    image_paths = []
    for cam_name, cam in lerobot_cameras.items():
        start = time.perf_counter()
        try:
            image_np = cam.async_read()
        except Exception as e:
            log.error(f"❌Error reading frame from {cam_name}:\n{e}")
            image_np = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        dt_ms = (time.perf_counter() - start) * 1e3
        log.debug(f"{cam_name} read frame: {dt_ms:.1f}ms")
        image_path = os.path.join(scan_dir, f"{cam_name}.png")
        Image.fromarray(image_np).save(image_path)
        log.info(f"✅ Saved frame to {image_path}")
        image_paths.append(image_path)

    cams: Cams = get_extrinsics(image_paths, scene.cams, scene.tags)
    log.info(f"cams: {cams}")
    log.info("✅ Done")
    return scan_name


if __name__ == "__main__":
    args = setup_log_with_config(ScanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    scan(args)
