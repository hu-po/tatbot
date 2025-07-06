import pyrealsense2 as rs
import logging
from dataclasses import dataclass
from typing import Optional

from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.data.cams import Cams, RealSenseCameraConfig

log = get_logger('bot.realsense', "👀")

@dataclass
class RealSenseConfig:
    debug: bool = False
    camera_name: Optional[str] = None
    serial_number: Optional[str] = None
    cams_yaml: str = "~/tatbot/config/cams/all.yaml"
    """Path to the camera config YAML file."""

def print_intrinsics(serial_number: str) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    try:
        log.debug(f"Starting pipeline for serial {serial_number}")
        pipeline.start(config)
        profile = pipeline.get_active_profile()
        video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = video_stream.get_intrinsics()

        log.info(f"Intrinsics for device {serial_number}:")
        log.info(f"  Width: {intrinsics.width}")
        log.info(f"  Height: {intrinsics.height}")
        log.info(f"  FX: {intrinsics.fx}")
        log.info(f"  FY: {intrinsics.fy}")
        log.info(f"  PPX (cx): {intrinsics.ppx}")
        log.info(f"  PPY (cy): {intrinsics.ppy}")
        log.info(f"  Distortion Model: {intrinsics.model}")
        log.info(f"  Distortion Coefficients: {intrinsics.coeffs}\n")
    finally:
        log.debug(f"Stopping pipeline for serial {serial_number}")
        pipeline.stop()

def get_realsense_config(cams: Cams, config: RealSenseConfig) -> Optional[RealSenseCameraConfig]:
    if config.camera_name:
        for cam in cams.realsenses:
            if cam.name == config.camera_name:
                return cam
        log.error(f"No RealSense camera found with name '{config.camera_name}'")
    elif config.serial_number:
        for cam in cams.realsenses:
            if cam.serial_number == config.serial_number:
                return cam
        log.error(f"No RealSense camera found with serial '{config.serial_number}'")
    else:
        if len(cams.realsenses) == 1:
            return cams.realsenses[0]
        elif len(cams.realsenses) > 1:
            log.error("Multiple RealSense cameras found. Please specify one with --camera_name or --serial_number.")
        else:
            log.error("No RealSense cameras found in config.")
    return None

if __name__ == "__main__":
    args = setup_log_with_config(RealSenseConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)

    cams = Cams.from_yaml(args.cams_yaml)
    log.info(f"Loaded {len(cams.realsenses)} RealSense and {len(cams.ipcameras)} IP cameras from config.")

    realsense_cfg = get_realsense_config(cams, args)
    if not realsense_cfg:
        log.error("No valid RealSense camera selected. Exiting.")
    else:
        log.info(f"Using RealSense camera: {realsense_cfg.name} (serial: {realsense_cfg.serial_number})")
        print_intrinsics(realsense_cfg.serial_number)