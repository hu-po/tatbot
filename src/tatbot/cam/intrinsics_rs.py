import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pyrealsense2 as rs

from tatbot.data.cams import Cams, RealSenseCameraConfig
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger("cam.intrinsics_rs", "📷")


@dataclass
class RealSenseConfig:
    debug: bool = False
    camera_name: Optional[str] = None
    serial_number: Optional[str] = None
    cams_yaml: str = "~/tatbot/config/cams/fast.yaml"
    """Path to the camera config YAML file."""


def get_local_realsense_serials() -> List[str]:
    context = rs.context()
    devices = context.query_devices()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
    log.info(f"Found {len(serials)} RealSense device(s) connected: {serials}")
    return serials


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


def match_realsense_devices(cams: Cams, local_serials: List[str]) -> Dict[str, Any]:
    config_serials = [cam.serial_number for cam in cams.realsenses]
    serial_to_cfg = {cam.serial_number: cam for cam in cams.realsenses}
    intersection = [serial_to_cfg[s] for s in local_serials if s in config_serials]
    only_in_config = [cam for cam in cams.realsenses if cam.serial_number not in local_serials]
    only_on_hardware = [s for s in local_serials if s not in config_serials]
    return {
        "intersection": intersection,
        "only_in_config": only_in_config,
        "only_on_hardware": only_on_hardware,
    }


def select_realsense_config(
    matched: Dict[str, List[RealSenseCameraConfig]], args: RealSenseConfig
) -> Optional[RealSenseCameraConfig]:
    intersection = matched["intersection"]
    if not intersection:
        log.error("No RealSense cameras are both connected and configured. Exiting.")
        return None
    if args.camera_name:
        for cam in intersection:
            if cam.name == args.camera_name:
                return cam
        log.error(f"No connected/configured RealSense camera found with name '{args.camera_name}'")
        return None
    if args.serial_number:
        for cam in intersection:
            if cam.serial_number == args.serial_number:
                return cam
        log.error(f"No connected/configured RealSense camera found with serial '{args.serial_number}'")
        return None
    if len(intersection) == 1:
        return intersection[0]
    log.error(
        "Multiple RealSense cameras are both connected and configured. Please specify one with --camera_name or --serial_number."
    )
    log.info(f"Available: {[cam.name + ' (' + cam.serial_number + ')' for cam in intersection]}")
    return None


if __name__ == "__main__":
    args = setup_log_with_config(RealSenseConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)

    cams = Cams.from_yaml(args.cams_yaml)
    log.info(f"Loaded {len(cams.realsenses)} RealSense and {len(cams.ipcameras)} IP cameras from config.")

    local_serials = get_local_realsense_serials()
    matched = match_realsense_devices(cams, local_serials)

    log.info(f"Connected & Configured: {[cam.serial_number for cam in matched['intersection']]}")
    log.info(f"Configured but not connected: {[cam.serial_number for cam in matched['only_in_config']]}")
    log.info(f"Connected but not configured: {matched['only_on_hardware']}")

    realsense_cfg = select_realsense_config(matched, args)
    if not realsense_cfg:
        log.error("No valid RealSense camera selected. Exiting.")
    else:
        log.info(f"Using RealSense camera: {realsense_cfg.name} (serial: {realsense_cfg.serial_number})")
        print_intrinsics(realsense_cfg.serial_number)
