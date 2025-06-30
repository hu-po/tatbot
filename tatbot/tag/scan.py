import glob
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from _bot import get_link_poses
from _log import TIME_FORMAT, get_logger, print_config, setup_log_with_config
from _scan import Scan
from _tag import TagTracker

log = get_logger('run_scan')


@dataclass
class RunScanConfig:
    debug: bool = False
    """Enable debug logging."""

    bot_scan_dir: str = ""
    """Path to the bot scan directory."""

    output_dir: str = "~/tatbot/output/scans"
    """Directory to save the scan."""

def run_scan(config: RunScanConfig) -> None:
    bot_scan_dir = os.path.expanduser(config.bot_scan_dir)
    frames_dir = os.path.join(bot_scan_dir, "frames")
    assert os.path.exists(frames_dir), f"Frames directory {frames_dir} does not exist"
    log.info(f"🔎🗃️ Ingesting bot scan at {bot_scan_dir} with frames")

    scan_name = f"{time.strftime(TIME_FORMAT, time.localtime())}"
    scan = Scan(name=scan_name)

    output_dir = os.path.expanduser(config.output_dir)
    output_dir = os.path.join(output_dir, scan_name)
    log.info(f"🔎🗃️ Creating output directory at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    log.info("🔎🗃️ Populating camera extrinsics from URDF...")
    link_names = scan.optical_frame_urdf_link_names.values()
    link_poses = get_link_poses(link_names, bot_config=scan.bot_config)
    for camera_name, link_name in scan.optical_frame_urdf_link_names.items():
        scan.extrinsics[camera_name].pos = np.array(link_poses[link_name][0])
        scan.extrinsics[camera_name].wxyz = np.array(link_poses[link_name][1])
        log.info(f"🔎🗃️ Camera {camera_name} extrinsics: {scan.extrinsics[camera_name]}")

    log.info("📡 Tracking tags in images...")
    tracker = TagTracker(scan.tag_config)
    for image_path in glob.glob(os.path.join(frames_dir, '*.png')):
        camera_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]
        output_path = os.path.join(output_dir, image_path.split('/')[-1].replace('.png', '_tagged.png'))
        log.info(f"🔎🗃️ Tracking tags in {image_path} and saving to {output_path}")
        # TODO: get camera_pos and camera_wxyz from URDF? initialize as identity?
        tracker.track_tags(
            image_path,
            scan.intrinsics[camera_name],
            scan.extrinsics[camera_name].pos,
            scan.extrinsics[camera_name].wxyz,
            output_path=output_path
        )

    # use origin tag to get camera extrinsics of realsense1, realsense2, camera2, camera3, camera4
    # use arm_l and arm_r tags to get extrinsics of camera1, camera5
    # use camera extrinsics to get palette and skin tags

    # update URDF file? save to scan metadata?

    scan.save(output_dir)
    log.info("🔎✅ Done")

if __name__ == "__main__":
    args = setup_log_with_config(RunScanConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
    print_config(args)
    run_scan(args)