from dataclasses import dataclass
import glob
import logging
import os
import shutil
import time
from io import StringIO

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotScanConfig
from lerobot.common.utils.control_utils import sanity_check_dataset_name
from lerobot.record import _init_rerun
import numpy as np

from _bot import urdf_joints_to_action, safe_loop, BotConfig, get_link_indices, get_link_poses
from _log import get_logger, setup_log_with_config, print_config, TIME_FORMAT, LOG_FORMAT
from _scan import Scan
from _tag import TagTracker

log = get_logger('bot_scan')

@dataclass
class BotScanConfig:
    debug: bool = False
    """Enable debug logging."""

    hf_username: str = "tatbot"
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    display_data: bool = False
    """Display data on screen using Rerun."""
    output_dir: str = "~/tatbot/output/record"
    """Directory to save the dataset."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""

def record_scan(config: BotScanConfig):
    log.info("🤖🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotScanConfig)
    robot.connect()

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"🤖🗃️ Creating output directory at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = config.dataset_name or f"scan-{time.strftime(TIME_FORMAT, time.localtime())}"
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    repo_id = f"{config.hf_username}/{dataset_name}"
    log.info(f"🤖📦🤗 Creating new LeRobot dataset at {repo_id}...")
    sanity_check_dataset_name(repo_id, None)
    dataset = LeRobotDataset.create(
        repo_id,
        config.fps,
        root=f"{output_dir}/{dataset_name}",
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )
    if config.display_data:
        _init_rerun(session_name="recording")

    scan_dir = f"{output_dir}/{dataset_name}/scan"
    log.info(f"🤖🗃️ Creating scan directory at {scan_dir}...")
    os.makedirs(scan_dir, exist_ok=True)

    logs_dir = f"{output_dir}/{dataset_name}/logs"
    log.info(f"🤖🗃️ Creating logs directory at {logs_dir}...")
    os.makedirs(logs_dir, exist_ok=True)
    episode_log_buffer = StringIO()

    class EpisodeLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            episode_log_buffer.write(msg + "\n")

    episode_handler = EpisodeLogHandler()
    episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
    logging.getLogger().addHandler(episode_handler)

    log.info(f"🤖 performing scan...")
    action = urdf_joints_to_action(BotConfig().rest_pose)
    sent_action = robot.send_action(action, goal_time=robot.config.goal_time_slow, block="both")
    action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
    observation = robot.get_observation()
    observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
    frame = {**observation_frame, **action_frame}
    dataset.add_frame(frame, task="scan with all cameras, arms at rest")

    log_path = os.path.join(logs_dir, f"logs.txt")
    log.info(f"🤖🗃️ Writing episode log to {log_path}")
    with open(log_path, "w") as f:
        f.write(episode_log_buffer.getvalue())

    # images get auto-deleted by lerobot, so copy them to local scan directory and un-nest them
    images_dir = os.path.expanduser(dataset.root / "images")
    assert os.path.isdir(images_dir), f"Images directory {images_dir} does not exist"
    shutil.copytree(images_dir, scan_dir, dirs_exist_ok=True)
    # Un-nest images from subdirectories and rename them to <camera_name>_<frame_idx>.png
    for subdir in glob.glob(os.path.join(scan_dir, 'observation.images.*')):
        if not os.path.isdir(subdir):
            continue
        camera_name = os.path.basename(subdir).replace('observation.images.', '')
        images = sorted(glob.glob(os.path.join(subdir, '**', '*.png'), recursive=True))
        for frame_idx, img_path in enumerate(images):
            new_name = f"{camera_name}_{frame_idx:03d}.png"
            new_path = os.path.join(scan_dir, new_name)
            shutil.copy2(img_path, new_path)
            log.debug(f"🤖🖼️  Un-nesting image {img_path} to {new_path}")

    dataset.save_episode()

    logging.getLogger().removeHandler(episode_handler)

    log.info("🤖✅ End")
    robot.disconnect()

    # track tags in each of the images
    scan = Scan()
    tracker = TagTracker(scan.tag_config)
    for image_path in glob.glob(os.path.join(scan_dir, '*.png')):
        camera_name = os.path.splitext(os.path.basename(image_path))[0].split('_')[0]  # e.g., 'camera1'
        # TODO: get camera_pos and camera_wxyz from URDF? initialize as identity?
        tracker.track_tags(
            image_path,
            scan.intrinsics[camera_name],
            np.array(scan.extrinsics[camera_name].pos),
            np.array(scan.extrinsics[camera_name].wxyz),
            output_path=image_path
        )

    # use origin tag to get camera extrinsics of realsense1, realsense2, camera2, camera3, camera4
    # use arm_l and arm_r tags to get extrinsics of camera1, camera5
    # use camera extrinsics to get palette and skin tags

    # update URDF file? save to scan metadata?

    scan.save()

    if config.push_to_hub:
        log.info("🤖📦🤗 Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

if __name__ == "__main__":
    args = setup_log_with_config(BotScanConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger('lerobot').setLevel(logging.DEBUG)
    safe_loop(record_scan, args)