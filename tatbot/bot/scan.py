import glob
import logging
import os
import shutil
import time
from dataclasses import dataclass
from io import StringIO

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotScanConfig
from lerobot.utils.control_utils import sanity_check_dataset_name
from lerobot.record import _init_rerun

from tatbot.bot.lerobot import safe_loop
from tatbot.bot.urdf import urdf_joints_to_action
from tatbot.data.pose import ArmPose, make_bimanual_joints
from tatbot.utils.log import (
    LOG_FORMAT,
    TIME_FORMAT,
    get_logger,
    print_config,
    setup_log_with_config,
)

log = get_logger('bot.scan', '🤖')

@dataclass
class BotScanConfig:
    debug: bool = False
    """Enable debug logging."""

    output_dir: str = "~/tatbot/nfs/bot"
    """Directory to save the dataset."""

    hf_username: str = "tatbot"
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    display_data: bool = False
    """Display data on screen using Rerun."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""
    num_images_per_camera: int = 3
    """Number of images to capture per camera."""

    left_arm_pose_name: str = "left/rest"
    """Name of the left arm pose (ArmPose)."""
    right_arm_pose_name: str = "right/rest"
    """Name of the right arm pose (ArmPose)."""

def record_scan(config: BotScanConfig):
    left_arm_pose: ArmPose = ArmPose.from_name(config.left_arm_pose_name)
    right_arm_pose: ArmPose = ArmPose.from_name(config.right_arm_pose_name)
    rest_pose = make_bimanual_joints(left_arm_pose, right_arm_pose)

    log.info("🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotScanConfig)
    robot.connect()

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"🗃️ Creating output directory at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = config.dataset_name or f"scan-{time.strftime(TIME_FORMAT, time.localtime())}"
    dataset_dir = os.path.join(output_dir, dataset_name)
    log.info(f"🗃️ Creating dataset directory at {dataset_dir}")
    os.makedirs(dataset_dir, exist_ok=True)

    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    repo_id = f"{config.hf_username}/{dataset_name}"
    log.info(f"📦🤗 Creating new LeRobot dataset at {repo_id}")
    sanity_check_dataset_name(repo_id, None)
    dataset = LeRobotDataset.create(
        repo_id,
        config.fps,
        root=dataset_dir,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
    )
    if config.display_data:
        _init_rerun(session_name="recording")

    frames_dir = os.path.join(dataset_dir, "frames")
    log.info(f"🗃️ Creating frames directory at {frames_dir}")
    os.makedirs(frames_dir, exist_ok=True)

    logs_dir = os.path.join(dataset_dir, "logs")
    log.info(f"🗃️ Creating logs directory at {logs_dir}")
    os.makedirs(logs_dir, exist_ok=True)
    episode_log_buffer = StringIO()

    class EpisodeLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            episode_log_buffer.write(msg + "\n")

    episode_handler = EpisodeLogHandler()
    episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
    logging.getLogger().addHandler(episode_handler)

    log.info("🤖 Sending robot to rest pose")
    action = urdf_joints_to_action(rest_pose)
    sent_action = robot.send_action(action, goal_time=robot.config.goal_time_slow, block="both")
    action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")

    for i in range(config.num_images_per_camera):
        log.info(f"🤖 Getting observation {i + 1} of {config.num_images_per_camera}")
        observation = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        frame = {**observation_frame, **action_frame}
        dataset.add_frame(frame, task=f"scan with all cameras, arms at rest, image {i + 1} of {config.num_images_per_camera}")

    log_path = os.path.join(logs_dir, "logs.txt")
    log.info(f"🗃️ Writing log to {log_path}")
    with open(log_path, "w") as f:
        f.write(episode_log_buffer.getvalue())

    # images get auto-deleted by lerobot, so copy them to local frames directory and un-nest the images
    images_dir = os.path.join(dataset_dir, "images")
    assert os.path.isdir(images_dir), f"LeRobot images directory {images_dir} does not exist"
    log.debug(f"🖼️ Copying images from {images_dir} to {frames_dir}")
    time.sleep(3) # wait for images to be written to disk
    for subdir in glob.glob(os.path.join(images_dir, 'observation.images.*')):
        log.debug(f"🖼️ Un-nesting images from {subdir}")
        if not os.path.isdir(subdir):
            continue
        camera_name = os.path.basename(subdir).replace('observation.images.', '')
        for i in range(config.num_images_per_camera):
            image_path = os.path.join(subdir, f"episode_000000", f"frame_{i:06d}.png")
            new_name = f"{camera_name}_{i:03d}.png"
            new_path = os.path.join(frames_dir, new_name)
            shutil.copy2(image_path, new_path)
            log.debug(f"🖼️ Copied {image_path} to {new_path}")

    dataset.save_episode()

    logging.getLogger().removeHandler(episode_handler)
    log.info("🤖✅ End")
    
    robot.disconnect()

    if config.push_to_hub:
        log.info("📦🤗 Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

if __name__ == "__main__":
    args = setup_log_with_config(BotScanConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)
        # logging.getLogger('lerobot').setLevel(logging.DEBUG)
    safe_loop(record_scan, args)