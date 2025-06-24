from dataclasses import dataclass
import logging
import os
import shutil
import time
from io import StringIO

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotScanConfig
from lerobot.common.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.record import _init_rerun

from _bot import urdf_joints_to_action, safe_loop, BotConfig
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
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    num_image_writer_processes: int = 0
    """
    Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    """
    num_image_writer_threads_per_camera: int = 4
    """
    Number of threads writing the frames as png images on disk, per camera.
    Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    Not enough threads might cause low camera fps.
    """
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""
    num_steps: int = 2
    """Number of steps to perform in one scan."""

def record_scan(config: BotScanConfig):
    scan = Scan()
    tracker = TagTracker(scan.tag_config)

    log.info("🤖🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotScanConfig)
    robot.connect()

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
        root=f"{config.output_dir}/{dataset_name}",
        robot_type=robot.name,
        features=dataset_features,
        use_videos=False, # we want images, not videos
        image_writer_processes=config.num_image_writer_processes,
        image_writer_threads=config.num_image_writer_threads_per_camera * len(robot.cameras),
    )
    if config.display_data:
        _init_rerun(session_name="recording")

    scan_dir = os.path.expanduser(f"{config.output_dir}/{dataset_name}/scan")
    log.info(f"🤖🗃️ Creating scan directory at {scan_dir}...")
    os.makedirs(scan_dir, exist_ok=True)
    scan.save(scan_dir)

    logs_dir = os.path.expanduser(f"{config.output_dir}/{dataset_name}/logs")
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
    for step in range(config.num_steps):
        log.debug(f"step: {step}/{config.num_steps}")
        start_loop_t = time.perf_counter()
        observation = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        action = urdf_joints_to_action(BotConfig().rest_pose)
        sent_action = robot.send_action(action, goal_time=robot.config.goal_time_slow, block="both")

        action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
        frame = {**observation_frame, **action_frame}
        dataset.add_frame(frame, task=f"scan, step {step} of {config.num_steps}")

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(max(0, dt_s))

    log_path = os.path.join(logs_dir, f"episode_{0:06d}.txt")
    log.info(f"🤖🗃️ Writing episode log to {log_path}")
    with open(log_path, "w") as f:
        f.write(episode_log_buffer.getvalue())

    dataset.save_episode()

    logging.getLogger().removeHandler(episode_handler)

    log.info("🤖✅ End")
    robot.disconnect()

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