"""
runs on trossen-ai
executes plans
records to lerobot dataset format
should be lightweight and fast, robot can fail if too many things are happening at once
"""
from dataclasses import dataclass
import logging
import os
import time
from typing import Callable, Any
from io import StringIO

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import log_say
from lerobot.record import _init_rerun

from _log import get_logger, setup_log_with_config, print_config, TIME_FORMAT, LOG_FORMAT
from _plan import Plan

log = get_logger('bot')

@dataclass
class PerformConfig:
    debug: bool = False
    """Enable debug logging."""

    plan_dir: str = os.path.expanduser("~/tatbot/output/plans/calibration")
    """Directory containing plan."""

    hf_username: str = os.environ.get("HF_USER", "hu-po")
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    display_data: bool = False
    """Display data on screen using Rerun."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = True
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
    play_sounds: bool = True
    """Whether to play sounds."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""
    max_episodes: int = 256
    """Maximum number of episodes to record."""
    resume: bool = False
    """If true, resumes recording from the last episode, dataset name must match."""


def safe_loop(loop: Callable, config: Any) -> None:
    try:
        loop(config)
    except Exception as e:
        log.error(f"Error: {e}")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected. Disconnecting robot...")
    finally:
        log.info("🛑🤖 Disconnecting robot...")
        robot = make_robot_from_config(TatbotConfig())
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()


def urdf_joints_to_action(urdf_joints: list[float]) -> dict[str, float]:
    _action = {
        "left.joint_0.pos": urdf_joints[0],
        "left.joint_1.pos": urdf_joints[1],
        "left.joint_2.pos": urdf_joints[2],
        "left.joint_3.pos": urdf_joints[3],
        "left.joint_4.pos": urdf_joints[4],
        "left.joint_5.pos": urdf_joints[5],
        "left.gripper.pos": urdf_joints[6],
        "right.joint_0.pos": urdf_joints[8],
        "right.joint_1.pos": urdf_joints[9],
        "right.joint_2.pos": urdf_joints[10],
        "right.joint_3.pos": urdf_joints[11],
        "right.joint_4.pos": urdf_joints[12],
        "right.joint_5.pos": urdf_joints[13],
        "right.gripper.pos": urdf_joints[14],
    }
    log.debug(f"🦾 Action: {_action}")
    return _action

def perform(config: PerformConfig):
    plan = Plan.from_yaml(config.plan_dir)
    
    log.info("🤖🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotConfig())
    robot.connect()

    log.info("⌨️ Adding keyboard listener...")
    listener, events = init_keyboard_listener()

    dataset_name = config.dataset_name or f"{plan.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    repo_id = f"{config.hf_username}/{dataset_name}"
    if config.resume:
        log.info(f"📦🤗 Resuming LeRobot dataset at {repo_id}...")
        dataset = LeRobotDataset(
            repo_id,
            root=f"{config.output_dir}/{dataset_name}",
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=config.num_image_writer_processes,
                num_threads=config.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, config.fps, dataset_features)
    else:
        log.info(f"📦🤗 Creating new LeRobot dataset at {repo_id}...")
        sanity_check_dataset_name(repo_id, None)
        dataset = LeRobotDataset.create(
            repo_id,
            config.fps,
            root=f"{config.output_dir}/{dataset_name}",
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=config.num_image_writer_processes,
            image_writer_threads=config.num_image_writer_threads_per_camera * len(robot.cameras),
        )
    if config.display_data:
        _init_rerun(session_name="recording")

    logs_dir = os.path.expanduser(f"{config.output_dir}/{dataset_name}/logs")
    log.info(f"🗃️ Creating logs directory at {logs_dir}...")
    os.makedirs(logs_dir, exist_ok=True)
    episode_log_buffer = StringIO()

    class EpisodeLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            episode_log_buffer.write(msg + "\n")

    episode_handler = EpisodeLogHandler()
    episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
    logging.getLogger().addHandler(episode_handler)

    log.info(f"Recording {len(plan.paths)} paths...")
    # one episode is a single path
    # when resuming, start from the idx of the next episode
    for path_idx, path in enumerate(plan.paths, start=dataset.num_episodes):
        # Reset in-memory log buffer for the new episode
        episode_log_buffer.seek(0)
        episode_log_buffer.truncate(0)

        if not robot.is_connected:
            log.warning("🤖⚠️ robot is not connected, attempting reconnect...")
            robot.connect()

        if path_idx >= config.max_episodes:
            log.info(f"⚠️ max episodes {config.max_episodes} exceeded, breaking...")
            log_say(f"max paths {config.max_episodes} exceeded", config.play_sounds, blocking=True)
            break

        log.info(f"recording path {path_idx} of {len(plan.paths)}")
        log_say(f"recording path {path_idx}", config.play_sounds)
        for pose_idx in range(len(path)):
            log.debug(f"pose_idx: {pose_idx}/{len(path)}")
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            joints = 0 # TODO: get joints from plan, need to be calculated from ik
            goal_time = 0 # TODO: get goal time from plan
            action = urdf_joints_to_action(joints)
            sent_action = robot.send_action(action, goal_time=goal_time, block="left")

            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            # TODO: add pattern closeup? progress image? patch?
            _task = f"{pattern.name} tattoo pattern"
            # _task += f", {color} {needle_type} line"
            _task += f", <{path_l.metric_coords[pose_idx][0]:.2f}, {path_l.metric_coords[pose_idx][1]:.2f}> m"
            _task += f", <{path_l.pixel_coords[pose_idx][0]:.2f}, {path_l.pixel_coords[pose_idx][1]:.2f}> px"
            _task += f", pose {pose_idx} of {len(path_l)-2}"
            _task += f", path {path_idx} of {len(pattern.paths)}"
            dataset.add_frame(frame, task=_task)

            if config.display_data:
                for obs, val in observation.items():
                    if isinstance(val, float):
                        rr.log(f"observation.{obs}", rr.Scalar(val))
                    elif isinstance(val, np.ndarray):
                        rr.log(f"observation.{obs}", rr.Image(val), static=True)
                for act, val in action.items():
                    if isinstance(val, float):
                        rr.log(f"action.{act}", rr.Scalar(val))

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / config.fps - dt_s)

            if events["exit_early"]:
                log.info("🛑 exit early")
                log_say("exit", config.play_sounds)
                events["exit_early"] = False
                break

        if events["rerecord_episode"]:
            log.info("🔄 re-recording episode")
            log_say("Re-record episode", config.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        log_path = os.path.join(logs_dir, f"episode_{path_idx:06d}.txt")
        log.info(f"🗃️ Writing episode log to {log_path}")
        with open(log_path, "w") as f:
            f.write(episode_log_buffer.getvalue())

        dataset.save_episode()

        if events["stop_recording"]:
            break

    logging.getLogger().removeHandler(episode_handler)

    log_say("End", config.play_sounds, blocking=True)

    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if config.push_to_hub:
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

    log_say("Aurevoir", config.play_sounds)


if __name__ == "__main__":
    args = setup_log_with_config(PerformConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    if args.debug:
        logging.getLogger('lerobot').setLevel(logging.DEBUG)
    safe_loop(perform, args)