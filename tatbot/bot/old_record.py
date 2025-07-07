import logging
import os
import shutil
import time
from dataclasses import dataclass
from io import StringIO

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait

from tatbot.data.plan import Plan
from tatbot.data.scene import Scene
from tatbot.data.stroke import StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.utils.log import (
    LOG_FORMAT,
    TIME_FORMAT,
    get_logger,
    print_config,
    setup_log_with_config,
)

log = get_logger('bot.plan', '🤖')

@dataclass
class BotPlanConfig:
    debug: bool = False
    """Enable debug logging."""

    scene_name: str = "default"
    """Name of the scene config to use (Scene)."""

    plan_name: str = "calib"
    """Name of the plan (Plan)."""
    plan_dir: str = "~/tatbot/nfs/plans"
    """Directory containing plan."""

    output_dir: str = "~/tatbot/nfs/bot"
    """Directory to save the dataset."""

    hf_username: str = "tatbot"
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
    tags: tuple[str, ...] = ("tatbot", "wxai", "trossen")
    """Tags to add to the dataset on Hugging Face."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""
    max_episodes: int = 256
    """Maximum number of episodes to record."""
    resume: bool = False
    """If true, resumes recording from the last episode, dataset name must match."""


def record_plan(config: BotPlanConfig):
    plan_dir = os.path.expanduser(config.plan_dir)
    plan_dir = os.path.join(plan_dir, config.plan_name)
    assert os.path.exists(plan_dir), f"❌ Plan directory {plan_dir} does not exist"
    log.debug(f"📂 Plan directory: {plan_dir}")

    plan: Plan = Plan.from_yaml(os.path.join(plan_dir, "plan.yaml"))
    strokebatch: StrokeBatch = StrokeBatch.load(os.path.join(plan_dir, "strokebatch.safetensors"))
    strokes: StrokeList = StrokeList.from_yaml(os.path.join(plan_dir, "strokes.yaml"))
    num_strokes = strokebatch.joints.shape[0]

    scene: Scene = Scene.from_name(config.scene_name)

    log.info("🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotConfig(
        ip_address_l=scene.arms.ip_address_l,
        ip_address_r=scene.arms.ip_address_r,
        arm_l_config_filepath=scene.arms.arm_l_config_filepath,
        arm_r_config_filepath=scene.arms.arm_r_config_filepath,
        goal_time_fast=scene.arms.goal_time_fast,
        goal_time_slow=scene.arms.goal_time_slow,
        connection_timeout=scene.arms.connection_timeout,
        home_pos_l=scene.home_pos_l.joints,
        home_pos_r=scene.home_pos_r.joints,
        cameras={
            cam.name : RealSenseCameraConfig(
                fps=cam.fps,
                width=cam.width,
                height=cam.height,
                serial_number_or_name=cam.serial_number,
            ) for cam in scene.cams.realsenses
        },
        cond_cameras={
            cam.name : OpenCVCameraConfig(
                fps=cam.fps,
                width=cam.width,
                height=cam.height,
                ip=cam.ip,
                username=cam.username,
                password=os.environ.get(cam.password, None),
                rtsp_port=cam.rtsp_port,
            ) for cam in scene.cams.ipcameras
        }
    ))
    robot.connect()

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"🗃️ Creating output directory at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    dataset_name = config.dataset_name or f"plan-{plan.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
    dataset_dir = f"{output_dir}/{dataset_name}"
    log.info(f"🗃️ Creating dataset directory at {dataset_dir}...")
    os.makedirs(dataset_dir, exist_ok=True)

    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    repo_id = f"{config.hf_username}/{dataset_name}"
    if config.resume:
        log.info(f"📦🤗 Resuming LeRobot dataset at {repo_id}...")
        dataset = LeRobotDataset(
            repo_id,
            root=dataset_dir,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4* len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, config.fps, dataset_features)
    else:
        log.info(f"📦🤗 Creating new LeRobot dataset at {repo_id}...")
        sanity_check_dataset_name(repo_id, None)
        dataset = LeRobotDataset.create(
            repo_id,
            config.fps,
            root=dataset_dir,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * (len(robot.cameras) + len(robot.cond_cameras)),
        )

    dataset_plan_dir = os.path.join(dataset_dir, "plan")
    log.info(f"🗃️ Creating plan directory inside dataset directory at {dataset_plan_dir}...")
    os.makedirs(dataset_plan_dir, exist_ok=True)
    shutil.copytree(plan_dir, dataset_plan_dir, dirs_exist_ok=True)

    dataset_cond_dir = os.path.join(dataset_dir, "cond")
    log.info(f"🗃️ Creating condition directory inside dataset directory at {dataset_cond_dir}...")
    os.makedirs(dataset_cond_dir, exist_ok=True)

    logs_dir = os.path.join(dataset_dir, "logs")
    log.info(f"🗃️ Creating logs directory inside dataset directory at {logs_dir}...")
    os.makedirs(logs_dir, exist_ok=True)
    episode_log_buffer = StringIO()

    class EpisodeLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            episode_log_buffer.write(msg + "\n")

    episode_handler = EpisodeLogHandler()
    episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
    logging.getLogger().addHandler(episode_handler)

    log.info(f"Recording {num_strokes} paths...")
    # one episode is a single path
    # when resuming, start from the idx of the next episode
    for stroke_idx in range(dataset.num_episodes, num_strokes):
        # reset in-memory log buffer for the new episode
        episode_log_buffer.seek(0)
        episode_log_buffer.truncate(0)

        if not robot.is_connected:
            log.warning("🤖⚠️ robot is not connected, attempting reconnect...")
            robot.connect()

        if stroke_idx >= config.max_episodes:
            log.info(f"🤖⚠️ max episodes {config.max_episodes} exceeded, breaking...")
            break

        # start every episode by sending arms to rest pose
        log.debug(f"🤖 sending arms to rest pose")
        action = robot._urdf_joints_to_action(scene.home_pos_full)
        robot.send_action(action, goal_time=plan.dt_slow, block="left")

        # Per-episode conditioning information is stored in seperate directory
        episode_cond = {}
        episode_cond_dir = os.path.join(dataset_cond_dir, f"episode_{stroke_idx:06d}")
        os.makedirs(episode_cond_dir, exist_ok=True)
        log.debug(f"🗃️ Creating episode-specific condition directory at {episode_cond_dir}...")
        cond_obs = robot.get_conditioning()
        for cam_key, obs in cond_obs.items():
            filepath = os.path.join(episode_cond_dir, f"{cam_key}.png")
            dataset._save_image(obs, filepath)
            episode_cond[cam_key] = filepath
        stroke_l, stroke_r = strokes.strokes[stroke_idx]
        episode_cond["stroke_l"] = stroke_l.to_dict()
        episode_cond["stroke_r"] = stroke_r.to_dict()
        if stroke_l.frame_path is not None:
            shutil.copy(
                os.path.join(dataset_plan_dir, "frames", stroke_l.frame_path),
                os.path.join(episode_cond_dir, "stroke_l.png")
            )
        if stroke_r.frame_path is not None:
            shutil.copy(
                os.path.join(dataset_plan_dir, "frames", stroke_r.frame_path),
                os.path.join(episode_cond_dir, "stroke_r.png")
            )

        log.info(f"🤖 recording path {stroke_idx} of {num_strokes}")
        for pose_idx in range(plan.stroke_length):
            log.debug(f"pose_idx: {pose_idx}/{plan.stroke_length}")
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            joints = strokebatch.joints[stroke_idx, pose_idx]
            action = robot._urdf_joints_to_action(joints)
            goal_time = strokebatch.dt[stroke_idx, pose_idx]
            sent_action = robot.send_action(action, goal_time=goal_time, block="left")

            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(max(0, goal_time - dt_s))

        log_path = os.path.join(logs_dir, f"episode_{stroke_idx:06d}.txt")
        log.info(f"🗃️ Writing episode log to {log_path}")
        with open(log_path, "w") as f:
            f.write(episode_log_buffer.getvalue())

        dataset.save_episode(episode_cond=episode_cond)

    logging.getLogger().removeHandler(episode_handler)

    log.info("🤖✅ End")
    robot.disconnect()

    if config.push_to_hub:
        log.info("📦🤗 Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

if __name__ == "__main__":
    args = setup_log_with_config(BotPlanConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger('lerobot').setLevel(logging.DEBUG)
    try:
        record_plan(args)
    except Exception as e:
        log.error(f"❌ Robot Loop Exit with Error:\n{e}")
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard interrupt detected")
    finally:
        log.info("🛑 Disconnecting robot...")
        scene: Scene = Scene.from_name(args.scene_name)
        robot = make_robot_from_config(TatbotConfig(
            ip_address_l=scene.arms.ip_address_l,
            ip_address_r=scene.arms.ip_address_r,
            arm_l_config_filepath=scene.arms.arm_l_config_filepath,
            arm_r_config_filepath=scene.arms.arm_r_config_filepath,
            goal_time_fast=1.0, # move very slowly
            goal_time_slow=5.0, # move very slowly
            connection_timeout=20.0,
            home_pos_l=scene.home_pos_l.joints,
            home_pos_r=scene.home_pos_r.joints,
        )) # no cameras
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()