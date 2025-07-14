import logging
import os
import shutil
import time
from dataclasses import dataclass
from io import StringIO
import traceback

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
from lerobot.teleoperators.gamepad import AtariTeleoperator, AtariTeleoperatorConfig

from tatbot.data.scene import Scene
from tatbot.data.pose import ArmPose
from tatbot.gen.strokes import load_make_strokes
from tatbot.utils.log import (
    LOG_FORMAT,
    TIME_FORMAT,
    get_logger,
    print_config,
    setup_log_with_config,
)

log = get_logger('bot.record', '🤖')

@dataclass
class RecordConfig:
    debug: bool = False
    """Enable debug logging."""

    scene: str = "align"
    """Name of the scene config to use (Scene)."""

    output_dir: str = "~/tatbot/nfs/recordings"
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
    fps: int = 30
    """Frames per second."""
    max_episodes: int = 256
    """Maximum number of episodes to record."""
    resume: bool = False
    """If true, resumes recording from the last episode, dataset name must match."""

    enable_conditioning_cameras: bool = False
    """Whether to enable conditioning cameras (this slows down the loop)."""
    enable_joystick: bool = True
    """Whether to enable joystick control."""


def record(config: RecordConfig):
    scene: Scene = Scene.from_name(config.scene)

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"🗃️ Creating output directory at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = config.dataset_name or f"{scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
    dataset_dir = f"{output_dir}/{dataset_name}"
    log.info(f"🗃️ Creating dataset directory at {dataset_dir}...")
    os.makedirs(dataset_dir, exist_ok=True)

    # copy the scene yaml to the output directory
    scene_path = os.path.join(dataset_dir, "scene.yaml")
    scene.to_yaml(scene_path)

    strokes, strokebatch = load_make_strokes(scene, dataset_dir, config.resume)
    num_strokes = len(strokes.strokes)

    log.info("🤗 Adding LeRobot robot...")
    robot = make_robot_from_config(TatbotConfig(
        ip_address_l=scene.arms.ip_address_l,
        ip_address_r=scene.arms.ip_address_r,
        arm_l_config_filepath=scene.arms.arm_l_config_filepath,
        arm_r_config_filepath=scene.arms.arm_r_config_filepath,
        goal_time_fast=scene.arms.goal_time_fast,
        goal_time_slow=scene.arms.goal_time_slow,
        connection_timeout=scene.arms.connection_timeout,
        home_pos_l=scene.sleep_pos_l.joints[:7],
        home_pos_r=scene.sleep_pos_r.joints[:7],
        cameras={},
        cond_cameras={},
        # cameras={
        #     cam.name : RealSenseCameraConfig(
        #         fps=cam.fps,
        #         width=cam.width,
        #         height=cam.height,
        #         serial_number_or_name=cam.serial_number,
        #     ) for cam in scene.cams.realsenses
        # },
        # cond_cameras={
        #     cam.name : OpenCVCameraConfig(
        #         fps=cam.fps,
        #         width=cam.width,
        #         height=cam.height,
        #         ip=cam.ip,
        #         username=cam.username,
        #         password=os.environ.get(cam.password, None),
        #         rtsp_port=cam.rtsp_port,
        #     ) for cam in scene.cams.ipcameras
        # }
    ))
    robot.connect()

    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    repo_id = f"{config.hf_username}/{dataset_name}"
    if config.resume:
        log.info(f"📦🤗 Resuming LeRobot dataset at {repo_id}...")
        dataset = LeRobotDataset(repo_id, root=dataset_dir)
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

    dataset_cond_dir = os.path.join(dataset_dir, "cond")
    log.info(f"🗃️ Creating condition directory inside dataset directory at {dataset_cond_dir}...")
    os.makedirs(dataset_cond_dir, exist_ok=True)
    if config.enable_conditioning_cameras:
        cond_obs = robot.get_conditioning()
        for cam_key, obs in cond_obs.items():
            filepath = os.path.join(dataset_cond_dir, f"{cam_key}.png")
            dataset._save_image(obs, filepath)

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

    if config.enable_joystick:
        log.info("🎮 Enabling joystick control...")
        atari_teleop = AtariTeleoperator(AtariTeleoperatorConfig())
        atari_teleop.connect()
    
    # offset index controls needle depth
    mid_offset_idx: int = scene.offset_num // 2
    offset_idx_l: int = mid_offset_idx
    offset_idx_r: int = mid_offset_idx
    # inkdip-specific offset index
    inkdip_offset_idx_l: int = mid_offset_idx
    inkdip_offset_idx_r: int = mid_offset_idx
    
    # one episode is a single path
    # when resuming, start from the idx of the next episode
    log.info(f"Recording {num_strokes} paths...")
    for stroke_idx in range(dataset.num_episodes, num_strokes):
        # reset in-memory log buffer for the new episode
        episode_log_buffer.seek(0)
        episode_log_buffer.truncate(0)

        if not robot.is_connected:
            log.warning("🤖⚠️ robot is not connected, attempting reconnect...")
            robot.connect()
            if not robot.is_connected:
                log.error("❌ Robot Loop Exit with Error:\n" + traceback.format_exc())
                raise Exception("🤖⚠️ robot is not connected, cannot record")

        if stroke_idx >= config.max_episodes:
            log.info(f"🤖⚠️ max episodes {config.max_episodes} exceeded, breaking...")
            break

        # get the strokes that will be executed this episode
        stroke_l, stroke_r = strokes.strokes[stroke_idx]

        # send the arms to the appropriate "ready" pose
        _joints_l = scene.ready_pos_l
        _joints_r = scene.ready_pos_r
        if stroke_l.is_inkdip:
            log.info(f"🤖 sending left arm to inkready pose")
            _joints_l = scene.inkready_pos_l
        if stroke_r.is_inkdip:
            log.info(f"🤖 sending right arm to inkready pose")
            _joints_r = scene.inkready_pos_r
        _full_joints = ArmPose.make_bimanual_joints(_joints_l, _joints_r)
        ready_action = robot._urdf_joints_to_action(_full_joints)
        robot.send_action(ready_action, goal_time=robot.config.goal_time_slow, block="left")

        # Per-episode conditioning information is stored in seperate directory
        # TODO: add conditioning cameras, but right now they slow down the loop too much
        episode_cond = {}
        episode_cond_dir = os.path.join(dataset_cond_dir, f"episode_{stroke_idx:06d}")
        os.makedirs(episode_cond_dir, exist_ok=True)
        log.debug(f"🗃️ Creating episode-specific condition directory at {episode_cond_dir}...")
        episode_cond["stroke_l"] = stroke_l.to_dict()
        episode_cond["stroke_r"] = stroke_r.to_dict()
        if stroke_l.frame_path is not None:
            shutil.copy(stroke_l.frame_path, os.path.join(episode_cond_dir, "stroke_l.png"))
        if stroke_r.frame_path is not None:
            shutil.copy(stroke_r.frame_path, os.path.join(episode_cond_dir, "stroke_r.png"))

        log.info(f"🤖 recording path {stroke_idx} of {num_strokes}")
        for pose_idx in range(scene.stroke_length):
            log.debug(f"pose_idx: {pose_idx}/{scene.stroke_length}")
            start_loop_t = time.perf_counter()

            if config.enable_joystick:
                action = atari_teleop.get_action()
                if action.get("red_button", False):
                    raise KeyboardInterrupt()
                if action.get("y", None) is not None:
                    if stroke_l.is_inkdip:
                        inkdip_offset_idx_l += int(action["y"])
                        inkdip_offset_idx_l = min(inkdip_offset_idx_l, scene.offset_num - 1)
                        inkdip_offset_idx_l = max(0, inkdip_offset_idx_l)
                    else:
                        offset_idx_l += int(action["y"])
                        offset_idx_l = min(offset_idx_l, scene.offset_num - 1)
                        offset_idx_l = max(0, offset_idx_l)
                if action.get("x", None) is not None:
                    if stroke_r.is_inkdip:
                        inkdip_offset_idx_r += int(action["x"])
                        inkdip_offset_idx_r = min(inkdip_offset_idx_r, scene.offset_num - 1)
                        inkdip_offset_idx_r = max(0, inkdip_offset_idx_r)
                    else:
                        offset_idx_r += int(action["x"])
                        offset_idx_r = min(offset_idx_r, scene.offset_num - 1)
                        offset_idx_r = max(0, offset_idx_r)

            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            if stroke_l.is_inkdip:
                _offset_idx_l = inkdip_offset_idx_l
                log.info(f"🎮 left inkdip offset index: {_offset_idx_l}")
            else:
                _offset_idx_l = offset_idx_l
                log.info(f"🎮 left offset index: {_offset_idx_l}")
            if stroke_r.is_inkdip:
                _offset_idx_r = inkdip_offset_idx_r
                log.info(f"🎮 right inkdip offset index: {_offset_idx_r}")
            else:
                _offset_idx_r = offset_idx_r
                log.info(f"🎮 right offset index: {_offset_idx_r}")
            joints = strokebatch.offset_joints(stroke_idx, pose_idx, _offset_idx_l, _offset_idx_r)
            robot_action = robot._urdf_joints_to_action(joints)
            goal_time = float(strokebatch.dt[stroke_idx, pose_idx, offset_idx_l]) # TODO: this is a hack, currently dt is the same for both arms
            sent_action = robot.send_action(robot_action, goal_time=goal_time, block="left")

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

        # re-send the arms to the appropriate "ready" pose
        robot.send_action(ready_action, goal_time=robot.config.goal_time_slow, block="left")

    logging.getLogger().removeHandler(episode_handler)

    log.info("✅ Done")
    robot.disconnect()

    if config.push_to_hub:
        log.info("📦🤗 Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

if __name__ == "__main__":
    args = setup_log_with_config(RecordConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger('lerobot').setLevel(logging.DEBUG)
    try:
        record(args)
    except Exception as e:
        log.error("❌ Robot Loop Exit with Error:\n" + traceback.format_exc())
    except KeyboardInterrupt:
        log.info("🛑⌨️ Keyboard/E-stop interrupt detected")
    finally:
        log.info("🛑 Disconnecting robot...")
        scene: Scene = Scene.from_name(args.scene)
        robot = make_robot_from_config(TatbotConfig(
            ip_address_l=scene.arms.ip_address_l,
            ip_address_r=scene.arms.ip_address_r,
            arm_l_config_filepath=scene.arms.arm_l_config_filepath,
            arm_r_config_filepath=scene.arms.arm_r_config_filepath,
            goal_time_fast=1.0, # move very slowly
            goal_time_slow=5.0, # move very slowly
            connection_timeout=20.0,
            home_pos_l=scene.sleep_pos_l.joints[:7],
            home_pos_r=scene.sleep_pos_r.joints[:7],
            # no cameras
            cameras={},
            cond_cameras={},
        ))
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()
        # double tap
        robot._connect_l()
        robot._connect_r()
        robot.disconnect()