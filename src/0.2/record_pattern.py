"""
Records episodes using a Pattern from JSON.
"""

from dataclasses import dataclass
import json
import logging
import os
import time
from io import StringIO

import cv2
import jax
import jax.numpy as jnp
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
import numpy as np
import PIL.Image
import pyroki as pk
import rerun as rr
import tyro
import viser
from viser.extras import ViserUrdf
import yourdfpy

from _ik import IKConfig, ik
from _bot import ik_solution_to_action, robot_safe_loop
from _log import COLORS, get_logger, setup_log_with_config, print_config, TIME_FORMAT, LOG_FORMAT
from _path import Pattern, offset_path, add_entry_exit_hover

log = get_logger('record_pattern')

@dataclass
class RecordPatternConfig:
    debug: bool = False
    """Enable debug logging."""
    seed: int = 42
    """Seed for random behavior."""

    pattern_dir: str = os.path.expanduser("~/tatbot/output/patterns/calibration")
    """Directory with pattern.json and image.png."""

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

    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the ee links in the URDF for left and right ik solving."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""
    robot_goal_time_slow: float = 2.8
    """Goal time for the robot when moving slowly."""
    robot_goal_time_fast: float = 0.1
    """Goal time for the robot when moving fast."""

    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""

    joint_pos_design_l: tuple[float, float, float, float, float, float, float] = (0.1, 1.23, 1.01, -1.35, 0.0, 0.0, 0.02)
    """Joint positions of the left arm for robot hovering over design."""
    joint_pos_design_r: tuple[float, float, float, float, float, float, float] = (3.05, 0.49, 1.09, -1.52, 0.0, 0.0, 0.04)
    """Joint positions of the rgiht arm for robot hovering over design."""

    ee_design_pos: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """position of the design ee transform."""
    ee_design_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the design ee transform."""

    hover_offset: tuple[float, float, float] = (0.0, 0.0, 0.006)
    """position offset when hovering over point, relative to current ee frame."""
    needle_offset: tuple[float, float, float] = (0.0, 0.0, -0.0065)
    """position offset to ensure needle touches skin, relative to current ee frame."""

    view_offset: tuple[float, float, float] = (0.0, -0.16, 0.16)
    """position offset when viewing design with right arm (relative to design ee frame)."""
    ee_view_wxyz: tuple[float, float, float, float] = (0.67360666, -0.25201478, 0.24747439, 0.64922119)
    """orientation quaternion (wxyz) of the view ee transform."""

    alignment_timeout: float = 20.0
    """Timeout for alignment in seconds."""
    alignment_interval: float = 1.0
    """Interval for alignment switching between design and inkcap."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""
    dip_offset: tuple[float, float, float] = (0.0, 0.0, -0.029)
    """position offset when dipping inkcap (relative to current ee frame)."""

    ink_dip_every_n_poses: int = 64
    """Dip ink every N poses, will complete the full path before dipping again."""


def record_pattern(config: RecordPatternConfig):
    log.info(f"🌱 Setting random seed to {config.seed}...")
    rng = jax.random.PRNGKey(config.seed)

    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    with server.gui.add_folder("Pattern"):
        image_path = os.path.join(config.pattern_dir, "image.png")
        assert os.path.exists(image_path), f"Pattern image not found at {image_path}"
        log.info(f"🖼️ Loading pattern image from {image_path}...")
        img_np = np.array(PIL.Image.open(image_path).convert("RGB"))
        viser_img = server.gui.add_image(
            label="pattern",
            image=img_np,
            format="png",
        )
        pathlen_image_path = os.path.join(config.pattern_dir, "pathlen.png")
        log.info(f"🖼️ Loading pathlen image from {pathlen_image_path}...")
        pathlen_img_np = np.array(PIL.Image.open(pathlen_image_path).convert("RGB"))
        server.gui.add_image(
            label="pathlen",
            image=pathlen_img_np,
            format="png",
        )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = config.view_camera_position
        client.camera.look_at = config.view_camera_look_at

    log.info("🦾 Adding vizer robot...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(config.urdf_path)
    pk_robot: pk.Robot = pk.Robot.from_urdf(urdf)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")

    log.info("🔍 Loading pattern...")
    pattern_path = os.path.join(config.pattern_dir, "pattern.json")
    assert os.path.exists(pattern_path), f"Pattern file not found at {pattern_path}"
    with open(pattern_path, "r") as f:
        pattern_json = json.load(f)
    pattern = Pattern.from_json(pattern_json)
    log.info(f"Loaded pattern '{pattern.name}' with {len(pattern.paths)} paths.")

    log.info("🦾 Adding lerobot robot...")
    robot = make_robot_from_config(TatbotConfig(
        goal_time_slow=config.robot_goal_time_slow,
        goal_time_fast=config.robot_goal_time_fast,
    ))
    robot.connect()
    listener, events = init_keyboard_listener()

    log.info("📦 Initializing dataset...")
    if config.display_data:
        _init_rerun(session_name="recording")
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    dataset_name = config.dataset_name or f"{pattern.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
    repo_id = f"{config.hf_username}/{dataset_name}"
    if config.resume:
        log.info("📦 Resuming dataset...")
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
        log.info("📦 Creating new dataset...")
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

    # convert to jnp arrays for jax operations
    # ee_ means end effector, safe to use as ik target
    ee_design_pos = jnp.array(config.ee_design_pos)
    ee_design_wxyz = jnp.array(config.ee_design_wxyz)
    hover_offset = jnp.array(config.hover_offset)
    needle_offset = jnp.array(config.needle_offset)
    view_offset = jnp.array(config.view_offset)
    ee_view_wxyz = jnp.array(config.ee_view_wxyz)
    ee_inkcap_pos = jnp.array(config.ee_inkcap_pos)
    ee_inkcap_wxyz = jnp.array(config.ee_inkcap_wxyz)
    dip_offset = jnp.array(config.dip_offset)
    target_link_indices = jnp.array([
        pk_robot.links.names.index(config.target_links_name[0]),
        pk_robot.links.names.index(config.target_links_name[1]),
    ])

    log.info("📐 Waiting for alignment, press right arrow key to continue...")
    start_time = time.time()
    while True:
        # if time.time() - start_time > config.alignment_timeout:
        #     log.error("❌📐 Alignment timeout")
        #     log_say("alignment timeout", config.play_sounds)
        #     raise RuntimeError("❌📐 Alignment timeout")
        # # TODO: keyboard listener does not work anymore?
        # if events["exit_early"]:
        #     log.info("✅📐 Alignment complete")
        #     log_say("alignment complete", config.play_sounds)
        #     break
        if time.time() - start_time > config.alignment_timeout:
            log.info("✅📐 Alignment complete")
            log_say("alignment complete", config.play_sounds, blocking=True)
            break
        log.info("📐 Aligning over design...")
        solution = ik(
            robot=pk_robot,
            target_link_indices=target_link_indices[0],
            target_wxyz=ee_design_wxyz,
            target_position=ee_design_pos,
            config=config.ik_config,
        )
        urdf_vis.update_cfg(np.array(solution))
        robot._set_positions_l(solution[:7], goal_time=robot.config.goal_time_slow)
        log_say("align design", config.play_sounds)
        time.sleep(config.alignment_interval)
        log.info("📐 Aligning over inkcap...")
        solution = ik(
            robot=pk_robot,
            target_link_indices=target_link_indices[0],
            target_wxyz=ee_inkcap_wxyz,
            target_position=ee_inkcap_pos,
            config=config.ik_config,
        )
        urdf_vis.update_cfg(np.array(solution))
        robot._set_positions_l(solution[:7], goal_time=robot.config.goal_time_slow)
        log_say("align inkcap", config.play_sounds)
        time.sleep(config.alignment_interval)

    ink_dip_tracker: int = 0
    has_dipped: bool = False
    log.info(f"🎨 ink dip tracker at {ink_dip_tracker}, threshold at {config.ink_dip_every_n_poses}")

    log.info(f"Recording {len(pattern.paths)} paths...")
    # when resuming, start from the idx of the next episode
    for path_idx, path in enumerate(pattern.paths, start=dataset.num_episodes):
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

        log_say("calculating paths", config.play_sounds)
        # path needs to be offset to the design position
        path_l = offset_path(path, ee_design_pos)
        # center the path in design frame
        path_l = offset_path(path_l, jnp.array([-pattern.width_m / 2, -pattern.height_m / 2, 0.0]))
        # append hover position to the beginnning and end of path
        path_l = add_entry_exit_hover(path_l, hover_offset)
        # add needle depth offset
        path_l = offset_path(path_l, needle_offset)
        # right hand follows left hand with a view offset
        path_r = offset_path(path_l, view_offset)

        perform_ink_dip: bool = False
        if not has_dipped:
            log.info("🎨 dipping ink for the first time")
            has_dipped = True
            perform_ink_dip = True
        elif ink_dip_tracker >= config.ink_dip_every_n_poses:
            log.info(f"🎨 dipping ink since ({ink_dip_tracker} >= {config.ink_dip_every_n_poses})")
            perform_ink_dip = True
        if perform_ink_dip:
            log.info("✒️ dipping ink...")
            log_say("dipping ink", config.play_sounds)
            for desc, (pose_l, pose_r) in [
                ("hover over inkcap", (ee_inkcap_pos, path_r.positions[0])),
                ("dip into inkcap", (ee_inkcap_pos + dip_offset, path_r.positions[0])),
                ("retract from inkcap", (ee_inkcap_pos, path_r.positions[0])),
                ("hover over path", (path_l.positions[0], path_r.positions[0])),
            ]:
                log_say(desc, config.play_sounds)
                solution = ik(
                    robot=pk_robot,
                    target_link_indices=target_link_indices,
                    target_wxyz=jnp.array([ee_design_wxyz, ee_view_wxyz]),
                    target_position=jnp.array([pose_l, pose_r]),
                    config=config.ik_config,
                )
                urdf_vis.update_cfg(np.array(solution))
                action = ik_solution_to_action(solution)
                robot.send_action(action, goal_time=robot.config.goal_time_slow, block="both")
            ink_dip_tracker = 0

        log.info(f"recording path {path_idx} of {len(pattern.paths)}")
        log_say(f"recording path {path_idx}", config.play_sounds)
        for pose_idx in range(len(path_l)):
            log.debug(f"pose_idx: {pose_idx}/{len(path_l)}")
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            solution = ik(
                robot=pk_robot,
                target_link_indices=target_link_indices,
                target_wxyz=jnp.array([ee_design_wxyz, ee_view_wxyz]),
                target_position=jnp.array([path_l.positions[pose_idx], path_r.positions[pose_idx]]),
                config=config.ik_config,
            )
            urdf_vis.update_cfg(np.array(solution))
            action = ik_solution_to_action(solution)
            if pose_idx <= 1 or pose_idx >= len(path_l) - 2:
                # move slowly into and out of hover positions
                sent_action = robot.send_action(action, goal_time=robot.config.goal_time_slow, block="both")
            else:
                sent_action = robot.send_action(action, goal_time=robot.config.goal_time_fast, block="left")
                ink_dip_tracker += 1

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

            log.info(f"🖼️ Updating visualization...")
            _img_np = img_np.copy()
            for pw, ph in path_l.pixel_coords[:pose_idx]:
                # small circle for all poses up to current pose
                cv2.circle(_img_np, (int(pw), int(ph)), 3, COLORS["green"], -1)
            # bigger circle indicating current pose
            cv2.circle(_img_np, (int(path_l.pixel_coords[pose_idx][0]), int(path_l.pixel_coords[pose_idx][1])), 6, COLORS["magenta"], -1)
            # even bigger circle indicating current pose for easier visual tracking
            cv2.circle(_img_np, (int(path_l.pixel_coords[pose_idx][0]), int(path_l.pixel_coords[pose_idx][1])), 64, COLORS["red"], -1)
            viser_img.image = _img_np

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
    args = setup_log_with_config(RecordPatternConfig)
    print_config(args)
    # TODO: waiting on https://github.com/TrossenRobotics/trossen_arm/issues/86#issue-3144375498
    logging.getLogger('trossen_arm').setLevel(logging.ERROR)
    logging.getLogger('lerobot').setLevel(logging.DEBUG)
    robot_safe_loop(record_pattern, args)