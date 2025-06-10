from dataclasses import asdict, dataclass
import json
import logging
import os
from pprint import pformat
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import rerun as rr
import viser
from viser.extras import ViserUrdf
import yourdfpy
from typing import Dict, Any
import tyro

from lerobot.record import _init_rerun, record_loop
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.common.utils.utils import log_say
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.control_utils import init_keyboard_listener, is_headless, sanity_check_dataset_name

from ik import IKConfig, ik

log = logging.getLogger('tatbot')

@dataclass
class ToolpathConfig:
    debug: bool = False
    """Enable debug logging."""

    toolpath_path: str = os.path.expanduser("~/tatbot/output/design/cat_toolpaths.json")
    """Local path to the toolpath file generated with design.py file."""

    dataset_name: str = f"test-{int(time.time())}"
    """Name of the dataset to record."""
    display_data: bool = False
    """Display data on screen using Rerun."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = False
    """Push the dataset to the Hugging Face Hub."""
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
    fps: int = 30
    """Frames per second."""

    seed: int = 42
    """Seed for random behavior."""
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_link_name: str = "left/tattoo_needle"
    """Name of the link to be controlled."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""

    view_camera_position: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial camera position in the Viser scene."""
    view_camera_look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera look_at in the Viser scene."""
    env_map_hdri: str = "forest"
    """HDRI for the environment map."""

    joint_pos_design_l: tuple[float, float, float, float, float, float, float] = (0.10, 1.23, 1.01, -1.35, 0, 0, 0.02)
    """Joint positions of the left arm for robot hovering over design."""
    joint_pos_design_r: tuple[float, float, float, float, float, float, float] = (3.05, 0.49, 1.09, -1.52, 0, 0, 0.04)
    """Joint positions of the rgiht arm for robot hovering over design."""

    ee_design_pos: tuple[float, float, float] = (0.08, 0.0, 0.04)
    """position of the design ee transform."""
    ee_design_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the design ee transform."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_dip: tuple[float, float, float] = (0.0, 0.0, -0.02)
    """dip vector when performing inkcap dip."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""


def main(config: ToolpathConfig):
    config = config
    log.info(f"🌱 Setting random seed to {config.seed}...")
    rng = jax.random.PRNGKey(config.seed)

    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = config.view_camera_position
        client.camera.look_at = config.view_camera_look_at

    log.info("🦾 Adding robots...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(config.urdf_path)
    viser_robot: pk.Robot = pk.Robot.from_urdf(urdf)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")

    if config.display_data:
        _init_rerun(session_name="recording")
    robot = make_robot_from_config(TatbotConfig())
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}

    sanity_check_dataset_name(f"hu-po/tatbot-toolpath-{config.dataset_name}", None)
    dataset = LeRobotDataset.create(
        f"hu-po/tatbot-toolpath-{config.dataset_name}",
        config.fps,
        root=f"{config.output_dir}/{config.dataset_name}",
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=config.num_image_writer_processes,
        image_writer_threads=config.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    robot.connect()
    listener, events = init_keyboard_listener()

    with open(config.toolpath_path, "r") as f:
        toolpaths = json.load(f)

    for relative_toolpath_segment in toolpaths:
        # The toolpath from the design file is relative to the design's origin.
        # We make it absolute by adding the design's position.
        absolute_toolpath_segment = [
            list(np.array(config.ee_design_pos) + np.array([p[0], p[1], 0.0])) for p in relative_toolpath_segment
        ]

        # Each segment is an episode, and starts with an ink dip.
        episode_toolpath = []
        # 1. Ink dip sequence.
        episode_toolpath.append(list(config.ee_inkcap_pos))
        episode_toolpath.append((np.array(config.ee_inkcap_pos) + np.array(config.ee_inkcap_dip)).tolist())
        episode_toolpath.append(list(config.ee_inkcap_pos))
        # 2. Hover over the general design area.
        episode_toolpath.append(list(config.ee_design_pos))
        # 3. Add the drawing path for the segment.
        episode_toolpath.extend(absolute_toolpath_segment)

        num_toolpoints = len(episode_toolpath)
        log_say(f"Recording episode {dataset.num_episodes}", config.play_sounds)
        for i, toolpoint in enumerate(episode_toolpath):
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            log.info(f"🔍 Solving IK for toolpoint: {toolpoint} (index: {i}/{num_toolpoints})")
            solution = ik(
                robot=viser_robot,
                target_link_indices=jnp.array([viser_robot.links.names.index(config.target_link_name)]),
                target_wxyz=jnp.array([config.ee_design_wxyz]),
                target_position=jnp.array([np.array(toolpoint)]),
                config=config.ik_config,
            )
            # hardcode the right arm
            solution = solution.at[8].set(config.joint_pos_design_r[0])
            solution = solution.at[9].set(config.joint_pos_design_r[1])
            solution = solution.at[10].set(config.joint_pos_design_r[2])
            solution = solution.at[11].set(config.joint_pos_design_r[3])
            solution = solution.at[12].set(config.joint_pos_design_r[4])
            solution = solution.at[13].set(config.joint_pos_design_r[5])
            solution = solution.at[14].set(config.joint_pos_design_r[6])
            urdf_vis.update_cfg(np.array(solution))
            action = {
                "left.joint_0.pos": solution[0],
                "left.joint_1.pos": solution[1],
                "left.joint_2.pos": solution[2],
                "left.joint_3.pos": solution[3],
                "left.joint_4.pos": solution[4],
                "left.joint_5.pos": solution[5],
                "left.gripper.pos": solution[6],
                "right.joint_0.pos": solution[8],
                "right.joint_1.pos": solution[9],
                "right.joint_2.pos": solution[10],
                "right.joint_3.pos": solution[11],
                "right.joint_4.pos": solution[12],
                "right.joint_5.pos": solution[13],
                "right.gripper.pos": solution[14],
            }
            log.debug(f"🦾 Action: {action}")
            # # first 5 actions (ink dipping, hovering) should be SLOW and blocking
            # if i < 5:
            #     sent_action = robot.send_action(action, goal_time=robot.config.goal_time_ready_sleep, blocking=True)
            # else:
            #     # rest of the actions should be FAST and non-blocking
            #     sent_action = robot.send_action(action)
            sent_action = action

            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=f"Tattoo path {i}")

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
                events["exit_early"] = False
                break

        if events["rerecord_episode"]:
            log_say("Re-record episode", config.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()

        if events["stop_recording"]:
            break

    log_say("Stop recording", config.play_sounds, blocking=True)

    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if config.push_to_hub:
        dataset.push_to_hub(tags=["tatbot", "wxai", "trossen"], private=config.private)

    log_say("Exiting", config.play_sounds)

if __name__ == "__main__":
    args = tyro.cli(ToolpathConfig)
    if args.debug:
        log.setLevel(logging.DEBUG)
        # logging.getLogger('lerobot').setLevel(logging.DEBUG)
        log.debug("🐛 Debug mode enabled.")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"💾 Saving output to {args.output_dir}")
    log.info(pformat(asdict(args)))
    main(args)