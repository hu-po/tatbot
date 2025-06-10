from dataclasses import asdict, dataclass
import json
import logging
import os
from pprint import pformat
import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
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

    design_dir: str = os.path.expanduser("~/tatbot/output/design/infinity")
    """Directory with design outputs from design.py."""

    # repo_id: str = f"hu-po/tatbot-test-{int(time.time())}"
    repo_id: str = f"hu-po/tatbot-infinity"
    """Hugging Face Hub repository ID, e.g. 'hf-username/my-dataset'."""
    display_data: bool = True
    """Display data on screen using Rerun."""
    output_dir: str = os.path.expanduser("~/tatbot/output/record")
    """Directory to save the dataset."""
    push_to_hub: bool = True
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
    fps: int = 5
    """Frames per second."""
    max_episodes: int = 100
    """Maximum number of episodes to record."""

    image_width_px: int = 256
    """Width of the design image (pixels)."""
    image_height_px: int = 256
    """Height of the design image (pixels)."""
    image_width_m: float = 0.06
    """Width of the design image (meters)."""
    image_height_m: float = 0.06
    """Height of the design image (meters)."""

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
    ee_design_hover_offset: tuple[float, float, float] = (0.0, 0.0, -0.0085)
    """offset of the design ee transform when hovering over a toolpoint."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_dip: tuple[float, float, float] = (0.0, 0.0, -0.032)
    """dip vector when performing inkcap dip."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""

    ink_dip_interval: int = 4
    """
    Dip ink every N toolpath segments.
    If N > 0, dips on segment 0, N, 2N, ...
    If N = 0, dips only on the first segment.
    If N < 0, never dips.
    """


def main(config: ToolpathConfig):
    config = config
    log.info(f"🌱 Setting random seed to {config.seed}...")
    rng = jax.random.PRNGKey(config.seed)

    log.info("🚀 Starting viser server...")
    server: viser.ViserServer = viser.ViserServer()
    server.scene.set_environment_map(hdri=config.env_map_hdri, background=True)

    with server.gui.add_folder("Design"):
        image_path = os.path.join(config.design_dir, "resized.png")
        if os.path.exists(image_path):
            log.info(f"🖼️ Loading design image from {image_path}...")
            img_pil = PIL.Image.open(image_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # Viser GUI expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            design_image_gui = server.gui.add_image(
                label="Tattoo Design",
                image=img_rgb,
                format="png",
            )
        else:
            log.warning(f"Design image not found at {image_path}, GUI image will be disabled.")
            img_bgr = None
            design_image_gui = None

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

    dataset_name = config.repo_id.split("/")[-1]
    sanity_check_dataset_name(config.repo_id, None)
    dataset = LeRobotDataset.create(
        config.repo_id,
        config.fps,
        root=f"{config.output_dir}/{dataset_name}",
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=config.num_image_writer_processes,
        image_writer_threads=config.num_image_writer_threads_per_camera * len(robot.cameras),
    )

    robot.connect()
    listener, events = init_keyboard_listener()

    toolpath_file_path = os.path.join(config.design_dir, "toolpaths.json")
    with open(toolpath_file_path, "r") as f:
        toolpaths = json.load(f)

    for toolpath_idx, relative_toolpath_segment in enumerate(toolpaths):

        if toolpath_idx >= config.max_episodes:
            log_say(f"Reached max episodes ({config.max_episodes})", config.play_sounds)
            break

        if img_bgr is not None and design_image_gui is not None:
            # Create a fresh copy for this segment's visualization
            segment_viz_img = img_bgr.copy()
            
            # Convert full segment to pixels and draw it
            segment_points_px = []
            scale_x = config.image_width_m / config.image_width_px
            scale_y = config.image_height_m / config.image_height_px
            for p_m in relative_toolpath_segment:
                px_x = int(p_m[0] / scale_x)
                px_y = int(p_m[1] / scale_y)
                segment_points_px.append((px_x, px_y))

            for k in range(len(segment_points_px) - 1):
                cv2.line(segment_viz_img, segment_points_px[k], segment_points_px[k+1], (255, 0, 0), 2) # Blue
            
            design_image_gui.image = cv2.cvtColor(segment_viz_img, cv2.COLOR_BGR2RGB)

        # The toolpath from the design file is relative to the design's origin.
        # We make it absolute by adding the design's position.
        absolute_toolpath_segment = [
            list(np.array(config.ee_design_pos) + np.array([p[0], p[1], 0.0]) + np.array(config.ee_design_hover_offset)) for p in relative_toolpath_segment
        ]

        # Each segment is an episode, and starts with an ink dip.
        episode_toolpath = []

        should_dip = (config.ink_dip_interval > 0 and toolpath_idx % config.ink_dip_interval == 0) or \
                     (config.ink_dip_interval == 0 and toolpath_idx == 0)

        if should_dip:
            log_say("Dipping ink", config.play_sounds)
            # Ink dip sequence
            episode_toolpath.append(list(config.ee_inkcap_pos))
            episode_toolpath.append((np.array(config.ee_inkcap_pos) + np.array(config.ee_inkcap_dip)).tolist())
            episode_toolpath.append(list(config.ee_inkcap_pos))

        # Hover over the first toolpoint
        episode_toolpath.append(list(absolute_toolpath_segment[0] - np.array(config.ee_design_hover_offset)))
        # Add the rest of the toolpath
        episode_toolpath.extend(absolute_toolpath_segment)
        len_prefix = len(episode_toolpath) - len(absolute_toolpath_segment)
        num_toolpoints = len(episode_toolpath)

        log_say(f"Recording tool path {toolpath_idx}", config.play_sounds)
        for toolpoint_idx, toolpoint in enumerate(episode_toolpath):
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            log.info(f"🔍 Solving IK for toolpoint: {toolpoint} (index: {toolpoint_idx}/{num_toolpoints})")
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
            # initial actions such as ink dipping and hovering should be SLOW and blocking
            # This includes the first drawing point.
            if toolpoint_idx < len_prefix + 1:
                sent_action = robot.send_action(action, goal_time=robot.config.goal_time_ready_sleep, blocking=True)
            else:
                # rest of the actions should be FAST and non-blocking
                sent_action = robot.send_action(action)

            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=f"Tattoo path {toolpoint_idx}")

            if img_bgr is not None and design_image_gui is not None:
                current_drawing_point_idx = toolpoint_idx - len_prefix
                if current_drawing_point_idx >= 0:
                    # new image for each step
                    step_image = segment_viz_img.copy()
                    
                    # draw path up to current point
                    path_to_draw_px = segment_points_px[:current_drawing_point_idx+1]
                    for k in range(len(path_to_draw_px) - 1):
                        cv2.line(step_image, path_to_draw_px[k], path_to_draw_px[k+1], (0, 255, 0), 2) # Green
                    
                    if path_to_draw_px:
                        cv2.circle(step_image, path_to_draw_px[-1], 5, (0, 0, 255), -1) # Red circle for current point

                    design_image_gui.image = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)

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
    try:
        main(args)
    except Exception as e:
        log.error(f"Error: {e}")
    except KeyboardInterrupt:
        log.info("🛑 Keyboard interrupt detected. Disconnecting robot...")
    finally:
        log.info("🛑 Disconnecting robot...")
        robot = make_robot_from_config(TatbotConfig())
        robot.connect(clear_error=False)
        log.error(robot._get_error_str())
        robot.disconnect()