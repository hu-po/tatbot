"""
Records episodes using a Pattern from JSON.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pprint import pformat
import time

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

from ik import IKConfig, ik
from pattern import COLORS, Pattern, offset_path

log = logging.getLogger('tatbot')

@dataclass
class PathConfig:
    debug: bool = False
    """Enable debug logging."""

    pattern_dir: str = os.path.expanduser("~/tatbot/output/patterns/calibration")
    """Directory with pattern.json and image.png."""

    hf_username: str = os.environ.get("HF_USER", "hu-po")
    """Hugging Face username."""
    dataset_name: str | None = None
    """Dataset will be saved to Hugging Face Hub repository ID, e.g. 'hf_username/dataset_name'."""
    display_data: bool = True
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
    play_sounds: bool = True
    """Whether to play sounds."""
    private: bool = False
    """Whether to push the dataset to a private repository."""
    fps: int = 5
    """Frames per second."""
    max_episodes: int = 100
    """Maximum number of episodes to record."""

    seed: int = 42
    """Seed for random behavior."""
    urdf_path: str = os.path.expanduser("~/tatbot/assets/urdf/tatbot.urdf")
    """Local path to the URDF file for the robot."""
    target_links_name: tuple[str, str] = ("left/tattoo_needle", "right/ee_gripper_link")
    """Names of the ee links in the URDF for left and right ik solving."""
    ik_config: IKConfig = IKConfig()
    """Configuration for the IK solver."""

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
    ee_design_hover_offset: tuple[float, float, float] = (0.0, 0.0, -0.0085)
    """offset of the design ee transform when hovering over a toolpoint."""
    ee_design_view_offset: tuple[float, float, float] = (0.0, -0.2, 0.2)
    """position of the design view ee transform (relative to design ee transform)."""
    ee_design_view_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """orientation quaternion (wxyz) of the design view ee transform."""

    ee_inkcap_pos: tuple[float, float, float] = (0.16, 0.0, 0.04)
    """position of the inkcap ee transform."""
    ee_inkcap_dip: tuple[float, float, float] = (0.0, 0.0, -0.034)
    """dip vector when performing inkcap dip."""
    ee_inkcap_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)
    """orientation quaternion (wxyz) of the inkcap ee transform."""

    alignment_timeout: float = 10.0
    """Timeout for alignment in seconds."""
    alignment_interval: float = 1.0
    """Interval for alignment switching between design and inkcap."""

    ink_dip_interval: int = 1
    """
    Dip ink every N path segments.
    If N > 0, dips on segment 0, N, 2N, ...
    If N = 0, dips only on the first segment.
    If N < 0, never dips.
    """


def main(config: PathConfig):
    config = config
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
            label="Pattern",
            image=img_np,
            format="png",
        )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = config.view_camera_position
        client.camera.look_at = config.view_camera_look_at

    log.info("🦾 Adding vizer robot...")
    urdf : yourdfpy.URDF = yourdfpy.URDF.load(config.urdf_path)
    viser_robot: pk.Robot = pk.Robot.from_urdf(urdf)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/root")

    log.info("🔍 Loading pattern...")
    pattern_path = os.path.join(config.pattern_dir, "pattern.json")
    assert os.path.exists(pattern_path), f"Pattern file not found at {pattern_path}"
    with open(pattern_path, "r") as f:
        pattern_json = json.load(f)
    pattern = Pattern.from_json(pattern_json)
    log.info(f"Loaded pattern '{pattern.name}' with {len(pattern.paths)} paths.")

    log.info("🔍 Initializing dataset...")
    if config.display_data:
        _init_rerun(session_name="recording")
    robot = make_robot_from_config(TatbotConfig())
    action_features = hw_to_dataset_features(robot.action_features, "action", True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
    dataset_features = {**action_features, **obs_features}
    dataset_name = config.dataset_name or f"{pattern.name}-{int(time.time())}"
    repo_id = f"{config.hf_username}/{dataset_name}"
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
    robot.connect()
    listener, events = init_keyboard_listener()

    # convert to jnp arrays for jax operations
    design_pos = jnp.array(config.ee_design_pos)
    design_wxyz = jnp.array(config.ee_design_wxyz)
    hover_offset = jnp.array(config.ee_design_hover_offset)
    view_offset = jnp.array(config.ee_design_view_offset)
    inkcap_pos = jnp.array(config.ee_inkcap_pos)
    inkcap_wxyz = jnp.array(config.ee_inkcap_wxyz)
    inkcap_dip = jnp.array(config.ee_inkcap_dip)
    target_link_indices = jnp.array([
        viser_robot.links.names.index(config.target_links_name[0]),
        viser_robot.links.names.index(config.target_links_name[1]),
    ])

    log.info("📐 Waiting for alignment, press right arrow key to continue...")
    start_time = time.time()
    while True:
        if time.time() - start_time > config.alignment_timeout:
            log.error("❌📐 Alignment timeout")
            log_say("alignment timeout", config.play_sounds)
            raise RuntimeError("❌📐 Alignment timeout")
        if events["exit_early"]:
            log.info("✅📐 Alignment complete")
            log_say("alignment complete", config.play_sounds)
            break
        log.info("📐 Aligning over design...")
        log_say("align design", config.play_sounds)
        solution = ik(
            robot=viser_robot,
            target_link_indices=target_link_indices[0],
            target_wxyz=design_wxyz,
            target_position=design_pos,
            config=config.ik_config,
        )
        robot._set_positions_l(solution, goal_time=robot.config.goal_time_slow, blocking=True)
        urdf_vis.update_cfg(np.array(solution))
        time.sleep(config.alignment_interval)
        log.info("📐 Aligning over inkcap...")
        log_say("align inkcap", config.play_sounds)
        solution = ik(
            robot=viser_robot,
            target_link_indices=target_link_indices[0],
            target_wxyz=inkcap_wxyz,
            target_position=inkcap_pos,
            config=config.ik_config,
        )
        robot._set_positions_l(solution, goal_time=robot.config.goal_time_slow, blocking=True)
        urdf_vis.update_cfg(np.array(solution))
        time.sleep(config.alignment_interval)

    log.info(f"Recording {len(pattern.paths)} paths...")
    log_say(f"recording paths", config.play_sounds)
    for path_idx, path in enumerate(pattern.paths):
        if path_idx >= config.max_episodes:
            log_say(f"max paths {config.max_episodes} exceeded", config.play_sounds)
            break

        log.info(f"🖼️ Updating visualization...")
        path_viz_img_np = img_np.copy()
        for pw, ph in path.pixel_coords:
            cv2.circle(path_viz_img_np, (pw, ph), 5, COLORS["green"], -1)
        viser_img.image = path_viz_img_np

        should_dip = (config.ink_dip_interval > 0 and path_idx % config.ink_dip_interval == 0) or (
            config.ink_dip_interval == 0 and path_idx == 0
        )
        if should_dip:
            log.info("✒️ dipping ink...")
            log_say("dipping ink", config.play_sounds)
            for desc, pose in [
                ("hover over inkcap", inkcap_pos),
                ("dip into inkcap", inkcap_pos + inkcap_dip),
                ("retract from inkcap", inkcap_pos),
                ("hover over path", path.positions[0] + hover_offset),
            ]:
                log_say(desc, config.play_sounds)
                solution = ik(
                    robot=viser_robot,
                    target_link_indices=target_link_indices[0],
                    target_wxyz=design_wxyz,
                    target_position=pose,
                    config=config.ik_config,
                )
                robot._set_positions_l(solution, goal_time=robot.config.goal_time_slow, blocking=True)
                urdf_vis.update_cfg(np.array(solution))

        # right hand follows left hand with an offset
        path_l = offset_path(path, design_pos)
        path_r = offset_path(path_l, view_offset)
        pathlen = len(path_l)

        log.info(f"recording path {path_idx} of {len(pattern.paths)}")
        log_say(f"recording path {path_idx} of {len(pattern.paths)}", config.play_sounds)
        for pose_idx in range(pathlen):
            log.info(f"pose_idx: {pose_idx}/{pathlen})")
            start_loop_t = time.perf_counter()
            observation = robot.get_observation()
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            solution = ik(
                robot=viser_robot,
                target_link_indices=target_link_indices,
                target_wxyz=jnp.array([
                    path_l.orientations[pose_idx],
                    path_r.orientations[pose_idx],
                ]),
                target_position=jnp.array([
                    path_l.positions[pose_idx],
                    path_r.positions[pose_idx],
                ]),
                config=config.ik_config,
            )
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
            sent_action = robot.send_action(action, goal_time=robot.config.goal_time_fast, block_mode="left")

            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            # TODO: add pattern closeup? progress image? patch?
            dataset.add_frame(frame, task=f"{pattern.name} tattoo pattern path {path_idx} of {len(pattern.paths)}")

            log.info(f"🖼️ Updating visualization...")
            step_viz_img_np = path_viz_img_np.copy()
            for pw, ph in path_l.pixel_coords[:pose_idx]:
                # small green circles for all poses up to current pose
                cv2.circle(step_viz_img_np, (pw, ph), 5, COLORS["green"], -1)
            # big red circle for current pose
            cv2.circle(step_viz_img_np, (path_l.pixel_coords[pose_idx][0], path_l.pixel_coords[pose_idx][1]), 5, COLORS["red"], -1)
            viser_img.image = step_viz_img_np


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

        dataset.save_episode()

        if events["stop_recording"]:
            break

    log_say("Stop recording", config.play_sounds, blocking=True)

    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if config.push_to_hub:
        dataset.push_to_hub(tags=list(config.tags), private=config.private)

    log_say("Exiting", config.play_sounds)

if __name__ == "__main__":
    args = tyro.cli(PathConfig)
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger('lerobot').setLevel(logging.DEBUG)
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
        robot._connect_l(clear_error=False)
        log.error(robot._get_error_str_l())
        robot._connect_r(clear_error=False)
        log.error(robot._get_error_str_r())
        robot.disconnect()