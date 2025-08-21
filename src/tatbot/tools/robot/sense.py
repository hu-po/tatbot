"""Sense tool for capturing environmental data from cameras and sensors."""

import logging
import os
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.utils.control_utils import sanity_check_dataset_name
from PIL import Image

from tatbot.bot.urdf import get_link_poses
from tatbot.cam.depth import DepthCamera
from tatbot.cam.extrinsics import get_extrinsics
from tatbot.cam.validation import compare_extrinsics_with_urdf
from tatbot.main import compose_and_validate_scene
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.robot.models import SenseInput, SenseOutput
from tatbot.utils.constants import NFS_RECORDINGS_DIR
from tatbot.utils.log import TIME_FORMAT, get_logger

# VGGT MCP constants
MCP_VGGT_TIMEOUT_S = 900
MCP_VGGT_RETRY_COUNT = 2

log = get_logger("tools.sense", "🔍")


@tool(
    name="sense",
    nodes=["hog"],
    description="Capture environmental data from cameras and sensors",
    input_model=SenseInput,
    output_model=SenseOutput,
)
async def sense_tool(input_data: SenseInput, ctx: ToolContext):
    """
    Capture environmental data from cameras and sensors.
    
    This tool captures images from all configured cameras, optionally calibrates
    camera extrinsics using AprilTags, and generates 3D pointclouds from depth
    cameras. It's used for environmental sensing and robot workspace mapping.
    
    Parameters:
    - scene (str, optional): Scene configuration to use. Default: "default"
    - debug (bool, optional): Enable debug logging. Default: false
    - meta (str, optional): Meta config to apply (e.g. "tatbotlogo"). Default: null
    - num_plys (int, optional): Number of PLY pointcloud files to capture. Default: 2
    - calibrate_extrinsics (bool, optional): Whether to calibrate camera extrinsics using AprilTags. Default: true
    - reference_tag_id (int, optional): Reference AprilTag ID for extrinsics calibration. Default: 0
    - max_deviation_warning (float, optional): Maximum deviation (m) from URDF before showing warning. Default: 0.05
    
    Returns:
    - success (bool): Whether sensing completed successfully
    - message (str): Status message
    - captured_files (list[str]): List of captured file paths
    """
    
    if input_data.debug:
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    
    dataset_dir = None
    dataset = None
    robot = None
    captured_files = []
    
    try:
        yield {"progress": 0.01, "message": "Loading scene configuration..."}
        
        # Load scene configuration (with optional meta)
        scene = compose_and_validate_scene(
            name=input_data.scene,
            meta=input_data.meta,
        )
        
        # Create output directory
        output_dir = NFS_RECORDINGS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        yield {"progress": 0.02, "message": f"Creating output directory at {output_dir}..."}
        
        # Create dataset directory with timestamp
        dataset_name = f"sense-{scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        log.info(f"Created dataset directory: {dataset_dir}")
        
        # Save scene configuration
        scene_path = dataset_dir / "scene.yaml"
        scene.to_yaml(str(scene_path))
        captured_files.append(str(scene_path))
        
        # Create standard subdirectories for artifacts
        images_dir = dataset_dir / "images"
        pointclouds_dir = dataset_dir / "pointclouds"
        colmap_dir = dataset_dir / "colmap"
        metadata_dir = dataset_dir / "metadata"
        for d in (images_dir, pointclouds_dir, colmap_dir, metadata_dir):
            d.mkdir(exist_ok=True)
        
        yield {"progress": 0.05, "message": "Creating robot with cameras..."}
        
        # Create robot with camera configurations
        robot: Robot = make_robot_from_config(
            TatbotConfig(
                ip_address_l=scene.arms.ip_address_l,
                ip_address_r=scene.arms.ip_address_r,
                arm_l_config_filepath=scene.arms.arm_l_config_filepath,
                arm_r_config_filepath=scene.arms.arm_r_config_filepath,
                goal_time=scene.arms.goal_time_slow,
                connection_timeout=scene.arms.connection_timeout,
                home_pos_l=scene.sleep_pos_l.joints,
                home_pos_r=scene.sleep_pos_r.joints,
                rs_cameras={
                    cam.name: RealSenseCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        serial_number_or_name=cam.serial_number,
                    ) for cam in scene.cams.realsenses
                },
                ip_cameras={
                    cam.name: OpenCVCameraConfig(
                        fps=cam.fps,
                        width=cam.width,
                        height=cam.height,
                        ip=cam.ip,
                        username=cam.username,
                        password=os.environ.get(cam.password, None),
                        rtsp_port=cam.rtsp_port,
                    )
                    for cam in scene.cams.ipcameras
                },
            )
        )
        
        yield {"progress": 0.06, "message": "Connecting to robot and cameras..."}
        
        # Connect to robot
        robot.connect()
        
        # Calculate camera threads
        num_camera_threads = 0
        if hasattr(robot, "rs_cameras") and len(robot.rs_cameras) > 0:
            num_camera_threads += 4 * len(robot.rs_cameras)
        if hasattr(robot, "ip_cameras") and len(robot.ip_cameras) > 0:
            num_camera_threads += 4 * len(robot.ip_cameras)
        
        log.info(f"Connected to robot with {num_camera_threads} camera threads")
        
        # Create dataset for recording
        action_features = hw_to_dataset_features(robot.action_features, "action", True)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", True)
        dataset_features = {**action_features, **obs_features}
        
        repo_id = f"tatbot/{dataset_name}"
        sanity_check_dataset_name(repo_id, None)
        
        dataset = LeRobotDataset.create(
            repo_id,
            fps=10,  # Default FPS
            root=str(dataset_dir),
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=num_camera_threads,
        )
        
        yield {"progress": 0.07, "message": f"Created LeRobot dataset at {repo_id}"}
        
        # Move robot to ready position
        yield {"progress": 0.08, "message": "Sending robot to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        # Capture images from all cameras
        yield {"progress": 0.3, "message": "🤖 Recording observation (png images)"}
        
        observation = robot.get_observation(full=True)
        for key, data in observation.items():
            if key in robot.rs_cameras or key in robot.ip_cameras:
                image_path = dataset_dir / f"{key}.png"
                Image.fromarray(data).save(str(image_path))
                captured_files.append(str(image_path))
                log.info(f"✅ Saved frame to {image_path}")
                # Also save under images/ for downstream VGGT/visualization
                image_path2 = images_dir / f"{key}.png"
                try:
                    Image.fromarray(data).save(str(image_path2))
                except Exception as e:
                    log.warning(f"Failed to save image copy to images/: {e}")
        
        yield {"progress": 0.31, "message": "✅ Saved image frames"}
        
        # Disconnect realsense cameras to free them for depth capture
        yield {"progress": 0.4, "message": "Disconnecting realsense cameras..."}
        for cam in robot.rs_cameras.values():
            cam.disconnect()
        
        # Calibrate camera extrinsics if enabled
        if getattr(input_data, 'calibrate_extrinsics', True):
            yield {"progress": 0.45, "message": "📐 Calibrating camera extrinsics using AprilTags..."}
            
            # Get image paths for all cameras
            image_paths = []
            for cam in scene.cams.realsenses:
                image_path = dataset_dir / f"{cam.name}.png"
                if image_path.exists():
                    image_paths.append(str(image_path))
            for cam in scene.cams.ipcameras:
                image_path = dataset_dir / f"{cam.name}.png"
                if image_path.exists():
                    image_paths.append(str(image_path))
            
            if image_paths:
                try:
                    # Run extrinsics calibration
                    calibrated_cams = get_extrinsics(
                        image_paths=image_paths,
                        cams=scene.cams,
                        tags=scene.tags,
                    )
                    
                    # Compare with URDF positions and show warnings
                    max_deviation = getattr(input_data, 'max_deviation_warning', 0.05)
                    compare_extrinsics_with_urdf(
                        calibrated_cams, 
                        scene, 
                        max_deviation
                    )
                    
                    # Persist AprilTag calibration for later visualization
                    try:
                        import json
                        frustums = []
                        cam_names: list[str] = []
                        cam_names.extend([cam.name for cam in calibrated_cams.ipcameras])
                        cam_names.extend([cam.name for cam in calibrated_cams.realsenses])
                        for cam_name in cam_names:
                            cam_cfg = calibrated_cams.get_camera(cam_name)
                            frustums.append({
                                "name": cam_name,
                                "pose": {
                                    "position": list(map(float, cam_cfg.extrinsics.pos.xyz)),
                                    "wxyz": list(map(float, cam_cfg.extrinsics.rot.wxyz)),
                                },
                                "intrinsic": {
                                    "fx": float(cam_cfg.intrinsics.fx),
                                    "fy": float(cam_cfg.intrinsics.fy),
                                    "ppx": float(cam_cfg.intrinsics.ppx),
                                    "ppy": float(cam_cfg.intrinsics.ppy),
                                }
                            })
                        (metadata_dir / "apriltag_frustums.json").write_text(json.dumps(frustums, indent=2))
                    except Exception:
                        log.warning("Failed to save AprilTag frustums JSON")
                    
                    yield {"progress": 0.48, "message": "✅ Camera extrinsics calibration completed"}
                    
                except Exception as e:
                    warning_msg = f"⚠️ Camera extrinsics calibration failed: {e}"
                    log.warning(warning_msg)
                    yield {"progress": 0.48, "message": warning_msg}
            else:
                yield {"progress": 0.48, "message": "⚠️ No camera images found for extrinsics calibration"}
        
        # Capture 3D pointclouds
        yield {"progress": 0.5, "message": "Connecting to Depth cameras..."}
        
        link_poses = get_link_poses(
            scene.urdf.path, scene.urdf.cam_link_names, scene.ready_pos_full.joints
        )
        
        depth_cameras = {}
        for realsense in scene.cams.realsenses:
            depth_cameras[realsense.name] = DepthCamera(
                realsense.serial_number,
                link_poses[realsense.urdf_link_name],
                save_prefix=f"{realsense.name}_",
                save_dir=str(dataset_dir),
            )
        
        num_plys = getattr(input_data, 'num_plys', 2)
        yield {"progress": 0.6, "message": f"Capturing {num_plys} pointclouds..."}
        
        for ply_idx in range(num_plys):
            log.info(f"Capturing pointcloud {ply_idx + 1}/{num_plys}...")
            for _, depth_cam in depth_cameras.items():
                expected_ply_path = Path(depth_cam.save_dir) / f"{depth_cam.save_prefix}{depth_cam.frame_idx:06d}.ply"
                depth_cam.get_pointcloud(save=True)
                if expected_ply_path.exists():
                    captured_files.append(str(expected_ply_path))
                    # Also copy/link into pointclouds/ for consistency
                    try:
                        import shutil
                        target = pointclouds_dir / expected_ply_path.name
                        if not target.exists():
                            shutil.copy2(expected_ply_path, target)
                    except Exception as e:
                        log.warning(f"Failed to copy PLY to pointclouds/: {e}")
            
            progress = 0.6 + (0.3 * (ply_idx + 1) / num_plys)
            yield {"progress": progress, "message": f"Captured pointcloud {ply_idx + 1}/{num_plys}"}

        # Save COLMAP intrinsics/extrinsics to config for this scene if available
        try:
            from tatbot.cam.vggt_runner import write_colmap_text
            from tatbot.data.cams import Cams

            # Prefer calibrated_cams if calibration succeeded, else scene.cams
            cams_to_save: Cams = calibrated_cams if 'calibrated_cams' in locals() else scene.cams
            names: list[str] = []
            intrinsics = []
            extrinsics = []
            for cam in cams_to_save.ipcameras:
                names.append(f"{cam.name}.png")
                K = np.array([[cam.intrinsics.fx, 0, cam.intrinsics.ppx],
                              [0, cam.intrinsics.fy, cam.intrinsics.ppy],
                              [0, 0, 1]], dtype=float)
                intrinsics.append(K)
                # Build camera-from-world 3x4 from world-from-camera (Pose) by inversion
                import jax.numpy as jnp
                import jaxlie
                T_wc = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array(cam.extrinsics.rot.wxyz)),
                    jnp.array(cam.extrinsics.pos.xyz),
                )
                T_cw = T_wc.inverse()
                R = np.array(T_cw.rotation().as_matrix())
                t = np.array(T_cw.translation())
                E = np.concatenate([R, t.reshape(3, 1)], axis=1)
                extrinsics.append(E)
            for cam in cams_to_save.realsenses:
                names.append(f"{cam.name}.png")
                K = np.array([[cam.intrinsics.fx, 0, cam.intrinsics.ppx],
                              [0, cam.intrinsics.fy, cam.intrinsics.ppy],
                              [0, 0, 1]], dtype=float)
                intrinsics.append(K)
                import jax.numpy as jnp
                import jaxlie
                T_wc = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(jnp.array(cam.extrinsics.rot.wxyz)),
                    jnp.array(cam.extrinsics.pos.xyz),
                )
                T_cw = T_wc.inverse()
                R = np.array(T_cw.rotation().as_matrix())
                t = np.array(T_cw.translation())
                E = np.concatenate([R, t.reshape(3, 1)], axis=1)
                extrinsics.append(E)
            # Write under dataset colmap dir and config/colmap/<scene>
            write_colmap_text(intrinsics, extrinsics, names, colmap_dir)
            # Resolve project root reliably (repo_root/src/tatbot/tools/robot/sense.py → repo_root)
            repo_root = Path(__file__).resolve().parents[4]
            config_colmap_dir = repo_root / "config" / "colmap" / scene.name
            write_colmap_text(intrinsics, extrinsics, names, config_colmap_dir)
        except Exception as e:
            log.warning(f"Failed to write COLMAP text files: {e}")
        
        # Optionally run VGGT reconstruction remotely on GPU node
        if getattr(input_data, 'enable_vggt', False):
            try:
                import hydra
                import yaml as _yaml
                from omegaconf import OmegaConf

                from tatbot.mcp.client import MCPClient

                # Load Hydra config for VGGT
                cfg = hydra.compose(config_name="config")
                cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
                preferred_node = cfg_dict.get('cam', {}).get('vggt', {}).get('preferred_gpu_node', 'ook')
                node_cfg_path = Path(__file__).resolve().parents[3] / "conf" / "mcp" / f"{preferred_node}.yaml"
                host = "localhost"; port = 8000
                if node_cfg_path.exists():
                    node_cfg = _yaml.safe_load(node_cfg_path.read_text())
                    host = node_cfg.get("host", host)
                    port = int(node_cfg.get("port", port))
                client = MCPClient(request_timeout_s=MCP_VGGT_TIMEOUT_S)
                ok, session_id, url = await client.establish_session(host, port)
                if ok and session_id and url:
                    tool_name = "vggt_reconstruct"
                    out_ply = str(pointclouds_dir / "vggt_dense.ply")
                    out_json = str(metadata_dir / "vggt_frustums.json")
                    out_colmap = str(colmap_dir)
                    arguments = {
                        "input_data": {
                            "image_dir": str(images_dir),
                            "output_pointcloud_path": out_ply,
                            "output_frustums_path": out_json,
                            "output_colmap_dir": out_colmap,
                            "scene": scene.name,
                            "meta": input_data.meta,
                            "vggt_conf_threshold": float(input_data.vggt_conf_threshold),
                            "shared_camera": False,
                        }
                    }
                    # Retry with backoff
                    for attempt in range(MCP_VGGT_RETRY_COUNT):
                        ok2, resp = await client.call_tool(url, session_id, tool_name, arguments)
                        if ok2:
                            captured_files.extend([out_ply, out_json])
                            yield {"progress": 0.94, "message": "✅ VGGT reconstruction finished on GPU node"}
                            break
                        await ctx.warn(f"VGGT call attempt {attempt + 1} failed; retrying...")
                else:
                    log.warning("VGGT GPU node MCP session failed; skipping VGGT run")
            except Exception as e:
                log.warning(f"VGGT remote reconstruction failed: {e}")

        # Return to ready position
        yield {"progress": 0.95, "message": "Returning robot to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        log.info(f"✅ Sensing operation completed successfully. Captured {len(captured_files)} files.")
        
        yield SenseOutput(
            success=True,
            message=f"✅ Sensing completed successfully. Captured {len(captured_files)} files.",
            captured_files=captured_files
        )
        
    except Exception as e:
        error_msg = f"❌ Sensing operation failed: {e}"
        log.error(error_msg)
        yield SenseOutput(
            success=False,
            message=error_msg,
            captured_files=captured_files
        )
        
    finally:
        # Cleanup
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                log.error(f"Error disconnecting robot: {e}")
