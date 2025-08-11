"""Sense tool for capturing environmental data from cameras and sensors."""

import logging
import os
import time
from pathlib import Path

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
    - scene_name (str, optional): Scene configuration to use. Default: "default"
    - debug (bool, optional): Enable debug logging. Default: false
    - num_plys (int, optional): Number of PLY pointcloud files to capture. Default: 2
    - calibrate_extrinsics (bool, optional): Whether to calibrate camera extrinsics using AprilTags. Default: true
    - reference_tag_id (int, optional): Reference AprilTag ID for extrinsics calibration. Default: 0
    - max_deviation_warning (float, optional): Maximum deviation (m) from URDF before showing warning. Default: 0.05
    
    Returns:
    - success (bool): Whether sensing completed successfully
    - message (str): Status message
    - captured_files (list[str]): List of captured file paths
    
    Example usage:
    {"scene_name": "default", "num_plys": 4}
    {}
    """
    
    if input_data.debug:
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    
    dataset_dir = None
    dataset = None
    robot = None
    captured_files = []
    
    try:
        yield {"progress": 0.01, "message": "Loading scene configuration..."}
        
        # Load scene configuration
        scene = compose_and_validate_scene(input_data.scene_name)
        
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
            
            progress = 0.6 + (0.3 * (ply_idx + 1) / num_plys)
            yield {"progress": progress, "message": f"Captured pointcloud {ply_idx + 1}/{num_plys}"}
        
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