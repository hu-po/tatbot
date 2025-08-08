"""Align robot tool for calibration operations."""

import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
)
from lerobot.utils.robot_utils import busy_wait

from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.align import make_align_strokes
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.main import compose_and_validate_scene
from tatbot.utils.gpu_conversion import GPUConversionService
from tatbot.utils.gpu import check_local_gpu
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.robot.models import AlignInput, AlignOutput
from tatbot.utils.log import TIME_FORMAT, get_logger

log = get_logger("tools.align", "📐")


@tool(
    name="align",
    nodes=["trossen-ai", "ook", "oop"],
    description="Generate and execute alignment strokes for robot calibration",
    input_model=AlignInput,
    output_model=AlignOutput,
)
async def align_tool(input_data: AlignInput, ctx: ToolContext):
    """
    Generate and execute alignment strokes for robot calibration.
    
    This tool creates alignment stroke patterns that help calibrate the robot's
    positioning and accuracy. It generates stroke trajectories, converts them to
    robot joint paths using GPU-accelerated inverse kinematics, and executes them
    on the physical robot while recording the motion data.
    
    Parameters:
    - scene_name (str, optional): Scene configuration to use. Default: "default"  
    - debug (bool, optional): Enable debug logging. Default: false
    
    Returns:
    - success (bool): Whether alignment completed successfully
    - message (str): Status message
    - stroke_count (int): Number of alignment strokes executed
    
    Example usage:
    {"scene_name": "default"}
    {}
    """
    
    if input_data.debug:
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    
    dataset_dir = None
    dataset = None
    robot = None
    
    try:
        yield {"progress": 0.01, "message": "Loading scene configuration..."}
        
        # Load scene configuration
        scene = compose_and_validate_scene(input_data.scene_name)
        
        # Create output directory
        output_dir = Path("~/tatbot/nfs/recordings").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        yield {"progress": 0.02, "message": f"Creating output directory at {output_dir}..."}
        
        # Create dataset directory with timestamp
        dataset_name = f"align-{scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        log.info(f"Created dataset directory: {dataset_dir}")
        
        # Save scene configuration
        scene_path = dataset_dir / "scene.yaml"
        scene.to_yaml(str(scene_path))
        
        yield {"progress": 0.05, "message": "Creating robot from config..."}
        
        # Create robot
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
                rs_cameras={},
                ip_cameras={},
            )
        )
        
        yield {"progress": 0.06, "message": "Connecting to robot..."}
        
        # Connect to robot
        robot.connect()
        
        # Calculate camera threads
        num_camera_threads = 0
        if hasattr(robot, "rs_cameras") and len(robot.rs_cameras) > 0:
            num_camera_threads += 4 * len(robot.rs_cameras)
        if hasattr(robot, "ip_cameras") and len(robot.ip_cameras) > 0:
            num_camera_threads += 4 * len(robot.ip_cameras)
        
        log.info(f"Connected to robot with {num_camera_threads} camera threads")
        
        # Create dataset
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
        
        # Generate alignment strokes
        yield {"progress": 0.2, "message": "Generating alignment strokes..."}
        
        strokes: StrokeList = make_align_strokes(scene)
        log.info(f"✅ Generated {len(strokes.strokes)} alignment stroke pairs")
        
        # Save strokes
        strokes_path = dataset_dir / "strokes.yaml"
        strokebatch_path = dataset_dir / "strokebatch.safetensors"
        
        log.info(f"💾 Saving strokes to {strokes_path}")
        strokes.to_yaml_with_arrays(str(strokes_path))
        
        if not strokes_path.exists():
            raise FileNotFoundError(f"strokes.yaml was not created at {strokes_path}")
        
        file_size = strokes_path.stat().st_size
        log.info(f"✅ strokes.yaml created successfully ({file_size} bytes)")
        
        # Wait for NFS sync
        log.info("⏳ Waiting for NFS sync before GPU conversion...")
        time.sleep(1.0)
        
        # Convert strokes to strokebatch (GPU or local)
        if check_local_gpu():
            log.info("Using local GPU for strokebatch conversion")
            strokebatch: StrokeBatch = strokebatch_from_strokes(scene, strokes, first_last_rest=False)
            strokebatch.save(str(strokebatch_path))
        else:
            log.info("Using remote GPU node for strokebatch conversion")
            gpu_proxy = GPUConversionService()
            
            success, _ = await gpu_proxy.convert_strokelist_remote(
                strokes_file_path=str(strokes_path),
                strokebatch_file_path=str(strokebatch_path),
                scene_name=scene.name,
                first_last_rest=False,
                use_ee_offsets=True
            )
            
            if not success:
                raise RuntimeError("Failed to convert strokes to strokebatch on remote GPU node")
            
            strokebatch = StrokeBatch.load(str(strokebatch_path))
        
        log.info(f"Strokebatch created with shape: {strokebatch.joints.shape}")
        
        # Execute alignment strokes
        offset_idx_l = scene.arms.offset_num - 1  # maximally retracted
        offset_idx_r = scene.arms.offset_num - 1
        
        for stroke_idx, (stroke_l, stroke_r) in enumerate(strokes.strokes):
            # Ensure robot is connected and in ready position
            if not robot.is_connected:
                log.warning("⚠️ Robot is not connected. Attempting to reconnect...")
                robot.connect()
                if not robot.is_connected:
                    raise RuntimeError("❌ Failed to connect to robot")
            
            robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
            
            stroke_msg = f"🔍 Executing stroke {stroke_idx + 1}/{len(strokes.strokes)}: left={stroke_l.description}, right={stroke_r.description}"
            log.info(stroke_msg)
            yield {
                'progress': 0.3 + (0.6 * stroke_idx / len(strokes.strokes)),
                'message': stroke_msg,
            }
            
            # Execute stroke poses
            for pose_idx in range(scene.stroke_length):
                start_loop_t = time.perf_counter()
                
                # Get observation
                observation = robot.get_observation()
                log.debug(f"observation: {observation}")
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                
                # Get target joints from strokebatch
                joints = strokebatch.offset_joints(stroke_idx, pose_idx, offset_idx_l, offset_idx_r)
                robot_action = robot._urdf_joints_to_action(joints)
                
                # Send action (slow for first/last poses, fast for middle)
                if pose_idx == 0 or pose_idx == scene.stroke_length - 1:
                    sent_action = robot.send_action(robot_action, scene.arms.goal_time_slow, safe=True)
                else:
                    sent_action = robot.send_action(robot_action, scene.arms.goal_time_fast)
                
                # Record data
                action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")
                
                # Maintain FPS
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / 10 - dt_s)  # 10 FPS
            
            # Save episode
            dataset.save_episode()
        
        # Return to ready position
        yield {"progress": 0.95, "message": "Returning robot to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        log.info("✅ Alignment operation completed successfully")
        
        yield AlignOutput(
            success=True,
            message=f"✅ Alignment completed successfully. Executed {len(strokes.strokes)} stroke pairs.",
            stroke_count=len(strokes.strokes)
        )
        
    except Exception as e:
        error_msg = f"❌ Alignment operation failed: {e}"
        log.error(error_msg)
        yield AlignOutput(
            success=False,
            message=error_msg,
            stroke_count=0
        )
        
    finally:
        # Cleanup
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                log.error(f"Error disconnecting robot: {e}")