"""Stroke tool for executing artistic strokes on paper/canvas."""

import logging
import shutil
import time
from io import StringIO

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import Robot, make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait

from tatbot.data.stroke import StrokeBatch, StrokeList
from tatbot.gen.batch import strokebatch_from_strokes
from tatbot.gen.gcode import make_gcode_strokes
from tatbot.main import compose_and_validate_scene
from tatbot.state.manager import StateManager
from tatbot.state.models import RobotState
from tatbot.tools.base import ToolContext
from tatbot.tools.registry import tool
from tatbot.tools.robot.models import StrokeInput, StrokeOutput
from tatbot.utils.constants import NFS_RECORDINGS_DIR
from tatbot.utils.gpu import check_local_gpu
from tatbot.utils.gpu_conversion import GPUConversionService
from tatbot.utils.log import LOG_FORMAT, TIME_FORMAT, get_logger

log = get_logger("tools.stroke", "🖌️")


@tool(
    name="stroke",
    nodes=["hog"],
    description="Execute a sequence of tattoo strokes",
    input_model=StrokeInput,
    output_model=StrokeOutput,
)
async def stroke_tool(input_data: StrokeInput, ctx: ToolContext):
    """
    Execute a sequence of tattoo strokes.
    
    This tool generates list of strokes (tuple for left and right) from the scene configuration,
    converts them to robot joint paths using GPU-accelerated inverse kinematics,
    and executes them on the physical robot while recording the motion data.
    It supports joystick control for real-time adjustments and can resume
    from previous executions.
    
    Parameters:
    - scene (str, optional): Scene configuration to use. Default: "default"  
    - debug (bool, optional): Enable debug logging. Default: false
    - meta (str, optional): Meta config to apply (e.g. "tatbotlogo"). Default: null
    - enable_joystick (bool, optional): Enable joystick for recording. Default: true
    - enable_realsense (bool, optional): Enable Intel RealSense cameras for recording. Default: true
    - resume (bool, optional): Resume from previous execution. Default: false
    - fps (int, optional): Frames per second for recording. Default: 10
    
    Returns:
    - success (bool): Whether stroke execution completed successfully
    - message (str): Status message
    - stroke_count (int): Number of strokes executed
    """
    
    if input_data.debug:
        logging.getLogger("lerobot").setLevel(logging.DEBUG)
    
    dataset_dir = None
    dataset = None
    robot = None
    atari_teleop = None
    episode_handler = None
    state_manager = None
    session_id = None
    
    try:
        yield {"progress": 0.01, "message": "Loading scene configuration..."}
        
        # Load scene configuration (with optional meta)
        scene = compose_and_validate_scene(
            name=input_data.scene,
            meta=input_data.meta,
        )
        
        # Initialize state manager
        state_manager = StateManager(node_id=ctx.node_name)
        await state_manager.connect()
        
        # Create output directory
        output_dir = NFS_RECORDINGS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        yield {"progress": 0.02, "message": f"Creating output directory at {output_dir}..."}
        
        # Create or resume dataset
        resume = getattr(input_data, 'resume', False)
        fps = getattr(input_data, 'fps', 10)
        
        if resume:
            # When resuming, find existing dataset directory
            dataset_name_pattern = f"stroke-{scene.name}-"
            existing_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(dataset_name_pattern)]
            if not existing_dirs:
                raise ValueError(f"No existing stroke dataset found for scene {scene.name}")
            dataset_dir = max(existing_dirs, key=lambda d: d.stat().st_mtime)  # Most recent
            log.info(f"🔄 Resuming from {dataset_dir}")
        else:
            # Create new dataset directory with timestamp
            dataset_name = f"stroke-{scene.name}-{time.strftime(TIME_FORMAT, time.localtime())}"
            dataset_dir = output_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            log.info(f"Created dataset directory: {dataset_dir}")
            
            # Save scene configuration
            scene_path = dataset_dir / "scene.yaml"
            scene.to_yaml(str(scene_path))
        
        yield {"progress": 0.05, "message": "Creating robot configuration..."}
        
        # Configure cameras based on input
        enable_realsense = getattr(input_data, 'enable_realsense', False)
        rs_cameras = {}
        if enable_realsense:
            log.info("Enabling Intel RealSense cameras for recording")
            from lerobot.cameras.realsense import RealSenseCameraConfig
            rs_cameras = {
                cam.name: RealSenseCameraConfig(
                    fps=cam.fps,
                    width=cam.width,
                    height=cam.height,
                    serial_number_or_name=cam.serial_number,
                ) for cam in scene.cams.realsenses
            }
        
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
                rs_cameras=rs_cameras,
                ip_cameras={},
            )
        )
        
        yield {"progress": 0.06, "message": "Connecting to robot..."}
        
        # Connect to robot
        robot.connect()
        
        # Update robot state
        robot_state = RobotState(
            node_id=ctx.node_name,
            is_connected_l=robot.is_connected,
            is_connected_r=robot.is_connected,
            current_pose="connecting",
        )
        await state_manager.update_robot_state(robot_state)
        
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
        
        repo_id = f"tatbot/{dataset_dir.name}"
        
        if resume:
            dataset = LeRobotDataset(repo_id, root=str(dataset_dir))
            if num_camera_threads > 0:
                dataset.start_image_writer(num_processes=0, num_threads=num_camera_threads)
            sanity_check_dataset_robot_compatibility(dataset, robot, fps, dataset_features)
        else:
            sanity_check_dataset_name(repo_id, None)
            dataset = LeRobotDataset.create(
                repo_id,
                fps=fps,
                root=str(dataset_dir),
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=num_camera_threads,
            )
        
        yield {"progress": 0.07, "message": f"Created/resumed LeRobot dataset at {repo_id}"}
        
        # Move robot to ready position
        yield {"progress": 0.08, "message": "Sending robot to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        # Update robot state
        robot_state.current_pose = "ready"
        await state_manager.update_robot_state(robot_state)
        
        # Setup episode logging
        yield {"progress": 0.2, "message": "Creating episode logger..."}
        
        logs_dir = dataset_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        episode_log_buffer = StringIO()
        
        class EpisodeLogHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                episode_log_buffer.write(msg + "\n")
        
        episode_handler = EpisodeLogHandler()
        episode_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=TIME_FORMAT))
        logging.getLogger().addHandler(episode_handler)
        
        # Setup joystick if enabled
        enable_joystick = getattr(input_data, 'enable_joystick', False)
        if enable_joystick:
            from lerobot.teleoperators.gamepad import (
                AtariTeleoperator,
                AtariTeleoperatorConfig,
            )
            
            yield {"progress": 0.21, "message": "Connecting to joystick..."}
            atari_teleop = AtariTeleoperator(AtariTeleoperatorConfig())
            atari_teleop.connect()
        
        # Load or generate strokes
        yield {"progress": 0.22, "message": "Loading strokes..."}
        
        strokes_path = dataset_dir / "strokes.yaml"
        strokebatch_path = dataset_dir / "strokebatch.safetensors"
        
        if resume:
            if not strokes_path.exists():
                raise FileNotFoundError(f"❌ Strokes file {strokes_path} does not exist")
            if not strokebatch_path.exists():
                raise FileNotFoundError(f"❌ Strokebatch file {strokebatch_path} does not exist")
            
            strokes: StrokeList = StrokeList.from_yaml_with_arrays(str(strokes_path))
            strokebatch: StrokeBatch = StrokeBatch.load(str(strokebatch_path))
        else:
            # Generate new strokes
            strokes: StrokeList = make_gcode_strokes(scene)
            strokes.to_yaml_with_arrays(str(strokes_path))
            
            # Convert to strokebatch (GPU or local)
            if check_local_gpu():
                log.info("Using local GPU for strokebatch conversion")
                strokebatch: StrokeBatch = strokebatch_from_strokes(scene, strokes)
                strokebatch.save(str(strokebatch_path))
            else:
                log.info("Using remote GPU node for strokebatch conversion")
                gpu_proxy = GPUConversionService()
                # Inform UI/TUI users that we are routing to a GPU node
                yield {"progress": 0.24, "message": "Routing stroke conversion to GPU node..."}
                
                success, _ = await gpu_proxy.convert_strokelist_remote(
                    strokes_file_path=str(strokes_path),
                    strokebatch_file_path=str(strokebatch_path),
                    scene=scene.name,
                    first_last_rest=True,
                    use_ee_offsets=True,
                    meta=input_data.meta
                )
                
                if not success:
                    raise RuntimeError("Failed to convert strokes to strokebatch on remote GPU node")
                
                strokebatch = StrokeBatch.load(str(strokebatch_path))
            
            log.info(f"Strokebatch created with shape: {strokebatch.joints.shape}")
        
        num_strokes = len(strokes.strokes)
        
        # Start stroke session in state manager
        session_id = await state_manager.start_stroke_session(
            total_strokes=num_strokes,
            stroke_length=scene.stroke_length,
            scene_name=scene.name
        )
        
        # Initialize offset indices for needle depth control
        mid_offset_idx: int = scene.arms.offset_num // 2
        offset_idx_l: int = mid_offset_idx
        offset_idx_r: int = mid_offset_idx
        inkdip_offset_idx_l: int = mid_offset_idx
        inkdip_offset_idx_r: int = mid_offset_idx
        
        # Execute strokes
        log.info(f"Recording {num_strokes} paths...")
        start_episode = dataset.num_episodes if resume else 0
        
        for stroke_idx in range(start_episode, num_strokes):
            # Reset log buffer for new episode
            episode_log_buffer.seek(0)
            episode_log_buffer.truncate(0)
            
            # Ensure robot is connected and in ready position
            if not robot.is_connected:
                log.warning("⚠️ Robot is not connected. Attempting to reconnect...")
                robot.connect()
                if not robot.is_connected:
                    raise RuntimeError("❌ Failed to connect to robot")
            
            robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
            
            # Get strokes for this episode
            stroke_l, stroke_r = strokes.strokes[stroke_idx]
            stroke_msg = f"Executing stroke {stroke_idx + 1}/{num_strokes}: left={stroke_l.description}, right={stroke_r.description}"
            log.info(stroke_msg)
            
            # Update stroke progress
            await state_manager.update_stroke_progress(
                stroke_idx=stroke_idx,
                pose_idx=0,
                stroke_description_l=stroke_l.description,
                stroke_description_r=stroke_r.description,
                offset_idx_l=offset_idx_l,
                offset_idx_r=offset_idx_r,
                session_id=session_id
            )
            
            yield {
                'progress': 0.3 + (0.6 * stroke_idx / num_strokes),
                'message': stroke_msg,
            }
            
            # Setup episode-specific conditioning
            episode_cond = {}
            episode_cond_dir = dataset_dir / f"episode_{stroke_idx:06d}"
            episode_cond_dir.mkdir(exist_ok=True)
            log.debug(f"🗃️ Creating episode-specific condition directory at {episode_cond_dir}")
            
            episode_cond["stroke_l"] = stroke_l.model_dump_for_yaml()
            episode_cond["stroke_r"] = stroke_r.model_dump_for_yaml()
            
            if stroke_l.frame_path is not None:
                shutil.copy(stroke_l.frame_path, episode_cond_dir / "stroke_l.png")
            if stroke_r.frame_path is not None:
                shutil.copy(stroke_r.frame_path, episode_cond_dir / "stroke_r.png")
            
            # Execute stroke poses
            log.info(f"🤖 recording path {stroke_idx} of {num_strokes}")
            for pose_idx in range(scene.stroke_length):
                start_loop_t = time.perf_counter()
                log.debug(f"pose_idx: {pose_idx}/{scene.stroke_length}")
                
                # Update pose progress
                if pose_idx % 5 == 0:  # Update every 5th pose to reduce overhead
                    await state_manager.update_stroke_progress(
                        stroke_idx=stroke_idx,
                        pose_idx=pose_idx,
                        stroke_description_l=stroke_l.description,
                        stroke_description_r=stroke_r.description,
                        offset_idx_l=offset_idx_l,
                        offset_idx_r=offset_idx_r,
                        session_id=session_id
                    )
                
                # Handle joystick input if enabled
                if enable_joystick and atari_teleop:
                    action = atari_teleop.get_action()
                    if action.get("red_button", False):
                        raise KeyboardInterrupt("User requested stop via joystick")
                    
                    # Adjust offset indices based on joystick input
                    if action.get("y", None) is not None:
                        if stroke_l.is_inkdip:
                            inkdip_offset_idx_l += int(action["y"])
                            inkdip_offset_idx_l = max(0, min(inkdip_offset_idx_l, scene.arms.offset_num - 1))
                        else:
                            offset_idx_l += int(action["y"])
                            offset_idx_l = max(0, min(offset_idx_l, scene.arms.offset_num - 1))
                    
                    if action.get("x", None) is not None:
                        if stroke_r.is_inkdip:
                            inkdip_offset_idx_r += int(action["x"])
                            inkdip_offset_idx_r = max(0, min(inkdip_offset_idx_r, scene.arms.offset_num - 1))
                        else:
                            offset_idx_r += int(action["x"])
                            offset_idx_r = max(0, min(offset_idx_r, scene.arms.offset_num - 1))
                
                # Get observation
                observation = robot.get_observation()
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                
                # Determine offset indices and movement order
                left_first = True  # default move left arm first
                if stroke_l.is_inkdip:
                    _offset_idx_l = inkdip_offset_idx_l
                    log.debug(f"left inkdip offset index: {_offset_idx_l}")
                else:
                    _offset_idx_l = offset_idx_l
                    log.debug(f"left offset index: {_offset_idx_l}")
                
                if stroke_r.is_inkdip:
                    left_first = False  # inkdip strokes move right arm first
                    _offset_idx_r = inkdip_offset_idx_r
                    log.debug(f"right inkdip offset index: {_offset_idx_r}")
                else:
                    _offset_idx_r = offset_idx_r
                    log.debug(f"right offset index: {_offset_idx_r}")
                
                # Get target joints and execute
                joints = strokebatch.offset_joints(stroke_idx, pose_idx, _offset_idx_l, _offset_idx_r)
                robot_action = robot._urdf_joints_to_action(joints)
                
                # Use slow movements for initial poses, fast for others
                if pose_idx in (0, 1):
                    sent_action = robot.send_action(
                        robot_action, 
                        scene.arms.goal_time_slow, 
                        safe=True, 
                        left_first=left_first
                    )
                else:
                    sent_action = robot.send_action(robot_action, scene.arms.goal_time_fast)
                
                # Record data
                action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                dataset.add_frame(frame, task=f"left: {stroke_l.description}, right: {stroke_r.description}")
                
                # Maintain FPS
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)
            
            # Save episode log
            log_path = logs_dir / f"episode_{stroke_idx:06d}.txt"
            log.info(f"🗃️ Writing episode log to {log_path}")
            with open(log_path, "w") as f:
                f.write(episode_log_buffer.getvalue())
            
            # Save episode data
            dataset.save_episode(episode_cond=episode_cond)
        
        # Return to ready position
        yield {"progress": 0.95, "message": "Returning robot to ready position..."}
        robot.send_action(robot._urdf_joints_to_action(scene.ready_pos_full.joints), safe=True)
        
        # Update robot state and end stroke session
        robot_state.current_pose = "ready"
        await state_manager.update_robot_state(robot_state)
        await state_manager.end_stroke_session(session_id)
        
        log.info("✅ Stroke operation completed successfully")
        
        yield StrokeOutput(
            success=True,
            message=f"✅ Stroke execution completed successfully. Executed {num_strokes} strokes.",
            stroke_count=num_strokes
        )
        
    except KeyboardInterrupt:
        log.info("🛑 Stroke operation interrupted by user")
        if state_manager and session_id:
            await state_manager.end_stroke_session(session_id)
        yield StrokeOutput(
            success=False,
            message="🛑 Stroke operation interrupted by user",
            stroke_count=0
        )
        return
    except Exception as e:
        error_msg = f"❌ Stroke operation failed: {e}"
        log.error(error_msg)
        if state_manager:
            await state_manager.report_error("stroke_execution", str(e), {"scene": input_data.scene})
            if session_id:
                await state_manager.end_stroke_session(session_id)
        yield StrokeOutput(
            success=False,
            message=error_msg,
            stroke_count=0
        )
        
    finally:
        # Cleanup
        if episode_handler is not None:
            try:
                logging.getLogger().removeHandler(episode_handler)
            except Exception as e:
                log.error(f"Error removing episode handler: {e}")
        
        if atari_teleop is not None:
            try:
                atari_teleop.disconnect()
            except Exception as e:
                log.error(f"Error disconnecting joystick: {e}")
        
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                log.error(f"Error disconnecting robot: {e}")
        
        if state_manager is not None:
            try:
                await state_manager.disconnect()
            except Exception as e:
                log.error(f"Error disconnecting state manager: {e}")
