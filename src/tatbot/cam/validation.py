import numpy as np

from tatbot.bot.urdf import get_link_poses
from tatbot.data.cams import Cams
from tatbot.data.scene import Scene
from tatbot.utils.log import get_logger

log = get_logger("cam.validation", "📊")


def compare_extrinsics_with_urdf(
    calibrated_cams: Cams,
    scene: Scene,
    max_deviation_warning: float = 0.05
) -> dict[str, float]:
    """Compare calibrated camera extrinsics with URDF-defined positions.
    
    Args:
        calibrated_cams: Camera configuration with calibrated extrinsics
        scene: Scene configuration containing URDF and camera definitions
        max_deviation_warning: Maximum deviation (m) before showing warning
        
    Returns:
        Dictionary mapping camera names to position deviations (in meters)
    """
    log.info("📊 Comparing calibrated extrinsics with URDF positions...")
    
    # Get URDF camera positions from optical frame links
    urdf_cam_links = []
    
    # Add IP camera optical frames
    for cam in scene.cams.ipcameras:
        urdf_cam_links.append(f"{cam.name}_optical_frame")
        
    # Add RealSense camera optical frames  
    for cam in scene.cams.realsenses:
        # Convert realsense_link to color_optical_frame
        base_name = cam.urdf_link_name.replace("_link", "")
        urdf_cam_links.append(f"{base_name}_color_optical_frame")
    
    urdf_cam_links = tuple(urdf_cam_links)
    deviations = {}
    
    try:
        urdf_link_poses = get_link_poses(
            scene.urdf.path,
            urdf_cam_links,
            scene.ready_pos_full.joints
        )
        
        # Compare each calibrated camera with URDF position
        # Iterate across known camera names based on configs
        cam_names: list[str] = []
        cam_names.extend([cam.name for cam in calibrated_cams.ipcameras])
        cam_names.extend([cam.name for cam in calibrated_cams.realsenses])
        for cam_name in cam_names:
            cam_config = calibrated_cams.get_camera(cam_name)
            calibrated_pos = np.array(cam_config.extrinsics.pos.xyz)
            
            # Find corresponding URDF link
            urdf_link_name = None
            
            # Check IP cameras
            for cam in scene.cams.ipcameras:
                if cam.name == cam_name:
                    urdf_link_name = f"{cam.name}_optical_frame"
                    break
                    
            # Check RealSense cameras
            for cam in scene.cams.realsenses:
                if cam.name == cam_name:
                    base_name = cam.urdf_link_name.replace("_link", "")
                    urdf_link_name = f"{base_name}_color_optical_frame"
                    break
            
            if urdf_link_name and urdf_link_name in urdf_link_poses:
                urdf_pos = np.array(urdf_link_poses[urdf_link_name].pos.xyz)
                deviation = np.linalg.norm(calibrated_pos - urdf_pos)
                deviations[cam_name] = deviation
                
                if deviation > max_deviation_warning:
                    log.warning(
                        f"⚠️ Camera {cam_name}: Large position deviation from URDF! "
                        f"Deviation: {deviation:.3f}m (>{max_deviation_warning:.3f}m threshold)"
                    )
                    log.warning(f"   URDF position: {urdf_pos}")
                    log.warning(f"   Calibrated position: {calibrated_pos}")
                else:
                    log.info(
                        f"✅ Camera {cam_name}: Position deviation {deviation:.3f}m (within tolerance)"
                    )
            else:
                log.warning(f"⚠️ Camera {cam_name}: Could not find corresponding URDF link for comparison")
                deviations[cam_name] = float('inf')  # Mark as invalid
                
    except Exception as e:
        log.warning(f"⚠️ Failed to compare with URDF positions: {e}")
        # Return empty dict on failure
        return {}
    
    return deviations


def validate_camera_setup(scene: Scene) -> bool:
    """Validate that all cameras in the scene have corresponding URDF links.
    
    Args:
        scene: Scene configuration to validate
        
    Returns:
        True if all cameras have valid URDF links, False otherwise
    """
    log.info("🔍 Validating camera setup against URDF...")
    
    # Get all link names from URDF
    try:
        from tatbot.bot.urdf import load_robot
        _, robot = load_robot(scene.urdf.path)
        all_link_names = set(robot.links.names)
    except Exception as e:
        log.error(f"❌ Failed to load URDF: {e}")
        return False
    
    valid = True
    
    # Check IP cameras
    for cam in scene.cams.ipcameras:
        expected_link = f"{cam.name}_optical_frame"
        if expected_link not in all_link_names:
            log.error(f"❌ IP Camera {cam.name}: Missing URDF link '{expected_link}'")
            valid = False
        else:
            log.debug(f"✅ IP Camera {cam.name}: Found URDF link '{expected_link}'")
    
    # Check RealSense cameras  
    for cam in scene.cams.realsenses:
        base_name = cam.urdf_link_name.replace("_link", "")
        expected_link = f"{base_name}_color_optical_frame"
        if expected_link not in all_link_names:
            log.error(f"❌ RealSense Camera {cam.name}: Missing URDF link '{expected_link}'")
            valid = False
        else:
            log.debug(f"✅ RealSense Camera {cam.name}: Found URDF link '{expected_link}'")
    
    if valid:
        log.info("✅ All cameras have valid URDF links")
    else:
        log.error("❌ Camera setup validation failed")
    
    return valid