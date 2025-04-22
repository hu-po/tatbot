import os
import logging
import yaml
from dotenv import load_dotenv

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# Load environment variables for camera credentials
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAMERA_CONFIG_PATH = '/root/cameras.yaml'

def generate_launch_description():
    # Verify that the RPI environment variable is set
    rpi_number = os.environ.get("RPI", None)
    if rpi_number is None:
        raise ValueError("Could not recognize rpi number (RPI environment variable not set)")
    else:
        rpi_number = int(rpi_number)
        logger.info(f"RPI number: {rpi_number}")
    
    # Load camera config
    if not os.path.exists(CAMERA_CONFIG_PATH):
        raise FileNotFoundError(f"Could not find camera config file at {CAMERA_CONFIG_PATH}")
    with open(CAMERA_CONFIG_PATH, 'r') as f:
        camera_config = yaml.safe_load(f)
    
    # Define each of the cameras from the YAML data
    cameras = []
    for cam_id, cam_info in camera_config.items():
        # Skip cameras not assigned to this RPI
        if cam_info.get("rpi_assignment") != rpi_number:
            continue
            
        # Skip disabled cameras
        if not cam_info.get("enabled", True):
            logger.info(f"Camera {cam_id} is disabled, skipping")
            continue
            
        # Get password from environment variable
        # Format with leading zeros to match the .env file format (e.g., CAMERA_002_PASSWORD)
        password_env_var = f"CAMERA_{cam_id:03d}_PASSWORD" if isinstance(cam_id, int) else f"CAMERA_{cam_id}_PASSWORD"
        password = os.environ.get(password_env_var)
        
        if not password:
            logger.error(f"No password found for camera {cam_id} (missing {password_env_var} environment variable)")
            continue
            
        cameras.append({
            "id": cam_id,
            "pipeline": f"rtspsrc location=rtsp://{cam_info['username']}:{password}@{cam_info['ip']}:554/cam/realmonitor?channel=1&subtype=0 ! decodebin ! videoconvert ! appsink",
            "rpi_assignment": cam_info["rpi_assignment"]
        })
    
    logger.info(f"Loaded {len(cameras)} cameras for RPI {rpi_number}")

    # Add gscam2 nodes which convert ip camera rtsp streams to ros2 topics
    nodes = []
    for cam in cameras:
        cam_id = cam["id"]
        formatted_cam_id = f"{cam_id:03d}" if isinstance(cam_id, int) else cam_id
        nodes.append(
            Node(
                package='gscam2',
                executable='gscam_main',
                name=f'camera_stream_{formatted_cam_id}',
                parameters=[{
                    'gscam_config': cam["pipeline"],
                    'image_topic': f'camera_{formatted_cam_id}_image_raw',
                    'camera_name': f'camera_{formatted_cam_id}',
                    'camera_info_url': f'file:///root/.ros/camera_info/{formatted_cam_id}.yaml',
                    'frame_id': f'camera_{formatted_cam_id}'
                }],
                output='screen'
            )
        )
    
    # Add AprilTag container with composable nodes - one set for each camera
    composable_nodes = []
    for cam in cameras:
        cam_id = cam["id"]
        # Image rectification node for this camera
        composable_nodes.append(
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name=f'rectify_{cam_id}',
                namespace=f'apriltag_{cam_id}',
                remappings=[
                    ('image', f'/camera_{cam_id}_image_raw')
                ],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        )
        
        # AprilTag node for this camera
        composable_nodes.append(
            ComposableNode(
                package='apriltag_ros',
                plugin='AprilTagNode',
                name=f'apriltag_{cam_id}',
                namespace=f'apriltag_{cam_id}',
                remappings=[
                    (f'/apriltag_{cam_id}/image_rect', f'/apriltag_{cam_id}/image_rect'),
                    (f'/apriltag_{cam_id}/camera_info', f'/apriltag_{cam_id}/camera_info')
                ],
                parameters=[{'from': '/root/apriltags.yaml'}],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        )
    
    # Only create the container if we have enabled cameras
    if composable_nodes:
        # Create the container with all the composable nodes
        apriltag_container = ComposableNodeContainer(
            name='apriltag_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=composable_nodes
        )
        
        # Add the container to our nodes list
        nodes.append(apriltag_container)
    else:
        logger.info("No cameras enabled, skipping AprilTag container")

    return LaunchDescription(nodes)