import cv2
import apriltag
import numpy as np
import jaxlie
import jax.numpy as jnp
import viser
import tyro
import logging
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
import math

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# CLIArgs Dataclass
@dataclass
class CLIArgs:
    camera_device_id: int = 0
    """ID of the camera device for cv2.VideoCapture."""
    camera_fx: float = 600.0
    """Focal length fx of the camera (in pixels)."""
    camera_fy: float = 600.0
    """Focal length fy of the camera (in pixels)."""
    camera_cx: float = 320.0 
    """Principal point cx of the camera (in pixels)."""
    camera_cy: float = 240.0 
    """Principal point cy of the camera (in pixels)."""
    tag_family: str = "tag36h11"
    """AprilTag family to detect."""
    tag_size_m: float = 0.16
    """Physical size of the AprilTag in meters (black square part)."""
    mat_tag_id: int = -1
    """Tag ID of the AprilTag that represents the mat. -1 for no specific mat tag."""
    debug_viser: bool = False
    """Enable Viser transform controls for debugging poses."""

# main function and if __name__ == "__main__": block
def main(args: CLIArgs):
    log.info(f"Starting AprilTag detection with JAXlie and OpenCV. Arguments: {args}")
    
    server = viser.ViserServer()
    log.info(f"Viser server started. Access it at: {server.get_url()}")

    log.info(f"Attempting to open camera device ID: {args.camera_device_id}")
    cap = cv2.VideoCapture(args.camera_device_id)
    if not cap.isOpened():
        log.error(f"Error: Could not open camera device ID {args.camera_device_id}.")
        return # Exit if camera fails to open
    
    # Optionally set camera properties like resolution, if needed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Camera opened successfully. Frame dimensions: {frame_width}x{frame_height}")

    # Viser Setup
    # The camera frustum visualizes the camera's field of view in the 3D scene.
    # It's placed at the origin (identity pose) because all tag poses are relative to the camera.
    # Viser's coordinate system: Y up, X right, looking down -Z.
    # Camera optical frame: Z forward, X right, Y down.
    # The frustum is defined using camera intrinsics (fx, fy) and frame dimensions.
    # fx, fy: focal lengths in pixels. cx, cy: principal point in pixels.
    # These intrinsics are crucial for accurate 3D pose estimation by the AprilTag library.
    log.info("Setting up Viser camera frustum and image output...")
    fov_y_radians = 2 * np.arctan(frame_height / (2 * args.camera_fy)) # Vertical FoV
    server.scene.add_camera_frustum(
        name="/camera", # Name of the camera frustum in Viser
        fov=fov_y_radians, # Field of view in y direction, in radians
        aspect=frame_width / frame_height, # Aspect ratio of the camera sensor
        scale=0.1, # Visual size of the frustum model in Viser
        # Default pose is identity (wxyz=[1,0,0,0], position=[0,0,0]), which is correct for our setup.
    )
    log.info(f"Added camera frustum to Viser. Frame: {frame_width}x{frame_height}, Fy: {args.camera_fy}, Calculated FoVy: {np.rad2deg(fov_y_radians):.2f} degrees.")

    # GUI image output for displaying the camera feed with annotations
    image_output_handle = server.add_gui_image_output(
        name="/camera_feed", # Name of the image output in Viser
        initial_image=np.zeros((frame_height, frame_width, 3), dtype=np.uint8), # Placeholder black image
        height=frame_height // 2 # Display height in Viser GUI, can be adjusted
    )
    log.info("Added Viser GUI image output.")

    # Initialize AprilTag detector
    # The user provides camera intrinsics (fx, fy, cx, cy) and tag size via CLI args.
    # These are essential for the detector to estimate the 3D pose of the tags.
    log.info(f"Initializing AprilTag detector with family: {args.tag_family}. Tag size: {args.tag_size_m}m.")
    log.info(f"Using camera intrinsics for detection: fx={args.camera_fx}, fy={args.camera_fy}, cx={args.camera_cx}, cy={args.camera_cy}")
    detector_options = apriltag.DetectorOptions(families=args.tag_family,
                                              nthreads=1, 
                                              quad_decimate=1.0, # No decimation
                                              quad_sigma=0.0,    # No blurring
                                              refine_edges=True, # Refine edges for better pose
                                              decode_sharpening=0.25, # Default sharpening
                                              )
    detector = apriltag.Detector(options=detector_options)
    log.info("AprilTag detector initialized.")

    try:
        log.info("Entering main processing loop (Ctrl+C to exit)...")
        while True:
            # Per-frame data storage
            current_frame_tag_poses: Dict[int, jaxlie.SE3] = {} # {tag_id: T_camera_tag}
            mat_pose_camera_frame: Optional[jaxlie.SE3] = None   # T_camera_mat

            # Read a frame from the OpenCV camera
            # cap.read() returns (boolean success, BGR_frame_as_numpy_array)
            ret, bgr_frame = cap.read() 
            if not ret:
                log.warning("Failed to grab frame. End of stream or camera error?")
                time.sleep(0.5) 
                continue 

            # Convert BGR frame to grayscale for AprilTag detection
            gray_image = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            
            # Camera parameters tuple for pupil-apriltags: (fx, fy, cx, cy)
            camera_params = (args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy)
            
            # Perform AprilTag detection
            # estimate_tag_pose=True: enables 3D pose estimation.
            # camera_params: required for pose estimation.
            # tag_size: physical size of the tag (black square) in meters, required for pose estimation.
            # Returns a list of 'Detection' objects.
            # Each detection can contain:
            #   - tag_id (int)
            #   - corners (4x2 np.ndarray, pixel coords)
            #   - center (1x2 np.ndarray, pixel coords)
            #   - pose_R (3x3 np.ndarray, rotation matrix of tag in camera frame)
            #   - pose_t (3x1 np.ndarray, translation vector of tag in camera frame, meters)
            detections = detector.detect(gray_image, 
                                         estimate_tag_pose=True, 
                                         camera_params=camera_params, 
                                         tag_size=args.tag_size_m)
            
            log.debug(f"Detected {len(detections)} AprilTags this frame.") # Changed from log.info
            for detection in detections:
                log.debug(f"  Tag ID: {detection.tag_id}, Pixel Center: {detection.center.astype(int)}")
                if detection.pose_R is not None and detection.pose_t is not None:
                    # Raw pose data from the detector
                    rotation_matrix = detection.pose_R      # 3x3 SO(3) matrix
                    translation_vector = detection.pose_t.flatten() # Ensure 1D array (3,)

                    try:
                        # Convert raw rotation and translation to a jaxlie.SE3 object.
                        # This represents the pose of the tag in the camera's coordinate frame (T_camera_tag).
                        # jaxlie.SE3.from_rotation_and_translation expects:
                        #   - rotation: jaxlie.SO3 object
                        #   - translation: 1D np.ndarray or jnp.ndarray of shape (3,)
                        tag_pose_in_camera_frame = jaxlie.SE3.from_rotation_and_translation(
                            rotation=jaxlie.SO3.from_matrix(rotation_matrix),
                            translation=translation_vector
                        )
                        current_frame_tag_poses[detection.tag_id] = tag_pose_in_camera_frame
                        log.debug(f"    Tag ID: {detection.tag_id} - Converted to jaxlie.SE3 (T_camera_tag): {tag_pose_in_camera_frame}")

                        # If this tag is the designated 'mat' tag, store its pose separately.
                        if detection.tag_id == args.mat_tag_id:
                            mat_pose_camera_frame = tag_pose_in_camera_frame
                            log.info(f"    Mat tag (ID: {args.mat_tag_id}) detected. Stored T_camera_mat: {mat_pose_camera_frame}")
                    
                    except Exception as e:
                        log.error(f"    Error converting raw pose for tag ID {detection.tag_id} to jaxlie.SE3: {e}")
                        log.debug(f"    Problematic Rotation Matrix:\n{rotation_matrix}")
                        log.debug(f"    Problematic Translation Vector:\n{translation_vector}")
                else:
                    log.debug(f"    Tag ID: {detection.tag_id} - Pose estimation failed (pose_R or pose_t is None).")

            # --- Viser Updates ---
            # Convert BGR frame from OpenCV to RGB for Viser display
            rgb_frame_for_viser = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            # Create a copy for drawing 2D annotations (boxes, IDs)
            annotated_frame_viser = rgb_frame_for_viser.copy()

            # Draw 2D annotations for each detected tag on the frame
            for det_id_loop, _ in current_frame_tag_poses.items(): # Iterate over successfully converted poses
                original_detection = next((d for d in detections if d.tag_id == det_id_loop), None)
                if original_detection is not None and original_detection.corners is not None:
                    corners = original_detection.corners.astype(int) # Pixel coordinates of corners
                    cv2.polylines(annotated_frame_viser, [corners], True, (0, 255, 0), 1) # Green box
                    # Place tag ID text near the first corner of the tag
                    cv2.putText(annotated_frame_viser, str(det_id_loop), tuple(corners[0]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1) # Blue text
            # Update the Viser GUI image output with the annotated frame
            image_output_handle.set_image(annotated_frame_viser)

            # Visualize 3D poses of all detected tags in Viser
            for tag_id_vis, tag_pose_vis_se3 in current_frame_tag_poses.items():
                # tag_pose_vis_se3 is T_camera_tag (tag's pose in camera frame)
                # Viser expects wxyz quaternion and xyz position.
                server.scene.add_frame(
                    name=f"/detected_tags/tag_{tag_id_vis}", # Unique name for each tag's frame
                    wxyz=np.array(tag_pose_vis_se3.rotation().wxyz()), # Get wxyz quaternion from SO3 part
                    position=np.array(tag_pose_vis_se3.translation()),  # Get xyz translation
                    axes_length=args.tag_size_m / 2.0, # Length of the visualized axes
                    axes_radius=0.002                  # Thickness of the visualized axes
                )
            
            # Visualize the mat's pose if detected
            if mat_pose_camera_frame is not None:
                # mat_pose_camera_frame is T_camera_mat
                mat_name = f"/mat_pose/tag_{args.mat_tag_id}" # Consistent name for the mat
                mat_wxyz = np.array(mat_pose_camera_frame.rotation().wxyz())
                mat_position = np.array(mat_pose_camera_frame.translation())
                
                if args.debug_viser:
                    # Use transform controls for interactive debugging of the mat's pose
                    server.scene.add_transform_controls(
                        name=mat_name,
                        wxyz=mat_wxyz,
                        position=mat_position,
                        scale=args.tag_size_m * 0.8, # Visual scale of the controls gizmo
                        axes_length=args.tag_size_m, # Length of axes for the frame itself
                        axes_radius=0.005
                    )
                else:
                    # Display as a static frame if not in debug mode
                    server.scene.add_frame(
                        name=mat_name,
                        wxyz=mat_wxyz,
                        position=mat_position,
                        axes_length=args.tag_size_m, # Larger axes for the mat tag
                        axes_radius=0.005
                    )
                log.debug(f"Visualizing mat pose (ID: {args.mat_tag_id}) in Viser at {mat_position}")
            else:
                # Mat not detected this frame.
                # If using transform controls with a handle (like in apriltag_jules.py),
                # here you would set mat_tf_handle.visible = False.
                # Since add_transform_controls is called each time, if the mat isn't detected,
                # the control for it simply isn't updated/re-added for this Viser message.
                # Viser's behavior for non-updated transform controls needs to be checked;
                # they might persist or disappear. Explicit management is safer for controls.
                # For add_frame, Viser typically removes frames not present in the current message bundle.
                log.debug(f"Mat tag (ID: {args.mat_tag_id}) not detected this frame, or pose conversion failed.")
            
            # OpenCV window display is removed as Viser is the primary interface
            # cv2.imshow('Live Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     log.info("'q' pressed, exiting loop.")
            #     break
            
            time.sleep(1.0 / 60) # Aim for ~60 FPS for Viser updates, adjust as needed
    except KeyboardInterrupt:
        log.info("Exiting due to Ctrl+C")
    finally:
        log.info("Releasing camera...")
        cap.release()
        # cv2.destroyAllWindows() # No longer needed as cv2.imshow is removed
        log.info("Script finished.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    main(args)
