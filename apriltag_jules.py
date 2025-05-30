import viser
import pyrealsense2 as rs
import apriltag
import numpy as np
import tyro
import cv2
import dataclasses
import logging
import time
import typing
import math

# Set up basic logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

@dataclasses.dataclass
class CLIArgs:
    """Command-line arguments for AprilTag detection."""
    realsense_serial_number: str = tyro.conf.arg(default="", help="Serial number of the RealSense camera. If empty, uses the first camera found.")
    tag_family: str = tyro.conf.arg(default="tag36h11", help="AprilTag family to detect.")
    tag_size_m: float = tyro.conf.arg(default=0.16, help="The physical size of the AprilTag in meters (typically the black square part).")
    debug_viser: bool = tyro.conf.arg(default=False, help="Enable Viser transform controls for debugging the mat pose.")
    mat_tag_id: int = tyro.conf.arg(default=-1, help="Tag ID of the AprilTag that represents the mat. If -1, mat pose will not be specifically identified or visualized with transform controls.")


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    # Converts a 3x3 rotation matrix to a WXYZ quaternion.
    # Source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = np.trace(R)
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


class RealSenseCamera:
    def __init__(self, serial_number: str = "", width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if serial_number:
            log.info(f"Attempting to use RealSense camera with serial number: {serial_number}")
            self.config.enable_device(serial_number)
        else:
            log.info("No RealSense serial number provided, using the first camera found.")
        
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        # Start the pipeline
        self.pipeline.start(self.config)

        # Retrieve camera intrinsics after starting the pipeline
        profile = self.pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()

        # fx, fy: focal lengths in pixels
        # cx, cy: principal point (center of projection) in pixels
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx 
        self.cy = intrinsics.ppy
        # Distortion coefficients (e.g., [k1, k2, p1, p2, k3] for Brown-Conrady model)
        self.coeffs = intrinsics.coeffs

        log.info(f"RealSense camera initialized. Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        log.debug(f"Distortion coefficients: {self.coeffs}")

    def get_color_frame(self) -> typing.Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        return image

    def stop(self):
        self.pipeline.stop()


def main(args: CLIArgs):
    """Main function for AprilTag detection."""
    log.info(f"Starting AprilTag detection with args: {args}")

    server = viser.ViserServer()
    camera = RealSenseCamera(
        serial_number=args.realsense_serial_number
        # Camera width/height are defaulted in its constructor.
        # If we wanted CLI args for camera width/height, they'd be passed here.
    )

    # Viser setup: Add a camera frustum to represent the RealSense camera in the 3D scene.
    # The FOV is calculated from the camera's focal length (fy) and sensor height.
    # The camera is placed at the origin of the Viser scene.
    server.scene.add_camera_frustum(
        name="/realsense_cam",
        fov=2 * np.arctan((camera.height / 2) / camera.fy), 
        aspect=camera.width / camera.height,
        scale=0.15, # Visual scale of the frustum model in Viser
        # Default pose is identity (wxyz=[1,0,0,0], position=[0,0,0]), which is correct.
    )
    # Add an image output to Viser to display the camera feed.
    image_output_handle = server.add_image_output(name="/image_feed", height=360)


    log.info(f"Initializing AprilTag detector with family: '{args.tag_family}'")
    # AprilTag Detector Parameters:
    #   families: The tag family (e.g., "tag36h11").
    #   nthreads: Number of CPU threads to use.
    #   quad_decimate: Detection of quads can be done on a lower-resolution image, improving speed. (e.g., 1.0 = full res).
    #   quad_sigma: Apply Gaussian blur to the quad image before processing.
    #   refine_edges: Spend more time trying to fit lines to edges of quads.
    #   decode_sharpening: How much to sharpen image before decoding.
    #   debug: If 1, saves debug images from the C library.
    detector = apriltag.Detector(families=args.tag_family,
                                   nthreads=1,          # Use 1 thread for now
                                   quad_decimate=1.0,   # No decimation
                                   quad_sigma=0.0,      # No blur
                                   refine_edges=1,      # Refine edges
                                   decode_sharpening=0.25,# Default sharpening
                                   debug=0)             # No C-level debug images
    log.info("AprilTag detector initialized.")

    # Handle for mat transform controls if debug_viser is true and a mat_tag_id is specified
    mat_tf_handle: typing.Optional[viser.TransformControlsHandle] = None
    if args.debug_viser and args.mat_tag_id != -1:
        log.info(f"Debug Viser mode enabled. Mat tag {args.mat_tag_id} will have transform controls.")
        # Initialize invisible; will be updated when/if the mat tag is seen.
        # The name includes the tag ID for clarity in Viser.
        mat_tf_handle = server.scene.add_transform_controls(
            name=f"/mat_pose_tf_debug/tag_{args.mat_tag_id}", 
            scale=args.tag_size_m, 
            opacity=0.5, 
            visible=False
        )

    mat_detected_this_frame = False

    try:
        while True:
            mat_detected_this_frame = False
            # These store the actual pose if the mat is detected in the current frame
            current_mat_pose_R: typing.Optional[np.ndarray] = None
            current_mat_pose_t: typing.Optional[np.ndarray] = None

            color_image = camera.get_color_frame()
            vis_image = None # For drawing annotations

            if color_image is not None:
                # Display the raw camera image in Viser
                image_output_handle.set_image(color_image) 
                vis_image = color_image.copy() # Create a copy for drawing 2D annotations

                # Convert to grayscale for AprilTag detection
                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                
                # Detect AprilTags.
                #   estimate_tag_pose=True: Enables 3D pose estimation.
                #   camera_params: (fx, fy, cx, cy) - camera intrinsics.
                #   tag_size: Physical size of the tag (square part) in meters.
                # This returns a list of `Detection` objects. Each detection includes:
                #   tag_id: The ID of the detected tag.
                #   corners: Pixel coordinates of the tag's four corners.
                #   center: Pixel coordinates of the tag's center.
                #   pose_R: 3x3 rotation matrix (tag frame to camera frame).
                #   pose_t: 3x1 translation vector (tag frame to camera frame, in meters).
                detections = detector.detect(gray_image,
                                             estimate_tag_pose=True,
                                             camera_params=(camera.fx, camera.fy, camera.cx, camera.cy),
                                             tag_size=args.tag_size_m)
                log.debug(f"Detected {len(detections)} AprilTags this frame.")

                # Viser updates by re-adding elements with the same name.
                # Tags detected in this frame will be added/updated.
                # Tags not detected in this frame will implicitly disappear from Viser
                # if they were added with `add_frame` and not re-added in this message.
                # TransformControls need explicit visibility management.

                for detection in detections:
                    # Draw 2D annotations on the `vis_image`
                    tag_corners_int = detection.corners.astype(np.int32)
                    cv2.polylines(vis_image, [tag_corners_int], isClosed=True, color=(0, 255, 0), thickness=1)
                    tag_center_int = detection.center.astype(np.int32)
                    cv2.putText(vis_image, str(detection.tag_id), tuple(tag_center_int - np.array([10,10])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Visualize 3D Pose in Viser
                    if detection.pose_R is not None and detection.pose_t is not None:
                        # Convert rotation matrix to WXYZ quaternion for Viser
                        wxyz = rotation_matrix_to_quaternion(detection.pose_R)
                        # Position is the translation vector (flattened to 1D for Viser)
                        position = detection.pose_t.flatten()
                        
                        # Add a coordinate frame for the detected tag.
                        # Viser updates existing frames if the name matches.
                        server.scene.add_frame(
                            name=f"/detected_tags/tag_{detection.tag_id}", # Unique name per tag
                            wxyz=wxyz,
                            position=position,
                            axes_length=args.tag_size_m / 2.5, # Smaller axes for individual tags
                            axes_radius=0.002
                        )

                        # Check if this detection is the specified mat tag
                        if detection.tag_id == args.mat_tag_id:
                            mat_detected_this_frame = True
                            log.debug(f"Mat tag (ID: {args.mat_tag_id}) detected with pose.")
                            current_mat_pose_R = detection.pose_R
                            current_mat_pose_t = detection.pose_t
                            # Ensure pose_t is (3,1) as some apriltag versions might return (3,)
                            if current_mat_pose_t is not None and current_mat_pose_t.shape == (3,):
                                current_mat_pose_t = current_mat_pose_t.reshape((3,1))
                    else:
                        log.debug(f"  Tag ID: {detection.tag_id}, pose not estimated (detection.pose_R or pose_t is None).")
                
                # Update the Viser image output with the annotated image
                if vis_image is not None: 
                    image_output_handle.set_image(vis_image)

            # Visualize Mat Pose (3D) if it was detected in this frame
            if current_mat_pose_R is not None and current_mat_pose_t is not None:
                mat_wxyz = rotation_matrix_to_quaternion(current_mat_pose_R)
                mat_position = current_mat_pose_t.flatten()

                if args.debug_viser and mat_tf_handle is not None:
                    # If debug Viser is on, update the transform controls
                    mat_tf_handle.visible = True
                    mat_tf_handle.wxyz = mat_wxyz
                    mat_tf_handle.position = mat_position
                    log.info(f"Mat tag {args.mat_tag_id} pose visualized with Viser transform controls.")
                else:
                    # If not in debug mode, or if mat_tf_handle isn't set (e.g. mat_tag_id=-1),
                    # add/update a simple frame for the mat if a mat_tag_id is specified.
                    if args.mat_tag_id != -1:
                        server.scene.add_frame(
                            name=f"/mat_pose/tag_{args.mat_tag_id}", # Consistent naming
                            wxyz=mat_wxyz,
                            position=mat_position,
                            axes_length=args.tag_size_m, # Larger axes for the mat
                            axes_radius=0.005
                        )
                        log.info(f"Mat tag {args.mat_tag_id} pose visualized with Viser frame.")
            
            # If mat_tag_id is specified but mat was not detected in this frame
            if args.mat_tag_id != -1 and not mat_detected_this_frame:
                log.debug(f"Specified mat tag {args.mat_tag_id} was not detected in this frame.")
                if args.debug_viser and mat_tf_handle is not None:
                    # Make transform control invisible if mat not seen
                    mat_tf_handle.visible = False
                # For non-debug simple frames, Viser handles removal if not re-added.
                # We could explicitly remove it: server.scene.remove_node(f"/mat_pose/tag_{args.mat_tag_id}")
                # but this is generally not needed if we re-add it every frame it's seen.

            time.sleep(1.0 / 30)  # Aim for roughly 30fps processing loop
            
    finally:
        camera.stop()
        log.info("RealSense camera stopped.")
        log.info("Program terminated.")

if __name__ == "__main__":
    args = tyro.cli(CLIArgs)
    main(args)
