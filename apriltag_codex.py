# INFO: This file detects AprilTags using the overhead RealSense camera.
# INFO: This python file requires dependencies defined in the pyproject.toml file.
# INFO: This file is a python script indended to be run directly with optional cli args.

from dataclasses import dataclass
import logging
import time
from typing import Tuple

import numpy as np
import pyrealsense2 as rs
import viser
from pupil_apriltags import Detector
import tyro


@dataclass
class Config:
    serial_number: str = "218622278376"
    tag_size: float = 0.04
    fps: int = 30


class RealSense:
    def __init__(self, serial_number: str, fps: int):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number:
            config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, rs.format.rgb8, fps)
        self.pipeline.start(config)

    def get_color_image(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())

    def get_intrinsics(self) -> Tuple[float, float, float, float]:
        profile = self.pipeline.get_active_profile()
        video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = video_stream.get_intrinsics()
        return intr.fx, intr.fy, intr.ppx, intr.ppy

    def shutdown(self) -> None:
        self.pipeline.stop()


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2])) / 2.0
    w4 = 4.0 * w
    x = (mat[2, 1] - mat[1, 2]) / w4
    y = (mat[0, 2] - mat[2, 0]) / w4
    z = (mat[1, 0] - mat[0, 1]) / w4
    return np.array([w, x, y, z], dtype=np.float32)


def main(config: Config) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(name)s|%(levelname)s|%(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    log.info("🚀 Starting viser server...")
    server = viser.ViserServer()
    server.scene.set_environment_map(hdri="forest", background=True)

    log.info("📷 Initializing RealSense camera...")
    camera = RealSense(config.serial_number, config.fps)
    fx, fy, cx, cy = camera.get_intrinsics()

    # Get the first frame to initialize the GUI image
    initial_img = camera.get_color_image()
    log.info(f"Initial image shape: {initial_img.shape}, dtype: {initial_img.dtype}")

    H, W, _ = initial_img.shape
    # fx, fy, cx, cy are already available from camera.get_intrinsics()
    fov_y_degrees = np.rad2deg(2 * np.arctan(H / (2 * fy)))

    scene_camera_handle = server.scene.add_camera_frustum(
        "/realsense_camera",
        fov=fov_y_degrees,
        aspect=W/H,
        scale=0.15, # Adjust scale as needed for visualization
        color=(200, 200, 200), # Light gray color for the frustum
        image=initial_img, # Display the initial image on the frustum's near plane
    )
    # Position the camera frame at the origin, default orientation.
    # AprilTag poses will be relative to this camera frame.
    scene_camera_handle.position = (0.0, 0.0, 0.0)
    scene_camera_handle.wxyz = (1.0, 0.0, 0.0, 0.0)


    # Add a GUI image of the color frame.
    gui_image_handle = server.gui.add_image(
        label="Color Image",
        image=initial_img,  # Use the first real frame here
        visible=True,
    )
    # log.info(f"gui_image_handle after creation: {gui_image_handle}")

    detector = Detector(families="tag16h5")

    tag_frames = []
    for i in range(3): # Create 3 frames for AprilTags
        frame = server.scene.add_frame(f"/apriltag/tag_{i}")
        frame.visible = False  # Initially hide them
        tag_frames.append(frame)
        # Add a simple square mesh to represent the tag
        server.scene.add_mesh_simple(
            name=f"/apriltag/tag_{i}/mesh", # Child of the tag frame
            vertices=np.array([
                [-config.tag_size / 2, -config.tag_size / 2, 0.0],
                [config.tag_size / 2, -config.tag_size / 2, 0.0],
                [config.tag_size / 2, config.tag_size / 2, 0.0],
                [-config.tag_size / 2, config.tag_size / 2, 0.0],
            ]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            color=(100, 200, 100),  # Greenish color for tags
        )

    try:
        while True:
            img = camera.get_color_image()
            gray = np.mean(img, axis=2).astype(np.uint8)

            # Update the GUI image.
            log.info(f"Updating image - shape: {img.shape}, dtype: {img.dtype}")
            gui_image_handle.value = img
            scene_camera_handle.image = img # Update image on the 3D camera frustum

            dets = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=config.tag_size,
            )
            log.info(f"Detections: {dets}")

            if dets:
                num_detections = len(dets)
                for i in range(len(tag_frames)):
                    if i < num_detections:
                        det = dets[i]
                        tag_frames[i].visible = True
                        R = det.pose_R
                        t = det.pose_t.squeeze()
                        tag_frames[i].position = t
                        tag_frames[i].wxyz = mat_to_quat(R)
                        
                        # Add/Update a label for the tag ID
                        # Viser will update the label if one with the same name exists,
                        # or create it if it doesn't.
                        # Labels are children of their respective tag frames, so they
                        # become visible/invisible with the parent frame.
                        server.scene.add_label(
                            name=f"/apriltag/tag_{i}/label", # Unique name for the label
                            text=f"ID: {det.tag_id}",
                            # Position the label slightly above the tag visualization if desired
                            # position=(0, 0, config.tag_size * 0.1) 
                        )
                        log.info(f"Tag {i} (ID: {det.tag_id}) visible at {t}")
                    else:
                        # Hide frames for which there is no detection
                        tag_frames[i].visible = False
            else:
                # No detections, hide all tag frames
                for i in range(len(tag_frames)):
                    tag_frames[i].visible = False
            
            time.sleep(0.01)
    finally:
        camera.shutdown()


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)