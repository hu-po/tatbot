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
    mat_tf = server.scene.add_frame("/mat")

    log.info("📷 Initializing RealSense camera...")
    camera = RealSense(config.serial_number, config.fps)
    fx, fy, cx, cy = camera.get_intrinsics()

    detector = Detector(families="tag36h11")

    try:
        while True:
            img = camera.get_color_image()
            gray = np.mean(img, axis=2).astype(np.uint8)
            dets = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=config.tag_size,
            )
            if dets:
                det = dets[0]
                R = det.pose_R
                t = det.pose_t.squeeze()
                mat_tf.position = t
                mat_tf.wxyz = mat_to_quat(R)
            time.sleep(0.01)
    finally:
        camera.shutdown()


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)

