from dataclasses import dataclass
from typing import List
import time

import cv2
import jax.numpy as jnp
import jaxlie
import numpy as np
import pupil_apriltags as apriltag

from _cam import CameraIntrinsics
from _log import get_logger

log = get_logger('_tag')

@dataclass
class TagConfig:
    apriltag_family: str = "tag16h5"
    """Family of AprilTags to use."""
    apriltag_size_m: float = 0.041
    """Size of AprilTags: distance between detection corners (meters)."""
    apriltags: dict[int, str] = {
        9: "palette",
        10: "origin",
        11: "skin",
    }
    """ AprilTag ID : Name mapping """
    apriltag_decision_margin: float = 20.0
    """Minimum decision margin for AprilTag detection filtering."""

class TagTracker:
    def __init__(self, config: TagConfig):
        self.config = config
        self.detector = apriltag.Detector(
            families=config.apriltag_family,
        )

    def track_tags(self, rgb_b: np.ndarray, intrinsics: CameraIntrinsics):
        log.debug("🏷️ Updating Realsense AprilTags...")
        apriltags_start_time = time.time()
        gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)
        detections: List[apriltag.Detection] = self.detector.detect(
            gray_b,
            # TODO: tune these params
            estimate_tag_pose=True,
            camera_params=(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy),
            tag_size=self.config.apriltag_size_m,
        )
        log.debug(f"🏷️ AprilTags detections: {detections}")
        log.debug(f"🏷️ AprilTags detections before filtering: {len(detections)}")
        detections = [d for d in detections if d.decision_margin >= self.config.apriltag_decision_margin]
        log.info(f"🏷️ AprilTags detections after filtering: {len(detections)}")
        gray_b_bgr = cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR)
        for d in detections:
            corners = np.int32(d.corners)
            cv2.polylines(gray_b_bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            center = tuple(np.int32(d.center))
            cv2.circle(gray_b_bgr, center, 4, (0, 0, 255), -1)
            cv2.putText(gray_b_bgr, str(d.tag_id), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        for d in detections:
            if d.tag_id in apriltag_frames_by_id:
                tag_in_cam = jnp.eye(4)
                tag_in_cam = tag_in_cam.at[:3, :3].set(jnp.array(d.pose_R))
                tag_in_cam = tag_in_cam.at[:3, 3].set(jnp.array(d.pose_t).flatten())
                tag_in_world = jnp.matmul(camera_transform_b.as_matrix(), tag_in_cam)
                pos = jnp.array(tag_in_world[:3, 3])
                wxyz = jaxlie.SO3.from_matrix(tag_in_world[:3, :3]).wxyz
                frame = apriltag_frames_by_id[d.tag_id]
                log.debug(f"🏷️ AprilTag {d.tag_id}:{frame.name} - pos: {pos}, wxyz: {wxyz}")
                frame.position = np.array(pos)
                frame.wxyz = np.array(wxyz)
        apriltags_elapsed_time = time.time() - apriltags_start_time
        apriltag_duration_ms.value = apriltags_elapsed_time * 1000