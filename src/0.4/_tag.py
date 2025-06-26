from dataclasses import dataclass, field
from typing import List
import time

import cv2
import jax.numpy as jnp
import jaxlie
import numpy as np
import pupil_apriltags as apriltag

from _cam import CameraIntrinsics
from _log import COLORS, get_logger

log = get_logger('_tag')

@dataclass
class TagConfig:
    family: str = "tag16h5"
    """Family of AprilTags to use."""
    size_m: float = 0.041
    """Size of AprilTags: distance between detection corners (meters)."""
    enabled_tags: dict[int, str] = field(default_factory=lambda: {
        6: "arm_l",
        7: "arm_r",
        9: "palette",
        10: "origin",
        11: "skin",
    })
    """ Dictionary of enabled AprilTag IDs."""
    urdf_link_names: dict[int, str] = field(default_factory=lambda: {
        6: "tag6",
        7: "tag7",
        9: "tag9",
        10: "tag10",
        11: "tag11",
    })
    """ Dictionary of AprilTag IDs to URDF link names."""
    decision_margin: float = 20.0
    """Minimum decision margin for AprilTag detection filtering."""

@dataclass
class TagPose:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    """Position of the tag in the world frame."""
    wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    """Orientation of the tag in the world frame."""

class TagTracker:
    def __init__(self, config: TagConfig):
        self.config = config
        self.detector = apriltag.Detector(
            families=config.family,
        )

    def track_tags(
        self,
        image_path: str,
        intrinsics: CameraIntrinsics,
        camera_pos: np.ndarray,
        camera_wxyz: np.ndarray,
        output_path: str | None = None
    ) -> dict[int, TagPose]:
        """
        Detect AprilTags in the image at image_path, draw detections, and optionally save to output_path.
        Returns a dictionary of detected tag poses.
        """
        log.debug("🏷️ Detecting AprilTags in image...")
        apriltags_start_time = time.time()
        image_np = cv2.imread(str(image_path))
        gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        detections: List[apriltag.Detection] = self.detector.detect(
            gray_image_np,
            # TODO: tune these params
            estimate_tag_pose=True,
            camera_params=(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy),
            tag_size=self.config.size_m,
        )
        log.debug(f"🏷️ Detected {len(detections)} AprilTags in image")
        detections = [d for d in detections if d.decision_margin >= self.config.decision_margin]
        log.debug(f"🏷️ Filtered down to {len(detections)} detections using decision margin {self.config.decision_margin}")

        camera_so3 = jaxlie.SO3.from_wxyz(jnp.array(camera_wxyz))
        camera_transform_b = jaxlie.SE3(
            rotation=camera_so3,
            translation=jnp.array(camera_pos)
        )

        detected_tags: dict[int, TagPose] = {}
        for d in detections:
            if d.tag_id in self.config.enabled_tags:
                tag_in_cam = jnp.eye(4)
                tag_in_cam = tag_in_cam.at[:3, :3].set(jnp.array(d.pose_R))
                tag_in_cam = tag_in_cam.at[:3, 3].set(jnp.array(d.pose_t).flatten())
                tag_in_world = jnp.matmul(camera_transform_b.as_matrix(), tag_in_cam)
                pos = jnp.array(tag_in_world[:3, 3])
                wxyz = jaxlie.SO3.from_matrix(tag_in_world[:3, :3]).wxyz
                detected_tags[d.tag_id] = TagPose(pos=pos, wxyz=wxyz)
                log.debug(f"🏷️ AprilTag {d.tag_id} - {self.config.enabled_tags[d.tag_id]} - pos: {pos}, wxyz: {wxyz}")

                if output_path is not None:
                    # draw detections on image
                    corners = np.int32(d.corners)
                    cv2.polylines(image_np, [corners], isClosed=True, color=COLORS["red"], thickness=5)
                    center = tuple(np.int32(d.center))
                    cv2.circle(image_np, center, 5, COLORS["red"], -1)
                    cv2.putText(image_np, str(d.tag_id), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["red"], 2)

        apriltags_elapsed_time = time.time() - apriltags_start_time
        log.debug(f"🏷️ AprilTag detection took {apriltags_elapsed_time * 1000:.2f}ms")
        if output_path is not None:
            cv2.imwrite(str(output_path), image_np)
        return detected_tags