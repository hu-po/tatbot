import time
from typing import List

import cv2
import jax.numpy as jnp
import jaxlie
import numpy as np
import pupil_apriltags as apriltag

from tatbot.data.cams import Intrinsics
from tatbot.data.pose import Pose, Rot, Pos
from tatbot.data.tags import Tags
from tatbot.utils.colors import COLORS
from tatbot.utils.log import get_logger

log = get_logger('cam.tracker', '🏷️')

class TagTracker:
    def __init__(self, config: Tags):
        self.config = config
        self.detector = apriltag.Detector(families=config.family)

    def track_tags(
        self,
        image_path: str,
        intrinsics: Intrinsics,
        camera_pos: np.ndarray,
        camera_wxyz: np.ndarray,
        output_path: str | None = None
    ) -> dict[int, Pose]:
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

        camera_transform_b = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3(camera_wxyz),
            translation=jnp.array(camera_pos)
        )

        detected_tags: dict[int, Pose] = {}
        for d in detections:
            if d.tag_id in self.config.enabled_tags:
                tag_rotation = jaxlie.SO3.from_matrix(jnp.array(d.pose_R))
                tag_translation = jnp.array(d.pose_t).flatten()
                tag_transform_cam = jaxlie.SE3.from_rotation_and_translation(tag_rotation, tag_translation)

                tag_in_world = camera_transform_b @ tag_transform_cam
                pos = tag_in_world.translation()
                wxyz = tag_in_world.rotation().wxyz
                detected_tags[d.tag_id] = Pose(pos=Pos(xyz=pos), rot=Rot(wxyz=wxyz))
                log.debug(f"🏷️ AprilTag {d.tag_id} - {self.config.enabled_tags[d.tag_id]} - pos: {pos}, wxyz: {wxyz}")

                if output_path is not None:
                    # draw detections on image
                    corners = np.int32(d.corners)
                    cv2.polylines(image_np, [corners], isClosed=True, color=COLORS["red"], thickness=8)
                    center = tuple(np.int32(d.center))
                    cv2.circle(image_np, center, 8, COLORS["red"], -1)
                    cv2.putText(image_np, str(d.tag_id), (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["red"], 2)

        apriltags_elapsed_time = time.time() - apriltags_start_time
        log.debug(f"🏷️ AprilTag detection took {apriltags_elapsed_time * 1000:.2f}ms")
        if output_path is not None:
            log.debug(f"🏷️ Saving image with detections to {output_path}")
            cv2.imwrite(str(output_path), image_np)
        return detected_tags