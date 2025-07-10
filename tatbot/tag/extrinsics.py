import glob
import os
import time

from tatbot.data.pose import Pose
from tatbot.data.cams import Cams
from tatbot.data.tags import Tags
from tatbot.tag.tracker import TagTracker
from tatbot.utils.log import get_logger

log = get_logger('tag.extrinsics', '🔎')

def get_extrinsics(
    image_paths: list[str],
    cams: Cams,
    tags: Tags,
) -> Cams:
    log.info("Calculating camera extrinsics...")
    log.debug(f"image_paths: {image_paths}")
    log.debug(f"cams: {cams}")
    log.debug(f"tags: {tags}")

    detected_tags: dict[int, Pose] = {}

    log.info("Tracking tags in images...")
    tracker = TagTracker(tags)
    for image_path in image_paths:
        camera_name = image_path.split('/')[-1].split('.')[0]
        log.info(f"Tracking tags in {image_path} for {camera_name}")
        # TODO: get camera_pos and camera_wxyz from URDF? initialize as identity?
        if "realsense" in camera_name:
            camera_pos = cams.realsenses[camera_name].extrinsics.pos
            camera_wxyz = cams.realsenses[camera_name].extrinsics.wxyz
            intrinsics = cams.realsenses[camera_name].intrinsics
        else:
            camera_pos = cams.ipcameras[camera_name].extrinsics.pos
            camera_wxyz = cams.ipcameras[camera_name].extrinsics.wxyz
            intrinsics = cams.ipcameras[camera_name].intrinsics

        _detected_tags = tracker.track_tags(
            image_path,
            intrinsics,
            camera_pos,
            camera_wxyz,
            output_path=os.path.dirname(image_path),
        )
        # TODO: don't overwrite existing detections
        detected_tags.update(_detected_tags)

    # TODO: assume tags are in the same position for all cameras, and current camera extrinsics are just a guess
    # TODO: do some kind of averaging/optimization to converge towards the correct extrinsics
    
    updated_cams = cams.copy()
    # TODO: populate new camera extrinsics

    log.info("✅ Done")
    return updated_cams