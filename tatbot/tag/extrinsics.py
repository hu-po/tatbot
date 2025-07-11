import os
from collections import defaultdict
from dataclasses import replace

import jax.numpy as jnp
import jaxlie
import numpy as np

from tatbot.data.pose import Pose, Pos, Rot
from tatbot.data.cams import Cams
from tatbot.data.tags import Tags
from tatbot.tag.tracker import TagTracker
from tatbot.utils.log import get_logger

log = get_logger('tag.extrinsics', '🔎')

def get_extrinsics(
    image_paths: list[str],
    cams: Cams,
    tags: Tags,
    max_iter: int = 20,
    epsilon: float = 1e-5,
) -> Cams:
    log.info("Calculating camera extrinsics...")
    log.debug(f"image_paths: {image_paths}")
    log.debug(f"cams: {cams}")
    log.debug(f"tags: {tags}")

    detected_tags: dict[str, dict[int, Pose]] = {}

    log.info("Tracking tags in images...")
    tracker = TagTracker(tags)
    for image_path in image_paths:
        camera_name = image_path.split('/')[-1].split('.')[0]
        log.info(f"Tracking tags in {image_path} for {camera_name}")
        camera_config = cams.get_camera(camera_name)
        # Create output path for annotated image
        output_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
        
        _detected_tags = tracker.track_tags(
            image_path,
            camera_config.intrinsics,
            camera_config.extrinsics.pos.xyz,
            camera_config.extrinsics.rot.wxyz,
            output_path=output_path,
        )
        detected_tags[camera_name] = _detected_tags

    # First, compute observed T_cam_tag for each detection using initial extrinsics
    observed_tag_cam: dict[str, dict[int, jaxlie.SE3]] = {}
    current_extrinsics: dict[str, jaxlie.SE3] = {}
    for camera_name in detected_tags:
        camera_config = cams.get_camera(camera_name)
        cam_ex = camera_config.extrinsics
        camera_pos = jnp.array(cam_ex.pos.xyz)
        camera_wxyz = jnp.array(cam_ex.rot.wxyz)
        T_world_cam = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(camera_wxyz),
            camera_pos
        )
        current_extrinsics[camera_name] = T_world_cam

        observed_tag_cam[camera_name] = {}
        for tag_id, world_pose in detected_tags[camera_name].items():
            tag_pos = jnp.array(world_pose.pos.xyz)
            tag_wxyz = jnp.array(world_pose.rot.wxyz)
            T_world_tag = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(tag_wxyz),
                tag_pos
            )
            T_cam_tag = T_world_cam.inverse() @ T_world_tag
            observed_tag_cam[camera_name][tag_id] = T_cam_tag

    # Optimization loop
    for it in range(max_iter):
        all_tag_world_estimates = defaultdict(list)
        for camera_name in detected_tags:
            T_world_cam = current_extrinsics[camera_name]
            for tag_id, T_cam_tag in observed_tag_cam[camera_name].items():
                T_world_tag = T_world_cam @ T_cam_tag
                all_tag_world_estimates[tag_id].append(T_world_tag)

        avg_tag_world = {}
        for tag_id, ests in all_tag_world_estimates.items():
            poss = jnp.stack([e.translation() for e in ests])
            quats = jnp.stack([e.rotation().wxyz for e in ests])
            avg_p = jnp.mean(poss, axis=0)
            avg_q = jnp.mean(quats, axis=0)
            avg_q = avg_q / jnp.linalg.norm(avg_q)
            avg_tag_world[tag_id] = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(avg_q), avg_p
            )

        new_extrinsics = {}
        max_delta = 0.0
        for camera_name in detected_tags:
            candidates = []
            for tag_id, T_cam_tag in observed_tag_cam[camera_name].items():
                if tag_id in avg_tag_world:
                    T_world_tag_avg = avg_tag_world[tag_id]
                    T_world_cam_cand = T_world_tag_avg @ T_cam_tag.inverse()
                    candidates.append(T_world_cam_cand)

            if candidates:
                poss = jnp.stack([c.translation() for c in candidates])
                quats = jnp.stack([c.rotation().wxyz for c in candidates])
                avg_p = jnp.mean(poss, axis=0)
                avg_q = jnp.mean(quats, axis=0)
                avg_q = avg_q / jnp.linalg.norm(avg_q)
                new_T = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(avg_q), avg_p
                )
                delta = jnp.max(jnp.abs(new_T.translation() - current_extrinsics[camera_name].translation()))
                if delta > max_delta:
                    max_delta = delta
                new_extrinsics[camera_name] = new_T
            else:
                new_extrinsics[camera_name] = current_extrinsics[camera_name]

        # Anchor to reference camera (set to identity)
        ref_cam = 'realsense1'
        if ref_cam in new_extrinsics:
            T_ref = new_extrinsics[ref_cam]
            T_correct = T_ref.inverse()
            for cam in new_extrinsics:
                new_extrinsics[cam] = T_correct @ new_extrinsics[cam]

        current_extrinsics = new_extrinsics
        log.info(f"Iteration {it + 1}/{max_iter}: max position delta = {max_delta:.6f}m")
        if max_delta < epsilon:
            log.info(f"Converged after {it + 1} iterations")
            break

    updated_cams = replace(cams)
    # populate new camera extrinsics
    for camera_name in current_extrinsics:
        T = current_extrinsics[camera_name]
        new_pos = np.array(T.translation())
        new_wxyz = np.array(T.rotation().wxyz)
        new_extrinsics = Pose(pos=Pos(xyz=new_pos), rot=Rot(wxyz=new_wxyz))
        camera_config = updated_cams.get_camera(camera_name)
        camera_config.extrinsics = new_extrinsics
        updated_cams.set_camera(camera_name, camera_config)

    log.info("✅ Done")
    return updated_cams