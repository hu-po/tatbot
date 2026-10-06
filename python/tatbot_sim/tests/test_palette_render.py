"""The rendered palette sticker is the inventory's tag, where and how the URDF turns it.

A camera straight above the palette reads the tag's id and its in-plane turn
the way the overhead cameras do. A tag rendered at the right spot but unturned
reads as a palette rotated a quarter turn.
"""

from __future__ import annotations

import json

import numpy as np
from tatbot_sim import palette as sim_palette
from tatbot_sim import tools

DICTIONARIES = {'apriltag_16h5': 'DICT_APRILTAG_16H5', 'apriltag_36h11': 'DICT_APRILTAG_36H11'}


def test_the_rendered_tag_reads_as_the_urdf_places_it():
    import cv2
    import sapien
    from tatbot_sim.env import add_palette_body
    from transforms3d.quaternions import mat2quat

    scene = sim_palette.load(tools.REPO)
    world = sapien.Scene()
    world.set_ambient_light([0.9, 0.9, 0.9])
    builder = world.create_actor_builder()
    add_palette_body(builder, scene, sapien.render.RenderMaterial(base_color=[0.05, 0.05, 0.055, 1.0]))
    builder.build_kinematic(name='palette')
    camera = world.add_camera('above', 800, 800, fovy=0.6, near=0.01, far=10)
    # SAPIEN cameras look along +x with +z up: look down -Z, image top toward +X.
    forward, up = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    camera.entity.set_pose(sapien.Pose(p=[*scene.tag_xyz_m[:2], 0.4],
                                       q=mat2quat(np.stack([forward, np.cross(up, forward), up], 1))))
    world.update_render()
    camera.take_picture()
    rgb = camera.get_picture('Color')[..., :3]
    # The picture format is process-global: a ManiSkill env built earlier in
    # this process may have switched Color from float to 8-bit.
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    inventory = json.loads((tools.REPO / scene.tag_inventory).read_text())
    target = inventory['targets']['palette']
    family = target.get('family', inventory.get('family'))
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARIES[family])))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    assert ids is not None and ids.ravel().tolist() == target['ids']
    top_left, top_right = corners[0].reshape(4, 2)[:2]
    assert np.allclose(corners[0].reshape(4, 2).mean(0), [399.5, 399.5], atol=1.5), 'centred on its frame'
    # image right is world -Y and image down is world -X, from the camera pose above
    du, dv = top_right - top_left
    seen = np.array([-dv, -du]) / np.hypot(du, dv)
    # The rendered pattern must follow the installed asset's tag yaw.
    assert np.allclose(seen, [np.cos(scene.tag_rpy[2]), np.sin(scene.tag_rpy[2])], atol=0.02), f'tag +x reads {seen} in the palette frame'
