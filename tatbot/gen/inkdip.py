import functools
from typing import Callable

import numpy as np

from tatbot.bot.urdf import get_link_poses
from tatbot.data.plan import Plan
from tatbot.data.pose import Pose
from tatbot.gen.ik import transform_and_offset
from tatbot.utils.log import get_logger
from tatbot.data.scene import Scene
from tatbot.data.inks import InkCap

log = get_logger('gen.inkdip', '💧')

def make_inkdip_func(scene: Scene, plan: Plan) -> Callable:
    """ Returns a function that can be used to generate inkdip positions for a given color. """
    
    inks_color_to_inkcap_name: dict[str, str] = {}
    inks_color_to_inkcap_pose: dict[str, Pose] = {}
    filled_inkcap_names: list[str] = []
    link_poses = get_link_poses(scene.urdf.path, scene.urdf.ink_link_names, scene.home_pos_full)
    for inkcap in scene.inks.inkcaps:
        assert inkcap.name in scene.urdf.ink_link_names, f"❌ Inkcap {inkcap.name} not found in URDF"
        if inkcap.ink is not None:
            filled_inkcap_names.append(inkcap.name)
            ink_color: str = inkcap.ink["name"]
            inks_color_to_inkcap_name[ink_color] = inkcap.name
            inks_color_to_inkcap_pose[ink_color] = link_poses[inkcap.name]
            log.debug(f"Inkcap {inkcap.name} is filled with {ink_color}")
        else:
            log.debug(f"Inkcap {inkcap.name} is empty")
    log.info(f"✅ Found {len(filled_inkcap_names)} filled inkcaps in {scene.inks.yaml_dir}")
    log.debug(f"Filled inkcaps in scene: {filled_inkcap_names}")
    
    @functools.lru_cache(maxsize=len(scene.inks.inkcaps))
    def inkdip_func(color: str, num_points: int = plan.stroke_length) -> np.ndarray:
        """Get <x, y, z> coordinates for an inkdip into a specific inkcap."""
        inkcap_pose: Pose = inks_color_to_inkcap_pose[color]
        # TODO: better selection algorithm for inkcap
        inkcap: InkCap = None
        for inkcap in scene.inks.inkcaps:
            if inkcap.ink is not None and inkcap.ink["name"] == color:
                inkcap = inkcap
                break
        if inkcap is None:
            raise ValueError(f"❌ No inkcap found for color {color}")
        # Split: 1/3 down, 1/3 wait, 1/3 up (adjust as needed)
        num_down = num_points // 3
        num_up = num_points // 3
        num_wait = num_points - num_down - num_up
        # dip down to inkcap depth
        down_z = np.linspace(0, inkcap.depth_m, num_down, endpoint=False)
        # wait at depth
        wait_z = np.full(num_wait, inkcap.depth_m)
        # retract back up
        up_z = np.linspace(inkcap.depth_m, 0, num_up, endpoint=True)
        # concatenate into offset array
        offsets = np.hstack([
            np.zeros((num_points, 2)), # x and y are 0
            -np.concatenate([down_z, wait_z, up_z]).reshape(-1, 1),
        ])
        offsets = offsets + plan.inkdip_hover_offset.xyz
        inkdip_pos = transform_and_offset(
            np.zeros((num_points, 3)), # <x, y, z>
            inkcap_pose.pos.xyz,
            inkcap_pose.rot.wxyz,
            offsets,
        )
        return inkdip_pos
    
    return inkdip_func, inks_color_to_inkcap_name, inks_color_to_inkcap_pose