import functools
from typing import Callable

import numpy as np

from tatbot.data.inks import InkCap
from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke
from tatbot.utils.log import get_logger

log = get_logger("gen.inkdip", "💧")


def make_inkdip_func(scene: Scene) -> Callable:
    """Returns a function that generates inkdip strokes given an ink name and arm."""

    @functools.lru_cache(maxsize=len(scene.inks.inkcaps))
    def inkdip_func(color: str, arm: str) -> Stroke:
        """Get <x, y, z> coordinates for an inkdip into a specific inkcap."""
        if arm == "left":
            inkcap: InkCap = scene.inkcaps_l[color]
        else:
            inkcap: InkCap = scene.inkcaps_r[color]
        # Split: 1/3 down, 1/3 wait, 1/3 up (adjust as needed)
        num_down = scene.stroke_length // 3
        num_up = scene.stroke_length // 3
        num_wait = scene.stroke_length - num_down - num_up
        depth_m = (
            inkcap.depth_m / 3
        )  # 1/3 depth # TODO: make this a function of the session duration? (stroke idx)
        # dip down to inkcap depth
        down_z = np.linspace(0, depth_m, num_down, endpoint=False)
        # wait at depth
        wait_z = np.full(num_wait, depth_m)
        # retract back up
        up_z = np.linspace(depth_m, 0, num_up, endpoint=True)
        # concatenate into final inkdip position array
        inkdip_pos = inkcap.pose.pos.xyz + np.hstack(
            [
                np.zeros((scene.stroke_length, 2)),  # x and y are 0
                -np.concatenate([down_z, wait_z, up_z]).reshape(-1, 1),
            ]
        )
        if arm == "left":
            inkdip_pos += scene.ee_offset_l.xyz
        else:
            inkdip_pos += scene.ee_offset_r.xyz
        return Stroke(
            description=f"{arm} arm inkdip into {inkcap.name} to fill with {color} ink",
            is_inkdip=True,
            inkcap=inkcap.name,
            color=color,
            ee_pos=inkdip_pos,
            arm=arm,
        )

    return inkdip_func
