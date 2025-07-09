import numpy as np

from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger
from tatbot.data.scene import Scene

log = get_logger('gen.align', '🖋️')

def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)

    # left arm and right arm strokes in order of execution on robot
    strokelist: StrokeList = StrokeList(strokes=[])

    # default time between poses is fast movement
    dt = np.full((scene.stroke_length, 1), scene.arms.goal_time_fast)
    # slow movement to and from hover positions
    dt[:2] = scene.arms.goal_time_slow
    dt[-2:] = scene.arms.goal_time_slow

    # hardcoded orientations for left and right arm end effectors
    ee_rot_l = np.tile(scene.ee_rot_l.wxyz, (scene.stroke_length, 1))
    ee_rot_r = np.tile(scene.ee_rot_r.wxyz, (scene.stroke_length, 1))

    for inkcap in scene.inkcaps_r.values():
        _inkdip_stroke: Stroke = inkdip_func(inkcap.ink.name, "right")
        _inkdip_stroke.ee_rot = ee_rot_r
        _inkdip_stroke.dt = dt
        strokelist.strokes.append(
            (
                Stroke(
                    description="left arm hovering over design",
                    arm="left",
                    ee_pos=np.tile(scene.needle_hover_offset.xyz, (scene.stroke_length, 1)),
                    ee_rot=ee_rot_l,
                    dt=dt,
                ),
                _inkdip_stroke,
            )
        )

    for inkcap in scene.inkcaps_l.values():
        _inkdip_stroke: Stroke = inkdip_func(inkcap.ink.name, "left")
        _inkdip_stroke.ee_rot = ee_rot_l
        _inkdip_stroke.dt = dt
        strokelist.strokes.append(
            (
                _inkdip_stroke,
                Stroke(
                    description="right arm hovering over design",
                    arm="right",
                    ee_pos=np.tile(scene.needle_hover_offset.xyz, (scene.stroke_length, 1)),
                    ee_rot=ee_rot_r,
                    dt=dt,
                ),
            )
        )

    return strokelist