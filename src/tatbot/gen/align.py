import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])
    strokelist.strokes.append(
        (
            Stroke(
                description="left arm hovering over (left) origin widget",
                arm="left",
                meter_coords=np.tile(scene.origin_widget_l_pos.xyz, (scene.stroke_length, 1)),
            ),
            Stroke(
                description="right arm hovering over (right) origin widget",
                arm="right",
                meter_coords=np.tile(scene.origin_widget_r_pos.xyz, (scene.stroke_length, 1)),
            ),
        )
    )
    for inkcap in scene.inkcaps_r.values():
        _inkdip_stroke: Stroke = inkdip_func(inkcap.ink.name, "right")
        strokelist.strokes.append(
            (
                Stroke(
                    description="left arm hovering over design",
                    arm="left",
                    meter_coords=np.tile(scene.arms.hover_offset.xyz, (scene.stroke_length, 1)),
                ),
                _inkdip_stroke,
            )
        )
    for inkcap in scene.inkcaps_l.values():
        _inkdip_stroke: Stroke = inkdip_func(inkcap.ink.name, "left")
        strokelist.strokes.append(
            (
                _inkdip_stroke,
                Stroke(
                    description="right arm hovering over design",
                    arm="right",
                    meter_coords=np.tile(scene.arms.hover_offset.xyz, (scene.stroke_length, 1)),
                ),
            )
        )
    return strokelist
