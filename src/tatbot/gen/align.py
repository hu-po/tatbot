import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])
    left_origin_widget_stroke = Stroke(
        description="left arm hovering over (left) origin widget",
        arm="left",
        ee_pos=np.tile(scene.origin_widget_l_pos.xyz, (scene.stroke_length, 1)),
        is_inkdip=True, # inkdip strokes are in final ee_pos
    )
    right_origin_widget_stroke = Stroke(
        description="right arm hovering over (right) origin widget",
        arm="right",
        ee_pos=np.tile(scene.origin_widget_r_pos.xyz, (scene.stroke_length, 1)),
        is_inkdip=True, # inkdip strokes are in final ee_pos
    )
    strokelist.strokes.append((left_origin_widget_stroke, right_origin_widget_stroke))
    for inkcap in scene.inkcaps_r.values():
        strokelist.strokes.append((left_origin_widget_stroke, inkdip_func(inkcap.ink.name, "right")))
    for inkcap in scene.inkcaps_l.values():
        strokelist.strokes.append((inkdip_func(inkcap.ink.name, "left"), right_origin_widget_stroke))
    left_design_stroke_xaxis = Stroke(
        description="left arm hovering over design pose, +1cm in X axis",
        arm="left",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
    )
    right_design_stroke_xaxis = Stroke(
        description="right arm hovering over design pose, -1cm in X axis",
        arm="right",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([-scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
    )
    left_design_stroke_yaxis = Stroke(
        description="left arm hovering over design pose, +1cm in Y axis",
        arm="left",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    )
    right_design_stroke_yaxis = Stroke(
        description="right arm hovering over design pose, -1cm in Y axis",
        arm="right",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, -scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    )
    strokelist.strokes.append((left_design_stroke_xaxis, right_design_stroke_xaxis))
    strokelist.strokes.append((left_design_stroke_yaxis, right_design_stroke_yaxis))
    strokelist.strokes.append((left_design_stroke_xaxis, right_design_stroke_xaxis))
    strokelist.strokes.append((left_design_stroke_yaxis, right_design_stroke_yaxis))
    return strokelist
