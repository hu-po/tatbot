import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])
    left_calibrator_stroke = Stroke(
        description="left arm hovering over calibrator",
        arm="left",
        ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
        is_inkdip=True, # inkdip strokes are in final ee_pos
    )
    right_calibrator_stroke = Stroke(
        description="right arm hovering over calibrator",
        arm="right",
        ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
        is_inkdip=True, # inkdip strokes are in final ee_pos
    )
    for inkcap in scene.inkcaps_r.values():
        strokelist.strokes.append((left_calibrator_stroke, inkdip_func(inkcap.ink.name, "right")))
    for inkcap in scene.inkcaps_l.values():
        strokelist.strokes.append((inkdip_func(inkcap.ink.name, "left"), right_calibrator_stroke))
    left_design_stroke_xaxis = Stroke(
        description=f"left arm hovering over design pose, -{scene.arms.align_x_size_m}cm in X axis",
        arm="left",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([-scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
    )
    right_design_stroke_xaxis = Stroke(
        description=f"right arm hovering over design pose, +{scene.arms.align_x_size_m}cm in X axis",
        arm="right",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
    )
    left_design_stroke_yaxis = Stroke(
        description=f"left arm hovering over design pose, -{scene.arms.align_x_size_m}cm in Y axis",
        arm="left",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, -scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    )
    right_design_stroke_yaxis = Stroke(
        description=f"right arm hovering over design pose, +{scene.arms.align_x_size_m}cm in Y axis",
        arm="right",
        meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    )
    strokelist.strokes.append((left_design_stroke_xaxis, right_design_stroke_xaxis))
    strokelist.strokes.append((left_design_stroke_yaxis, right_design_stroke_yaxis))
    return strokelist
