import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])
    left_rest_stroke = Stroke(
        description="left arm at rest",
        arm="left",
        meter_coords=np.zeros((scene.stroke_length, 3)),
        is_rest=True,
    )
    right_rest_stroke = Stroke(
        description="right arm at rest",
        arm="right",
        meter_coords=np.zeros((scene.stroke_length, 3)),
        is_rest=True,
    )

    # First hover arms over the calibrator
    strokelist.strokes.append((
        Stroke(
            description="left arm hovering over calibrator",
            arm="left",
            ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        ),
        right_rest_stroke,
    ))
    strokelist.strokes.append((
        left_rest_stroke,
        Stroke(
            description="right arm hovering over calibrator",
            arm="right",
            ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        ),
    ))

    # Then hover arms over each of the inkcaps
    for inkcap in scene.inkcaps_r.values():
        strokelist.strokes.append((left_rest_stroke, inkdip_func(inkcap.ink.name, "right")))
    for inkcap in scene.inkcaps_l.values():
        strokelist.strokes.append((inkdip_func(inkcap.ink.name, "left"), right_rest_stroke))
    
    # Finally hover over the "alignment X", which is a red cross centered on the design pose
    strokelist.strokes.append((
        Stroke(
            description=f"left arm hovering over design pose, -{scene.arms.align_x_size_m}cm in X axis",
            arm="left",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([-scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
        ), 
        Stroke(
            description=f"right arm hovering over design pose, +{scene.arms.align_x_size_m}cm in X axis",
            arm="right",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([scene.arms.align_x_size_m, 0, 0]), (scene.stroke_length, 1)),
        ),
    ))
    strokelist.strokes.append((
        Stroke(
            description=f"left arm hovering over design pose, -{scene.arms.align_x_size_m}cm in Y axis",
            arm="left",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, -scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
        ),
        Stroke(
            description=f"right arm hovering over design pose, +{scene.arms.align_x_size_m}cm in Y axis",
            arm="right",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
        ),
    ))
    return strokelist
