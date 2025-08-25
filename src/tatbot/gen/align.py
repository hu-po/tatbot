import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])

    # Create rest strokes, which will be used on alternate arms
    rest_stroke_l = Stroke(
        description="left arm at rest",
        meter_coords=np.zeros((scene.stroke_length, 3)),
        arm="left",
        is_rest=True,
    )
    rest_stroke_r = Stroke(
        description="right arm at rest",
        meter_coords=np.zeros((scene.stroke_length, 3)),
        arm="right",
        is_rest=True,
    )

    # First hover arms over the calibrator
    strokelist.strokes.append((
        Stroke(
            description="left arm hovering over calibrator",
            arm="left",
            ee_pos=np.tile(scene.calibrator_pose.pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        ),
        rest_stroke_r
    ))
    strokelist.strokes.append((
        rest_stroke_l,
        Stroke(
            description="right arm hovering over calibrator",
            arm="right",
            ee_pos=np.tile(scene.calibrator_pose.pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        )
    ))

    # Then hover arms over each of the inkcaps
    for inkcap_l in scene.inkcaps_l.values():
        strokelist.strokes.append((inkdip_func(inkcap_l.ink.name, "left"), rest_stroke_r))
    for inkcap_r in scene.inkcaps_r.values():
        strokelist.strokes.append((rest_stroke_l, inkdip_func(inkcap_r.ink.name, "right")))        
    
    # Finally hover over the lasercross: a red cross which acts as the origin of the tattoo design
    lasercross_halflen_m = scene.arms.lasercross_len_mm / 2000 # convert to meters and divide by 2
    strokelist.strokes.append((
        Stroke(
            description=f"left arm hovering over lasercross pose, -{lasercross_halflen_m}m in X axis",
            arm="left",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([-lasercross_halflen_m, 0, 0]), (scene.stroke_length, 1)),
        ), 
        Stroke(
            description=f"right arm hovering over lasercross pose, +{lasercross_halflen_m}m in X axis",
            arm="right",
            meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([lasercross_halflen_m, 0, 0]), (scene.stroke_length, 1)),
        ),
    ))
    return strokelist
