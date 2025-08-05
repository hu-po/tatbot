import numpy as np

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger

log = get_logger("gen.align", "📐")


def make_align_strokes(scene: Scene) -> StrokeList:
    inkdip_func = make_inkdip_func(scene)
    strokelist: StrokeList = StrokeList(strokes=[])

    # First hover arms over the calibrator, alternate arm hovers over large inkcap
    strokelist.strokes.append((
        Stroke(
            description="left arm hovering over calibrator",
            arm="left",
            ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        ),
        inkdip_func(list(scene.inkcaps_r.values())[0].ink.name, "right"),
    ))
    strokelist.strokes.append((
        inkdip_func(list(scene.inkcaps_l.values())[0].ink.name, "left"),
        Stroke(
            description="right arm hovering over calibrator",
            arm="right",
            ee_pos=np.tile(scene.calibrator_pos.xyz, (scene.stroke_length, 1)),
            is_inkdip=True, # inkdip strokes are in final ee_pos
        ),
    ))

    # Then hover arms over each of the inkcaps
    for inkcap_l, inkcap_r in zip(scene.inkcaps_l.values(), scene.inkcaps_r.values(), strict=False):
        strokelist.strokes.append((inkdip_func(inkcap_l.ink.name, "left"), inkdip_func(inkcap_r.ink.name, "right")))
    
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
    # TODO: the arms collide when doing this...
    # strokelist.strokes.append((
    #     Stroke(
    #         description=f"left arm hovering over design pose, -{scene.arms.align_x_size_m}cm in Y axis",
    #         arm="left",
    #         meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, -scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    #     ),
    #     Stroke(
    #         description=f"right arm hovering over design pose, +{scene.arms.align_x_size_m}cm in Y axis",
    #         arm="right",
    #         meter_coords=np.tile(scene.arms.hover_offset.xyz + np.array([0, scene.arms.align_x_size_m, 0]), (scene.stroke_length, 1)),
    #     ),
    # ))
    return strokelist
