import collections
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import svgpathtools
from PIL import Image

from tatbot.data.plan import Plan
from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.ik import transform_and_offset
from tatbot.gen.strokebatch import strokebatch_from_strokes
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.data.scene import Scene

log = get_logger('gen.svg', '🖋️')

@dataclass
class GenCalibPlanConfig():
    debug: bool = False
    """Enable debug logging."""

    plan_name: str = "default"
    """Name of the plan (Plan)."""
    scene_name: str = "default"
    """Name of the scene config (Scene)."""

    output_dir: str = "~/tatbot/nfs/plans"
    """Directory to save the plan."""


def gen_calib_plan(config: GenCalibPlanConfig):
    log.info(f"Generating {config.name} ...")

    output_dir = os.path.expanduser(config.output_dir)
    output_dir = os.path.join(output_dir, config.name)
    log.info(f"📂 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    plan: Plan = Plan.from_name(config.plan_name)
    scene: Scene = Scene.from_name(config.scene_name)
    inkdip_func, inks_color_to_inkcap_name, inks_color_to_inkcap_pose = make_inkdip_func(scene, plan)

    # left arm and right arm strokes in order of execution on robot
    strokelist: StrokeList = StrokeList(strokes=[])

    # default time between poses is fast movement
    dt = np.full((plan.stroke_length, 1), scene.arms.goal_time_fast)
    # slow movement to and from hover positions
    dt[:2] = scene.arms.goal_time_slow
    dt[-2:] = scene.arms.goal_time_slow

    # hardcoded orientations for left and right arm end effectors
    ee_rot_l = np.tile(plan.ee_rot_l.wxyz, (plan.stroke_length, 1))
    ee_rot_r = np.tile(plan.ee_rot_r.wxyz, (plan.stroke_length, 1))

    # start with "alignment" strokes
    alignment_inkcap_color_r = pen_paths_r[0][0]
    alignment_inkcap_color_l = pen_paths_l[0][0]
    alignment_inkcap_pose_r: Pose = inks_color_to_inkcap_pose[alignment_inkcap_color_r]
    alignment_inkcap_pose_l: Pose = inks_color_to_inkcap_pose[alignment_inkcap_color_l]
    strokelist.strokes.append(
        (
            Stroke(
                description="left arm over design",
                ee_pos=transform_and_offset(
                    np.zeros((plan.stroke_length, 3)),
                    scene.skin.design_pose.pos.xyz,
                    scene.skin.design_pose.rot.wxyz,
                    plan.needle_hover_offset.xyz,
                ),
                ee_rot=ee_rot_l,
                dt=dt,
                is_alignment=True,
                arm="left",
            ),
            Stroke(
                description=f"right arm over {alignment_inkcap_color_r} inkcap",
                ee_pos=transform_and_offset(
                    np.zeros((plan.stroke_length, 3)),
                    alignment_inkcap_pose_r.pos.xyz,
                    alignment_inkcap_pose_r.rot.wxyz,
                    plan.needle_hover_offset.xyz,
                ),
                ee_rot=ee_rot_r,
                dt=dt,
                is_alignment=True,
                arm="right",
            ),
        )
    )
    # same but switch the arms
    strokelist.strokes.append(
        (
            Stroke(
                description=f"left arm over {alignment_inkcap_color_l} inkcap",
                ee_pos=transform_and_offset(
                    np.zeros((plan.stroke_length, 3)),
                    alignment_inkcap_pose_l.pos.xyz,
                    alignment_inkcap_pose_l.rot.wxyz,
                    plan.needle_hover_offset.xyz,
                ),
                ee_rot=ee_rot_l,
                dt=dt,
                is_alignment=True,
                arm="left",
            ),
            Stroke(
                description="right arm over design",
                ee_pos=transform_and_offset(
                    np.zeros((plan.stroke_length, 3)),
                    scene.skin.design_pose.pos.xyz,
                    scene.skin.design_pose.rot.wxyz,
                    plan.needle_hover_offset.xyz,
                ),
                ee_rot=ee_rot_r,
                dt=dt,
                is_alignment=True,
                arm="right",
            ),
        )
    )

    strokes_path = os.path.join(output_dir, "strokes.yaml")
    log.info(f"💾 Saving strokes to {strokes_path}")
    strokelist.to_yaml(strokes_path)

    strokebatch: StrokeBatch = strokebatch_from_strokes(
        strokelist=strokelist,
        stroke_length=plan.stroke_length,
        batch_size=plan.ik_batch_size,
        joints=scene.sleep_pos_full,
        urdf_path=scene.urdf.path,
        link_names=scene.urdf.ee_link_names,
        design_pose=scene.skin.design_pose,
        needle_hover_offset=plan.needle_hover_offset,
        needle_offset_l=plan.needle_offset_l,
        needle_offset_r=plan.needle_offset_r,
    )
    strokebatch_path = os.path.join(output_dir, f"strokebatch.safetensors")
    log.info(f"💾 Saving strokebatch to {strokebatch_path}")
    strokebatch.save(strokebatch_path)

    # copy the plan yaml to the output directory
    plan.name = config.name # override the plan name
    plan_path = os.path.join(output_dir, "plan.yaml")
    log.info(f"💾 Saving plan yaml to {plan_path}")
    plan.to_yaml(plan_path)

if __name__ == "__main__":
    args = setup_log_with_config(GenCalibPlanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    gen_calib_plan(args)