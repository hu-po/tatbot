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
class GenSVGPlanConfig():
    debug: bool = False
    """Enable debug logging."""

    name: str = "calib"
    """Name of the SVG file"""
    design_dir: str = "~/tatbot/nfs/designs"
    """Directory containing the design svg (per pen) and png file."""
    output_dir: str = "~/tatbot/nfs/plans"
    """Directory to save the plan."""

    pens_config_path: str = "~/tatbot/config/drawingbotv3/pens/full.json"
    """Path to the DrawingBotV3 Pens config file."""
    
    plan_name: str = "calib"
    """Name of the plan (Plan)."""
    scene_name: str = "default"
    """Name of the scene config (Scene)."""

def gen_svg_plan(config: GenSVGPlanConfig):
    log.info(f"Generating {config.name} ...")
    
    design_dir = os.path.expanduser(config.design_dir)
    design_dir = os.path.join(design_dir, config.name)
    assert os.path.exists(design_dir), f"❌ Design directory {design_dir} does not exist"
    log.debug(f"📂 Design directory: {design_dir}")

    output_dir = os.path.expanduser(config.output_dir)
    output_dir = os.path.join(output_dir, config.name)
    log.info(f"📂 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    plan: Plan = Plan.from_name(config.plan_name)
    scene: Scene = Scene.from_name(config.scene_name)
    inkdip_func, inks_color_to_inkcap_name, inks_color_to_inkcap_pose = make_inkdip_func(scene, plan)

    svg_files = []
    pens: dict[str, str] = {}
    for file in os.listdir(design_dir):
        if file.endswith('.svg'):
            svg_path = os.path.join(design_dir, file)
            svg_files.append(svg_path)
            match = re.match(r".*_pen\d+_(\w+)\.svg$", file)
            if not match:
                raise ValueError(f"❌ Could not extract pen name from filename: {file}")
            pen_name = match.group(1)
            pens[pen_name] = svg_path
    log.info(f"✅ Found {len(pens)} pens in {design_dir}")
    log.debug(f"Pens in design: {pens.keys()}")

    pens_config_path = os.path.expanduser(config.pens_config_path)
    assert os.path.exists(pens_config_path), f"❌ Pens config file {pens_config_path} does not exist"
    log.info(f"📂 Loading pens from config file: {pens_config_path}")
    with open(pens_config_path, 'r') as f:
        pens_config = json.load(f)
    config_pens = {pen["name"]: pen for pen in pens_config["data"]["pens"]}
    log.info(f"✅ Found {len(config_pens)} pens in {config.pens_config_path}")
    log.debug(f"Pens in config: {config_pens.keys()}")

    # create directory for image files
    frames_dir = os.path.join(output_dir, "frames")
    log.info(f"🗃️ Creating frames directory at {frames_dir}")
    os.makedirs(frames_dir, exist_ok=True)

    # Copy the final design image to frames/full.png
    final_design_img = None
    for file in os.listdir(design_dir):
        if file.endswith('.png') and '_F' not in file:
            final_design_img = file
            break
    if final_design_img is not None:
        shutil.copy(os.path.join(design_dir, final_design_img), os.path.join(frames_dir, 'full.png'))
        log.info(f"Copied final design image {final_design_img} to frames/full.png")
    else:
        log.warning(f"No final design image found in {design_dir}")
    final_design_img = Image.open(os.path.join(design_dir, final_design_img))
    plan.image_width_px = final_design_img.width
    plan.image_height_px = final_design_img.height
    log.info(f"Design image is {plan.image_width_px}x{plan.image_height_px}")

    # --- Build mapping from (arm, pen, stroke index) to original PNG filename ---
    # Example filename: a6f9q4thkhrm80cqrv9rebavmc_plotted_1_set1_pen3_true_blue_F000001.png
    stroke_img_map = collections.defaultdict(list)  # (pen_name) -> list of (frame_num, filename)
    for file in os.listdir(design_dir):
        m = re.match(r".*_pen\d+_([a-zA-Z0-9_]+)_F(\d+)\.png$", file)
        if m:
            pen_name = m.group(1)
            frame_num = int(m.group(2))
            stroke_img_map[pen_name].append((frame_num, file))
    if stroke_img_map == {}:
        raise ValueError(f"❌ No stroke images found in {design_dir}, did you export intermediate frames?")
    # Sort by frame number for each pen
    for pen in stroke_img_map:
        stroke_img_map[pen].sort()

    pen_paths_l: list[tuple[str, svgpathtools.Path]] = []
    pen_paths_r: list[tuple[str, svgpathtools.Path]] = []
    for pen_name, svg_path in pens.items():
        assert pen_name in config_pens, f"❌ Pen {pen_name} not found in pens config"
        assert config_pens[pen_name]["name"] == pen_name, f"❌ Pen {pen_name} not found in pens config"
        assert pen_name in inks_color_to_inkcap_name, f"❌ Pen {pen_name} not found in ink palette"
        log.info(f"Processing svg file at: {svg_path}")
        paths, _, _ = svgpathtools.svg2paths2(svg_path)
        log.info(f"Found {len(paths)} paths")
        for path in paths:
            if path.length() == 0:
                log.warning(f"❌ Path {path} is empty, skipping")
                continue
            if pen_name in plan.left_arm_pen_names:
                pen_paths_l.append((pen_name, path))
            if pen_name in plan.right_arm_pen_names:
                pen_paths_r.append((pen_name, path))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        log.error("No paths found for left or right arm")

    def coords_from_path(path: svgpathtools.Path) -> tuple[np.ndarray, np.ndarray]:
        """Resample path evenly along the path and convert to pixel and meter coordinates."""
        total_length = path.length()
        distances = np.linspace(0, total_length, plan.stroke_length)
        points = [path.point(path.ilength(d)) for d in distances]
        pixel_coords = np.array([[p.real, p.imag] for p in points])
        meter_coords = np.hstack([(
            pixel_coords * np.array([ # convert pixel coordinates to meter coordinates
                plan.image_width_m / plan.image_width_px,
                plan.image_height_m / plan.image_height_px,
            ], dtype=np.float32)
            - np.array([ # center the meter coordinates in the image
                plan.image_width_m / 2,
                plan.image_height_m / 2,
            ], dtype=np.float32)
        ), np.zeros((plan.stroke_length, 1), dtype=np.float32)]) # z axis is 0
        return pixel_coords, meter_coords

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

    # # start with "alignment" strokes
    # alignment_inkcap_color_r = pen_paths_r[0][0]
    # alignment_inkcap_color_l = pen_paths_l[0][0]
    # alignment_inkcap_pose_r: Pose = inks_color_to_inkcap_pose[alignment_inkcap_color_r]
    # alignment_inkcap_pose_l: Pose = inks_color_to_inkcap_pose[alignment_inkcap_color_l]
    # strokelist.strokes.append(
    #     (
    #         Stroke(
    #             description="left arm over design",
    #             ee_pos=transform_and_offset(
    #                 np.zeros((plan.stroke_length, 3)),
    #                 scene.skin.design_pose.pos.xyz,
    #                 scene.skin.design_pose.rot.wxyz,
    #                 plan.needle_hover_offset.xyz,
    #             ),
    #             ee_rot=ee_rot_l,
    #             dt=dt,
    #             is_alignment=True,
    #             arm="left",
    #         ),
    #         Stroke(
    #             description=f"right arm over {alignment_inkcap_color_r} inkcap",
    #             ee_pos=transform_and_offset(
    #                 np.zeros((plan.stroke_length, 3)),
    #                 alignment_inkcap_pose_r.pos.xyz,
    #                 alignment_inkcap_pose_r.rot.wxyz,
    #                 plan.needle_hover_offset.xyz,
    #             ),
    #             ee_rot=ee_rot_r,
    #             dt=dt,
    #             is_alignment=True,
    #             arm="right",
    #         ),
    #     )
    # )
    # # same but switch the arms
    # strokelist.strokes.append(
    #     (
    #         Stroke(
    #             description=f"left arm over {alignment_inkcap_color_l} inkcap",
    #             ee_pos=transform_and_offset(
    #                 np.zeros((plan.stroke_length, 3)),
    #                 alignment_inkcap_pose_l.pos.xyz,
    #                 alignment_inkcap_pose_l.rot.wxyz,
    #                 plan.needle_hover_offset.xyz,
    #             ),
    #             ee_rot=ee_rot_l,
    #             dt=dt,
    #             is_alignment=True,
    #             arm="left",
    #         ),
    #         Stroke(
    #             description="right arm over design",
    #             ee_pos=transform_and_offset(
    #                 np.zeros((plan.stroke_length, 3)),
    #                 scene.skin.design_pose.pos.xyz,
    #                 scene.skin.design_pose.rot.wxyz,
    #                 plan.needle_hover_offset.xyz,
    #             ),
    #             ee_rot=ee_rot_r,
    #             dt=dt,
    #             is_alignment=True,
    #             arm="right",
    #         ),
    #     )
    # )
    # next lets add inkdip on left arm, right arm will be at rest
    first_color_l = pen_paths_l[0][0]
    inkcap_name_l = inks_color_to_inkcap_name[first_color_l]
    strokelist.strokes.append(
        (
            Stroke(
                description=f"left arm inkdip into {inkcap_name_l}",
                is_inkdip=True,
                inkcap=inkcap_name_l,
                ee_pos=inkdip_func(first_color_l),
                ee_rot=ee_rot_l,
                dt=dt,
                arm="left",
            ),
            Stroke(
                description="right arm at rest",
                ee_pos=np.zeros((plan.stroke_length, 3)),
                ee_rot=ee_rot_r,
                dt=dt,
                arm="right",
            ),
        )
    )
    inkcap_name_r = None # these will be used to determine when to inkdip
    ptr_l: int = 0
    ptr_r: int = 0
    stroke_idx: int = len(strokelist.strokes) # strokes list already contains strokes
    max_paths = max(len(pen_paths_l), len(pen_paths_r))
    for _ in range(max_paths):
        color_l = pen_paths_l[ptr_l][0] if ptr_l < len(pen_paths_l) else None
        path_l = pen_paths_l[ptr_l][1] if ptr_l < len(pen_paths_l) else None

        color_r = pen_paths_r[ptr_r][0] if ptr_r < len(pen_paths_r) else None
        path_r = pen_paths_r[ptr_r][1] if ptr_r < len(pen_paths_r) else None

        # LEFT ARM LOGIC
        if path_l is None:
            stroke_l = Stroke(
                description="left arm at rest",
                ee_pos=np.zeros((plan.stroke_length, 3)),
                ee_rot=ee_rot_l,
                dt=dt,
                arm="left",
            )
        elif inkcap_name_l is not None:
            os.symlink(
                os.path.join(design_dir, stroke_img_map[color_l][ptr_l][1]),
                os.path.join(frames_dir, f"arm_l_color_{color_l}_stroke_{stroke_idx:04d}.png"),
            )
            pixel_coords, meter_coords = coords_from_path(path_l)
            stroke_l = Stroke(
                description=f"left arm stroke using left arm",
                arm="left",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_l,
                dt=dt,
                svg_path_obj=str(path_l),
                inkcap=inkcap_name_l,
                is_inkdip=False,
                frame_path=new_frame_name,
                color=color_l,
            )
            inkcap_name_l = None # inkdip on next stroke
            ptr_l += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_l is not None:
                inkcap_name_l = inks_color_to_inkcap_name[color_l]
                stroke_l = Stroke(
                    description=f"left arm inkdip into {inkcap_name_l}",
                    is_inkdip=True,
                    inkcap=inkcap_name_l,
                    ee_pos=inkdip_func(color_l),
                    ee_rot=ee_rot_l,
                    dt=dt,
                    arm="left",
                    frame_path=None,
                )
            else:
                stroke_l = Stroke(
                    description="left arm at rest",
                    ee_pos=np.zeros((plan.stroke_length, 3)),
                    ee_rot=ee_rot_l,
                    dt=dt,
                    arm="left",
                    frame_path=None,
                )

        # RIGHT ARM LOGIC
        if path_r is None:
            stroke_r = Stroke(
                description="right arm at rest",
                ee_pos=np.zeros((plan.stroke_length, 3)),
                ee_rot=ee_rot_r,
                dt=dt,
                arm="right",
                frame_path=None,
            )
        elif inkcap_name_r is not None:
            pixel_coords, meter_coords = coords_from_path(path_r)
            new_frame_name = f"arm_r_color_{color_r}_stroke_{stroke_idx:04d}.png"
            os.symlink(
                os.path.join(design_dir, stroke_img_map[color_r][ptr_r][1]),
                os.path.join(frames_dir, new_frame_name),
            )
            stroke_r = Stroke(
                description=f"right arm stroke using right arm",
                arm="right",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_r,
                dt=dt,
                svg_path_obj=str(path_r),
                inkcap=inkcap_name_r,
                is_inkdip=False,
                frame_path=new_frame_name,
                color=color_r,
            )
            inkcap_name_r = None # inkdip on next stroke
            ptr_r += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_r is not None:
                inkcap_name_r = inks_color_to_inkcap_name[color_r]
                stroke_r = Stroke(
                    description=f"right arm inkdip into {inkcap_name_r}",
                    is_inkdip=True,
                    inkcap=inkcap_name_r,
                    ee_pos=inkdip_func(color_r),
                    ee_rot=ee_rot_r,
                    dt=dt,
                    arm="right",
                    frame_path=None,
                )
            else:
                stroke_r = Stroke(
                    description="right arm at rest",
                    ee_pos=np.zeros((plan.stroke_length, 3)),
                    ee_rot=ee_rot_r,
                    dt=dt,
                    arm="right",
                    frame_path=None,
                )
        strokelist.strokes.append((stroke_l, stroke_r))
        stroke_idx += 1

    strokes_path = os.path.join(output_dir, "strokes.yaml")
    log.info(f"💾 Saving strokes to {strokes_path}")
    strokelist.to_yaml(strokes_path)

    # strokebatch: StrokeBatch = strokebatch_from_strokes(
    #     strokelist=strokelist,
    #     stroke_length=plan.stroke_length,
    #     batch_size=plan.ik_batch_size,
    #     joints=scene.ready_pos_full,
    #     urdf_path=scene.urdf.path,
    #     link_names=scene.urdf.ee_link_names,
    #     design_pose=scene.skin.design_pose,
    #     needle_hover_offset=plan.needle_hover_offset,
    #     needle_offset_l=plan.needle_offset_l,
    #     needle_offset_r=plan.needle_offset_r,
    # )
    # strokebatch_path = os.path.join(output_dir, f"strokebatch.safetensors")
    # log.info(f"💾 Saving strokebatch to {strokebatch_path}")
    # strokebatch.save(strokebatch_path)

    # copy the plan yaml to the output directory
    plan.name = config.name # override the plan name
    plan_path = os.path.join(output_dir, "plan.yaml")
    log.info(f"💾 Saving plan yaml to {plan_path}")
    plan.to_yaml(plan_path)

if __name__ == "__main__":
    args = setup_log_with_config(GenSVGPlanConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    gen_svg_plan(args)