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
        if pen_name in plan.left_arm_pen_names:
            arm = "left"
        elif pen_name in plan.right_arm_pen_names:
            arm = "right"
        else:
            raise ValueError(f"❌ Pen {pen_name} not found in plan config")
        log.info(f"Processing svg file at: {svg_path}")
        paths, _, _ = svgpathtools.svg2paths2(svg_path)
        log.info(f"Found {len(paths)} paths")
        for path in paths:
            if arm == "left":
                pen_paths_l.append((pen_name, path))
            elif arm == "right":
                pen_paths_r.append((pen_name, path))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        if len(pen_paths_l) == 0:
            log.warning("No paths found for left arm, duplicating right arm paths")
            pen_paths_l = pen_paths_r.copy()
        if len(pen_paths_r) == 0:
            log.warning("No paths found for right arm, duplicating left arm paths")
            pen_paths_r = pen_paths_l.copy()

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

    # --- 1:1 mapping: Copy all frame images and create a stroke for each ---
    strokelist: StrokeList = StrokeList(strokes=[])
    dt = np.full((plan.stroke_length, 1), scene.arms.goal_time_fast)
    dt[:2] = scene.arms.goal_time_slow
    dt[-2:] = scene.arms.goal_time_slow
    ee_rot_l = np.tile(plan.ee_rot_l.wxyz, (plan.stroke_length, 1))
    ee_rot_r = np.tile(plan.ee_rot_r.wxyz, (plan.stroke_length, 1))

    # For each pen, for each frame, create a stroke and copy the frame image
    for pen_name, frames in stroke_img_map.items():
        # Determine arm (left/right) for this pen
        if pen_name in plan.left_arm_pen_names:
            arm = "l"
            ee_rot = ee_rot_l
        elif pen_name in plan.right_arm_pen_names:
            arm = "r"
            ee_rot = ee_rot_r
        else:
            log.warning(f"Pen {pen_name} not found in plan config, skipping.")
            continue
        # Get all paths for this pen (if any)
        pen_paths = [p for p in (pen_paths_l if arm == "l" else pen_paths_r) if p[0] == pen_name]
        for idx, (frame_num, filename) in enumerate(frames):
            new_frame_name = f"arm_{arm}_color_{pen_name}_stroke_{idx+1:04d}.png"
            shutil.copy(os.path.join(design_dir, filename), os.path.join(frames_dir, new_frame_name))
            # If a path exists for this index, use it; otherwise, create a dummy stroke
            if idx < len(pen_paths):
                path = pen_paths[idx][1]
                pixel_coords, meter_coords = coords_from_path(path)
                stroke = Stroke(
                    description=f"{arm} arm stroke for {pen_name} frame {idx+1}",
                    arm="left" if arm == "l" else "right",
                    pixel_coords=pixel_coords,
                    ee_pos=meter_coords,
                    ee_rot=ee_rot,
                    dt=dt,
                    svg_path_obj=str(path),
                    inkcap=inks_color_to_inkcap_name[pen_name],
                    is_inkdip=False,
                    frame_path=new_frame_name,
                    color=pen_name,
                )
            else:
                # Dummy stroke (at rest)
                stroke = Stroke(
                    description=f"{arm} arm at rest (no path) for {pen_name} frame {idx+1}",
                    arm="left" if arm == "l" else "right",
                    ee_pos=np.zeros((plan.stroke_length, 3)),
                    ee_rot=ee_rot,
                    dt=dt,
                    frame_path=new_frame_name,
                    color=pen_name,
                )
            # Always append as a tuple (stroke, None) or (None, stroke) for left/right arm
            if arm == "l":
                strokelist.strokes.append((stroke, None))
            else:
                strokelist.strokes.append((None, stroke))

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