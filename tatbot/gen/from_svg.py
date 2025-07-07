import collections
import functools
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass

import numpy as np
import svgpathtools
import yaml
from lxml import etree
from PIL import Image

from tatbot.bot.urdf import get_link_poses
from tatbot.data.plan import Plan
from tatbot.data.pose import Pose
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.ik import transform_and_offset
from tatbot.gen.strokebatch import strokebatch_from_strokes
from tatbot.utils.log import get_logger, print_config, setup_log_with_config
from tatbot.data.scene import Scene

log = get_logger('gen.from_svg', '🖋️')

@dataclass
class FromSVGConfig:
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
    
    plan_name: str = "default"
    """Name of the plan (Plan)."""
    scene_name: str = "default"
    """Name of the scene config (Scene)."""

def gen_from_svg(config: FromSVGConfig):
    log.info(f"Generating {config.name} ...")
    
    design_dir = os.path.expanduser(config.design_dir)
    design_dir = os.path.join(design_dir, config.name)
    assert os.path.exists(design_dir), f"❌ Design directory {design_dir} does not exist"
    log.debug(f"📂 Design directory: {design_dir}")

    output_dir = os.path.expanduser(config.output_dir)
    output_dir = os.path.join(output_dir, config.name)
    log.info(f"📂 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load the scene abstraction
    scene: Scene = Scene.from_name(config.scene_name)
    plan: Plan = Plan.from_name(config.plan_name)

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

    ink_name_to_inkcap_name: dict[str, str] = {}
    ink_name_to_inkcap_pose: dict[str, Pose] = {}
    link_poses = get_link_poses(scene.urdf.path, scene.urdf.ink_link_names, scene.home_pos_full)
    for inkcap in scene.inks.inkcaps:
        assert inkcap.name in scene.urdf.ink_link_names, f"❌ Inkcap {inkcap.name} not found in URDF"
        ink_name = inkcap.ink["name"]
        ink_name_to_inkcap_name[ink_name] = inkcap.name
        ink_name_to_inkcap_pose[ink_name] = link_poses[inkcap.name]
    log.info(f"✅ Found {len(ink_name_to_inkcap_name)} inkcaps in {scene.inks.yaml_dir}")
    log.debug(f"Inkcaps in scene: {ink_name_to_inkcap_name.keys()}")

    # create directory for image files
    frames_dir = os.path.join(output_dir, "frames")
    log.info(f"🗃️ Creating frames directory at {frames_dir}")
    os.makedirs(frames_dir, exist_ok=True)

    # Copy the final design image to frames/full.png
    final_design_img = None
    for file in os.listdir(design_dir):
        if file.endswith('.png') and '_pen' not in file and '_F' not in file:
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
    # Sort by frame number for each pen
    for pen in stroke_img_map:
        stroke_img_map[pen].sort()

    pen_paths_l: list[tuple[str, svgpathtools.Path]] = []
    pen_paths_r: list[tuple[str, svgpathtools.Path]] = []
    for pen_name, svg_path in pens.items():
        assert pen_name in config_pens, f"❌ Pen {pen_name} not found in pens config"
        assert config_pens[pen_name]["name"] == pen_name, f"❌ Pen {pen_name} not found in pens config"
        assert pen_name in ink_name_to_inkcap_name, f"❌ Pen {pen_name} not found in ink palette"
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
    
    @functools.lru_cache(maxsize=len(scene.inks.inkcaps))
    def make_inkdip_pos(color: str, num_points: int = plan.stroke_length) -> np.ndarray:
        """Get <x, y, z> coordinates for an inkdip into a specific inkcap."""
        inkcap_pose: Pose = ink_name_to_inkcap_pose[color]
        inkcap = next(ic for ic in scene.inks.inkcaps if ic.ink.name == color)
        # Split: 1/3 down, 1/3 wait, 1/3 up (adjust as needed)
        num_down = num_points // 3
        num_up = num_points // 3
        num_wait = num_points - num_down - num_up
        # dip down to inkcap depth
        down_z = np.linspace(0, inkcap.depth_m, num_down, endpoint=False)
        # wait at depth
        wait_z = np.full(num_wait, inkcap.depth_m)
        # retract back up
        up_z = np.linspace(inkcap.depth_m, 0, num_up, endpoint=True)
        # concatenate into offset array
        offsets = np.hstack([
            np.zeros((num_points, 2)), # x and y are 0
            -np.concatenate([down_z, wait_z, up_z]).reshape(-1, 1),
        ])
        offsets = offsets + plan.inkdip_hover_offset.xyz
        inkdip_pos = transform_and_offset(
            np.zeros((num_points, 3)), # <x, y, z>
            inkcap_pose.pos.xyz,
            inkcap_pose.rot.wxyz,
            offsets,
        )
        return inkdip_pos
    
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
    alignment_inkcap_pose_r: Pose = ink_name_to_inkcap_pose[alignment_inkcap_color_r]
    alignment_inkcap_pose_l: Pose = ink_name_to_inkcap_pose[alignment_inkcap_color_l]
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
    # next lets add inkdip on left arm, right arm will be at rest
    first_color_l = pen_paths_l[0][0]
    first_inkcap_l = ink_name_to_inkcap_name[first_color_l]
    strokelist.strokes.append(
        (
            Stroke(
                description=f"left arm inkdip into {first_inkcap_l}",
                is_inkdip=True,
                inkcap=first_inkcap_l,
                ee_pos=make_inkdip_pos(first_color_l),
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
    right_arm_inkcap_name = None # these will be used to determine when to inkdip
    left_arm_ptr: int = 0
    right_arm_ptr: int = 0
    stroke_idx: int = len(strokelist.strokes) # strokes list already contains strokes
    max_paths = max(len(pen_paths_l), len(pen_paths_r))
    for _ in range(max_paths):
        color_l = pen_paths_l[left_arm_ptr][0] if left_arm_ptr < len(pen_paths_l) else None
        path_l = pen_paths_l[left_arm_ptr][1] if left_arm_ptr < len(pen_paths_l) else None

        color_r = pen_paths_r[right_arm_ptr][0] if right_arm_ptr < len(pen_paths_r) else None
        path_r = pen_paths_r[right_arm_ptr][1] if right_arm_ptr < len(pen_paths_r) else None

        # LEFT ARM LOGIC
        if path_l is None:
            stroke_l = Stroke(
                description="left arm at rest",
                ee_pos=np.zeros((plan.stroke_length, 3)),
                ee_rot=ee_rot_l,
                dt=dt,
                arm="left",
            )
        elif first_inkcap_l is not None:
            old_frame_path = stroke_img_map[color_l][left_arm_ptr][1]
            new_frame_name = f"arm_l_color_{color_l}_stroke_{stroke_idx:04d}.png"
            shutil.copy(os.path.join(design_dir, old_frame_path), os.path.join(frames_dir, new_frame_name))
            pixel_coords, meter_coords = coords_from_path(path_l)
            stroke_l = Stroke(
                description=f"left arm stroke using left arm",
                arm="left",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_l,
                dt=dt,
                svg_path_obj=str(path_l),
                inkcap=first_inkcap_l,
                is_inkdip=False,
                frame_path=new_frame_name,
                color=color_l,
            )
            first_inkcap_l = None # inkdip on next stroke
            left_arm_ptr += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_l is not None:
                first_inkcap_l = ink_name_to_inkcap_name[color_l]
                stroke_l = Stroke(
                    description=f"left arm inkdip into {first_inkcap_l}",
                    is_inkdip=True,
                    inkcap=first_inkcap_l,
                    ee_pos=make_inkdip_pos(color_l),
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
        elif right_arm_inkcap_name is not None:
            pixel_coords, meter_coords = coords_from_path(path_r)
            old_frame_path = stroke_img_map[color_r][right_arm_ptr][1]
            new_frame_name = f"arm_r_color_{color_r}_stroke_{stroke_idx:04d}.png"
            shutil.copy(os.path.join(design_dir, old_frame_path), os.path.join(frames_dir, new_frame_name))
            stroke_r = Stroke(
                description=f"right arm stroke using right arm",
                arm="right",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_r,
                dt=dt,
                svg_path_obj=str(path_r),
                inkcap=right_arm_inkcap_name,
                is_inkdip=False,
                frame_path=new_frame_name,
                color=color_r,
            )
            right_arm_inkcap_name = None # inkdip on next stroke
            right_arm_ptr += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_r is not None:
                right_arm_inkcap_name = ink_name_to_inkcap_name[color_r]
                stroke_r = Stroke(
                    description=f"right arm inkdip into {right_arm_inkcap_name}",
                    is_inkdip=True,
                    inkcap=right_arm_inkcap_name,
                    ee_pos=make_inkdip_pos(color_r),
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

    strokebatch: StrokeBatch = strokebatch_from_strokes(
        strokelist=strokelist,
        stroke_length=plan.stroke_length,
        batch_size=plan.ik_batch_size,
        joints=scene.home_pos_full,
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
    args = setup_log_with_config(FromSVGConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    gen_from_svg(args)