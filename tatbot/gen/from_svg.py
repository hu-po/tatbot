import json
import logging
import os
import re
from dataclasses import dataclass
import functools

import numpy as np
import svgpathtools
import yaml
from lxml import etree
from PIL import Image

from tatbot.bot.urdf import get_link_poses
from tatbot.data.ink import InkPalette, InkCap
from tatbot.data.plan import Plan
from tatbot.data.stroke import Stroke
from tatbot.data.pose import ArmPose, Pose
from tatbot.data.urdf import URDF
from tatbot.data.strokebatch import StrokeBatch
from tatbot.gen.strokebatch import strokebatch_from_strokes
from tatbot.gpu.ik import transform_and_offset
from tatbot.utils.log import get_logger, print_config, setup_log_with_config

log = get_logger('gen.from_svg', '🖋️')

@dataclass
class FromSVGConfig:
    debug: bool = False
    """Enable debug logging."""

    name: str = "yawning_cat"
    """Name of the SVG file"""
    design_dir: str = f"~/tatbot/assets/designs/{name}"
    """Directory containing the design svg (per pen) and png file."""
    output_dir: str = os.path.expanduser(f"~/tatbot/output/plans/{name}")
    """Directory to save the plan."""

    pens_config_path: str = "~/tatbot/config/pens/fullcolor.json"
    """Path to the pens config file."""
    
    plan_name: str = "default"
    """Name of the plan (Plan)."""
    urdf_name: str = "default"
    """Name of the urdf (URDF)."""
    ink_palette_name: str = "default"
    """Name of the ink palette (InkPalette)."""
    left_arm_pose_name: str = "left/rest"
    """Name of the left arm pose (ArmPose)."""
    righ_arm_pose_name: str = "right/rest"
    """Name of the right arm pose (ArmPose)."""

def gen_from_svg(config: FromSVGConfig):
    log.info(f"Generating {config.name} ...")
    
    design_dir = os.path.expanduser(config.design_dir)
    assert os.path.exists(design_dir), f"❌ Design directory {design_dir} does not exist"
    log.debug(f"📂 Design directory: {design_dir}")

    output_dir = os.path.expanduser(config.output_dir)
    log.info(f"📂 Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    plan: Plan = Plan.from_name(config.plan_name)
    log.info(f"✅ Loaded plan: {plan}")
    log.debug(f"Plan: {plan}")
    urdf: URDF = URDF.from_name(config.urdf_name)
    log.info(f"✅ Loaded URDF: {urdf}")
    log.debug(f"URDF: {urdf}")
    left_arm_pose: ArmPose = ArmPose.from_name(config.left_arm_pose_name)
    log.info("✅ Loaded left arm pose")
    log.debug(f"Left arm pose: {left_arm_pose}")
    right_arm_pose: ArmPose = ArmPose.from_name(config.righ_arm_pose_name)
    log.info("✅ Loaded right arm pose")
    log.debug(f"Right arm pose: {right_arm_pose}")
    rest_pose: np.ndarray = np.concatenate([right_arm_pose.joints, left_arm_pose.joints])

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

    ink_palette: InkPalette = InkPalette.from_name(config.ink_palette_name)
    log.info(f"✅ Loaded ink palette: {ink_palette}")
    log.debug(f"Ink palette: {ink_palette}")
    inkpalette_color_to_name: dict[str, str] = {}
    inkpalette_color_to_pose: dict[str, Pose] = {}
    for inkcap in ink_palette.inkcaps:
        assert inkcap.name in urdf.ink_link_names, f"❌ Inkcap {inkcap.name} not found in URDF"
        inkpalette_color_to_name[inkcap.ink.name] = inkcap.name
        link_poses = get_link_poses(urdf.path, urdf.ink_link_names, rest_pose)
        inkpalette_color_to_pose[inkcap.ink.name] = link_poses[inkcap.name]

    inkpalette_colors = {inkcap.ink.name: inkcap.name for inkcap in ink_palette.inkcaps}
    log.info(f"✅ Found {len(inkpalette_colors)} colors in ink palette")
    log.debug(f"Ink palette colors: {inkpalette_colors}")

    image_path: str = None
    for file in os.listdir(design_dir):
        if file.endswith('.png'):
            if image_path is not None:
                log.warning(f"⚠️ Multiple image files found in {design_dir}")
            image_path = os.path.join(design_dir, file)
    assert image_path is not None, f"❌ No image file found in {design_dir}"
    log.info(f"📂 Found image file: {image_path}")
    image = Image.open(image_path)
    image_width_px = image.width
    image_height_px = image.height
    assert image_width_px == plan.image_width_px, f"❌ Image width {image_width_px} does not match plan width {plan.image_width_px}"
    assert image_height_px == plan.image_height_px, f"❌ Image height {image_height_px} does not match plan height {plan.image_height_px}"
    width_scale_m = plan.image_width_m / image_width_px
    height_scale_m = plan.image_height_m / image_height_px
    log.info(f"Loaded image: {image_path} with size {image_width_px}x{image_height_px}")

    def resample_path(path: svgpathtools.Path, num_points: int = plan.path_length) -> np.ndarray:
        """Resample path to num_points evenly along the path."""
        total_length = path.length()
        distances = np.linspace(0, total_length, num_points)
        points = [path.point(path.ilength(d)) for d in distances]
        coords = np.array([[p.real, p.imag] for p in points])
        return coords
    
    @functools.lru_cache(maxsize=len(ink_palette.inkcaps))
    def inkdip_xyz(color: str, num_points: int = plan.path_length) -> np.ndarray:
        """Get <x, y, z> coordinates for an inkdip into a specific inkcap."""
        inkcap_pose: Pose = inkpalette_color_to_pose[color]
        # hover over the inkcap
        inkdip_pos = transform_and_offset(
            np.zeros((num_points, 3)), # <x, y, z>
            inkcap_pose.pos,
            inkcap_pose.wxyz,
            plan.inkdip_hover_offset,
        )
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
        inkdip_pos = transform_and_offset(
            inkdip_pos,
            inkcap_pose.pos,
            inkcap_pose.wxyz,
            offsets,
        )
        return inkdip_pos
    
    strokes: list[Stroke] = []

    for pen_name, svg_path in pens.items():
        assert pen_name in config_pens, f"❌ Pen {pen_name} not found in pens config"
        assert config_pens[pen_name]["name"] == pen_name, f"❌ Pen {pen_name} not found in pens config"
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
            pixel_coords = resample_path(path)
            norm_coords = pixel_coords / np.array([image_width_px, image_height_px], dtype=np.float32)
            norm_center = np.mean(norm_coords, axis=0)
            meter_coords_2d = (
                pixel_coords * np.array([width_scale_m, height_scale_m], dtype=np.float32)
                - np.array([plan.image_width_m / 2, plan.image_height_m / 2], dtype=np.float32)
            )
            meter_coords = np.hstack([meter_coords_2d, np.zeros((pixel_coords.shape[0], 1), dtype=np.float32)])
            meters_center = np.mean(meter_coords, axis=0)
            stroke = Stroke(
                description=f"{pen_name} stroke using {arm} arm",
                arm=arm,
                color=config_pens[pen_name]["color"],
                pixel_coords=pixel_coords,
                meter_coords=meter_coords,
                meters_center=meters_center,
                norm_coords=norm_coords,
                norm_center=norm_center,
                inkcap=None,
                is_inkdip=False,
            )
            strokes.append(stroke)

#     # sort strokes along the -Y axis in normalized coords
#     sorted_strokes = sorted(_strokes.items(), key=lambda x: x[1].norm_center[1], reverse=True)
#     left_arm_pointer = 0 # left arm starts with leftmost stroke, moves rightwards
#     half_length = len(sorted_strokes) // 2
#     right_arm_pointer = half_length # right arm starts from middle moves rightwards
#     total_strokes = len(sorted_strokes)
#     while left_arm_pointer <= half_length and right_arm_pointer < total_strokes:
#         stroke_name_l, stroke_l = sorted_strokes[left_arm_pointer]
#         assert stroke_name_l in self.strokes, f"⚙️❌ Stroke {stroke_name_l} not found in plan"
#         stroke_name_r, stroke_r = sorted_strokes[right_arm_pointer]
#         assert stroke_name_r in self.strokes, f"⚙️❌ Stroke {stroke_name_r} not found in plan"
#         log.debug(f"⚙️ Building path from strokes left: {stroke_name_l} and right: {stroke_name_r}...")
        
#         # create a new empty path for this stroke
#         path = Path.empty(self.path_length)
#         # set the default time between poses to fast movement
#         path.dt[:] = self.path_dt_fast
#         # slow movement to and from hover positions
#         path.dt[:2] = self.path_dt_slow
#         path.dt[-2:] = self.path_dt_slow
#         # TODO: for now orientation is just design orientation (for inkdips as well)
#         path.ee_wxyz_l[:, :] = np.tile(self.ee_wxyz_l, (self.path_length, 1))
#         path.ee_wxyz_r[:, :] = np.tile(self.ee_wxyz_r, (self.path_length, 1))

#         # left arm pointer hits a stroke with no inkcap
#         if self.strokes[stroke_name_l].inkcap is None:
#             # get ink color from stroke, left arm will dip for this path
#             inkcap_name = self.ink_config.find_best_inkcap(self.strokes[stroke_name_l].color)
#             self.strokes[stroke_name_l].inkcap = inkcap_name
#             path.ee_pos_l = self.make_inkdip_pos(inkcap_name)
#             # make a new stroke object for the inkdip path
#             stroke_l = Stroke(
#                 description=f"left arm inkdip into {inkcap_name}",
#                 is_inkdip=True,
#                 inkcap=inkcap_name,
#             )
#             if left_arm_pointer == 0:
#                 # this is the first stroke of the session, keep right arm at rest for this path
#                 stroke_r = Stroke(description="right arm rest")
#                 self.path_idx_to_strokes.append([stroke_l, stroke_r])
#                 paths.append(path)
#                 continue
#         elif left_arm_pointer < half_length: # still haven't hit the last left-arm stroke
#             # left arm pointer hits a stroke with an inkcap
#             # transform to design frame, add needle offset
#             path.ee_pos_l[1:-1, :] = transform_and_offset(
#                 self.strokes[stroke_name_l].meter_coords,
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.needle_offset_l,
#             )
#             # add hover positions to start and end
#             path.ee_pos_l[0, :] = transform_and_offset(
#                 np.expand_dims(self.strokes[stroke_name_l].meter_coords[0], axis=0),
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.hover_offset,
#             )
#             path.ee_pos_l[-1, :] = transform_and_offset(
#                 np.expand_dims(self.strokes[stroke_name_l].meter_coords[-1], axis=0),
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.hover_offset,
#             )
#             left_arm_pointer += 1
#         else:
#             stroke_l = Stroke(description="left arm rest")

#         if self.strokes[stroke_name_r].inkcap is None:
#             inkcap_name = self.ink_config.find_best_inkcap(self.strokes[stroke_name_r].color)
#             self.strokes[stroke_name_r].inkcap = inkcap_name
#             path.ee_pos_r = self.make_inkdip_pos(inkcap_name)
#             # make a new stroke object for the inkdip path
#             stroke_r = Stroke(
#                 description=f"right arm inkdip into {inkcap_name}",
#                 is_inkdip=True,
#                 inkcap=inkcap_name,
#             )
#         else:
#             # right arm pointer hits a stroke with an inkcap
#             # transform to design frame, add needle offset
#             path.ee_pos_r[1:-1, :] = transform_and_offset(
#                 self.strokes[stroke_name_r].meter_coords,
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.needle_offset_r,
#             )
#             # add hover positions to start and end
#             path.ee_pos_r[0, :] = transform_and_offset(
#                 np.expand_dims(self.strokes[stroke_name_r].meter_coords[0], axis=0),
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.hover_offset,
#             )
#             path.ee_pos_r[-1, :] = transform_and_offset(
#                 np.expand_dims(self.strokes[stroke_name_r].meter_coords[-1], axis=0),
#                 self.design_pos,
#                 self.design_wxyz,
#                 self.hover_offset,
#             )
#             right_arm_pointer += 1

#         stroke_l.arm = "left"
#         stroke_r.arm = "right"
#         log.debug(f"⚙️ Adding path from strokes\n left: {stroke_l.description}\n right: {stroke_r.description}...")
#         self.path_idx_to_strokes.append([stroke_l, stroke_r])
#         paths.append(path)

#     # HACK: add two paths at the beginning where left arm is centered over design and right arm is centered over black inkcap
#     # use twice the hover offset to ensure enough space for adjustments
#     path = Path.empty(self.path_length)
#     path.ee_pos_l = transform_and_offset(
#         np.zeros((self.path_length, 3)),
#         self.design_pos,
#         self.design_wxyz,
#         self.hover_offset * 2,
#     )
#     path.ee_wxyz_l = np.tile(self.ee_wxyz_l, (self.path_length, 1))
#     path.ee_pos_r = transform_and_offset(
#         np.tile([0, 0, 0], (self.path_length, 1)),
#         self.ink_config.inkpalette_pos,
#         self.ink_config.inkpalette_wxyz,
#         self.ink_config.inkdip_hover_offset * 2,
#     )
#     path.ee_wxyz_r = np.tile(self.ee_wxyz_r, (self.path_length, 1))
#     path.dt[:2] = self.path_dt_slow
#     path.dt[-2:] = self.path_dt_slow
#     paths.insert(0, path)
#     stroke_l = Stroke(description="left over design with twice the hover offset")
#     stroke_r = Stroke(description="right over large black inkcap with twice the hover offset")
#     self.path_idx_to_strokes.insert(0, [stroke_l, stroke_r])

#     # HACK: second hack path
#     path = Path.empty(self.path_length)
#     path.ee_pos_l = transform_and_offset(
#         np.tile([0, 0, 0], (self.path_length, 1)),
#         self.ink_config.inkpalette_pos,
#         self.ink_config.inkpalette_wxyz,
#         self.ink_config.inkdip_hover_offset * 2,
#     )
#     path.ee_wxyz_l = np.tile(self.ee_wxyz_l, (self.path_length, 1))
#     path.ee_pos_r = transform_and_offset(
#         np.zeros((self.path_length, 3)),
#         self.design_pos,
#         self.design_wxyz,
#         self.hover_offset * 2,
#     )
#     path.ee_wxyz_r = np.tile(self.ee_wxyz_r, (self.path_length, 1))
#     path.dt[:2] = self.path_dt_slow
#     path.dt[-2:] = self.path_dt_slow
#     paths.insert(0, path)
#     stroke_l = Stroke(description="left over large black inkcap with twice the hover offset")
#     stroke_r = Stroke(description="right over design with twice the hover offset")
#     self.path_idx_to_strokes.insert(0, [stroke_l, stroke_r])

    strokebatch: StrokeBatch = strokebatch_from_strokes(
        strokes=strokes,
        path_length=plan.path_length,
        batch_size=plan.ik_batch_size,
        joints=urdf.joints,
        urdf_path=urdf.path,
        link_names=urdf.ee_link_names,
    )
    strokebatch_path = os.path.join(output_dir, f"strokebatch.safetensors")
    log.info(f"💾 Saving strokebatch to {strokebatch_path}")
    strokebatch.save(strokebatch_path)

    # TODO: save image
    # TODO: save svg
    # TODO: save pen info
    # TODO: save inkcap info
    # TODO: save stroke info
#     self.save_pathbatch(pathbatch)
#     self.save() # update metadata


if __name__ == "__main__":
    args = setup_log_with_config(FromSVGConfig)
    print_config(args)
    if args.debug:
        log.setLevel(logging.DEBUG)
    gen_from_svg(args)