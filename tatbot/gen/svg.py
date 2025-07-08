import collections
import json
import os
import re
import shutil

import numpy as np
import svgpathtools
from PIL import Image

from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger
from tatbot.data.scene import Scene

log = get_logger('gen.svg', '🖋️')

def make_svg_strokes(scene: Scene) -> StrokeList:
    assert scene.design_dir is not None, "❌ Design directory is not set, is this a SVG scene?"
    svg_files = []
    svg_pens: dict[str, str] = {}
    for file in os.listdir(scene.design_dir):
        if file.endswith('.svg'):
            svg_path = os.path.join(scene.design_dir, file)
            svg_files.append(svg_path)
            match = re.match(r".*_pen\d+_(\w+)\.svg$", file)
            if not match:
                raise ValueError(f"❌ Could not extract pen name from filename: {file}")
            pen_name = match.group(1)
            svg_pens[pen_name] = svg_path
    log.info(f"✅ Found {len(svg_pens)} pens in {scene.design_dir}")
    log.debug(f"Pens in design: {svg_pens.keys()}")

    pen_paths_l: list[tuple[str, svgpathtools.Path]] = []
    pen_paths_r: list[tuple[str, svgpathtools.Path]] = []
    for pen_name, svg_path in svg_pens.items():
        assert pen_name in scene.pens_config, f"❌ Pen {pen_name} not found in pens config"
        log.info(f"Processing svg file at: {svg_path}")
        paths, _, _ = svgpathtools.svg2paths2(svg_path)
        log.info(f"Found {len(paths)} paths")
        for path in paths:
            if path.length() == 0:
                log.warning(f"❌ Path {path} is empty, skipping")
                continue
            if pen_name in scene.pen_names_l:
                pen_paths_l.append((pen_name, path))
            if pen_name in scene.pen_names_r:
                pen_paths_r.append((pen_name, path))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        raise ValueError("No paths found for left or right arm")

    # --- Build mapping from (arm, pen, stroke index) to original PNG filename ---
    # Example filename: a6f9q4thkhrm80cqrv9rebavmc_plotted_1_set1_pen3_true_blue_F000001.png
    stroke_img_map = collections.defaultdict(list)  # (pen_name) -> list of (frame_num, filename)
    for file in os.listdir(scene.design_dir):
        m = re.match(r".*_pen\d+_([a-zA-Z0-9_]+)_F(\d+)\.png$", file)
        if m:
            pen_name = m.group(1)
            assert pen_name in scene.pens_config, f"❌ Pen {pen_name} not found in pens config"
            assert pen_name in scene.pen_names_l or pen_name in scene.pen_names_r, f"❌ Pen {pen_name} not found in left or right pen names"
            frame_num = int(m.group(2))
            stroke_img_map[pen_name].append((frame_num, file))
    if stroke_img_map == {}:
        raise ValueError(f"❌ No stroke images found in {scene.design_dir}, did you export intermediate frames?")
    # Sort by frame number for each pen
    for pen in stroke_img_map:
        stroke_img_map[pen].sort()

    def coords_from_path(path: svgpathtools.Path) -> tuple[np.ndarray, np.ndarray]:
        """Resample path evenly along the path and convert to pixel and meter coordinates."""
        total_length = path.length()
        distances = np.linspace(0, total_length, scene.stroke_length)
        points = [path.point(path.ilength(d)) for d in distances]
        pixel_coords = np.array([[p.real, p.imag] for p in points])
        meter_coords = np.hstack([(
            pixel_coords * np.array([ # convert pixel coordinates to meter coordinates
                scene.skin.design_pose.pos.xyz[0] / scene.skin.design_pose.pos.xyz[0],
                scene.skin.design_pose.pos.xyz[1] / scene.skin.design_pose.pos.xyz[1],
            ], dtype=np.float32)
            - np.array([ # center the meter coordinates in the image
                scene.skin.design_pose.pos.xyz[0] / 2,
                scene.skin.design_pose.pos.xyz[1] / 2,
            ], dtype=np.float32)
        ), np.zeros((scene.stroke_length, 1), dtype=np.float32)]) # z axis is 0
        return pixel_coords, meter_coords

    # left arm and right arm strokes in order of execution on robot
    strokelist: StrokeList = StrokeList(strokes=[])

    # default time between poses is fast movement
    dt = np.full((scene.stroke_length, 1), scene.arms.goal_time_fast)
    # slow movement to and from hover positions
    dt[:2] = scene.arms.goal_time_slow
    dt[-2:] = scene.arms.goal_time_slow

    # hardcoded orientations for left and right arm end effectors
    ee_rot_l = np.tile(scene.ee_rot_l.wxyz, (scene.stroke_length, 1))
    ee_rot_r = np.tile(scene.ee_rot_r.wxyz, (scene.stroke_length, 1))

    inkdip_func = make_inkdip_func(scene)

    # start with inkdip on left arm, right arm will be at rest
    first_color_l = pen_paths_l[0][0]
    _inkdip_stroke = inkdip_func(first_color_l, "left")
    _inkdip_stroke.ee_rot = ee_rot_l
    _inkdip_stroke.dt = dt
    strokelist.strokes.append(
        (
            _inkdip_stroke,
            Stroke(
                description="right arm at rest",
                ee_pos=np.zeros((scene.stroke_length, 3)),
                ee_rot=ee_rot_r,
                dt=dt,
                arm="right",
            ),
        )
    )
    inkcap_name_l = _inkdip_stroke.inkcap
    inkcap_name_r = None # when None, indicates that an inkdip stroke is required
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
                ee_pos=np.zeros((scene.stroke_length, 3)),
                ee_rot=ee_rot_l,
                dt=dt,
                arm="left",
            )
        elif inkcap_name_l is not None:
            pixel_coords, meter_coords = coords_from_path(path_l)
            stroke_l = Stroke(
                description=f"left arm stroke after inkdip in {inkcap_name_l}",
                arm="left",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_l,
                dt=dt,
                svg_path_obj=str(path_l),
                inkcap=inkcap_name_l,
                is_inkdip=False,
                frame_path=os.path.join(scene.design_dir, stroke_img_map[color_l][ptr_l][1]),
                color=color_l,
            )
            inkcap_name_l = None # inkdip on next stroke
            ptr_l += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_l is not None:
                stroke_l = inkdip_func(color_l, "left")
                stroke_l.ee_rot = ee_rot_l
                stroke_l.dt = dt
                stroke_l.inkcap = inkcap_name_l
            else:
                stroke_l = Stroke(
                    description="left arm at rest",
                    ee_pos=np.zeros((scene.stroke_length, 3)),
                    ee_rot=ee_rot_l,
                    dt=dt,
                    arm="left",
                    frame_path=None,
                )

        # RIGHT ARM LOGIC
        if path_r is None:
            stroke_r = Stroke(
                description="right arm at rest",
                ee_pos=np.zeros((scene.stroke_length, 3)),
                ee_rot=ee_rot_r,
                dt=dt,
                arm="right",
                frame_path=None,
            )
        elif inkcap_name_r is not None:
            pixel_coords, meter_coords = coords_from_path(path_r)
            stroke_r = Stroke(
                description=f"right arm stroke after inkdip in {inkcap_name_r}",
                arm="right",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_r,
                dt=dt,
                svg_path_obj=str(path_r),
                inkcap=inkcap_name_r,
                is_inkdip=False,
                frame_path=os.path.join(scene.design_dir, stroke_img_map[color_r][ptr_r][1]),
                color=color_r,
            )
            inkcap_name_r = None # inkdip on next stroke
            ptr_r += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_r is not None:
                stroke_r = inkdip_func(color_r, "right")
                stroke_r.ee_rot = ee_rot_r
                stroke_r.dt = dt
                stroke_r.inkcap = inkcap_name_r
            else:
                stroke_r = Stroke(
                    description="right arm at rest",
                    ee_pos=np.zeros((scene.stroke_length, 3)),
                    ee_rot=ee_rot_r,
                    dt=dt,
                    arm="right",
                    frame_path=None,
                )
        strokelist.strokes.append((stroke_l, stroke_r))
        stroke_idx += 1

    return strokelist