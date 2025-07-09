import collections
import os
import re

import cv2
import numpy as np
from PIL import Image

from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.log import get_logger
from tatbot.utils.colors import COLORS
from tatbot.data.scene import Scene

log = get_logger('gen.gcode', '🖋️')

def parse_gcode_file(gcode_path: str, scene: Scene) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Parse a G-code file and convert each path to meter and pixel coordinates.
    Returns list of tuples: (meter_coords, pixel_coords, gcode_text)
    
    G-code format:
    - G0: Rapid movement (pen up) - separator
    - G1: Linear movement (pen down) - part of path
    - X, Y coordinates in millimeters
    """
    # Load design image to get dimensions for pixel coordinate calculation
    if scene.design_img_path and os.path.exists(scene.design_img_path):
        with Image.open(scene.design_img_path) as img:
            img_width, img_height = img.size
    
    paths = []
    current_path_points = []
    current_path_gcode = []
    pen_down = False
    
    with open(gcode_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            command = parts[0].upper()
            
            if command == 'G0':  # Rapid movement (pen up) - separator
                if pen_down and current_path_points:
                    # End current path and convert coordinates
                    if len(current_path_points) >= 1:
                        gcode_coords_mm = np.array(current_path_points, dtype=np.float32)
                        gcode_text = '\n'.join(current_path_gcode)
                        
                        # Convert mm to meters for robot coordinates
                        meter_coords = np.hstack([gcode_coords_mm / 1000.0, np.zeros((len(gcode_coords_mm), 1), dtype=np.float32)])
                        
                        # Convert to pixel coordinates
                        gcode_coords_m = gcode_coords_mm / 1000.0  # mm to meters
                        skin_coords = np.zeros_like(gcode_coords_m)
                        
                        # For landscape mode G-code: swap X and Y coordinates
                        # G-code X (width) maps to skin Y dimension (zone_width_m)
                        # G-code Y (height) maps to skin Z dimension (zone_height_m)
                        skin_coords[:, 0] = gcode_coords_m[:, 1] + scene.skin.zone_height_m / 2.0  # G-code Y → skin X
                        skin_coords[:, 1] = gcode_coords_m[:, 0] + scene.skin.zone_width_m / 2.0   # G-code X → skin Y
                        
                        pixel_coords = np.zeros_like(skin_coords)
                        pixel_coords[:, 0] = (skin_coords[:, 0] / scene.skin.zone_height_m) * scene.skin.image_height_px
                        pixel_coords[:, 1] = (skin_coords[:, 1] / scene.skin.zone_width_m) * scene.skin.image_width_px
                        
                        # Clamp to image bounds
                        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, img_width - 1)
                        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, img_height - 1)
                        
                        paths.append((meter_coords, pixel_coords, gcode_text))
                    current_path_points = []
                    current_path_gcode = []
                
                # Extract coordinates from G0 command to use as starting point for next path
                x_coord = None
                y_coord = None
                
                for part in parts[1:]:
                    if part.startswith('X'):
                        try:
                            x_coord = float(part[1:])
                        except ValueError:
                            log.warning(f"Invalid X coordinate in line {line_num}: {part}")
                    elif part.startswith('Y'):
                        try:
                            y_coord = float(part[1:])
                        except ValueError:
                            log.warning(f"Invalid Y coordinate in line {line_num}: {part}")
                
                if x_coord is not None and y_coord is not None:
                    point = np.array([x_coord, y_coord], dtype=np.float32)
                    current_path_points = [point]  # Start new path with G0 coordinates
                    current_path_gcode = [line]  # Include G0 line in gcode text
                else:
                    current_path_points = []
                    current_path_gcode = []
                
                pen_down = False
                
            elif command == 'G1':  # Linear movement (pen down)
                if not pen_down:
                    # If we don't have any points yet, start fresh
                    if not current_path_points:
                        current_path_points = []
                        current_path_gcode = []
                pen_down = True
                current_path_gcode.append(line)
                
                # Extract coordinates
                x_coord = None
                y_coord = None
                
                for part in parts[1:]:
                    if part.startswith('X'):
                        try:
                            x_coord = float(part[1:])
                        except ValueError:
                            log.warning(f"Invalid X coordinate in line {line_num}: {part}")
                    elif part.startswith('Y'):
                        try:
                            y_coord = float(part[1:])
                        except ValueError:
                            log.warning(f"Invalid Y coordinate in line {line_num}: {part}")
                
                if x_coord is not None and y_coord is not None:
                    point = np.array([x_coord, y_coord], dtype=np.float32)
                    current_path_points.append(point)
    
    # Handle final path
    if pen_down and current_path_points and len(current_path_points) >= 1:
        gcode_coords_mm = np.array(current_path_points, dtype=np.float32)
        gcode_text = '\n'.join(current_path_gcode)
        
        # Convert mm to meters for robot coordinates
        meter_coords = np.hstack([gcode_coords_mm / 1000.0, np.zeros((len(gcode_coords_mm), 1), dtype=np.float32)])
        
        # Convert to pixel coordinates
        gcode_coords_m = gcode_coords_mm / 1000.0  # mm to meters
        skin_coords = np.zeros_like(gcode_coords_m)
        
        # For landscape mode G-code: swap X and Y coordinates
        # G-code X (width) maps to skin Y dimension (zone_width_m)
        # G-code Y (height) maps to skin Z dimension (zone_height_m)
        skin_coords[:, 0] = gcode_coords_m[:, 1] + scene.skin.zone_height_m / 2.0  # G-code Y → skin X
        skin_coords[:, 1] = gcode_coords_m[:, 0] + scene.skin.zone_width_m / 2.0   # G-code X → skin Y
        
        pixel_coords = np.zeros_like(skin_coords)
        pixel_coords[:, 0] = (skin_coords[:, 0] / scene.skin.zone_height_m) * scene.skin.image_height_px
        pixel_coords[:, 1] = (skin_coords[:, 1] / scene.skin.zone_width_m) * scene.skin.image_width_px
        
        # Clamp to image bounds
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, img_width - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, img_height - 1)
        
        paths.append((meter_coords, pixel_coords, gcode_text))
    
    return paths


def generate_stroke_frame_image(scene: Scene, pen_name: str, path_idx: int, pixel_coords: np.ndarray, arm: str) -> str:
    """
    Generate a frame image for a specific stroke by drawing the path on the base design image.
    
    Args:
        scene: The scene containing design information
        pen_name: Name of the pen/color being used
        path_idx: Index of the path within the pen's paths
        pixel_coords: Pixel coordinates of the path
        arm: Which arm is drawing ("left" or "right")
    
    Returns:
        Path to the generated frame image
    """
    # Load the base design image
    if not scene.design_img_path or not os.path.exists(scene.design_img_path):
        raise ValueError(f"Design image not found at {scene.design_img_path}")
    
    # Read the base image
    base_img = cv2.imread(scene.design_img_path)
    if base_img is None:
        raise ValueError(f"Could not read design image at {scene.design_img_path}")
    
    # Create a copy to draw on
    frame_img = base_img.copy()
    
    # Define colors for visualization (BGR format for OpenCV)
    path_color = COLORS["red"]  # Red for the path
    
    # Draw the complete path in red
    log.debug(f"Drawing path with {len(pixel_coords)} points for {arm} arm {pen_name}")
    for i in range(len(pixel_coords) - 1):
        # For landscape mode: pixel_coords[:, 0] is height (Y), pixel_coords[:, 1] is width (X)
        # OpenCV expects (x, y) where x is column (width) and y is row (height)
        pt1 = (int(pixel_coords[i, 1]), int(pixel_coords[i, 0]))  # (width, height)
        pt2 = (int(pixel_coords[i + 1, 1]), int(pixel_coords[i + 1, 0]))  # (width, height)
        cv2.line(frame_img, pt1, pt2, path_color, 2)
    
    # Draw points along the path
    for px, py in pixel_coords:
        # For landscape mode: px is height (Y), py is width (X)
        cv2.circle(frame_img, (int(py), int(px)), 3, path_color, -1)  # (width, height)
    
    # Add text indicating the pen and arm near the stroke but biased towards center
    if len(pixel_coords) > 0:
        # Calculate stroke center (in pixel coordinates)
        stroke_center_x = np.mean(pixel_coords[:, 1])  # width coordinate
        stroke_center_y = np.mean(pixel_coords[:, 0])  # height coordinate
        
        # Get image center
        img_center_x = frame_img.shape[1] / 2  # width
        img_center_y = frame_img.shape[0] / 2  # height
        
        # Calculate text position: 70% towards stroke center, 30% towards image center
        placement_ratio = 0.8
        text_x = int(placement_ratio * stroke_center_x + (1 - placement_ratio) * img_center_x)
        text_y = int(placement_ratio * stroke_center_y + (1 - placement_ratio) * img_center_y)
        
        # Ensure text stays within image bounds with some padding
        text_x = max(10, min(text_x, frame_img.shape[1] - 150))
        text_y = max(30, min(text_y, frame_img.shape[0] - 10))
        
        # Draw three lines of text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_height = 25
        
        cv2.putText(frame_img, arm.upper(), (text_x, text_y), font, font_scale, COLORS["red"], thickness)
        cv2.putText(frame_img, pen_name, (text_x, text_y + line_height), font, font_scale, COLORS["red"], thickness)
        cv2.putText(frame_img, f"{path_idx:04d}", (text_x, text_y + 2 * line_height), font, font_scale, COLORS["red"], thickness)
    
    # Generate filename
    frame_filename = f"stroke_{pen_name}_{arm}_{path_idx:04d}.png"
    frame_path = os.path.join(scene.design_dir, frame_filename)
    
    # Save the frame image
    cv2.imwrite(frame_path, frame_img)
    
    return frame_path


def make_gcode_strokes(scene: Scene) -> StrokeList:
    assert scene.design_dir is not None, "❌ Design directory is not set, does this scene have a design?"
    gcode_files = []
    gcode_pens: dict[str, str] = {}
    for file in os.listdir(scene.design_dir):
        if file.endswith('.gcode'):
            gcode_path = os.path.join(scene.design_dir, file)
            gcode_files.append(gcode_path)
            match = re.match(r".*_pen\d+_(\w+)\.gcode$", file)
            if not match:
                raise ValueError(f"❌ Could not extract pen name from filename: {file}")
            pen_name = match.group(1)
            gcode_pens[pen_name] = gcode_path
    log.info(f"✅ Found {len(gcode_pens)} pens in {scene.design_dir}")
    log.debug(f"Pens in design: {gcode_pens.keys()}")

    pen_paths_l: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    pen_paths_r: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for pen_name, gcode_path in gcode_pens.items():
        assert pen_name in scene.pens_config, f"❌ Pen {pen_name} not found in pens config"
        log.info(f"Processing gcode file at: {gcode_path}")
        paths = parse_gcode_file(gcode_path, scene)
        log.info(f"Found {len(paths)} paths")
        for meter_coords, pixel_coords, gcode_text in paths:
            if len(meter_coords) < 1:
                log.warning(f"❌ Path has no points, skipping")
                continue
            if pen_name in scene.pen_names_l:
                pen_paths_l.append((pen_name, meter_coords, pixel_coords, gcode_text))
            if pen_name in scene.pen_names_r:
                pen_paths_r.append((pen_name, meter_coords, pixel_coords, gcode_text))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        raise ValueError("No paths found for left or right arm")

    # Generate frame images for each stroke
    log.info("Generating frame images for strokes...")
    stroke_frame_paths = {}  # (pen_name, arm, path_idx) -> frame_path
    
    # Generate frames for left arm paths
    for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_l):
        frame_path = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "left")
        stroke_frame_paths[(pen_name, "left", path_idx)] = frame_path
        log.debug(f"Generated frame for left arm {pen_name} path {path_idx}: {frame_path}")
    
    # Generate frames for right arm paths
    for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_r):
        frame_path = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "right")
        stroke_frame_paths[(pen_name, "right", path_idx)] = frame_path
        log.debug(f"Generated frame for right arm {pen_name} path {path_idx}: {frame_path}")
    
    log.info(f"Generated {len(stroke_frame_paths)} frame images")

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
        path_l = pen_paths_l[ptr_l][1] if ptr_l < len(pen_paths_l) else None
        path_r = pen_paths_r[ptr_r][1] if ptr_r < len(pen_paths_r) else None

        color_l = pen_paths_l[ptr_l][0] if ptr_l < len(pen_paths_l) else None
        color_r = pen_paths_r[ptr_r][0] if ptr_r < len(pen_paths_r) else None

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
            meter_coords = pen_paths_l[ptr_l][1]
            pixel_coords = pen_paths_l[ptr_l][2]
            gcode_text = pen_paths_l[ptr_l][3]
            frame_path = stroke_frame_paths.get((color_l, "left", ptr_l))
            stroke_l = Stroke(
                description=f"left arm stroke after inkdip in {inkcap_name_l}",
                arm="left",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_l,
                dt=dt,
                gcode_text=gcode_text,
                inkcap=inkcap_name_l,
                is_inkdip=False,
                frame_path=frame_path,
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
                inkcap_name_l = stroke_l.inkcap
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
            meter_coords = pen_paths_r[ptr_r][1]
            pixel_coords = pen_paths_r[ptr_r][2]
            gcode_text = pen_paths_r[ptr_r][3]
            frame_path = stroke_frame_paths.get((color_r, "right", ptr_r))
            stroke_r = Stroke(
                description=f"right arm stroke after inkdip in {inkcap_name_r}",
                arm="right",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_r,
                dt=dt,
                gcode_text=gcode_text,
                inkcap=inkcap_name_r,
                is_inkdip=False,
                frame_path=frame_path,
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
                inkcap_name_r = stroke_r.inkcap
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