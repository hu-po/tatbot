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

def parse_gcode_file(gcode_path: str) -> list[tuple[np.ndarray, str]]:
    """
    Parse a G-code file and convert it to a list of tuples (numpy array, gcode text).
    Each tuple represents a sequence of G1 movements (pen down) and the corresponding G-code text.
    G0 movements (pen up) act as separators between paths.
    
    G-code format:
    - G0: Rapid movement (pen up) - separator
    - G1: Linear movement (pen down) - part of path
    - X, Y coordinates in millimeters
    """
    paths = []
    current_path_points = []
    current_path_gcode = []
    pen_down = False
    
    with open(gcode_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            # Parse G-code command
            parts = line.split()
            if not parts:
                continue
                
            command = parts[0].upper()
            
            if command == 'G0':  # Rapid movement (pen up) - separator
                if pen_down and current_path_points:
                    # End current path
                    if len(current_path_points) >= 1:
                        path_array = np.array(current_path_points, dtype=np.float32)
                        gcode_text = '\n'.join(current_path_gcode)
                        paths.append((path_array, gcode_text))
                    current_path_points = []
                    current_path_gcode = []
                pen_down = False
                
            elif command == 'G1':  # Linear movement (pen down)
                if not pen_down:
                    # Start new path
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
            
            # Add point to current path if we have coordinates
            if x_coord is not None and y_coord is not None:
                point = np.array([x_coord, y_coord], dtype=np.float32)
                current_path_points.append(point)
    
    # Handle final path
    if pen_down and current_path_points and len(current_path_points) >= 1:
        path_array = np.array(current_path_points, dtype=np.float32)
        gcode_text = '\n'.join(current_path_gcode)
        paths.append((path_array, gcode_text))
    
    return paths

def coords_from_path(scene: Scene, path_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample path evenly along the path and convert to pixel and meter coordinates."""
    # Load design image to get dimensions for pixel coordinate calculation
    if scene.design_img_path and os.path.exists(scene.design_img_path):
        with Image.open(scene.design_img_path) as img:
            img_width, img_height = img.size
    else:
        # Fallback to default dimensions if image not available
        img_width, img_height = 1000, 1000
        log.warning(f"Design image not found at {scene.design_img_path}, using default dimensions")
    
    # Calculate cumulative distances along the path
    if len(path_array) < 2:
        # Single point or empty path
        metric_coords = np.tile(path_array[0] if len(path_array) == 1 else np.array([0.0, 0.0]), (scene.stroke_length, 1))
    else:
        # Calculate distances between consecutive points
        diffs = np.diff(path_array, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        total_length = np.sum(segment_lengths)
        
        if total_length == 0:
            # Path has no length, use first point
            metric_coords = np.tile(path_array[0], (scene.stroke_length, 1))
        else:
            # Resample evenly along the path
            distances = np.linspace(0, total_length, scene.stroke_length)
            metric_coords = np.zeros((scene.stroke_length, 2), dtype=np.float32)
            
            current_segment = 0
            accumulated_length = 0.0
            
            for i, target_distance in enumerate(distances):
                # Find which segment this distance falls into
                while (current_segment < len(segment_lengths) and 
                        accumulated_length + segment_lengths[current_segment] < target_distance):
                    accumulated_length += segment_lengths[current_segment]
                    current_segment += 1
                
                if current_segment >= len(segment_lengths):
                    # Past the end of the path, use last point
                    metric_coords[i] = path_array[-1]
                else:
                    # Interpolate within the current segment
                    segment_start = path_array[current_segment]
                    segment_end = path_array[current_segment + 1]
                    segment_progress = (target_distance - accumulated_length) / segment_lengths[current_segment]
                    metric_coords[i] = segment_start + segment_progress * (segment_end - segment_start)
    
    # Convert metric coordinates (mm) to pixel coordinates
    # Assuming the G-code coordinates span the full image dimensions
    # Find the bounding box of the G-code coordinates to determine the scale
    min_x, min_y = np.min(metric_coords, axis=0)
    max_x, max_y = np.max(metric_coords, axis=0)
    gcode_width = max_x - min_x
    gcode_height = max_y - min_y
    
    # Calculate scale factors
    scale_x = img_width / gcode_width if gcode_width > 0 else 1.0
    scale_y = img_height / gcode_height if gcode_height > 0 else 1.0
    
    # Convert to pixel coordinates
    pixel_coords = np.zeros_like(metric_coords)
    pixel_coords[:, 0] = (metric_coords[:, 0] - min_x) * scale_x
    pixel_coords[:, 1] = (metric_coords[:, 1] - min_y) * scale_y
    
    # Convert mm to meters for the robot
    meter_coords = np.hstack([metric_coords / 1000.0, np.zeros((scene.stroke_length, 1), dtype=np.float32)])  # z axis is 0
    
    return pixel_coords, meter_coords


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
    arm_color = COLORS["blue"] if arm == "left" else COLORS["purple"]  # Blue for left, Purple for right
    
    # Draw the complete path in red
    for i in range(len(pixel_coords) - 1):
        pt1 = (int(pixel_coords[i, 0]), int(pixel_coords[i, 1]))
        pt2 = (int(pixel_coords[i + 1, 0]), int(pixel_coords[i + 1, 1]))
        cv2.line(frame_img, pt1, pt2, path_color, 2)
    
    # Draw points along the path
    for px, py in pixel_coords:
        cv2.circle(frame_img, (int(px), int(py)), 3, path_color, -1)
    
    # Add text indicating the pen and arm
    text = f"{arm.upper()} - {pen_name}"
    cv2.putText(frame_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["black"], 2)
    
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

    pen_paths_l: list[tuple[str, np.ndarray, str]] = []
    pen_paths_r: list[tuple[str, np.ndarray, str]] = []
    for pen_name, gcode_path in gcode_pens.items():
        assert pen_name in scene.pens_config, f"❌ Pen {pen_name} not found in pens config"
        log.info(f"Processing gcode file at: {gcode_path}")
        paths = parse_gcode_file(gcode_path)
        log.info(f"Found {len(paths)} paths")
        for path_array, gcode_text in paths:
            if len(path_array) < 1:
                log.warning(f"❌ Path has no points, skipping")
                continue
            if pen_name in scene.pen_names_l:
                pen_paths_l.append((pen_name, path_array, gcode_text))
            if pen_name in scene.pen_names_r:
                pen_paths_r.append((pen_name, path_array, gcode_text))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        raise ValueError("No paths found for left or right arm")

    # Generate frame images for each stroke
    log.info("Generating frame images for strokes...")
    stroke_frame_paths = {}  # (pen_name, arm, path_idx) -> frame_path
    
    # Generate frames for left arm paths
    for path_idx, (pen_name, path_array, _) in enumerate(pen_paths_l):
        pixel_coords, _ = coords_from_path(scene, path_array)
        frame_path = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "left")
        stroke_frame_paths[(pen_name, "left", path_idx)] = frame_path
        log.debug(f"Generated frame for left arm {pen_name} path {path_idx}: {frame_path}")
    
    # Generate frames for right arm paths
    for path_idx, (pen_name, path_array, _) in enumerate(pen_paths_r):
        pixel_coords, _ = coords_from_path(scene, path_array)
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
            pixel_coords, meter_coords = coords_from_path(scene, path_l)
            frame_path = stroke_frame_paths.get((color_l, "left", ptr_l))
            stroke_l = Stroke(
                description=f"left arm stroke after inkdip in {inkcap_name_l}",
                arm="left",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_l,
                dt=dt,
                gcode_text=pen_paths_l[ptr_l][2],
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
            pixel_coords, meter_coords = coords_from_path(scene, path_r)
            frame_path = stroke_frame_paths.get((color_r, "right", ptr_r))
            stroke_r = Stroke(
                description=f"right arm stroke after inkdip in {inkcap_name_r}",
                arm="right",
                pixel_coords=pixel_coords,
                ee_pos=meter_coords,
                ee_rot=ee_rot_r,
                dt=dt,
                gcode_text=pen_paths_r[ptr_r][2],
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