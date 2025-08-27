import os
import re
import json

import cv2
import numpy as np
from PIL import Image

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.colors import COLORS, argb_to_bgr
from tatbot.utils.log import get_logger

log = get_logger("gen.gcode", "⚙️")


def parse_gcode_file(gcode_path: str, scene: Scene) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Parse a G-code file into a list of strokes:
        (meter_coords, pixel_coords, gcode_text)

    *  G-code coordinates are millimetres, origin at image centre.
    *  `meter_coords` → shape (scene.stroke_length, 3)
    *  `pixel_coords` → shape (scene.stroke_length, 2)  (row, col)

    Each stroke is **re-sampled** so that *all* strokes have exactly
    `scene.stroke_length` points, evenly spaced along arc-length.
    """

    # ----------------------------------------------------------------—— helpers
    def mm2m(arr_mm: np.ndarray) -> np.ndarray:
        return arr_mm / 1_000.0

    def resample_path(points: np.ndarray, n: int) -> np.ndarray:
        """
        Even-arc-length resampling of an (N, D) polyline to *n* points.
        If N == 1 the single point is repeated.
        """
        if len(points) == 1:
            return np.repeat(points, n, axis=0)

        # cumulative arc-length
        seg = np.diff(points, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.insert(np.cumsum(seg_len), 0, 0.0)

        # target arc-lengths
        target = np.linspace(0.0, cum_len[-1], n)
        resampled = np.zeros((n, points.shape[1]), dtype=np.float32)

        seg_idx = 0
        for i, t in enumerate(target):
            while seg_idx < len(cum_len) - 2 and t > cum_len[seg_idx + 1]:
                seg_idx += 1

            t0, t1 = cum_len[seg_idx], cum_len[seg_idx + 1]
            if t1 == t0:  # duplicate points
                resampled[i] = points[seg_idx]
            else:
                alpha = (t - t0) / (t1 - t0)
                resampled[i] = (1 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
        return resampled

    # ----------------------------------------------------------------—— image info
    img_w_m: float = scene.skin.image_width_m
    img_h_m: float = scene.skin.image_height_m

    assert scene.design_img_path and os.path.exists(scene.design_img_path), (
        f"Design image not found at {scene.design_img_path}"
    )

    with Image.open(scene.design_img_path) as _im:
        img_w_px, img_h_px = _im.size

    # Prefer design canvas size from a .drawingbotv3 project if present.
    # DBV3 often uses a different canvas size (in mm) than the skin config,
    # and the preview PNG is rendered using that canvas. Using the skin size
    # here causes scale mismatch when overlaying strokes on the preview image.
    try:
        if scene.design_dir and os.path.isdir(scene.design_dir):
            dbv3_files = [f for f in os.listdir(scene.design_dir) if f.endswith(".drawingbotv3")]
            if len(dbv3_files) > 1:
                raise ValueError(
                    f"Multiple .drawingbotv3 files found in {scene.design_dir}: {dbv3_files}. "
                    "Please keep only one per design directory."
                )
            if len(dbv3_files) == 1:
                dbv3_path = os.path.join(scene.design_dir, dbv3_files[0])
                with open(dbv3_path, "r") as fh:
                    dbv3 = json.load(fh)
                canvas = (
                    dbv3
                    .get("data", {})
                    .get("settings", {})
                    .get("drawingState", {})
                    .get("canvas", {})
                )
                # Units conversion to millimetres
                units = str(canvas.get("units", "MILLIMETRES")).upper()
                unit_to_mm = {
                    "MILLIMETRES": 1.0,
                    "MM": 1.0,
                    "CENTIMETRES": 10.0,
                    "CM": 10.0,
                    "INCHES": 25.4,
                    "IN": 25.4,
                }
                conv = unit_to_mm.get(units)
                if conv is None:
                    log.warning(f"⚠️ Unknown DBV3 canvas units '{units}'; falling back to skin size")
                else:
                    drawing_w = float(canvas.get("drawingWidth", 0.0))
                    drawing_h = float(canvas.get("drawingHeight", 0.0))
                    drawing_w_mm = drawing_w * conv
                    drawing_h_mm = drawing_h * conv
                    if drawing_w_mm > 0 and drawing_h_mm > 0:
                        img_w_m = drawing_w_mm / 1000.0
                        img_h_m = drawing_h_mm / 1000.0
                        log.info(
                            "Using DBV3 canvas for pixel mapping: "
                            f"{drawing_w_mm:.1f}x{drawing_h_mm:.1f} mm (units={units})"
                        )
                    else:
                        log.warning(
                            "⚠️ DBV3 canvas has non-positive dimensions; falling back to skin size"
                        )
    except Exception as e:
        log.warning(f"⚠️ Failed to use .drawingbotv3 canvas; using skin size: {e}")

    # Warn on aspect ratio mismatch between the background PNG and the chosen canvas (DBV3 or skin)
    try:
        img_aspect = img_w_px / img_h_px
        canvas_aspect = img_w_m / img_h_m if (img_w_m > 0 and img_h_m > 0) else None
        if canvas_aspect is None:
            log.warning("⚠️ Canvas dimensions are non-positive; aspect ratio check skipped")
        else:
            if abs(img_aspect - canvas_aspect) > 1e-6:
                log.warning(
                    "⚠️ Background PNG aspect ratio does not match canvas: "
                    f"png={img_w_px}x{img_h_px} (aspect={img_aspect:.4f}) vs "
                    f"canvas={img_w_m*1000:.0f}x{img_h_m*1000:.0f} mm (aspect={canvas_aspect:.4f})"
                )
    except Exception as e:
        log.warning(f"⚠️ Failed aspect ratio check: {e}")

    # ----------------------------------------------------------------—— parsing state
    paths: list[tuple[np.ndarray, np.ndarray, str]] = []
    cur_pts_mm: list[list[float]] = []
    cur_gcode: list[str] = []
    pen_down = False
    n_target = scene.stroke_length  # convenience

    def _flush_current():
        """Convert current path to robot/pixel coords and append to *paths*."""
        if not pen_down or len(cur_pts_mm) < 1:
            return

        pts_mm = np.asarray(cur_pts_mm, dtype=np.float32)
        pts_m = mm2m(pts_mm)  # (N,2)  centre-origin
        pts_m = resample_path(pts_m, n_target)  # uniform length

        # ─ Robot space (metres, z=0) ──────────────────────────────────────────
        meter_coords = np.hstack([pts_m, np.zeros((n_target, 1), dtype=np.float32)])

        # ─ Pixel space (row, col) ─────────────────────────────────────────────
        # Map meter coordinates to pixel coordinates
        # Use actual image dimensions for proper scaling to avoid misalignment
        # The issue is that if image aspect ratio doesn't match skin config,
        # the coordinates will be misaligned
        # Fix: Use actual image dimensions to determine correct scaling
        # Instead of using configured meter dimensions, use actual image dimensions
        # Calculate the correct scaling based on actual image dimensions
        scale_x = img_w_px / img_w_m
        scale_y = img_h_px / img_h_m
        col = (pts_m[:, 0] + img_w_m / 2) * scale_x
        row = (img_h_m / 2 - pts_m[:, 1]) * scale_y  # flip Y

        pixel_coords = np.stack([row, col], axis=1).astype(np.float32)
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, img_h_px - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, img_w_px - 1)

        paths.append((meter_coords, pixel_coords, "\n".join(cur_gcode)))

    # ----------------------------------------------------------------—— read file
    with open(gcode_path, "r") as fh:
        for line_num, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            tokens = line.split()
            cmd = tokens[0].upper()

            if cmd == "G0":  # pen up  (separator)
                _flush_current()
                cur_pts_mm.clear()
                cur_gcode.clear()
                pen_down = False

                x_mm = y_mm = None
                for tok in tokens[1:]:
                    if tok.startswith("X"):
                        x_mm = float(tok[1:])
                    elif tok.startswith("Y"):
                        y_mm = float(tok[1:])
                if x_mm is not None and y_mm is not None:
                    cur_pts_mm.append([x_mm, y_mm])
                    cur_gcode.append(line)

            elif cmd == "G1":  # pen down (draw)
                pen_down = True
                cur_gcode.append(line)

                x_mm = y_mm = None
                for tok in tokens[1:]:
                    if tok.startswith("X"):
                        x_mm = float(tok[1:])
                    elif tok.startswith("Y"):
                        y_mm = float(tok[1:])
                if x_mm is not None and y_mm is not None:
                    cur_pts_mm.append([x_mm, y_mm])

            else:
                log.warning(f"⚠️ Unrecognised G-code '{cmd}' on line {line_num}")

    _flush_current()
    return paths


def generate_stroke_frame_image(
    scene: Scene,
    pen_name: str,
    path_idx: int,
    pixel_coords: np.ndarray,
    arm: str,
    pen_color: tuple[int, int, int] | None = None,
) -> str:
    if not scene.design_img_path or not os.path.exists(scene.design_img_path):
        raise ValueError(f"Design image not found at {scene.design_img_path}")

    base_img = cv2.imread(scene.design_img_path)
    if base_img is None:
        raise ValueError(f"Could not read design image at {scene.design_img_path}")

    frame_img = base_img.copy()

    # Use provided pen color or fall back to red
    if pen_color is not None:
        path_color = pen_color
    else:
        path_color = COLORS["red"]

    # poly-line ---------------------------------------------------------------
    for i in range(len(pixel_coords) - 1):
        r1, c1 = pixel_coords[i]
        r2, c2 = pixel_coords[i + 1]
        cv2.line(frame_img, (int(c1), int(r1)), (int(c2), int(r2)), path_color, 2)

    # nodes -------------------------------------------------------------------
    for r, c in pixel_coords:
        cv2.circle(frame_img, (int(c), int(r)), 3, path_color, -1)

    frame_filename = f"stroke_{pen_name}_{arm}_{path_idx:04d}.png"
    frame_path = os.path.join(scene.design_dir, frame_filename)
    cv2.imwrite(frame_path, frame_img)
    return frame_path


def make_gcode_strokes(scene: Scene) -> StrokeList:
    assert scene.design_dir is not None, "❌ Design directory is not set, does this scene have a design?"
    gcode_files = []
    gcode_pens: dict[str, str] = {}
    for file in os.listdir(scene.design_dir):
        if file.endswith(".gcode"):
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
                log.warning("❌ Path has no points, skipping")
                continue
            if pen_name in scene.pen_names_l:
                pen_paths_l.append((pen_name, meter_coords, pixel_coords, gcode_text))
            if pen_name in scene.pen_names_r:
                pen_paths_r.append((pen_name, meter_coords, pixel_coords, gcode_text))

    if len(pen_paths_l) == 0 or len(pen_paths_r) == 0:
        raise ValueError("No paths found for left or right arm")

    # Clear existing stroke frame images
    log.info("Clearing existing stroke frame images...")
    for file in os.listdir(scene.design_dir):
        if file.startswith("stroke_") and file.endswith(".png"):
            file_path = os.path.join(scene.design_dir, file)
            try:
                os.remove(file_path)
                log.debug(f"Removed existing frame: {file}")
            except OSError as e:
                log.warning(f"Could not remove {file}: {e}")

    # Generate frame images for each stroke
    log.info("Generating frame images for strokes...")
    stroke_frame_paths = {}
    for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_l):
        pen_color = argb_to_bgr(scene.pens_config[pen_name]["argb"])
        fp = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "left", pen_color)
        stroke_frame_paths[(pen_name, "left", path_idx)] = fp
    for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_r):
        pen_color = argb_to_bgr(scene.pens_config[pen_name]["argb"])
        fp = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "right", pen_color)
        stroke_frame_paths[(pen_name, "right", path_idx)] = fp
    log.info(f"Generated {len(stroke_frame_paths)} frame images")

    # left arm and right arm strokes in order of execution on robot
    strokelist: StrokeList = StrokeList(strokes=[])

    inkdip_func = make_inkdip_func(scene)

    # start with inkdip on left arm, right arm will be at rest
    first_color_l = pen_paths_l[0][0]
    _inkdip_stroke = inkdip_func(first_color_l, "left")
    strokelist.strokes.append(
        (
            _inkdip_stroke,
            Stroke(
                description="right arm at rest",
                meter_coords=np.zeros((scene.stroke_length, 3)),
                arm="right",
                is_rest=True,
            ),
        )
    )
    inkcap_name_l = _inkdip_stroke.inkcap
    inkcap_name_r = None  # when None, indicates that an inkdip stroke is required
    ptr_l = ptr_r = 0
    num_paths_l = len(pen_paths_l)
    num_paths_r = len(pen_paths_r)
    log.info(f"Left arm has {num_paths_l} paths, right arm has {num_paths_r} paths")
    for _ in range(num_paths_l + num_paths_r):
        path_l = pen_paths_l[ptr_l][1] if ptr_l < len(pen_paths_l) else None
        path_r = pen_paths_r[ptr_r][1] if ptr_r < len(pen_paths_r) else None
        color_l = pen_paths_l[ptr_l][0] if ptr_l < len(pen_paths_l) else None
        color_r = pen_paths_r[ptr_r][0] if ptr_r < len(pen_paths_r) else None

        # LEFT ARM
        if path_l is None:
            stroke_l = Stroke(
                description="left arm at rest",
                meter_coords=np.zeros((scene.stroke_length, 3)),
                arm="left",
                is_rest=True,
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
                meter_coords=meter_coords,
                gcode_text=gcode_text,
                inkcap=inkcap_name_l,
                is_inkdip=False,
                frame_path=frame_path,
                color=color_l,
            )
            inkcap_name_l = None  # inkdip on next stroke
            ptr_l += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_l is not None:
                stroke_l = inkdip_func(color_l, "left")
                inkcap_name_l = stroke_l.inkcap
            else:
                stroke_l = Stroke(
                    description="left arm at rest",
                    meter_coords=np.zeros((scene.stroke_length, 3)),
                    arm="left",
                    is_rest=True,
                )

        # RIGHT ARM
        if path_r is None:
            stroke_r = Stroke(
                description="right arm at rest",
                meter_coords=np.zeros((scene.stroke_length, 3)),
                arm="right",
                is_rest=True,
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
                meter_coords=meter_coords,
                gcode_text=gcode_text,
                inkcap=inkcap_name_r,
                is_inkdip=False,
                frame_path=frame_path,
                color=color_r,
            )
            inkcap_name_r = None  # inkdip on next stroke
            ptr_r += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_r is not None:
                stroke_r = inkdip_func(color_r, "right")
                inkcap_name_r = stroke_r.inkcap
            else:
                stroke_r = Stroke(
                    description="right arm at rest",
                    meter_coords=np.zeros((scene.stroke_length, 3)),
                    arm="right",
                    is_rest=True,
                )
        strokelist.strokes.append((stroke_l, stroke_r))

    return strokelist
