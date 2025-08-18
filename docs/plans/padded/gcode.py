import os
import re

import cv2
import numpy as np
from PIL import Image

from tatbot.data.scene import Scene
from tatbot.data.stroke import Stroke, StrokeList
from tatbot.gen.analyze import analyze_design_files
from tatbot.gen.inkdip import make_inkdip_func
from tatbot.utils.colors import COLORS, argb_to_bgr
from tatbot.utils.log import get_logger

log = get_logger("gen.gcode", "⚙️")


def parse_gcode_file(
    gcode_path: str,
    scene: Scene,
    spacing_m: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Parse a G-code file into a list of strokes:
        (meter_coords, pixel_coords, gcode_text)

    *  G-code coordinates are millimetres, origin at image centre.
    Depending on configuration, each stroke is re-sampled by either:
    - fixed count: `scene.stroke_length` points (legacy behaviour), or
    - fixed spacing: approximately `spacing_m` meters between points, producing
      a variable number of points which will be padded later for batching.
    """

    # ----------------------------------------------------------------—— helpers
    def mm2m(arr_mm: np.ndarray) -> np.ndarray:
        return arr_mm / 1_000.0

    def resample_path_by_count(points: np.ndarray, n: int) -> np.ndarray:
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

    def resample_path_by_spacing(points: np.ndarray, spacing: float) -> np.ndarray:
        """
        Even-arc-length resampling of an (N, D) polyline to ~uniform spacing.
        Returns at least 2 points when possible; repeats when degenerate.
        """
        if len(points) == 0:
            return points.astype(np.float32)
        if len(points) == 1:
            return np.repeat(points, 2, axis=0).astype(np.float32)

        seg = np.diff(points, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.insert(np.cumsum(seg_len), 0, 0.0)
        total = float(cum_len[-1])
        if total <= 1e-9:
            return np.repeat(points[:1], 2, axis=0).astype(np.float32)

        # number of samples ensures max gap <= spacing
        n = max(2, int(np.ceil(total / max(spacing, 1e-6))) + 1)
        target = np.linspace(0.0, total, n)
        resampled = np.zeros((n, points.shape[1]), dtype=np.float32)

        seg_idx = 0
        for i, t in enumerate(target):
            while seg_idx < len(cum_len) - 2 and t > cum_len[seg_idx + 1]:
                seg_idx += 1
            t0, t1 = cum_len[seg_idx], cum_len[seg_idx + 1]
            if t1 == t0:
                resampled[i] = points[seg_idx]
            else:
                alpha = (t - t0) / (t1 - t0)
                resampled[i] = (1 - alpha) * points[seg_idx] + alpha * points[seg_idx + 1]
        return resampled

    # ----------------------------------------------------------------—— image info
    img_w_m: float = scene.skin.image_width_m
    img_h_m: float = scene.skin.image_height_m

    assert scene.design_img_path and os.path.exists(scene.design_img_path), f"Design image not found at {scene.design_img_path}"
    
    with Image.open(scene.design_img_path) as _im:
        img_w_px, img_h_px = _im.size

    # ----------------------------------------------------------------—— parsing state
    paths: list[tuple[np.ndarray, np.ndarray, str]] = []
    cur_pts_mm: list[list[float]] = []
    cur_gcode: list[str] = []
    pen_down = False
    n_target = scene.stroke_length  # convenience for fixed-count mode

    def _flush_current():
        """Convert current path to robot/pixel coords and append to *paths*."""
        if not pen_down or len(cur_pts_mm) < 1:
            return

        pts_mm = np.asarray(cur_pts_mm, dtype=np.float32)
        pts_m = mm2m(pts_mm)  # (N,2)  centre-origin
        # Resample by either fixed count or fixed spacing
        if spacing_m is not None:
            pts_m = resample_path_by_spacing(pts_m, spacing_m)
            n_eff = len(pts_m)
        else:
            pts_m = resample_path_by_count(pts_m, n_target)
            n_eff = n_target

        # ─ Robot space (metres, z=0) ──────────────────────────────────────────
        meter_coords = np.hstack([pts_m, np.zeros((n_eff, 1), dtype=np.float32)])

        # ─ Pixel space (row, col) ─────────────────────────────────────────────
        col = (pts_m[:, 0] + img_w_m / 2) / img_w_m * img_w_px
        row = (img_h_m / 2 - pts_m[:, 1]) / img_h_m * img_h_px  # flip Y

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
    
    # Determine sampling mode and spacing
    mode = getattr(scene, "stroke_sample_mode", "fixed")
    spacing_m: float | None = getattr(scene, "stroke_point_spacing_m", None)
    auto_config = getattr(scene, "auto_sample_config", None)
    
    if mode not in ("fixed", "distance", "auto"):
        log.warning(f"Unknown stroke_sample_mode '{mode}', defaulting to 'fixed'")
        mode = "fixed"

    # Handle auto-detection mode
    if mode == "auto":
        log.info("Auto-detecting optimal spacing parameters...")
        # Analyze the design first
        gcode_paths = list(gcode_pens.values())
        stats = analyze_design_files(gcode_paths)
        
        # Apply auto-config constraints if provided
        if auto_config:
            min_spacing = auto_config.get("min_spacing_m", 0.0005)
            max_spacing = auto_config.get("max_spacing_m", 0.010)
            spacing_m = np.clip(stats.suggested_spacing_m, min_spacing, max_spacing)
            
            # Adjust max_points if needed
            if hasattr(scene, 'stroke_max_points'):
                scene.stroke_max_points = min(scene.stroke_max_points, stats.suggested_max_points)
        else:
            spacing_m = stats.suggested_spacing_m
            if hasattr(scene, 'stroke_max_points'):
                scene.stroke_max_points = stats.suggested_max_points
        
        # Switch to distance mode with detected spacing
        mode = "distance"
        scene.stroke_point_spacing_m = spacing_m
        log.info(f"Auto-detected spacing: {spacing_m:.4f}m, complexity: {stats.complexity_score:.2f}")
    
    elif mode == "distance":
        if spacing_m is None:
            raise ValueError("stroke_sample_mode='distance' requires stroke_point_spacing_m")
    else:
        # fixed mode: leave spacing_m as None to use fixed-count resampling
        spacing_m = None

    # Parse with chosen spacing/mode
    for pen_name, gcode_path in gcode_pens.items():
        assert pen_name in scene.pens_config, f"❌ Pen {pen_name} not found in pens config"
        log.info(f"Processing gcode file at: {gcode_path}")
        paths = parse_gcode_file(gcode_path, scene, spacing_m=spacing_m)
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

    # Determine padded stroke length (mode-dependent)
    def _compute_max_len() -> int:
        _max_len = 0
        for _, m, _, _ in pen_paths_l:
            _max_len = max(_max_len, int(m.shape[0]))
        for _, m, _, _ in pen_paths_r:
            _max_len = max(_max_len, int(m.shape[0]))
        return _max_len

    max_len = _compute_max_len()
    if max_len <= 0:
        raise ValueError("No valid paths found in design")

    if mode == "distance":
        # If exceeding cap, increase spacing and re-parse once
        cap = int(getattr(scene, "stroke_max_points", 256))
        if max_len > cap:
            scale = float(max_len) / float(cap)
            if spacing_m is not None:  # Type check to satisfy mypy
                spacing_m *= scale
                log.info(
                    f"Max stroke len {max_len} exceeds cap {cap}; increasing spacing by x{scale:.2f} to {spacing_m:.4f}m and re-parsing"
                )
                
                # Clear paths and frame images
                pen_paths_l.clear()
                pen_paths_r.clear()
                stroke_frame_paths.clear()
                
                # Re-parse with new spacing
                for pen_name, gcode_path in gcode_pens.items():
                    paths = parse_gcode_file(gcode_path, scene, spacing_m=spacing_m)
                    for meter_coords, pixel_coords, gcode_text in paths:
                        if len(meter_coords) < 1:
                            continue
                        if pen_name in scene.pen_names_l:
                            pen_paths_l.append((pen_name, meter_coords, pixel_coords, gcode_text))
                        if pen_name in scene.pen_names_r:
                            pen_paths_r.append((pen_name, meter_coords, pixel_coords, gcode_text))
                
                max_len = _compute_max_len()
                
                # Regenerate frame images with new pixel coordinates
                log.info("Regenerating frame images with adjusted spacing...")
                for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_l):
                    pen_color = argb_to_bgr(scene.pens_config[pen_name]["argb"])
                    fp = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "left", pen_color)
                    stroke_frame_paths[(pen_name, "left", path_idx)] = fp
                for path_idx, (pen_name, _, pixel_coords, _) in enumerate(pen_paths_r):
                    pen_color = argb_to_bgr(scene.pens_config[pen_name]["argb"])
                    fp = generate_stroke_frame_image(scene, pen_name, path_idx, pixel_coords, "right", pen_color)
                    stroke_frame_paths[(pen_name, "right", path_idx)] = fp
                log.info(f"Regenerated {len(stroke_frame_paths)} frame images")

        # Set padded length for batching/visualization
        final_length = int(min(max_len, cap))
        if hasattr(scene, 'stroke_length'):
            scene.stroke_length = final_length
        log.info(f"Set stroke_length to {final_length} for distance mode")
    else:
        # fixed: keep configured scene.stroke_length untouched
        pass

    # left arm and right arm strokes in order of execution on robot
    strokelist: StrokeList = StrokeList(strokes=[])

    inkdip_func = make_inkdip_func(scene)

    # start with inkdip on left arm, right arm will be at rest
    first_color_l = pen_paths_l[0][0]
    _inkdip_stroke = inkdip_func(first_color_l, "left")
    rest_length = scene.stroke_length if mode == "fixed" else max_len
    strokelist.strokes.append(
        (
            _inkdip_stroke,
            Stroke(
                description="right arm at rest",
                meter_coords=np.zeros((rest_length, 3)),
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
            rest_length = scene.stroke_length if mode == "fixed" else max_len
            stroke_l = Stroke(
                description="left arm at rest",
                meter_coords=np.zeros((rest_length, 3)),
                arm="left",
                is_rest=True,
            )
        elif inkcap_name_l is not None:
            meter_coords = pen_paths_l[ptr_l][1]
            pixel_coords = pen_paths_l[ptr_l][2]
            gcode_text = pen_paths_l[ptr_l][3]
            frame_path = stroke_frame_paths.get((color_l, "left", ptr_l))
            
            # Calculate arc length and metadata
            arc_length = 0.0
            if len(meter_coords) > 1:
                segments = np.diff(meter_coords, axis=0)
                arc_length = float(np.linalg.norm(segments, axis=1).sum())
            
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
                actual_points=len(meter_coords),
                arc_length_m=arc_length,
                point_spacing_m=spacing_m if spacing_m else None,
            )
            inkcap_name_l = None  # inkdip on next stroke
            ptr_l += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_l is not None:
                stroke_l = inkdip_func(color_l, "left")
                inkcap_name_l = stroke_l.inkcap
            else:
                rest_length = scene.stroke_length if mode == "fixed" else max_len
                stroke_l = Stroke(
                    description="left arm at rest",
                    meter_coords=np.zeros((rest_length, 3)),
                    arm="left",
                    is_rest=True,
                )

        # RIGHT ARM
        if path_r is None:
            rest_length = scene.stroke_length if mode == "fixed" else max_len
            stroke_r = Stroke(
                description="right arm at rest",
                meter_coords=np.zeros((rest_length, 3)),
                arm="right",
                is_rest=True,
            )
        elif inkcap_name_r is not None:
            meter_coords = pen_paths_r[ptr_r][1]
            pixel_coords = pen_paths_r[ptr_r][2]
            gcode_text = pen_paths_r[ptr_r][3]
            frame_path = stroke_frame_paths.get((color_r, "right", ptr_r))
            
            # Calculate arc length and metadata
            arc_length = 0.0
            if len(meter_coords) > 1:
                segments = np.diff(meter_coords, axis=0)
                arc_length = float(np.linalg.norm(segments, axis=1).sum())
            
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
                actual_points=len(meter_coords),
                arc_length_m=arc_length,
                point_spacing_m=spacing_m if spacing_m else None,
            )
            inkcap_name_r = None  # inkdip on next stroke
            ptr_r += 1
        else:
            # Only perform inkdip if a stroke will follow
            if path_r is not None:
                stroke_r = inkdip_func(color_r, "right")
                inkcap_name_r = stroke_r.inkcap
            else:
                rest_length = scene.stroke_length if mode == "fixed" else max_len
                stroke_r = Stroke(
                    description="right arm at rest",
                    meter_coords=np.zeros((rest_length, 3)),
                    arm="right",
                    is_rest=True,
                )
        strokelist.strokes.append((stroke_l, stroke_r))

    return strokelist
