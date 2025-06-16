from dataclasses import dataclass, field
import json
import math
import os

import cv2
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from _path import Pose, Path, Pattern, make_pathviz_image, make_pathlen_image
from _log import setup_log_with_config, get_logger

log = get_logger('pattern_calib')

@dataclass
class CalibrationPatternConfig:
    debug: bool = False
    """Enable debug logging."""
    output_dir: str = os.path.expanduser("~/tatbot/output/patterns/calibration")
    """Directory to save the calibration pattern and paths."""
    image_width_px: int = 256
    """Width of the calibration pattern image in pixels."""
    image_height_px: int = 256
    """Height of the calibration pattern image in pixels."""
    image_width_m: float = 0.04
    """Width of the calibration pattern image in meters."""
    image_height_m: float = 0.04
    """Height of the calibration pattern image in meters."""
    grid_cols: int = 2
    """Number of columns in the calibration pattern grid."""
    grid_rows: int = 2
    """Number of rows in the calibration pattern grid."""
    background_color: str = "white"
    """Background color of the calibration pattern image."""
    thickness: int = 4
    """Thickness of the calibration pattern lines."""

@dataclass
class VerticalLineConfig:
    """Parameters for a vertical line."""
    length: int = 120
    num_points: int = 64
    """Number of points to generate for the path."""

@dataclass
class HorizontalLineConfig:
    """Parameters for a horizontal line."""
    length: int = 120
    num_points: int = 64
    """Number of points to generate for the path."""

@dataclass
class CircleConfig:
    """Parameters for a full circle."""
    radius: int = 64
    num_points: int = 64
    """Number of points to generate for the path."""

@dataclass
class WaveConfig:
    """Parameters for a continuous wavy line."""
    amplitude: int = 16
    frequency: float = 0.08
    length: int = 108
    num_points: int = 64
    """Number of points to generate for the path."""

@dataclass
class VerticalLineGroupConfig:
    """Parameters for a group of parallel vertical lines."""
    num_lines: int = 5
    spacing: int = 16
    """pixels between adjacent line center-lines"""
    num_points: list[int] = field(default_factory=lambda: [16, 16, 32, 32, 64])
    length: list[int] = field(default_factory=lambda: [16, 32, 48, 64, 64])

@dataclass
class HorizontalLineGroupConfig:
    """Parameters for a group of parallel horizontal lines."""
    num_lines: int = 5
    spacing: int = 16
    num_points: list[int] = field(default_factory=lambda: [16, 16, 32, 32, 64])
    length: list[int] = field(default_factory=lambda: [16, 32, 48, 64, 64])

@dataclass
class CircleGroupConfig:
    """Parameters for a group of concentric circles."""
    radii: list[int] = field(default_factory=lambda: [16, 32, 48])
    num_points: list[int] = field(default_factory=lambda: [16, 32, 48])

@dataclass
class WaveGroupConfig:
    """Parameters for a group of parallel waveforms."""
    num_waves: int = 3
    spacing: int = 24
    num_points: list[int] = field(default_factory=lambda: [WaveConfig.num_points, WaveConfig.num_points + 20, WaveConfig.num_points + 40])
    amplitude: int = WaveConfig.amplitude
    frequency: float = WaveConfig.frequency
    length: int = WaveConfig.length

def linspace_points(p1: tuple[int, int], p2: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """Generates `n` equally spaced points between two points."""
    return [tuple(int(round(p1[i] + (p2[i] - p1[i]) * (j / (n - 1)))) for i in range(2)) for j in range(n)]

def generate_vertical_line_path(config: VerticalLineConfig, origin: tuple[int, int], thickness: int) -> list[tuple[int, int]]:
    """Generates `num_points` equally spaced points for a vertical line."""
    if config.num_points < 2:
        raise ValueError("num_points must be at least 2.")
    log.debug(f"Generating vertical line with {config.num_points} points and thickness {thickness}.")

    x0, y0 = origin
    p1 = (x0, y0 - config.length // 2)
    p2 = (x0, y0 + config.length // 2)
    return linspace_points(p1, p2, config.num_points)

def generate_horizontal_line_path(config: HorizontalLineConfig, origin: tuple[int, int], thickness: int) -> list[tuple[int, int]]:
    """Generates `num_points` equally spaced points for a horizontal line."""
    if config.num_points < 2:
        raise ValueError("num_points must be at least 2.")
    log.debug(f"Generating horizontal line with {config.num_points} points and thickness {thickness}.")

    x0, y0 = origin
    p1 = (x0 - config.length // 2, y0)
    p2 = (x0 + config.length // 2, y0)
    return linspace_points(p1, p2, config.num_points)

def generate_circle_path(config: CircleConfig, origin: tuple[int, int], thickness: int) -> list[tuple[int, int]]:
    """Generates `num_points` equally spaced points for a circle."""
    if config.num_points < 2:
        raise ValueError("num_points must be at least 2.")
    log.debug(f"Generating circle with {config.num_points} points and thickness {thickness}.")
    
    x0, y0 = origin
    points = []
    for i in range(config.num_points):
        angle = 2 * math.pi * i / config.num_points
        x = x0 + config.radius * math.cos(angle)
        y = y0 + config.radius * math.sin(angle)
        points.append((round(x), round(y)))
    return points

def generate_wave_path(config: WaveConfig, origin: tuple[int, int], thickness: int) -> list[tuple[int, int]]:
    """Generates `num_points` equally spaced points for a continuous horizontal wave."""
    if config.num_points < 2:
        raise ValueError("num_points must be at least 2.")
    log.debug(f"Generating wave with {config.num_points} points and thickness {thickness}.")

    x0, y0 = origin
    points = []
    for i in range(config.num_points):
        progress = i / (config.num_points - 1)
        x_offset = -config.length / 2 + config.length * progress
        y_offset = config.amplitude * math.sin(x_offset * config.frequency)
        points.append((round(x0 + x_offset), round(y0 + y_offset)))
    return points

# New helper functions to generate 3-part variants of each primitive shape

def generate_vertical_line_paths(config: VerticalLineGroupConfig, origin: tuple[int, int], thickness: int) -> list[list[tuple[int, int]]]:
    """Generates parallel vertical lines with specified point counts and lengths."""
    spacing = config.spacing
    x0, y0 = origin
    deltas = [int((i - (config.num_lines - 1) / 2) * spacing) for i in range(config.num_lines)]
    paths = []
    for i, x_offset in enumerate(deltas):
        num_points = config.num_points[i] if i < len(config.num_points) else config.num_points[-1]
        length = config.length[i] if i < len(config.length) else config.length[-1]
        p1 = (x0 + x_offset, y0 - length // 2)
        p2 = (x0 + x_offset, y0 + length // 2)
        paths.append(linspace_points(p1, p2, num_points))
    return paths


def generate_horizontal_line_paths(config: HorizontalLineGroupConfig, origin: tuple[int, int], thickness: int) -> list[list[tuple[int, int]]]:
    """Generates parallel horizontal lines with specified point counts and lengths."""
    spacing = config.spacing
    x0, y0 = origin
    deltas = [int((i - (config.num_lines - 1) / 2) * spacing) for i in range(config.num_lines)]
    paths = []
    for i, y_offset in enumerate(deltas):
        num_points = config.num_points[i] if i < len(config.num_points) else config.num_points[-1]
        length = config.length[i] if i < len(config.length) else config.length[-1]
        p1 = (x0 - length // 2, y0 + y_offset)
        p2 = (x0 + length // 2, y0 + y_offset)
        paths.append(linspace_points(p1, p2, num_points))
    return paths


def generate_circle_paths(config: CircleGroupConfig, origin: tuple[int, int], thickness: int) -> list[list[tuple[int, int]]]:
    """Generates three concentric circles with varying radii and point counts."""
    x0, y0 = origin
    paths = []
    radii = config.radii
    np_list = config.num_points if config.num_points else [16] * len(radii)
    for i, r in enumerate(radii):
        num_points = np_list[i] if i < len(np_list) else np_list[-1]
        circle_cfg = CircleConfig(radius=r, num_points=num_points)
        paths.append(generate_circle_path(circle_cfg, origin, thickness))
    return paths


def generate_wave_paths(config: WaveGroupConfig, origin: tuple[int, int], thickness: int) -> list[list[tuple[int, int]]]:
    """Generates parallel wave paths with specified point counts."""
    spacing = config.spacing
    x0, y0 = origin
    deltas = [int((i - (config.num_waves - 1) / 2) * spacing) for i in range(config.num_waves)]
    paths = []
    for i, y_offset in enumerate(deltas):
        num_points = config.num_points[i] if i < len(config.num_points) else config.num_points[-1]
        wave_cfg = WaveConfig(
            amplitude=config.amplitude,
            frequency=config.frequency,
            length=config.length,
            num_points=num_points,
        )
        paths.append(generate_wave_path(wave_cfg, (x0, y0 + y_offset), thickness))
    return paths


def make_calibration_pattern(config: CalibrationPatternConfig):
    # Create a blank canvas for visualization
    image_strokes = Image.new("RGB", (config.image_width_px, config.image_height_px), config.background_color)
    draw = ImageDraw.Draw(image_strokes)

    strokes_to_generate = [
        (generate_vertical_line_paths, VerticalLineGroupConfig()),
        (generate_horizontal_line_paths, HorizontalLineGroupConfig()),
        (generate_circle_paths, CircleGroupConfig()),
        (generate_wave_paths, WaveGroupConfig()),
    ]

    all_paths = []
    cell_width = config.image_width_px // config.grid_cols
    cell_height = config.image_height_px // config.grid_rows

    for i, (generate_func, stroke_config) in enumerate(strokes_to_generate):
        if i >= config.grid_cols * config.grid_rows:
            break

        col = i % config.grid_cols
        row = i // config.grid_cols
        cell_center_x = col * cell_width + cell_width // 2
        cell_center_y = row * cell_height + cell_height // 2
        
        paths = generate_func(stroke_config, (cell_center_x, cell_center_y), config.thickness)
        if not paths:
            continue

        for path in paths:
            if not path:
                continue
            all_paths.append(path)

            if len(path) > 1:
                is_curve = isinstance(stroke_config, (WaveGroupConfig, CircleGroupConfig))
                draw.line(path, fill="black", width=config.thickness, joint="curve" if is_curve else None)

    log.info(f"Generated {len(all_paths)} paths.")

    # Save visualization images
    strokes_path = os.path.join(config.output_dir, "image.png")
    image_strokes.save(strokes_path)
    log.info(f"🖼️ Saved stroke patterns image to {strokes_path}")

    # Convert paths to meters and save to JSON
    scale_x = config.image_width_m / config.image_width_px
    scale_y = config.image_height_m / config.image_height_px

    paths = []
    for path_px in all_paths:
        if not path_px:
            continue

        num_points = len(path_px)
        positions_list = [[p[0] * scale_x, p[1] * scale_y, 0.0] for p in path_px]

        paths.append(
            Path(
                positions=jnp.array(positions_list),
                orientations=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (num_points, 1)),
                pixel_coords=jnp.array(path_px, dtype=jnp.int32),
                metric_coords=jnp.zeros((num_points, 2)),
            )
        )

    pattern = Pattern(
        name="calibration",
        paths=paths,
        width_m=config.image_width_m,
        height_m=config.image_height_m,
        width_px=config.image_width_px,
        height_px=config.image_height_px,
    )

    # Generate and save path visualization
    path_viz_np = make_pathviz_image(pattern)
    path_viz_path = os.path.join(config.output_dir, "pathviz.png")
    cv2.imwrite(path_viz_path, path_viz_np)
    log.info(f"🖼️ Saved tool path visualization to {path_viz_path}")

    # Generate and save path length visualization
    pathlen_img = make_pathlen_image(pattern)
    pathlen_path = os.path.join(config.output_dir, "pathlen.png")
    cv2.imwrite(pathlen_path, pathlen_img)
    log.info(f"🖼️ Saved path length visualization to {pathlen_path}")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.ndarray, jnp.ndarray)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    paths_path = os.path.join(config.output_dir, "pattern.json")
    with open(paths_path, "w") as f:
        json_data = {
            "name": pattern.name,
            "width_m": pattern.width_m,
            "height_m": pattern.height_m,
            "width_px": pattern.width_px,
            "height_px": pattern.height_px,
            "paths": [],
        }
        for path in pattern.paths:
            # Convert JAX arrays to numpy arrays first to avoid slow iteration
            positions = np.asarray(path.positions)
            orientations = np.asarray(path.orientations)
            pixel_coords = np.asarray(path.pixel_coords)
            metric_coords = np.asarray(path.metric_coords)

            poses = [
                {
                    "pos": positions[i].tolist(),
                    "wxyz": orientations[i].tolist(),
                    "pixel_coords": pixel_coords[i].tolist(),
                    "metric_coords": metric_coords[i].tolist(),
                }
                for i in range(len(path))
            ]
            json_data["paths"].append({"poses": poses})
        json.dump(json_data, f, indent=4, cls=NumpyEncoder)
    log.info(f"💾 Saved {len(pattern.paths)} tool paths to {paths_path}")

if __name__ == "__main__":
    args = setup_log_with_config(CalibrationPatternConfig)
    make_calibration_pattern(args)