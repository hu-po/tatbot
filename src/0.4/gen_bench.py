from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from _log import get_logger, setup_log_with_config
from _plan import Plan
from _path import Stroke

log = get_logger('gen_bench')

@dataclass
class VerticalLineGroupConfig:
    """Parameters for a group of parallel vertical lines."""
    num_lines: int = 3
    spacing: int = 20
    """pixels between adjacent line center-lines"""
    length: tuple[int, ...] = (32, 48, 64)
    num_points: tuple[int, ...] = (32, 32, 64)
    """Number of individual points per line"""

@dataclass
class HorizontalLineGroupConfig:
    """Parameters for a group of parallel horizontal lines."""
    num_lines: int = 3
    spacing: int = 20
    length: tuple[int, ...] = (32, 48, 64)
    num_points: tuple[int, ...] = (32, 32, 64)

@dataclass
class CircleGroupConfig:
    """Parameters for a group of concentric circles."""
    radii: tuple[int, ...] = (16, 32, 48)
    num_points: tuple[int, ...] = (32, 64, 96)

@dataclass
class WaveGroupConfig:
    """Parameters for a group of parallel waveforms."""
    num_waves: int = 3
    spacing: int = 24
    amplitude: int = 16
    frequency: float = 0.08
    length: tuple[int, ...] = (108, 108, 108)
    num_points: tuple[int, ...] = (48, 64, 96)

@dataclass
class BenchmarkPlanConfig:
    debug: bool = False
    """Enable debug logging."""

    output_dir: str = os.path.expanduser("~/tatbot/output/plans/bench")
    """Directory to save the plan."""

    image_width_px: int = 256
    """Width of the image in pixels."""
    image_height_px: int = 256
    """Height of the image in pixels."""
    background_color: str = "white"
    """Background color of the image."""

    thickness: int = 4
    """Thickness of the lines."""
    

def linspace_points(p1: tuple[int, int], p2: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """Generates `n` equally spaced points between two points."""
    return [tuple(int(round(p1[i] + (p2[i] - p1[i]) * (j / (n - 1)))) for i in range(2)) for j in range(n)]


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
    """Generates concentric circles with varying radii and point counts."""
    x0, y0 = origin
    paths = []
    radii = config.radii
    np_list = config.num_points if config.num_points else [16] * len(radii)
    for i, r in enumerate(radii):
        num_points = np_list[i] if i < len(np_list) else np_list[-1]
        points = []
        for j in range(num_points):
            angle = 2 * math.pi * j / num_points
            x = x0 + r * math.cos(angle)
            y = y0 + r * math.sin(angle)
            points.append((round(x), round(y)))
        paths.append(points)
    return paths


def generate_wave_paths(config: WaveGroupConfig, origin: tuple[int, int], thickness: int) -> list[list[tuple[int, int]]]:
    """Generates parallel wave paths with specified point counts."""
    spacing = config.spacing
    x0, y0 = origin
    deltas = [int((i - (config.num_waves - 1) / 2) * spacing) for i in range(config.num_waves)]
    paths = []
    for i, y_offset in enumerate(deltas):
        num_points = config.num_points[i] if i < len(config.num_points) else config.num_points[-1]
        length = config.length[i] if i < len(config.length) else config.length[-1]
        amplitude = config.amplitude
        frequency = config.frequency
        points = []
        for j in range(num_points):
            progress = j / (num_points - 1)
            x_offset = -length / 2 + length * progress
            y_wave = amplitude * math.sin(x_offset * frequency)
            points.append((round(x0 + x_offset), round(y0 + y_offset + y_wave)))
        paths.append(points)
    return paths

def plan_from_calib(config: BenchmarkPlanConfig):
    plan = Plan(
        name="bench",
        dirpath=config.output_dir,
        image_width_px=config.image_width_px,
        image_height_px=config.image_height_px,
    )

    log.info(f"🖼️ Creating blank canvas of shape {config.image_width_px}x{config.image_height_px}")
    image = Image.new("RGB", (config.image_width_px, config.image_height_px), config.background_color)
    draw = ImageDraw.Draw(image)

    strokes_to_generate = [
        (generate_vertical_line_paths, VerticalLineGroupConfig()),
        (generate_horizontal_line_paths, HorizontalLineGroupConfig()),
        (generate_circle_paths, CircleGroupConfig()),
        (generate_wave_paths, WaveGroupConfig()),
    ]

    all_paths = []
    cell_width = config.image_width_px // 2
    cell_height = config.image_height_px // 2

    for i, (generate_func, stroke_config) in enumerate(strokes_to_generate):
        col = i % 2
        row = i // 2
        cell_center_x = col * cell_width + cell_width // 2
        cell_center_y = row * cell_height + cell_height // 2
        
        paths = generate_func(stroke_config, (cell_center_x, cell_center_y), config.thickness)
        if not paths:
            continue

        for j, path in enumerate(paths):
            if not path:
                continue
            # Build description string
            if isinstance(stroke_config, VerticalLineGroupConfig):
                length = stroke_config.length[j] if j < len(stroke_config.length) else stroke_config.length[-1]
                desc = f"vertical_line length={length} thickness={config.thickness} index={j}"
            elif isinstance(stroke_config, HorizontalLineGroupConfig):
                length = stroke_config.length[j] if j < len(stroke_config.length) else stroke_config.length[-1]
                desc = f"horizontal_line length={length} thickness={config.thickness} index={j}"
            elif isinstance(stroke_config, CircleGroupConfig):
                radius = stroke_config.radii[j] if j < len(stroke_config.radii) else stroke_config.radii[-1]
                desc = f"circle radius={radius} thickness={config.thickness} index={j}"
            elif isinstance(stroke_config, WaveGroupConfig):
                length = stroke_config.length[j] if j < len(stroke_config.length) else stroke_config.length[-1]
                amplitude = stroke_config.amplitude
                frequency = stroke_config.frequency
                desc = f"wave length={length} amplitude={amplitude} frequency={frequency} thickness={config.thickness} index={j}"
            else:
                desc = f"unknown type thickness={config.thickness} index={j}"
            all_paths.append(Stroke(pixel_coords=path, description=desc, color="black"))

            if len(path) > 1:
                is_curve = isinstance(stroke_config, (WaveGroupConfig, CircleGroupConfig))
                draw.line(path, fill="black", width=config.thickness, joint="curve" if is_curve else None)

    log.info(f"✍️ Generated {len(all_paths)} paths.")
    plan.add_strokes(all_paths, image)

if __name__ == "__main__":
    args = setup_log_with_config(BenchmarkPlanConfig)
    plan_from_calib(args)