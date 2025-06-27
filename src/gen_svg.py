from dataclasses import dataclass
import os

import svgpathtools
import numpy as np
from PIL import Image
import cairosvg
import xml.etree.ElementTree as ET

from _log import get_logger, setup_log_with_config
from _plan import Plan
from _path import Stroke

log = get_logger('gen_svg')

@dataclass
class SVGPlanConfig:
    debug: bool = False
    """Enable debug logging."""

    output_dir: str = os.path.expanduser("~/tatbot/output/plans/svg")
    """Directory to save the plan."""

    svg_path: str = "~/tatbot/assets/designs/zorya.svg"
    """Path to the SVG file."""

    points_per_path: int = 108
    """Number of points to sample per SVG path."""

    image_width_px: int = 640
    """ Width of the design image (pixels)."""
    image_height_px: int = 640
    """ Height of the design image (pixels)."""

def ensure_svg_size(svg_path, width, height, viewbox):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    changed = False

    if 'width' not in root.attrib:
        root.set('width', str(width))
        changed = True
    if 'height' not in root.attrib:
        root.set('height', str(height))
        changed = True
    if 'viewBox' not in root.attrib:
        root.set('viewBox', viewbox)
        changed = True

    if changed:
        temp_svg_path = svg_path + ".temp"
        tree.write(temp_svg_path)
        return temp_svg_path
    return svg_path

def plan_from_svg(config: SVGPlanConfig):
    plan = Plan(
        name="svg",
        dirpath=config.output_dir,
    )

    log.info(f"🖋️📂 Loading SVG file from {config.svg_path}")
    svg_path = os.path.expanduser(config.svg_path)
    paths, attributes, svg_attr = svgpathtools.svg2paths2(svg_path)

    strokes: list[Stroke] = []
    for i, path in enumerate(paths):
        if path.length() == 0:
            continue

        ts = np.linspace(0, 1, config.points_per_path)
        points = np.array([path.point(t) for t in ts])
        pixel_coords = np.round(np.column_stack((points.real, points.imag))).astype(int)

        stroke = Stroke(
            pixel_coords=pixel_coords,
            description=f"svg_path_{i}",
            color="black"
        )
        strokes.append(stroke)

    log.info(f"🖋️ Generated {len(strokes)} strokes.")

    # Render PNG from SVG if no image context is already available
    raster_path = os.path.join(config.output_dir, "image.png")
    svg_path_with_size = ensure_svg_size(
        svg_path,
        config.image_width_px,
        config.image_height_px,
        f"0 0 {config.image_width_px} {config.image_height_px}"
    )
    cairosvg.svg2png(url=svg_path_with_size, write_to=raster_path)
    if svg_path_with_size != svg_path:
        os.remove(svg_path_with_size)
    image = Image.open(raster_path)

    plan.add_strokes(strokes, image)

if __name__ == "__main__":
    args = setup_log_with_config(SVGPlanConfig)
    plan_from_svg(args)