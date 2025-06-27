from dataclasses import dataclass
import os

import svgpathtools
import numpy as np
from PIL import Image

from _log import get_logger, setup_log_with_config
from _plan import Plan
from _path import Stroke

log = get_logger('gen_svg')

@dataclass
class SVGPlanConfig:
    debug: bool = False
    """Enable debug logging."""

    svg_name: str = "roaring_cat"
    """Name of the SVG file."""

    output_dir: str = os.path.expanduser(f"~/tatbot/output/plans/svg/{svg_name}")
    """Directory to save the plan."""

    svg_path: str = f"~/tatbot/assets/designs/{svg_name}.svg"
    """Path to the SVG file."""

    png_path: str | None = f"~/tatbot/assets/designs/{svg_name}.png"
    """Path to the PNG file."""

    points_per_path: int = 108
    """Number of points to sample per SVG path."""

    image_width_px: int = 640
    """ Width of the design image (pixels)."""
    image_height_px: int = 640
    """ Height of the design image (pixels)."""

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

    if config.png_path is not None:
        image = Image.open(os.path.expanduser(config.png_path))
        plan.image_width_px = image.width
        plan.image_height_px = image.height

    plan.add_strokes(strokes, image)

if __name__ == "__main__":
    args = setup_log_with_config(SVGPlanConfig)
    plan_from_svg(args)