#!/usr/bin/env python3
"""Seeded flower-of-life frame with a fresh print code; Python 3.10+ and Pillow.

All geometry uses millimeters. The floral artwork uses a SHA-256 counter stream
independent of Python RNG versions; the print-instance ID uses OS randomness.
SVG is the canonical vector artwork; PNG is a strictly black/white rendering.
"""

import argparse
import hashlib
import json
import math
import secrets
import sys
from pathlib import Path

import stencil_instance
import stencil_work_index

VERSION = "1"
PILLOW_REQUIREMENT = "Pillow==12.1.1"
TAU = math.tau


class SeedStream:
    def __init__(self, seed):
        self.key = str(seed).encode("utf-8")
        self.counter = 0

    def unit(self):
        digest = hashlib.sha256(
            b"floral-stencil-v1\0" + self.key + b"\0"
            + self.counter.to_bytes(8, "big")
        ).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big") / 2**64


def arc(x, y, radius, start, stop):
    # Samples are spaced <= 0.09 mm; SVG and PNG use identical geometry.
    steps = max(8, math.ceil(abs(stop - start) * radius / 0.09))
    return [(x + radius * math.cos(start + (stop - start) * i / steps),
             y + radius * math.sin(start + (stop - start) * i / steps))
            for i in range(steps + 1)]


def build(args):
    rng = SeedStream(args.seed)
    shapes = []
    radius = args.spacing_mm
    margin = args.margin_mm
    w, h = args.width_mm, args.height_mm

    def path(points, stroke=0.0, fill=False):
        shapes.append({"points": points, "stroke": stroke, "fill": fill})

    def disk(x, y, r, solid=True):
        path(arc(x, y, r, 0, TAU), 0 if solid else args.stroke_mm, solid)

    # Extra circles outside the crop ensure full coverage up to its edges.
    centers = []
    for row in range(-2, math.ceil(h / (radius * math.sqrt(3) / 2)) + 2):
        for col in range(-2, math.ceil(w / radius) + 2):
            x = margin + (col + (row % 2) / 2) * radius
            y = margin + row * radius * math.sqrt(3) / 2
            centers.append((row, col, x, y))

    # Narrow vesica-piscis lenses occur between circles sqrt(3)*radius apart.
    # Filled lenses vary across the sheet; no repeated motif tile is copied.
    lookup = {(row, col): (x, y) for row, col, x, y in centers}
    for row, col, x, y in centers:
        # Only these three forward lattice neighbors are sqrt(3)*radius away.
        # Keep their traversal order stable: it is part of the seed contract.
        neighbors = [(row + 1, col - 2 + row % 2),
                     (row + 1, col + 1 + row % 2), (row + 2, col)]
        for key in neighbors:
            if key not in lookup:
                continue
            xx, yy = lookup[key]
            distance = math.hypot(xx - x, yy - y)
            if abs(distance - math.sqrt(3) * radius) > 1e-6:
                continue
            if rng.unit() >= args.fill_rate:
                continue
            theta = math.atan2(yy - y, xx - x)
            alpha = math.acos(distance / (2 * radius))
            points = arc(x, y, radius, theta - alpha, theta + alpha)
            points += arc(xx, yy, radius, theta + math.pi - alpha,
                          theta + math.pi + alpha)
            path(points, fill=True)

    for _, _, x, y in centers:
        for sector in range(6):
            if rng.unit() < args.gap_rate:
                continue
            start = sector * TAU / 6
            # Some sectors have a clear internal break; never a hairline gap.
            if rng.unit() < 0.13:
                gap = 0.9 / radius
                mid = start + TAU / 12
                path(arc(x, y, radius, start, mid - gap / 2), args.stroke_mm)
                path(arc(x, y, radius, mid + gap / 2, start + TAU / 6),
                     args.stroke_mm)
            else:
                path(arc(x, y, radius, start, start + TAU / 6), args.stroke_mm)

    # Triangle centers occupy open spaces between the circle intersections.
    # Random oriented clusters create local asymmetry at a second feature scale.
    for _, _, x, y in centers:
        for direction in (-1, 1):
            cx = x + radius / 2
            cy = y + direction * radius * math.sqrt(3) / 6
            if not (margin + 1.5 < cx < w - margin - 1.5
                    and margin + 1.5 < cy < h - margin - 1.5):
                continue
            choice = rng.unit()
            if choice < 0.48:
                continue
            angle = rng.unit() * TAU
            if choice < 0.66:
                disk(cx, cy, 0.65, solid=False)
            elif choice < 0.82:
                disk(cx, cy, 0.45 + rng.unit() * 0.25)
            else:
                count = 2 if choice < 0.93 else 3
                for k in range(count):
                    offset = (k - (count - 1) / 2) * 1.3
                    disk(cx + offset * math.cos(angle),
                         cy + offset * math.sin(angle), 0.36)
    # Omit invisible paths without changing any seeded choices or draw order.
    inset = margin + args.frame_mm
    visible = []
    for shape in shapes:
        xs, ys = zip(*shape["points"], strict=True)
        pad = shape["stroke"] / 2
        x0, x1, y0, y1 = min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad
        if x1 < margin or x0 > w-margin or y1 < margin or y0 > h-margin:
            continue
        if x0 > inset and x1 < w-inset and y0 > inset and y1 < h-inset:
            continue
        visible.append(shape)
    return visible


def _validate_issue_request(args, issue_index):
    if issue_index is not None:
        if args.instance_id is not None or args.unmarked:
            raise ValueError("original-work issuance needs a newly minted marked print")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("qualified print output must be a new empty directory")


def render(args, shapes, *, issue_index=None):
    from PIL import Image, ImageDraw
    from PIL import __version__ as pillow_version

    _validate_issue_request(args, issue_index)

    scale = args.dpi / 25.4
    size = (round(args.width_mm * scale), round(args.height_mm * scale))
    raster = Image.new("1", size, 1)
    draw = ImageDraw.Draw(raster)
    m = args.margin_mm
    inset = m + args.frame_mm
    iw, ih = args.width_mm - 2*inset, args.height_mm - 2*inset
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{args.width_mm:g}mm" height="{args.height_mm:g}mm" '
           f'viewBox="0 0 {args.width_mm:g} {args.height_mm:g}">',
           '<rect width="100%" height="100%" fill="white"/>',
           # Four strips work in renderers that ignore even-odd clipping holes.
           '<defs><clipPath id="crop">'
           f'<rect x="{m:g}" y="{m:g}" width="{args.width_mm-2*m:g}" height="{args.frame_mm:g}"/>'
           f'<rect x="{m:g}" y="{inset+ih:g}" width="{args.width_mm-2*m:g}" height="{args.frame_mm:g}"/>'
           f'<rect x="{m:g}" y="{inset:g}" width="{args.frame_mm:g}" height="{ih:g}"/>'
           f'<rect x="{inset+iw:g}" y="{inset:g}" width="{args.frame_mm:g}" height="{ih:g}"/>'
           '</clipPath></defs>',
           '<g clip-path="url(#crop)" stroke-linecap="round" '
           'stroke-linejoin="round">']
    for shape in shapes:
        pts = shape["points"]
        pixel = [(round(x * scale), round(y * scale)) for x, y in pts]
        coords = " ".join(f"{x:.4f},{y:.4f}" for x, y in pts)
        if shape["fill"]:
            draw.polygon(pixel, fill=0)
            svg.append(f'<polygon points="{coords}" fill="black"/>')
        else:
            line_width = max(1, round(shape["stroke"] * scale))
            draw.line(pixel, fill=0, width=line_width, joint="curve")
            end_radius = line_width / 2
            for x, y in (pixel[0], pixel[-1]):
                draw.ellipse((x - end_radius, y - end_radius,
                              x + end_radius, y + end_radius), fill=0)
            svg.append(f'<polyline points="{coords}" fill="none" '
                       f'stroke="black" stroke-width="{shape["stroke"]:g}"/>')
    svg.extend(['</g>', '</svg>'])
    artwork_svg_sha256 = hashlib.sha256(("\n".join(svg) + "\n").encode()).hexdigest()
    # Match the SVG clipping rectangle, keeping the margin completely blank.
    left, top = round(m * scale), round(m * scale)
    right = round((args.width_mm - m) * scale)
    bottom = round((args.height_mm - m) * scale)
    canvas = Image.new("1", size, 1)
    canvas.paste(raster.crop((left, top, right, bottom)), (left, top))
    ImageDraw.Draw(canvas).rectangle(
        (round(inset*scale), round(inset*scale),
         round((inset+iw)*scale)-1, round((inset+ih)*scale)-1), fill=1)
    instance_id = None
    mark = None
    if not args.unmarked:
        mark = stencil_instance.spec(args.width_mm, args.margin_mm, args.frame_mm)
        instance_id = args.instance_id or secrets.token_hex(stencil_instance.NONCE_BYTES)
        x, y, unit = mark['x_mm'], mark['y_mm'], mark['module_mm']
        x2, y2 = x+mark['cols']*unit, y+mark['rows']*unit
        ink = ImageDraw.Draw(canvas)
        ink.rectangle((round(x*scale), round(y*scale), round(x2*scale)-1,
                       round(y2*scale)-1), fill=1)
        for row, col, black in stencil_instance.modules(instance_id):
            if not black:
                continue
            left, top = x+col*unit, y+row*unit
            ink.rectangle((round(left*scale), round(top*scale),
                           round((left+unit)*scale)-1, round((top+unit)*scale)-1), fill=0)
        svg[-1:-1] = ['<g id="tatbot-instance-mark">',
                      *stencil_instance.svg_elements(mark, instance_id), '</g>']
    args.output.mkdir(parents=True, exist_ok=True)
    png_path = args.output / "stencil.png"
    svg_path = args.output / "stencil.svg"
    canvas.save(png_path, dpi=(args.dpi, args.dpi), optimize=False)
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    settings = {key: value for key, value in vars(args).items()
                if key not in ("output", "issue_work_index")}
    settings.update(generator_version=VERSION, pixels=list(size),
                    pillow_version=pillow_version,
                    clear_center_mm=[iw, ih],
                    artwork_svg_sha256=artwork_svg_sha256,
                    physical_instance_id=instance_id, instance_mark=mark,
                    black_fraction=round(canvas.histogram()[0] / (size[0]*size[1]), 5),
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (png_path, svg_path)})
    (args.output / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    from stencil_reference import export
    reference = export(args.output / "settings.json")
    if issue_index is not None:
        receipt_copy = args.output / "issue-receipt.json"
        path = issue_index.issue(settings, reference, (args.output / "tracking.json").read_bytes())
        with receipt_copy.open("xb") as file:
            file.write(path.read_bytes())
            file.flush()
            import os
            os.fsync(file.fileno())
        settings["issue_receipt"] = str(receipt_copy)
    return settings


# The frame's own options and their defaults: everything but --seed and --output.
FRAME_OPTIONS = (("--width-mm", float, 100), ("--height-mm", float, 150), ("--margin-mm", float, 5),
                 ("--frame-mm", float, 14), ("--spacing-mm", float, 7), ("--stroke-mm", float, 0.45),
                 ("--dpi", int, 300), ("--gap-rate", float, 0.22), ("--fill-rate", float, 0.14))


def add_arguments(parser):
    """Shared by the standalone backend and the stdlib-only CLI registry."""
    parser.add_argument("--seed", default="tatbot-42", help="Any UTF-8 text or number")
    parser.add_argument("--output", required=True, help="directory for PNG, SVG and settings; issued session prints need a new empty directory")
    add_frame_arguments(parser)
    parser.add_argument("--issue-work-index", action="store_true", help=argparse.SUPPRESS)


def add_frame_arguments(parser, *, defaults=True):
    """The frame's own options. With `defaults=False` an omitted option stays None, so a
    caller offering several designs (stencil_generate.py) can tell the given ones apart."""
    scope = "" if defaults else "flower-of-life design; "
    for flag, kind, default in FRAME_OPTIONS:
        parser.add_argument(flag, type=kind, default=default if defaults else None,
                            help=f"{scope}default {default:g}")
    parser.add_argument("--instance-id", help=f"{scope}24 lowercase hex digits for a specific printed file; "
                        "omitted generates a fresh random ID")
    parser.add_argument("--unmarked", action="store_true",
                        help=f"{scope}legacy unqualified artwork without an instance mark")


def validate(args):
    """Reject invalid geometry before planning dependencies or creating files."""
    for name in ("width_mm", "height_mm", "spacing_mm", "stroke_mm", "frame_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(args.margin_mm) or args.margin_mm < 0:
        raise ValueError("margin must be finite and nonnegative")
    if min(args.width_mm, args.height_mm) <= 2 * args.margin_mm + args.spacing_mm:
        raise ValueError("page interior must be larger than the circle spacing")
    if min(args.width_mm, args.height_mm) <= 2 * (args.margin_mm + args.frame_mm):
        raise ValueError("frame must leave a nonempty center")
    if not 72 <= args.dpi <= 1200:
        raise ValueError("DPI must be between 72 and 1200")
    if max(args.width_mm, args.height_mm) > 500 or args.spacing_mm < 4:
        raise ValueError("page dimensions must be <= 500 mm and spacing >= 4 mm")
    if args.width_mm * args.height_mm * (args.dpi / 25.4)**2 > 50_000_000:
        raise ValueError("raster is too large; use lower DPI (maximum 50 million pixels)")
    stencil_instance.validate_generation(args)
    if not 0 <= args.gap_rate <= 1 or not 0 <= args.fill_rate <= 1:
        raise ValueError("gap and fill rates must be between 0 and 1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        validate(args)
    except ValueError as error:
        parser.error(str(error))
    args.output = Path(args.output).expanduser()
    try:
        if args.issue_work_index:
            with stencil_work_index.IssueIndex(stencil_work_index.default_root()) as index:
                result = render(args, build(args), issue_index=index)
        else:
            result = render(args, build(args))
    except stencil_work_index.IndexBusyError as error:
        print(f"tatbot: busy (original-work index): {error}", file=sys.stderr)
        return 6
    except (stencil_work_index.IndexEvidenceError, OSError) as error:
        print(f"tatbot: gate refused (original-work index): {error}", file=sys.stderr)
        return 3
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
