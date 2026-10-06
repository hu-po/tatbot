#!/usr/bin/env python3
"""Print sheets for a generated stencil: the artwork page with millimetre rulers and one label line.

The sheet puts the stencil page (its own white margin included) between a ruler
down its left edge, a ruler along its bottom edge and one small line of text
centred over its top: the pattern and the page size. The rulers read 0 at the
page's top and left edges and run its full height and width, so a print, or
its transfer on skin, can be checked for scale both ways. All three sit outside
the page, so the page coordinates in tracking.json are unchanged; `sheet.json`
records where the page sits on the sheet. The strips are narrow (8 mm left,
4 mm top, 6 mm bottom) so a sheet fits a skin: an 83 x 127 mm page makes a
91 x 137 mm sheet, and two fit side by side on a 185 x 140 mm silicone skin
cut in half.

Outputs, in `<output>/print/`:
- `stencil-app.png`: 1-bit, for a thermal stencil printer's phone app. Set the
  whole image to the size recorded in `sheet.json`.
- `paper-a4.pdf` and `paper-letter.pdf`: the sheet at 100 %, centred on a paper
  page. Print at actual size; the rulers must read the page's size.
- `paper-fit.pdf` with `--fit-area WxH`: the page is exactly a printer's
  printable area, for driverless printers that ignore "no scaling" and stretch
  any page to fill that area (so the stretch is 1).
- `sheet.svg`: the same sheet in millimetres.
- `sheet.json`: geometry, source hashes and print instructions.

Needs Pillow; generating a coded stencil (`--seed`) also needs numpy.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

RULER_MM = 8.0          # left strip: the ruler down the page's height
LABEL_MM = 4.0          # top strip: one line, the pattern and the page size
FOOT_MM = 6.0           # bottom strip: the ruler along the page's width
GAP_MM = .5             # a ruler's spine from the page edge
TICKS_MM = (2.6, 2.0, 1.2)   # tick lengths every 10, 5 and 1 mm
NUMBER_MM = 2.0         # ruler numbers
LABEL_FONT_MM = 2.4
LINE_MM = .2            # ruler line and tick width
PAPERS_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}
SCHEMA = "tatbot.stencil-print-sheet/1"
SVG_ANCHOR = {"l": "start", "m": "middle", "r": "end", "a": "hanging"}   # Pillow anchor letters


def _font(size_px):
    from PIL import ImageFont
    return ImageFont.load_default(size=max(8, round(size_px)))


def label_line(settings, reference):
    pattern = reference["pattern_id"].removeprefix("stencil-")[:12]
    return f"pattern {pattern}   {settings['width_mm']:g} x {settings['height_mm']:g} mm"


def ruler_marks(length_mm):
    """(mm from the page edge, tick length mm, label or None) for every millimetre."""
    return [(mm, TICKS_MM[0], str(mm)) if mm % 10 == 0 else (mm, TICKS_MM[1 if mm % 5 == 0 else 2], None)
            for mm in range(int(length_mm)+1)]


def sheet_layout(width, height, label):
    """(sheet mm, lines, texts) in sheet mm: a line is ((x0, y0), (x1, y1)), a text (x, y, text, size, anchor)
    with Pillow's anchor letters (horizontal l/m/r, then a = ascender or m = middle)."""
    sheet = (RULER_MM+width, LABEL_MM+height+FOOT_MM)
    x_spine, y_spine = RULER_MM-GAP_MM, LABEL_MM+height+GAP_MM
    number_at = TICKS_MM[0]+.4
    lines = [((x_spine, LABEL_MM), (x_spine, LABEL_MM+height)), ((RULER_MM, y_spine), (RULER_MM+width, y_spine))]
    texts = [(RULER_MM+width/2, .8, label, LABEL_FONT_MM, "ma")]   # centred over the page
    for mm, tick, text in ruler_marks(height):
        lines.append(((x_spine-tick, LABEL_MM+mm), (x_spine, LABEL_MM+mm)))
        if text:
            texts.append((x_spine-number_at, LABEL_MM+mm, text, NUMBER_MM, "rm"))
    for mm, tick, text in ruler_marks(width):
        lines.append(((RULER_MM+mm, y_spine), (RULER_MM+mm, y_spine+tick)))
        if text:   # a number at the right edge moves in by its half width, about .3 em a digit
            x = min(RULER_MM+mm, sheet[0]-.2-.3*NUMBER_MM*len(text))
            texts.append((x, y_spine+number_at, text, NUMBER_MM, "ma"))
    return sheet, lines, texts


def _page(artwork_dir):
    artwork_dir = Path(artwork_dir)
    settings = json.loads((artwork_dir/"settings.json").read_text())
    reference = json.loads((artwork_dir/"tracking.json").read_text())
    return settings, reference, label_line(settings, reference)


def render_sheet(artwork_dir, dpi):
    """(1-bit sheet image, sheet size mm, page offset mm, label line)."""
    from PIL import Image, ImageDraw
    settings, _, label = _page(artwork_dir)
    width, height = settings["width_mm"], settings["height_mm"]
    sheet_mm, lines, texts = sheet_layout(width, height, label)
    scale = dpi/25.4
    page = Image.open(Path(artwork_dir)/"stencil.png").convert("L")
    page = page.resize((round(width*scale), round(height*scale)), Image.Resampling.NEAREST)
    sheet = Image.new("L", (round(sheet_mm[0]*scale), round(sheet_mm[1]*scale)), 255)
    sheet.paste(page, (round(RULER_MM*scale), round(LABEL_MM*scale)))
    draw = ImageDraw.Draw(sheet)

    def px(x, y):
        return min(x*scale, sheet.width-1), min(y*scale, sheet.height-1)

    for start, end in lines:
        draw.line([px(*start), px(*end)], fill=0, width=max(1, round(LINE_MM*scale)))
    for x, y, text, size, anchor in texts:
        draw.text(px(x, y), text, fill=0, font=_font(size*scale), anchor=anchor)
    return sheet.point(lambda v: 255 if v >= 128 else 0).convert("1"), sheet_mm, (RULER_MM, LABEL_MM), label


def sheet_svg(artwork_dir):
    """The sheet as millimetre SVG: the artwork's own SVG nested at the page offset."""
    settings, _, label = _page(artwork_dir)
    width, height = settings["width_mm"], settings["height_mm"]
    sheet_mm, lines, texts = sheet_layout(width, height, label)
    inner = (Path(artwork_dir)/"stencil.svg").read_text().strip().splitlines()
    body = [f'<svg x="{RULER_MM:g}" y="{LABEL_MM:g}" width="{width:g}" height="{height:g}" '
            f'viewBox="0 0 {width:g} {height:g}">', *inner[1:]]
    ink = [f'<line x1="{x0:g}" y1="{y0:g}" x2="{x1:g}" y2="{y1:g}" stroke="black" stroke-width="{LINE_MM:g}"/>'
           for (x0, y0), (x1, y1) in lines]
    ink += [f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" text-anchor="{SVG_ANCHOR[anchor[0]]}" '
            f'dominant-baseline="{SVG_ANCHOR[anchor[1]]}" font-family="sans-serif">{text}</text>'
            for x, y, text, size, anchor in texts]
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_mm[0]:g}mm" '
                      f'height="{sheet_mm[1]:g}mm" viewBox="0 0 {sheet_mm[0]:g} {sheet_mm[1]:g}">',
                      '<rect width="100%" height="100%" fill="white"/>', *ink, *body, "</svg>"])+"\n"


def paper_pdf(sheet, dpi, paper_mm, path):
    """The sheet at 100 %, centred on a paper page."""
    from PIL import Image
    scale = dpi/25.4
    page = Image.new("L", (round(paper_mm[0]*scale), round(paper_mm[1]*scale)), 255)
    left = round((page.width-sheet.width)/2)
    top = round((page.height-sheet.height)/2)
    page.paste(sheet.convert("L"), (left, top))
    page.save(path, "PDF", resolution=dpi)


def write_sheets(artwork_dir, output, dpi=300, fit_area_mm=None):
    artwork_dir, output = Path(artwork_dir), Path(output)
    settings, reference, _ = _page(artwork_dir)
    sheet, sheet_mm, offset, label = render_sheet(artwork_dir, dpi)
    if fit_area_mm and not (sheet_mm[0] <= fit_area_mm[0] and sheet_mm[1] <= fit_area_mm[1]):
        raise ValueError(f"--fit-area {fit_area_mm[0]:g}x{fit_area_mm[1]:g} cannot hold the "
                         f"{sheet_mm[0]:g} x {sheet_mm[1]:g} mm sheet")
    output.mkdir(parents=True, exist_ok=True)
    files = {"stencil-app.png": output/"stencil-app.png", "sheet.svg": output/"sheet.svg"}
    sheet.save(files["stencil-app.png"], dpi=(dpi, dpi), optimize=False)
    files["sheet.svg"].write_text(sheet_svg(artwork_dir), encoding="utf-8")
    papers = dict(PAPERS_MM, **({"fit": fit_area_mm} if fit_area_mm else {}))
    for paper, size in papers.items():
        files[f"paper-{paper}.pdf"] = output/f"paper-{paper}.pdf"
        paper_pdf(sheet, dpi, size, files[f"paper-{paper}.pdf"])
    width, height = settings["width_mm"], settings["height_mm"]
    manifest = {"schema": SCHEMA, "artwork": str(artwork_dir.resolve()), "pattern_id": reference["pattern_id"],
                "reference_id": reference["reference_id"], "sheet_mm": list(sheet_mm), "page_offset_mm": list(offset),
                "dpi": dpi, "label": label,
                "stencil_app": f"set the whole image to {sheet_mm[0]:g} x {sheet_mm[1]:g} mm; do not crop the white edges",
                "paper": f"print the PDF at 100 % / actual size; the rulers must read {height:g} mm down the side "
                         f"and {width:g} mm along the bottom",
                "files": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}}
    (output/"sheet.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artwork", help="existing generated stencil directory (settings.json, tracking.json, stencil.png/svg)")
    source.add_argument("--seed", help="generate a coded flower-of-life stencil (the default design) with this seed first")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="coded generator setting with --seed, e.g. bit=side or knot_mm=2.4; repeatable")
    parser.add_argument("--output", required=True, help="directory for the artwork (with --seed) and print/")
    parser.add_argument("--dpi", type=int, default=300, help="sheet resolution (default 300)")
    parser.add_argument("--fit-area", metavar="WxH",
                        help="also write paper-fit.pdf whose page is exactly a printer's printable area in mm "
                             "(e.g. 203.2x269.4): for printers that stretch any page to fill that area")


def validate(args):
    if args.set and not args.seed:
        raise ValueError("--set applies to --seed generation, not to --artwork")
    if not 150 <= args.dpi <= 1200:
        raise ValueError("dpi must be within 150-1200")
    if args.fit_area is not None:
        fit_area(args.fit_area)
    for pair in args.set or ():
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")


def fit_area(text):
    """'WxH' in mm -> (w, h); write_sheets refuses an area that cannot hold the sheet."""
    try:
        width, height = (float(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"--fit-area expects WxH in mm, got {text!r}") from None
    if not (0 < width <= 500 and 0 < height <= 500):
        raise ValueError("--fit-area must be positive and at most 500 mm")
    return width, height


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        validate(args)
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output).expanduser()
    artwork = Path(args.artwork).expanduser() if args.artwork else output/"artwork"
    if args.seed:
        import stencil_coded
        settings = {}
        for pair in args.set or ():
            key, _, value = pair.partition("=")
            try:
                settings[key.replace("-", "_")] = float(value)
            except ValueError:
                settings[key.replace("-", "_")] = value
        stencil_coded.generate(args.seed, artwork, **settings)
    try:
        manifest = write_sheets(artwork, output/"print", args.dpi, fit_area(args.fit_area) if args.fit_area else None)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
