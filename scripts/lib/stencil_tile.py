#!/usr/bin/env python3
"""The fiducial-tile skin stencil: a grid of AprilTag 16h5 markers with rulers on the left and bottom.

Three layouts (LAYOUTS):
  forearm   3 x 4 markers of 2 mm cells (12 mm) at a 16.5 mm pitch, centred in a 45 x 63 mm tile; its twelve
            codes come from FOREARM_CODES, sixteen 16h5 codes at least 6 bits apart under every turn
  forearm18 2 x 3 markers of 3 mm cells (18 mm) at a 22 mm pitch, 40 x 62 mm: two columns near the forearm's
            crest, cells the overhead D555 reads at ~3 px from 0.62 m (the forearm tile's are ~2 px), and codes
            FOREARM18_CODES, none of the forearm tile's, so a transfer of each on one arm cannot be mistaken
  silicone  5 x 7 markers of 3 mm cells (18 mm) at a 23.25 mm pitch, 111 x 157.5 mm, inside the 128 x 172 mm a
            silicone skin takes; each corner's 2 x 2 markers are one 40 mm anchor the overhead D555 can decode
            (6.7 mm cells, ~6 px at 1280 x 720 colour), so the sheet is found and named without the wrist
Every code on a sheet is distinct, so one decoded marker names its place. The seed picks the codes and their
order. Marker black is printed SHRINK_MM smaller per edge; a skin transfer bleeds it back (2026-10-17 forearm
test). A small label under the bottom ruler gives the layout, seed and sizes.

`generate(layout, seed, output)` writes `<layout>-seed<seed>.json` (every marker's id, role, grid place and
corners in mm: origin the tile's top-left, x right, y down, corners TL/TR/BR/BL of the upright marker, as
OpenCV and the AprilTag C library both report them) and `<layout>-seed<seed>-<dpi>dpi.png` for the stencil
printer app (it mis-scales PDFs). `artwork(layout, seed, output)` writes the print as a stencil reference
(`tatbot vision stencil generate --design fiducial-tile`): stencil.png (the tile, the print's page),
tile.json (the layout), settings.json and tracking.json, with the printable PNG in print/. Needs numpy,
OpenCV >= 4.7 and Pillow.

    python3 stencil_tile.py --layout forearm --seed 0 --output DIR
"""

import argparse
import json
import sys
from pathlib import Path

FAMILY = "apriltag_16h5"
GENERATOR = "tile-16h5-1"     # stencil_reference.TILE_GENERATORS
ARTWORK_PPM = 20              # stencil.png's px per mm
CODES = 30                    # 16h5's code count
# The largest set of 16h5 codes at least 6 bits apart under every relative turn (no set of 4+ reaches 7); the
# whole family is 5 apart. tests/test_stencil_tile.py checks the distance.
FOREARM_CODES = (0, 2, 4, 7, 8, 11, 12, 14, 15, 20, 21, 22, 23, 26, 27, 28)
# six 16h5 codes 6 bits apart under every turn and 5 from every code of the forearm tile's seed-0 print
FOREARM18_CODES = (1, 3, 6, 9, 16, 17)
SHRINK_MM = 0.4               # marker black printed this much smaller per edge
PPM = 40                      # internal render px per mm; downsampled to the print dpi at the end
DPI = 300
PILLOW_REQUIREMENT = "Pillow==12.3.0"   # the CLI's managed environment for generating
MARGIN_MM = 1.0               # clear edge above and right of the tile
# Clear skin between the tile and the rulers: at the overhead's 0.86 px/mm a 2 mm gap joins the edge markers to
# the ruler (the silicone sheet's three ruler-side anchors read in no synthetic view), at the wrist's ~4 px/mm it
# does not.
RULER_MM = 5.0                # ruler band width
TICK_MM = 0.5                 # tick line width
LABEL_MM = 2.2                # label cap height
LAYOUTS = {
    "forearm": {"cols": 3, "rows": 4, "cell_mm": 2.0, "pitch_mm": 16.5, "tile_mm": (45.0, 63.0), "anchor_mm": None,
                "gap_mm": 2.0, "codes": FOREARM_CODES},
    "forearm18": {"cols": 2, "rows": 3, "cell_mm": 3.0, "pitch_mm": 22.0, "tile_mm": None, "anchor_mm": None,
                  "gap_mm": 2.0, "codes": FOREARM18_CODES},
    "silicone": {"cols": 5, "rows": 7, "cell_mm": 3.0, "pitch_mm": 23.25, "tile_mm": None, "anchor_mm": 40.0,
                 "gap_mm": 4.0},
}
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "Arial.ttf")


def places(spec):
    """(role, row, col, x_mm, y_mm, side_mm) of every marker in the tile's grid frame, row-major; an anchor
    takes a corner's 2 x 2 grid places and sits flush with the tile's outer corner."""
    cols, rows, pitch = spec["cols"], spec["rows"], spec["pitch_mm"]
    side = 6 * spec["cell_mm"]
    anchor = spec["anchor_mm"]
    out = []
    corners = {(0, 0), (0, cols - 2), (rows - 2, 0), (rows - 2, cols - 2)} if anchor else set()
    covered = {(r + dr, c + dc) for r, c in corners for dr in (0, 1) for dc in (0, 1)}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in corners:
                x = c * pitch + (pitch + side - anchor if c else 0)
                y = r * pitch + (pitch + side - anchor if r else 0)
                out.append(("anchor", r, c, x, y, anchor))
            elif (r, c) not in covered:
                out.append(("marker", r, c, c * pitch, r * pitch, side))
    return out


def choose_codes(layout, seed, count, anchors):
    """The sheet's codes in place order: from the layout's own well-separated set (LAYOUTS codes), else anchors first
    from FOREARM_CODES, then the grid."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if pool := LAYOUTS[layout].get("codes"):
        return [int(i) for i in rng.choice(pool, count, replace=False)]
    anchor_ids = [int(i) for i in rng.choice(FOREARM_CODES, anchors, replace=False)]
    rest = [i for i in range(CODES) if i not in anchor_ids]
    return anchor_ids + [int(i) for i in rng.choice(rest, count - anchors, replace=False)]


def build(layout, seed):
    """The sheet at PPM px/mm (uint8, 0 = ink) and its layout record."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    spec = LAYOUTS[layout]
    cell, pitch = spec["cell_mm"], spec["pitch_mm"]
    marks = places(spec)
    grid_w = (spec["cols"] - 1) * pitch + 6 * cell
    grid_h = (spec["rows"] - 1) * pitch + 6 * cell
    tw, th = spec["tile_mm"] or (grid_w, grid_h)
    ox, oy = (tw - grid_w) / 2, (th - grid_h) / 2
    n_anchor = sum(role == "anchor" for role, *_ in marks)
    codes = choose_codes(layout, seed, len(marks), n_anchor)
    # anchors take the first codes; the grid's follow in place order
    order = sorted(range(len(marks)), key=lambda k: marks[k][0] != "anchor")
    code_of = {k: codes[j] for j, k in enumerate(order)}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    gap = spec["gap_mm"]
    left, top = RULER_MM + gap, MARGIN_MM
    width = left + tw + MARGIN_MM
    one_line = layout == "silicone"   # its 172 mm height has room for one label line
    narrow = width < 50.0             # the forearm18 sheet's 48 mm: four shorter lines
    n_lines = 1 if one_line else 4 if narrow else 3
    height = top + th + gap + RULER_MM + 1.0 + LABEL_MM * 1.45 * n_lines
    size = f"stencil {width:.0f} x {height:.0f} mm"
    lines = ([f"{layout}  seed {seed}  apriltag 16h5  cell {cell:g} mm  {size}"] if one_line else
             [f"{layout}  seed {seed}", f"apriltag 16h5  cell {cell:g} mm", f"tile {tw:g} x {th:g} mm", size] if narrow
             else [f"{layout}  seed {seed}  apriltag 16h5", f"cell {cell:g} mm  tile {tw:g} x {th:g} mm", size])
    img = np.full((round(height * PPM), round(width * PPM)), 255, np.uint8)
    markers = []
    for k, (role, r, c, x, y, side) in enumerate(marks):
        x, y = ox + x, oy + y
        px = round(side * PPM)
        draw = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker   # OpenCV < 4.7: drawMarker
        tag = cv2.resize(draw(dictionary, code_of[k], 6), (px, px),
                         interpolation=cv2.INTER_NEAREST)
        col, row = round((left + x) * PPM), round((top + y) * PPM)
        img[row:row + px, col:col + px] = tag
        markers.append({"id": code_of[k], "role": role, "row": r, "col": c, "side_mm": side,
                        "corners_mm": [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]})
    r_px = max(1, round(SHRINK_MM * PPM))
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1)))

    def bar(x0, y0, x1, y1):
        img[round(y0 * PPM):round(y1 * PPM), round(x0 * PPM):round(x1 * PPM)] = 0

    def tick(v):
        return RULER_MM if v % 50 == 0 else RULER_MM * (0.75 if v % 10 == 0 else 0.4)

    t, by = TICK_MM / 2, top + th + gap
    bar(RULER_MM - t, top, RULER_MM + t, top + th + t)
    bar(left - t, by - t, left + tw, by + t)
    for v in np.arange(0, th + 1e-6, 5.0):
        bar(RULER_MM - tick(v), top + v - t, RULER_MM, top + v + t)
    for v in np.arange(0, tw + 1e-6, 5.0):
        bar(left + v - t, by, left + v + t, by + tick(v))
    pil = Image.fromarray(img)
    draw, font = ImageDraw.Draw(pil), _font(round(LABEL_MM * 1.35 * PPM), ImageFont)
    for i, line in enumerate(lines):
        draw.text((round(left * PPM), round((by + RULER_MM + 1.0 + i * LABEL_MM * 1.45) * PPM)), line, fill=0,
                  font=font)
    record = {"layout": layout, "seed": seed, "family": FAMILY, "cell_mm": cell, "pitch_mm": pitch,
              "marker_mm": 6 * cell, "anchor_mm": spec["anchor_mm"], "black_preshrink_mm": SHRINK_MM,
              "tile_mm": [tw, th], "stencil_mm": [round(width, 2), round(height, 2)],
              "tile_origin_in_stencil_mm": [left, top],
              "rulers": "left (down from the tile top) and bottom (right from the tile left); a tick every 5 mm, "
                        "longer at 10, longest at 50",
              "markers": markers}
    return np.array(pil), record


def _font(px, image_font):
    for name in FONTS:
        try:
            return image_font.truetype(name, px)
        except OSError:
            pass
    return image_font.load_default()


def to_dpi(img, dpi):
    import cv2
    import numpy as np
    scale = dpi / 25.4 / PPM
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.where(out < 128, 0, 255).astype(np.uint8)


def generate(layout, seed, output, dpi=DPI):
    """Write the layout JSON and the print PNG; return their paths and the layout record."""
    from PIL import Image
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    img, record = build(layout, seed)
    base = output / f"{layout}-seed{seed}"
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(record, indent=1) + "\n")
    png_path = output / f"{layout}-seed{seed}-{dpi}dpi.png"
    Image.fromarray(to_dpi(img, dpi)).convert("1").save(png_path, dpi=(dpi, dpi))
    return {"layout": str(json_path), "png": str(png_path), "record": record}


def artwork(layout, seed, output, dpi=DPI):
    """The print as a stencil reference in `output`, its printable PNG in output/print/; the tracking manifest."""
    import hashlib

    import cv2
    import stencil_reference
    from PIL import Image

    output = Path(output).expanduser()
    (output / "print").mkdir(parents=True, exist_ok=True)
    img, record = build(layout, seed)
    (left, top), (tw, th) = record["tile_origin_in_stencil_mm"], record["tile_mm"]
    page = img[round(top * PPM):round((top + th) * PPM), round(left * PPM):round((left + tw) * PPM)]
    page = cv2.resize(page, (round(tw * ARTWORK_PPM), round(th * ARTWORK_PPM)), interpolation=cv2.INTER_AREA)
    Image.fromarray(page).save(output / "stencil.png")
    (output / stencil_reference.TILE_FILE).write_text(json.dumps(record, indent=1) + "\n")
    printable = output / "print" / f"{layout}-seed{seed}-{dpi}dpi.png"
    Image.fromarray(to_dpi(img, dpi)).convert("1").save(printable, dpi=(dpi, dpi))

    def sha(name):
        return hashlib.sha256((output / name).read_bytes()).hexdigest()

    settings = {"generator_version": GENERATOR, "layout": layout, "seed": seed, "family": FAMILY,
                "width_mm": tw, "height_mm": th, "margin_mm": 0.0, "frame_mm": 0.0,
                "pixels": [page.shape[1], page.shape[0]], "stencil_mm": record["stencil_mm"],
                "files": {"stencil.png": sha("stencil.png"), stencil_reference.TILE_FILE: sha("tile.json")},
                "print": str(printable.relative_to(output))}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return stencil_reference.export(output / "settings.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--output", required=True, help="output folder, outside the repo")
    args = parser.parse_args(argv)
    out = generate(args.layout, args.seed, args.output, args.dpi)
    print(json.dumps({k: out[k] for k in ("layout", "png")} | {"stencil_mm": out["record"]["stencil_mm"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
