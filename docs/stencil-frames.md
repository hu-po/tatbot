---
summary: "The skin stencil, a fiducial tile of AprilTag 16h5 markers: its layouts, printing, and how the wrist and overhead read it"
tags: [vision, stencil, reference]
updated: 2026-10-19
audience: [dev, artist]
---

# The skin stencil

The stencil transferred onto skin is a **fiducial tile**: a grid of AprilTag
16h5 markers with millimetre rulers on the left and bottom and a small label
([`scripts/lib/stencil_tile.py`](../scripts/lib/stencil_tile.py)). Every code
on a sheet is distinct, so one decoded marker names its place on the print and
no fit can land a repeat away. The whole tile is the page and all of it is
clear: the design is drawn over the markers.

## Layouts

| `--preset` | Tile | Markers | Notes |
|---|---|---|---|
| `forearm` (default) | 45 × 63 mm (sheet 53 × 82) | 3 × 4 of 2 mm cells (12 mm) at 16.5 mm | codes at least 6 bits apart under every turn |
| `silicone` | 111 × 157.5 mm (sheet 121 × 172) | 19 of 3 mm cells (18 mm) at 23.25 mm, four 40 mm corner anchors | the anchors decode overhead at 1280x720; a 4 mm ruler gap keeps the edge markers apart from the rulers |

`--seed` (an integer) picks the codes and their order. Marker black is printed
0.4 mm smaller per edge; a skin transfer bleeds it back. The silicone layout's
seed 0 is the drawing stack's default page (`page.pattern_id` in
`ros/tatbot_bringup/config/stack.yaml`).

## Generate, print and install

```sh
scripts/tatbot vision stencil generate --preset silicone --seed 0 \
  --output ~/tatbot-logs/stencils/silicone-tile-0
```

The artwork goes into `--output`: `stencil.png` (the tile, the print's page),
`tile.json` (every marker's code and printed corners), `settings.json` and
`tracking.json` (the reference manifest, which binds `tile.json` by hash). The
printable PNG is `print/<layout>-seed<seed>-300dpi.png`, for the stencil printer
app (it mis-scales PDFs); print it at 100%.

Install the reference on the node that runs the stencil observer (the
overhead camera's owner), where the drawing stack's `ros up --pattern` finds it:

```sh
scripts/tatbot vision stencil reference --settings ~/tatbot-logs/stencils/silicone-tile-0/settings.json --install
```

A reference installs atomically: the image, `tile.json` and `settings.json`
land first and the manifest last, so the observer never pairs a manifest with
files it does not describe. A manifest that is not a tile's is refused by name.

## How it is read

[`scripts/vision/fiducials/stencil.py`](../scripts/vision/fiducials/stencil.py)
reads a tile with the fiducial module's detector: the green channel with its
contrast evened (a violet transfer absorbs green, skin reflects it), the
`skin` detector profile, and only the print's own codes.

- **Wrist D405** (`ros/README.md` §4.4): the gauge holds and every hover and
  lift reading find the markers in the camera's own frames and carry each
  corner onto the page, or a curved page's skin, through the map the page's
  rectification samples by. One rigid move that most markers agree with
  (2.5 mm) is the reading, accepted at corner rms up to 1.4 mm with two markers,
  or one whose read move is under half the way to its neighbour. The locate
  views before the holds rectify a flat page's frames onto the plane of the
  overhead's pose and read the markers there; on a curved page they lift each
  corner by the depth about its own pixel instead and carry it onto the page
  along its normal, as a frame whose gauge reads no surface is read. At a
  first setup that page is still the overhead pose's plane (no field exists
  yet), and the holds' own turn and shift then place the page; a page kept
  through a mid-draw re-setup reads them through its field's chart.
- **Overhead D555** ([vision](vision.md)): the stencil observer lifts each
  corner of every read of the tile's codes to the depth about its own pixel,
  keeps the reads within 15% of their printed size (a false read of a code
  elsewhere never hides the print's own) whose corners lie as far apart as
  printed, allowing for the curve of the skin the layout goes on (a 35 mm
  forearm for the forearm tile, 100 mm for the silicone tile; of equal groups,
  the one nearest its printed distances), and fits the tile's pose to them. No
  surface is fitted to the depth: the arm over part of a curved print moves
  the rest's points by the depth under 0.2 mm (one plane over the visible part
  moved them 2-6 mm). The pose itself is the layout laid round a cylinder
  through the markers in view, its origin the print's centre on the skin and
  its z the skin's normal there (docs/vision.md); with markers along one row
  or column, or under three, the flat layout's rigid fit (their chord plane).
  Fewer than two agreeing markers is no pose for
  that turn (the arm over the sheet hides most of them). The support lists the
  agreeing markers' ids and their corners (`markers.ids[k]` names
  `anchor_points_m[4k:4k+4]`, each on its corner's camera ray): the drawing
  session judges a moved sheet by those same markers, never by the pose's tilt
  or height (`ros/README.md`, Page motion). A tile whose markers
  are under 15 mm (the forearm layout's 12 mm read at ~10 px) is read at twice
  the colour's resolution with a lower perimeter floor and threshold constant
  and a stronger contrast evening, in a window about its last placement and
  the whole frame when that finds nothing; the silicone layout keeps the plain
  reading.

Under full code correction other 4x4 patterns in a scene (another marker
sheet, an AprilTag 36h11 face) can decode as one of the tile's codes; the
layout and size checks are what refuse them.
