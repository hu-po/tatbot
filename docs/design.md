# `tatbot design` — acquired DBV3 artwork to a portable design

Use [DrawingBot V3](drawingbot.md) to acquire ordered physical paths first.
Inkmap and ROS preparation require `dbv3-batik-paths/1` artwork with a frozen
recipe identity. The former SVG/raster `tatbot design trace` route is retired
and refuses with a regeneration action.

```sh
tatbot design generate "an orbital drawing" --out source.png
tatbot drawingbot generate job.json --out acquisition --app /path/to/drawingbotv3
tatbot design place acquisition/artwork.json --target plane --out design.json
tatbot design check design.json --preview preview.svg
tatbot ros compile design.json -o prepared
```

`design generate` produces a source PNG and provenance sidecar. Supply those
bytes and their exact hash to a version 3 DBV3 job with explicit dimensions,
pen table, PFM settings and seed. It does not produce finished path artwork.
The bundled `web/inkmap/public/designs/dbv3-orbit/job.json` is a working example,
with its source in `input/source.png`. Job paths resolve relative to the job.

`design place` and `design check` require acquired DBV3 records. Placement
changes translation, rotation and mirror; changing physical dimensions or
pen settings requires a new acquisition. Paths retain native order, direction,
subpaths, closedness and repeated passes. Preview SVGs are derived output,
not accepted source records. A design contains artwork, placement and nominal
chart geometry; ROS preparation binds the fitted tool and registered page.

**`place`** puts one artwork on a plane or cylinder chart and writes the design.
`--canvas-mm` is the chart extent, defaulting to the smallest chart that holds
the artwork, its offset and its margin. `--offset-mm`, `--rotation-deg`,
`--margin-mm` and `--mirror` are the placement intent. Two refusals live here:

- a **cylinder chart closing its full circumference**, where the band would
  meet itself and stop being a chart; the paper cylinder's own band is three
  quarters of the way round, everything but the bottom quarter it rests on;
- **artwork that leaves its canvas** once rotation, mirroring and stroke width
  are compiled. Every stroke goes through the target's own margin rule.

**`check`** validates a design with the browser's reader, recomputes every
digest in Python, and reports the path footprint about chart zero: the
target, the number of scheduled material strokes, the bounds, and the
enclosing radius. Acquired paths retain their native sequence and direction;
no fill planner, deduplication or travel optimizer substitutes for those paths.
`--preview` writes the artwork's checked SVG export. It does not predict deposition.

## `--radius-mm` is nominal

The radius and canvas a design carries are what the design assumes about its
target. They are not a measurement. `tatbot ros compile` accepts plane charts
only (a 100 × 150 mm chart is the stencil page); a cylinder design waits for
curved surfaces. Choosing a radius here never establishes registration,
calibration accuracy, or reachability.

## Where these verbs run

Placement and validation shell out to Node 22 with `web/inkmap`'s dependencies, so
every `design` verb carries the `design` role and hops to a node that has them
(`config/nodes.json`). The arm node does not need npm.

## Drawing a design

```text
tatbot ros compile design.json          # program.json + preview.svg, offline
tatbot ros draw program.json            # on the ROS 2 stack's arm
```

The ROS 2 stack registers the page, touches it and draws; `ros/README.md`
section 4 is the path from a design to paper, and
[design format](design-format.md#portable-design-handoff) the file itself.

The same file draws in the simulator: `tatbot sim generate
paper-draw -- --design design.json ...` records episodes of the design's
material strokes with the factory's own cameras and ink dips, on the flat pad
or, for a cylinder design, the paper cylinder. See
[simulation](simulation.md#drawing-a-portable-design).

## Checking it without hardware

```text
scripts/check inkgen cli docs
uvx --with-requirements scripts/tests/requirements.txt pytest -q scripts/tests/test_cli_design.py
cd python/tatbot_sim && uv run pytest -q tests/test_design_build.py
cd web/inkmap && node --experimental-strip-types --test tests/design.test.ts
```

The acquired catalogue binds each source image, native recipe, project, pen table
and output receipt. Browser and Python tests verify the unchanged records through
placement and save/restore. ROS tests verify path traversal and explicit rejection
of legacy artwork and physical resizing without regeneration.
