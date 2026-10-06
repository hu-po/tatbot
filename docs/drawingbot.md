# DrawingBotV3 experiments

## Single physical acquisition

`tatbot drawingbot generate JOB --out NEW_DIRECTORY --app INSTALLATION`
acquires one drawing without loading Inkmap, ROS or the simulator. A
`tatbot.dbv3-job/3` document supplies `id`, `source` (`file`, exact `sha256`,
`provenance` in the shared ArtworkSource format), `size_mm: [width, height]`, `pen_width_mm`, `pfm`, native
`settings` including an explicit integer `Random Seed`, `drawing_set`, and `state` with all
drawing-area and export settings. Source paths resolve relative to the job.
Use `scripts/lib/drawingbot/defaults/state.json` as the explicit state template.
Use `scripts/lib/drawingbot/defaults/drawing-set.json` as the explicit pen-table
template. Each pen has a unique `id`, a human `name`, a `type` label, `enabled`,
8-bit `rgba`, integer `weight`, and `stroke_factor`. The set records its
`name`, `type`, `distribution_order`, `distribution_type`, `color_separation`
and ordered `pens`. One set with fixed opaque colors and `Default` separation
is supported. Unsupported native distribution choices fail readback.

Native pen names carry the logical IDs so the normal `%NAME%` SVG layer
export preserves identity even when human names and colors repeat. Effective
readback records the actual native name/type, set/row IDs and export group;
`requested_name` retains the human label. `svgLayerNaming` must be `%NAME%`.
Keep disabled and zero-path pens in the requested table; their settings remain
in recipe evidence. Changing colors or factors requires a new acquisition.

The worker verifies source bytes before launching the app, reads back page
dimensions, physical pen width, the full pen table and requested PFM settings, and checks the
export's physical dimensions. A successful acquisition publishes `result.json`
only after these checks. Outputs retain a portable job, native
`tatbot.dbv3-recipe/3` bundle, effective settings, raw/normalized export,
`artwork.json` (shared `InkmapArtwork/2` containing `TattooProgram/1` metric paths), a derived
`preview.svg`, and `decoding.json` with decoder/runtime identities; failures retain diagnostic artifacts. An output directory is never
reused. Portable native projects omit cached drawings, UI state and temporary
batch directories. This command is native acquisition; production artwork preparation runs separately and no robot is contacted.

The bounded native decoder preserves exported pen groups, path order,
direction, subpaths, closedness and repeated passes. It admits absolute
M/L/Q/C/Z centerlines and uniform matrix transforms; unsupported paint or
geometry fails acquisition. Quadratic/cubic subdivision has a 5 micrometre
chord bound and a finite point/depth budget. It also checks control-polygon
length so collinear reversals survive. This bound concerns decoding the
exported curves, not the upstream DBV3 simplification error or physical ink.
SVG's rounded display widths are checked against each pen's native generation width
(with a recorded 1% serialization tolerance); saved path widths use that
explicit base width multiplied by the effective native stroke factor. The
decoder refuses unknown/disabled groups, duplicate group IDs, color drift and
width drift. Measured deposited width remains a tool property.

Job and recipe version 2 inputs are refused with a version error. For a new
single-black job, change its schema to version 3 and add the pen-table template;
existing frozen artwork remains valid input to ROS preparation. A historical
recipe needs a new acquisition rather than silently adding missing pen evidence.

## Prepare and review

```sh
tatbot drawingbot generate /path/to/job.json --out /path/to/new-acquisition --app /path/to/drawingbotv3
tatbot ros compile /path/to/new-acquisition/artwork.json --at=0,0 --width 20 --speed 3.5 -o /path/to/prepared
```

Install and activate Premium 1.6.22 normally. The version-specific bridge uses
Java 21, JavaFX and Xvfb; it is not an official headless API. Application and
activation remain private. Native acquisition needs `javac`, `java` and
`xvfb-run`. Install `uv` for the drawing commands: `tatbot drawingbot`,
`tatbot research` and local ROS artwork preparation resolve the same locked
Python 3.12.14 environment with NumPy 2.5.3 and PyYAML 6.0.3. The first command
downloads missing runtime dependencies into uv's cache; subsequent commands
reuse them. The repository and an unrelated active virtual environment do not
supply those dependencies. Neither route requires Inkmap's Node dependencies
or the simulator.

The runtime definition is `scripts/lib/drawing_python.py` and its adjacent
lockfile. Change them together with
`uv lock --script scripts/lib/drawing_python.py --default-index https://pypi.org/simple`;
a stale lock refuses execution. The launcher resolves the interpreter first,
then executes Python directly so uv is absent from the cancellation path.
ROS node processes retain their deployed ROS environment.

Open `artwork.json` in Inkmap, or inspect the acquired `preview.svg` and the
prepared `preview.svg`. Preparation preserves DBV3 order and physical size;
`--width` is an assertion, and changing size requires regeneration. Multiple
pens require explicit ink bindings. See ROS preparation in `ros/README.md`, section 4.2.

Study comparisons use **File → Review study…** in Inkmap. Build a portable
review with `tatbot research review`, then import downloaded human observations
with `tatbot research feedback`; see [research](research.md). Both review and
ordinary placement render the shared acquired artwork. The separate gallery
server and its CLI have been removed. Original experiment galleries and their
preferences remain historical files, without a maintained legacy review route.

## Native replay and container

```sh
tatbot drawingbot export /path/to/acquisition/recipe --out /path/to/new-replay --app /path/to/drawingbotv3
```

Each job owns a fresh JavaFX/Xvfb worker. A saved recipe contains source bytes,
a portable native project, explicit export/global settings, application and
bridge hashes, Java identity and rendering flags. Replay verifies these
inputs and reports exact normalized-export matches separately from identity.
It does not compile a robot program or transfer appearance preferences.
`--migrate-runtime` explicitly records a new native runtime baseline.

`tatbot drawingbot container build|smoke|export --out DIRECTORY` retains the
private mounted-application container workflow. Use `--app`, `--recipe` and
a dedicated activated `--state` directory for export. Never copy a host's
activation into an image. A container smoke test verifies JavaFX/bridge
startup; licensed generation requires a separate successful job.

## Physical assumptions

Generation width is the DBV3 drawing-area pen width, distinct from its pen
factor and the measured deposited line. The worker retains the requested logical pen table and verifies native
readback, including disabled and zero-path pens. Size, seed and native PFM settings are explicit job inputs.
No stock source count, fixed 50 × 75 mm size or 24-variant requirement remains
in acquisition/preparation. Unknown image-model versions and seeds stay null.

Ballpoints use cartridge ink and never dip. A 3RL designation supplies no
measured deposited width or replenishment budget. Future dip/lift policies
need measurements and an implementation in the existing ROS executor.

Paired robot trials use [DBV3 research](research.md): frozen candidates, both
prepared sides, persistent slot ownership and the existing ROS draw/inspect
workflow. The legacy compiler-selection runner is not part of that route.

## Bundled acquired examples

Inkmap's default catalogue is `dbv3-acquired-v1`: three public-safe original
raster sources acquired with the installed Premium 1.6.22 native worker. Each
30 × 30 mm acquisition uses a 0.3 mm black pen and retains the unmodified
`InkmapArtwork/2`, version 3 job/recipe, effective pen table, raw export,
normalized export and decoding receipt under `web/inkmap/public/designs/`.
Application and activation data are excluded. Regenerate the included job into
a new directory to change physical dimensions or pen settings; previews cannot
serve as acquisition input to Inkmap or ROS preparation.
