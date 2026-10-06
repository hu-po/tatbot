"""`tatbot design` prepares source images or places acquired native DBV3 artwork.

`generate` supplies source imagery for a version 3 DBV3 job; `drawingbot generate`
acquires the finished paths. The retired `trace` verb refuses with that action.
`place` and `check` accept only acquired DBV3 records and use Inkmap's reader.
These offline design tools carry no motion authority.
"""
from __future__ import annotations

from tatbot_cli.registry import OFFLINE, verb

from ._common import uvmod

SIM_PROJECT = "python/tatbot_sim"
DESIGN_DOC = "docs/design.md"
DESIGN_INV = (
    "A design carries artwork, placement and nominal chart geometry only: no tool, no pose, no authority to move anything.",
    "Nominal chart dimensions are not measured pad dimensions; the session's compile rung binds a measured surface and refuses a chart that disagrees with it.",
    "Every artwork record and design is materialized and validated by the same reader the Inkmap editor uses; a rehashed envelope cannot detach a placement from its artwork.",
)


def _generate_args(p):
    p.add_argument("subject", help='what to draw, e.g. "a swallow carrying a rose"')
    p.add_argument("--out", required=True, help="PNG path; a .json sidecar records prompt, seed and model")
    p.add_argument("--seed", type=int, help="repeat an earlier image")
    p.add_argument("--style", help="style phrase replacing the default look descriptors")
    p.add_argument("--api-url", help="generator base URL (default: the node with role inkgen)")
    p.add_argument("--space", action="store_true",
                   help="use the public hosted generator on purpose")
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--no-autostart", action="store_true",
                   help="refuse (exit 5) instead of starting a stopped fleet generator")


@verb(effects=('read_files', 'write_files', 'network', 'start_process', 'gpu'), visibility="public",
      noun="design", verb="generate", tier=OFFLINE,
      summary="ask the generator for one tattoo-flash PNG (starting it if it is not up)",
      role="design", auto_hop=True, args=_generate_args,
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/design_cli.py", "web/inkgen/app.py"),
      passthrough="tatbot_sim.inkmap.design_cli generate",
      example=("a swallow carrying a rose", "--out", "/tmp/tatbot-design/swallow.png"), doc=DESIGN_DOC,
      invariants=DESIGN_INV + (
          "Only the fleet generator is ever started: an address given with --api-url is probed, never started or woken.",
          "With no `inkgen` role configured and no --api-url, an interactive generation uses the public Space; --space selects it on purpose.",
          "A start is refused (exit 6) when the GPU has less free memory than the model needs, naming what holds it.",
          "The generator stops itself when idle; nothing here keeps it up."))
def design_generate(ctx, ns, rest):
    argv = ["generate", ns.subject, "--out", ns.out, "--timeout-s", str(ns.timeout_s)]
    for flag, value in (("--seed", ns.seed), ("--style", ns.style), ("--api-url", ns.api_url)):
        if value is not None:
            argv += [flag, str(value)]
    if ns.no_autostart:
        argv.append("--no-autostart")
    if ns.space:
        argv.append("--space")
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.design_cli", *argv, *rest,
                 notes=["supply the image to a version 3 DBV3 job: `tatbot drawingbot generate` acquires its paths"])


def _trace_args(p):
    p.add_argument("image", nargs="?", default="retired-source.svg", help="PNG, JPEG, or SVG")
    p.add_argument("--size-mm", nargs=2, type=float, default=(30, 30), metavar=("W", "H"),
                   help="physical size; a raster keeps its aspect ratio inside this box")
    p.add_argument("--name", help="artwork name (default: the file's stem)")
    p.add_argument("--width-mm", type=float, help="planning stroke width (default: the tool's recorded line width, else 0.3)")
    p.add_argument("--tool", help="tool id whose datasheet `line:` width is the planning width when --width-mm is absent")
    p.add_argument("--strokes", choices=("outline", "centerline"), default=None,
                   help="a stroked SVG path as a filled outline the planner rings (default), "
                        "or as one centerline drawn once at the planning width")
    p.add_argument("--generation", help="generation sidecar JSON (default: the one beside the image)")
    p.add_argument("--out", default="retired-artwork.json", help="artwork record JSON to write")


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public",
      noun="design", verb="trace", tier=OFFLINE,
      summary="retired tracer: use drawingbot generate and import acquired artwork.json",
      role="design", auto_hop=True, args=_trace_args,
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/design_cli.py", "web/inkmap/tools/artwork.ts"),
      passthrough="tatbot_sim.inkmap.design_cli trace",
      doc=DESIGN_DOC,
      invariants=DESIGN_INV + (
          "This retired verb refuses with a DBV3 generation action; it produces no traced artwork.",))

def design_trace(ctx, ns, rest):
    argv = ["trace", ns.image, "--size-mm", str(ns.size_mm[0]), str(ns.size_mm[1]), "--out", ns.out]
    for flag, value in (("--name", ns.name), ("--width-mm", ns.width_mm), ("--generation", ns.generation),
                        ("--tool", ns.tool), ("--strokes", ns.strokes)):
        if value is not None:
            argv += [flag, str(value)]
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.design_cli", *argv, *rest)


def _place_args(p):
    p.add_argument("artwork", help="acquired artwork.json from `tatbot drawingbot generate`")
    p.add_argument("--target", choices=("plane", "cylinder"), default="plane")
    p.add_argument("--radius-mm", type=float, help="nominal cylinder radius; not a measurement")
    p.add_argument("--canvas-mm", nargs=2, type=float, metavar=("U", "V"),
                   help="chart extent (default: the smallest chart holding the artwork)")
    p.add_argument("--offset-mm", nargs=2, type=float, metavar=("U", "V"), help="artwork offset in the chart")
    p.add_argument("--rotation-deg", type=float, help="rotate the artwork in the chart")
    p.add_argument("--margin-mm", type=float, help="keep-clear border inside the chart")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--name", help="design name (default: the artwork's)")
    p.add_argument("--placement-id", help="placement id inside the design (default placement-1)")
    p.add_argument("--out", required=True, help="design JSON to write")


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public",
      noun="design", verb="place", tier=OFFLINE,
      summary="place one artwork on a plane or cylinder chart as a portable design",
      role="design", auto_hop=True, args=_place_args,
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/design_cli.py", "web/inkmap/tools/artwork.ts"),
      passthrough="tatbot_sim.inkmap.design_cli place",
      example=("/tmp/tatbot-design/artwork.json", "--target", "cylinder", "--radius-mm", "40",
               "--canvas-mm", "80", "110", "--out", "/tmp/tatbot-design/design.json"),
      doc=DESIGN_DOC,
      invariants=DESIGN_INV + (
          "--radius-mm is the design's assumption about the target; the mapper's fitted radius governs at preparation.",
          "A cylinder chart closing its full circumference is refused: a band that meets itself is no longer a chart. The paper cylinder's band is three quarters of the way round.",
          "Artwork that leaves its canvas once rotation, mirroring and stroke width are compiled is refused here, not clamped."))
def design_place(ctx, ns, rest):
    argv = ["place", ns.artwork, "--target", ns.target, "--out", ns.out]
    for flag, value in (("--radius-mm", ns.radius_mm), ("--rotation-deg", ns.rotation_deg),
                        ("--margin-mm", ns.margin_mm), ("--name", ns.name),
                        ("--placement-id", ns.placement_id)):
        if value is not None:
            argv += [flag, str(value)]
    for flag, pair in (("--canvas-mm", ns.canvas_mm), ("--offset-mm", ns.offset_mm)):
        if pair is not None:
            argv += [flag, str(pair[0]), str(pair[1])]
    if ns.mirror:
        argv.append("--mirror")
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.design_cli", *argv, *rest)


def _check_args(p):
    p.add_argument("design", help="design JSON")
    p.add_argument("--preview", help="write the artwork's checked preview SVG here")
    p.add_argument("--fill-style", choices=("concentric", "hatch"),
                   help="paint planner to report: concentric inset rings (default) or contour-then-hatch")


@verb(effects=('read_files', 'write_files', 'start_process'), visibility="public", output="json",
      noun="design", verb="check", tier=OFFLINE,
      summary="validate a design with the browser reader and report its material footprint",
      args=_check_args,
      wraps=("python/tatbot_sim/src/tatbot_sim/inkmap/design_cli.py", "web/inkmap/tools/artwork.ts"),
      passthrough="tatbot_sim.inkmap.design_cli check",
      example=("/tmp/tatbot-design/design.json",), doc=DESIGN_DOC,
      invariants=DESIGN_INV + (
          "The footprint is chart geometry about chart zero; it establishes no measured pose, registration, or reachability.",
          "The preview is the artwork's own checked preview SVG, never a prediction of what ink will do."))
def design_check(ctx, ns, rest):
    argv = ["check", ns.design]
    if ns.preview:
        argv += ["--preview", ns.preview]
    if ns.fill_style:
        argv += ["--fill-style", ns.fill_style]
    return uvmod(ctx, SIM_PROJECT, "tatbot_sim.inkmap.design_cli", *argv, *rest)
