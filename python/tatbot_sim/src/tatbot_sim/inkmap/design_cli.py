"""Prepare source images or place immutable acquired native DBV3 artwork.

`generate` writes source PNGs; native `drawingbot generate` acquires their paths.
The retired `trace` verb refuses. Placement and checking carry no motion authority.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from tatbot_sim.human_rep.contracts import FILL_STYLES
from tatbot_sim.inkmap.artwork import artwork_preview
from tatbot_sim.inkmap.design import scan_coverage, validate_design
from tatbot_sim.inkmap.design_build import (
    DEFAULT_WIDTH_MM,
    STROKE_MODES,
    DesignBuildError,
    design_from_artwork,
    write_json,
)

EXIT_REFUSED = 3
EXIT_UNREACHABLE = 5
EXIT_BUSY = 6


def _fail(message: str, code: int = EXIT_REFUSED) -> int:
    print(f"tatbot design: {message}", file=sys.stderr)
    return code


def _read(path: Path) -> bytes:
    try:
        return path.expanduser().read_bytes()
    except OSError as exc:
        raise DesignBuildError(f"cannot read {path}: {exc}") from exc


def _read_json(path: str, *, schema: str | None = None, produced_by: str = "") -> dict[str, Any]:
    """Read one JSON artifact, refusing rather than tracebacking on anything else."""
    try:
        document = json.loads(_read(Path(path)))
    except json.JSONDecodeError as exc:
        raise DesignBuildError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DesignBuildError(f"{path} is not a JSON object")
    if schema is not None and document.get("schema") != schema:
        raise DesignBuildError(f"{path} is {document.get('schema')!r}, expected {schema!r}"
                               + (f" from `{produced_by}`" if produced_by else ""))
    return document


def _generate(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.inkgen_client import InkgenBusyError, InkgenUnreachableError, generate

    try:
        reply = generate(args.subject, seed=args.seed, style=args.style, api_url=args.api_url,
                         timeout_s=args.timeout_s, autostart=not args.no_autostart, space=args.space)
    except InkgenBusyError as exc:  # the GPU is someone else's right now
        return _fail(str(exc), EXIT_BUSY)
    except InkgenUnreachableError as exc:
        return _fail(str(exc), EXIT_UNREACHABLE)
    except DesignBuildError as exc:
        return _fail(str(exc))
    out = Path(args.out).expanduser()
    if out.exists():
        return _fail(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(reply["png"])
    sidecar = {"schema": "tatbot.design-generation/1", "subject": args.subject,
               "seed": reply["seed"], "prompt": reply["prompt"], "model": reply["model"],
               "model_revision": reply.get("model_revision"), "settings": reply.get("settings"),
               "backend": reply.get("backend"),
               "api_url": reply["api_url"], "png": out.name,
               "png_sha256": reply["png_sha256"], "seconds": reply.get("seconds")}
    write_json(out.with_suffix(".json"), sidecar)
    revision = reply.get("model_revision")
    print(f"wrote {out} ({len(reply['png'])} bytes, seed {reply['seed']}, model {reply['model']}"
          f"{'@' + revision[:12] if revision else ''}, {reply.get('backend', 'unknown')} backend)")
    print("next: supply this source to a version 3 DBV3 job; tatbot drawingbot generate JOB --out ACQUISITION --app INSTALLATION")
    return 0


def _trace(args: argparse.Namespace) -> int:
    return _fail("The SVG/raster tracer is retired. Use `tatbot drawingbot generate JOB --out ACQUISITION --app INSTALLATION`, then `tatbot design place ACQUISITION/artwork.json`.")


def tool_line_width_mm(tool_id: str) -> tuple[float, dict[str, Any]]:
    """The fitted tool's recorded drawn-line width, from its datasheet."""
    from tatbot_sim import tools
    from tatbot_sim.repo import repo_root

    try:
        tool = tools.registry().load_tool(tool_id, repo_root())
    except (OSError, ValueError) as exc:
        raise DesignBuildError(f"cannot load tool {tool_id!r}: {exc}") from exc
    if tool.line_width_m is None:
        raise DesignBuildError(f"tool {tool_id!r} records no `line:` width in its datasheet; give --width-mm "
                               "or record the drawn line (docs/tools.md)")
    return tool.line_width_m * 1000.0, dict(tool.line)


def _planning_width_mm(args: argparse.Namespace) -> tuple[float, str]:
    """`--width-mm` when given; else the tool's recorded line; else the default."""
    if args.width_mm is not None:
        return float(args.width_mm), "--width-mm"
    if args.tool:
        width, line = tool_line_width_mm(args.tool)
        return width, f"{args.tool} line, {line['status']} {line['utc']}"
    return DEFAULT_WIDTH_MM, "default"



def _require_acquired(record):
    from tatbot_contracts.artwork import validate_path_artwork
    validate_path_artwork(record)
    if (record['conversion']['adapter'] != 'dbv3-batik-paths/1' or record['conversion']['recipe_sha256'] is None
            or record['program']['provenance']['producer'] != 'dbv3-batik-paths/1'):
        raise DesignBuildError('Generate with DrawingBot V3 and import artwork.json; the legacy tracer is retired')

def _place(args: argparse.Namespace) -> int:
    # Argument consistency first: a caller who named an impossible target should
    # hear about that, not about whichever file happened to be missing too.
    if args.target == "cylinder" and args.radius_mm is None:
        return _fail("a cylinder target needs --radius-mm (the nominal radius of the target)")
    if args.target == "plane" and args.radius_mm is not None:
        return _fail("--radius-mm belongs to a cylinder target")
    record = _read_json(args.artwork, schema="tatbot.inkmap-artwork/2", produced_by="tatbot drawingbot generate")
    _require_acquired(record)
    design = design_from_artwork(
        record, name=args.name, placement_id=args.placement_id, kind=args.target,
        radius_m=None if args.radius_mm is None else args.radius_mm / 1000,
        canvas_m=None if args.canvas_mm is None else tuple(v / 1000 for v in args.canvas_mm),
        anchor_uv_m=tuple(v / 1000 for v in args.offset_mm), margin_m=args.margin_mm / 1000,
        rotation_rad=math.radians(args.rotation_deg), mirrored=args.mirror)
    out = write_json(Path(args.out).expanduser(), design)
    target = design["placements"][0]["placement"]["target"]
    canvas = [v * 1000 for v in target["canvas_m"]]
    print(f"wrote {out}: {design['name']}, {target['kind']} chart {canvas[0]:.3f} x {canvas[1]:.3f} mm")
    if target["kind"] == "cylinder":
        print(f"  nominal radius {target['radius_m'] * 1000:.3f} mm — nominal only; "
              "`tatbot ros compile` accepts plane charts")
    print(f"next: tatbot design check {out}")
    return 0


def schedule_summary(schedules: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-tier table and dedup totals of a design's stroke schedule, in mm."""
    tiers: dict[int, dict[str, float]] = {}
    for report in schedules:
        for row in report["tiers"]:
            entry = tiers.setdefault(int(row["tier"]), {"strokes": 0, "contact_mm": 0.0,
                                                        "dedup_dropped_strokes": 0, "dedup_dropped_mm": 0.0})
            entry["strokes"] += int(row["strokes"])
            entry["contact_mm"] += float(row["length_m"]) * 1000.0
            entry["dedup_dropped_strokes"] += int(row["dropped_strokes"])
            entry["dedup_dropped_mm"] += float(row["dropped_m"]) * 1000.0
    total_mm = sum(entry["contact_mm"] for entry in tiers.values())
    return {
        "tiers": [{"tier": tier, **entry, "share": entry["contact_mm"] / total_mm if total_mm else 0.0}
                  for tier, entry in sorted(tiers.items())],
        "dedup_dropped_strokes": sum(int(r["dedup_dropped_strokes"]) for r in schedules),
        "dedup_dropped_mm": sum(float(r["dedup_dropped_m"]) for r in schedules) * 1000.0,
        "pen_up_travel_mm": {key: sum(float(r["pen_up_travel_m"][key]) for r in schedules) * 1000.0
                             for key in ("generation_order", "scheduled")},
    }


def _tier_table(report: dict[str, Any]) -> str:
    lines = ["tier  strokes  contact_mm  share  dropped  dropped_mm",
             *(f"{row['tier']:>4}  {row['strokes']:>7}  {row['contact_mm']:>10.0f}  {row['share']:>5.0%}  "
               f"{row['dedup_dropped_strokes']:>7}  {row['dedup_dropped_mm']:>10.0f}" for row in report["tiers"])]
    travel = report["pen_up_travel_mm"]
    lines.append(f"pen-up travel {travel['scheduled']:.0f} mm scheduled ({travel['generation_order']:.0f} mm in "
                 f"generation order); {report['dedup_dropped_strokes']} redundant fill strokes dropped "
                 f"({report['dedup_dropped_mm']:.0f} mm)")
    return "\n".join(lines)


def _check(args: argparse.Namespace) -> int:
    document = _read_json(args.design, schema="tatbot.inkmap-design/1", produced_by="tatbot design place")
    design = validate_design(document)
    for record in design["artworks"].values():
        _require_acquired(record)
    coverage = scan_coverage(design, fill_style=args.fill_style)
    report: dict[str, Any] = {
        "schema": "tatbot.design-check/1", "name": design["name"], "fill_style": args.fill_style,
        "design_sha256": design["content_sha256"],
        "artworks": sorted(design["artworks"]),
        "placements": [item["id"] for item in design["placements"]],
        "target": coverage["target"], "material_strokes": coverage["material_strokes"],
        "bounds_uv_mm": [[v * 1000 for v in pair] for pair in coverage["bounds_uv_m"]],
        "radius_mm": coverage["radius_m"] * 1000,
        **schedule_summary(coverage["schedule"]),
    }
    if args.preview:
        # The record's own checked preview, recompiled and compared by the
        # reader above. It shows the artwork, never a predicted deposition.
        first = design["placements"][0]["artwork_id"]
        Path(args.preview).expanduser().write_text(artwork_preview(design["artworks"][first]))
        report["preview"] = str(args.preview)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(_tier_table(report), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tatbot design", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="ask the generator for one tattoo-flash PNG")
    generate.add_argument("subject")
    generate.add_argument("--out", required=True, help="PNG path; a .json sidecar records prompt/seed/model")
    generate.add_argument("--seed", type=int)
    generate.add_argument("--style", help="style phrase replacing the default look descriptors")
    generate.add_argument("--api-url", help="generator base URL; only the fleet generator is ever started")
    generate.add_argument("--space", action="store_true",
                          help="use the public hosted generator on purpose")
    generate.add_argument("--timeout-s", type=float, default=180.0)
    generate.add_argument("--no-autostart", action="store_true",
                          help="refuse instead of starting a stopped fleet generator")
    generate.set_defaults(func=_generate)

    trace = sub.add_parser("trace", help="retired tracer: generate with native DBV3 and import artwork.json")
    trace.add_argument("image", help="PNG, JPEG, or SVG")
    trace.add_argument("--size-mm", nargs=2, type=float, required=True, metavar=("W", "H"),
                       help="physical size; a raster keeps its aspect ratio inside this box")
    trace.add_argument("--name")
    trace.add_argument("--width-mm", type=float, help="planning stroke width (default: the tool's recorded line, else 0.3)")
    trace.add_argument("--tool", help="tool id whose datasheet `line:` width sets the planning width when --width-mm is absent")
    trace.add_argument("--strokes", choices=STROKE_MODES, default="outline",
                       help="a stroked SVG path as a filled outline the planner rings (default), "
                            "or as one centerline drawn once at the planning width")
    trace.add_argument("--generation", help="generation sidecar JSON (default: alongside the image)")
    trace.add_argument("--out", required=True)
    trace.set_defaults(func=_trace)

    place = sub.add_parser("place", help="place one artwork on a plane or cylinder chart")
    place.add_argument("artwork")
    place.add_argument("--target", choices=("plane", "cylinder"), default="plane")
    place.add_argument("--radius-mm", type=float, help="nominal cylinder radius; not a measurement")
    place.add_argument("--canvas-mm", nargs=2, type=float, metavar=("U", "V"),
                       help="chart extent (default: the smallest chart holding the artwork)")
    place.add_argument("--offset-mm", nargs=2, type=float, default=[0.0, 0.0], metavar=("U", "V"))
    place.add_argument("--rotation-deg", type=float, default=0.0)
    place.add_argument("--margin-mm", type=float, default=0.0)
    place.add_argument("--mirror", action="store_true")
    place.add_argument("--name")
    place.add_argument("--placement-id", default="placement-1")
    place.add_argument("--out", required=True)
    place.set_defaults(func=_place)

    check = sub.add_parser("check", help="validate a design with the browser reader and report its footprint")
    check.add_argument("design")
    check.add_argument("--preview", help="write the artwork's checked preview SVG here")
    check.add_argument("--fill-style", choices=FILL_STYLES, default="concentric",
                       help="paint planner: concentric inset rings (default) or contour-then-hatch")
    check.set_defaults(func=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DesignBuildError as exc:
        return _fail(str(exc))
    except ValueError as exc:  # contract refusals from the shared readers
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
