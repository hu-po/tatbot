"""python -m tatbot_ink compile DESIGN [DESIGN ...] [--arm right] [--ee-tool ID] [--speed MM_S] [--inks FILE] [--stencil DIR] [--min-stroke-mm MM] [--mark-width-mm MM] [--keep-region=X,Y;X,Y;...] [-o DIR]

Writes DIR/program.json and DIR/preview.svg (DIR defaults to the current directory) and prints the
program's stats as one JSON line. --speed is mm/s like the CLI (`tatbot ros compile --speed 3.5`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tatbot_ink import CompileError, compile, write_preview, write_program
from tatbot_ink.place import MIN_STROKE_M


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tatbot_ink", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("compile", help="acquired artwork or placed design -> program.json + preview.svg")
    run.add_argument("design", type=Path, nargs="+",
                     help="acquired artwork or placed design; several acquired artworks are layers, drawn in order")
    run.add_argument("--arm", default="right", help="whose fitted tool (config/workspace.yaml); default right")
    run.add_argument("--ee-tool", dest="tool", default=None, help="config/tools/<id>.yaml instead of the fitted tool")
    run.add_argument("--speed", type=float, default=3.5, help="draw speed, mm/s (default 3.5)")
    run.add_argument("--inks", type=Path, default=None, help="inks file (format tatbot-inks)")
    run.add_argument("--max-segment-s", type=float, default=60.0, help="planned Cartesian seconds per computational chunk (default 60)")
    run.add_argument("--max-stroke-mm", type=float, default=None, metavar="MM",
                     help="draw a longer path as strokes of at most this length, each lifted and read again (default: whole)")
    run.add_argument("--repo", type=Path, default=None, help="checkout holding config/ (default $TATBOT_REPO)")
    run.add_argument("--at", default=None, metavar="X,Y",
                     help="centre the design here, mm in the page frame (x right, y toward the top of the print)")
    run.add_argument("--width", type=float, default=None, help="assert acquired canvas width in mm; resizing requires DBV3 regeneration")
    run.add_argument("--stencil", type=Path, default=None, metavar="DIR",
                     help="the generated print it is drawn on (its directory): place it inside that print's clear centre "
                          "instead of the nominal 128 x 172 mm page's")
    run.add_argument("--min-stroke-mm", type=float, default=MIN_STROKE_M * 1000, metavar="MM",
                     help=f"drop a path no wider than this that lies on its own ink's line (default {MIN_STROKE_M * 1000:g}; 0 keeps all)")
    run.add_argument("--mark-width-mm", type=float, default=None, metavar="MM",
                     help="count the pen's passes at this mark width (default: the tool's line width, else each path's generation width)")
    run.add_argument("--keep-region", default=None, metavar="X,Y;X,Y;...",
                     help="keep only the paths inside this polygon, mm in the page frame (x right, y toward the top of the "
                          "print), as placed; write --keep-region=... for a negative first x")
    run.add_argument("-o", "--out", type=Path, default=Path("."), help="output directory (default .)")
    args = parser.parse_args(argv)
    try:
        at_m = None if args.at is None else [float(v) / 1000.0 for v in args.at.split(",")]
        if at_m is not None and len(at_m) != 2:
            raise ValueError(f"--at takes X,Y in mm, got {args.at!r}")
        region_m = None if args.keep_region is None else [[float(v) / 1000.0 for v in pair.split(",")]
                                                          for pair in args.keep_region.replace(";", " ").split()]
        program = compile(args.design[0] if len(args.design) == 1 else args.design, arm=args.arm, tool_id=args.tool, repo=args.repo, speed_m_s=args.speed / 1000.0,
                          inks_path=args.inks, max_segment_s=args.max_segment_s,
                          at_m=at_m, width_m=None if args.width is None else args.width / 1000.0, stencil=args.stencil,
                          min_stroke_m=args.min_stroke_mm / 1000.0,
                          mark_width_m=None if args.mark_width_mm is None else args.mark_width_mm / 1000.0,
                          keep_region_m=region_m,
                          max_stroke_m=None if args.max_stroke_mm is None else args.max_stroke_mm / 1000.0)
    except (CompileError, OSError, KeyError, ValueError) as error:  # a malformed design is a refusal too
        print(f"tatbot_ink compile: {error}", file=sys.stderr)
        return 1
    program_path = write_program(program, args.out / "program.json")
    preview_path = write_preview(program, args.out / "preview.svg")
    print(json.dumps({**program["stats"], "program": str(program_path), "preview": str(preview_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
