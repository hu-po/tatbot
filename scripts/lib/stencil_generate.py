#!/usr/bin/env python3
"""Generate the skin stencil: the fiducial tile of `stencil_tile.py`.

A grid of AprilTag 16h5 markers, every code on the sheet distinct, with rulers.
`--preset` names its layout (forearm, the default, or silicone); `--seed` (an
integer) picks its codes, a fresh one minted without it. The artwork
(stencil.png, tile.json, settings.json, tracking.json) goes into `--output`;
the printable PNG is print/<layout>-seed<seed>-300dpi.png, for the stencil
printer app. Stdlib at import, so the CLI plans with it; generating needs
numpy, OpenCV and Pillow.
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

import stencil_tile

DESIGNS = ("fiducial-tile",)
TILE = DESIGNS[0]


def add_arguments(parser):
    """Shared by the backend and the stdlib-only CLI registry."""
    parser.add_argument("--design", choices=DESIGNS, default=TILE,
                        help="fiducial-tile (the only design): a grid of AprilTag 16h5 markers")
    parser.add_argument("--preset", choices=sorted(stencil_tile.LAYOUTS),
                        help="the tile's layout: forearm (the default, 3 x 4 markers in 45 x 63 mm) or silicone "
                             "(5 x 7 places with corner anchors, 111 x 157.5 mm)")
    parser.add_argument("--seed", help="a non-negative integer that picks the codes; omitted mints a fresh one")
    parser.add_argument("--output", required=True, help="directory for the artwork; the printable PNG goes into its print/")


def validate(args):
    """Refuse what the design does not take, before planning or writing anything."""
    if args.seed is not None and not str(args.seed).isdigit():
        raise ValueError("a fiducial-tile seed is a non-negative integer")


def forward(args):
    """Backend argv for parsed options: every given option and nothing defaulted."""
    argv = ["--design", args.design, "--output", args.output]
    argv += [] if args.seed is None else ["--seed", args.seed]
    argv += [] if args.preset is None else ["--preset", args.preset]
    return argv


def generate(args):
    """Write the artwork and its printable PNG; return the summary main prints."""
    output = Path(args.output).expanduser()
    seed = secrets.randbelow(1_000_000) if args.seed is None else int(args.seed)
    layout = args.preset or "forearm"
    reference = stencil_tile.artwork(layout, seed, output)
    return {"design": args.design, "seed": str(seed), "preset": layout, "output": str(output),
            "pattern_id": reference["pattern_id"], "reference": reference,
            "sheets": {"png": str(output/"print"/f"{layout}-seed{seed}-{stencil_tile.DPI}dpi.png")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        validate(args)
        result = generate(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
