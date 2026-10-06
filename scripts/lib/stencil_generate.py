#!/usr/bin/env python3
"""Generate a stencil frame: the coded flower-of-life (default) or the legacy floral frame.

`--design coded-flower-of-life` (the default) writes the bead-and-teardrop frame of
`stencil_coded.py`: a bead on every lattice junction and a teardrop bit on every
edge. The print ID seeds the whole code, so the frame is the print code and there
is no separate print-ID grid. `--seed` names the print (its print ID derives from
it); without one a fresh seed is minted, so every ordinary print is distinct.
Settings are `--set key=value`, keys from `stencil_coded.DEFAULTS`.

`--design flower-of-life` writes the legacy seeded floral frame of
`stencil_frame.py` with its own options (`--frame-mm`, `--unmarked`, ...) and
default seed `tatbot-42`.

The artwork (stencil.png, stencil.svg, settings.json, tracking.json, and a coded
print's coded.json) goes directly into `--output`; print sheets go into
`<output>/print/` (`stencil_print.py`) unless `--no-sheets`. Stdlib at import, so
the CLI plans with it; generating needs Pillow, and the coded design numpy.
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

import stencil_frame
import stencil_print

DESIGNS = ("coded-flower-of-life", "flower-of-life")
CODED = DESIGNS[0]
# stencil_coded.DEFAULTS's keys, repeated here so this module stays stdlib-only
# (stencil_coded imports numpy); a test pins them together.
CODED_SETTINGS = frozenset({
    "width_mm", "height_mm", "margin_mm", "frame_mm", "spacing_mm", "stroke_mm", "knot_mm", "dpi",
    "window_radius", "optimize_rounds", "arcs", "bit", "knot", "ornament", "ornament_rate", "halo_mm",
    "teardrop_from", "teardrop_to", "seed_at"})
# stencil_frame's valued options, taken only with --design flower-of-life (plus --unmarked).
LEGACY_OPTIONS = (*(flag[2:].replace("-", "_") for flag, _, _ in stencil_frame.FRAME_OPTIONS), "instance_id")


def add_arguments(parser):
    """Shared by the backend and the stdlib-only CLI registry."""
    parser.add_argument("--design", choices=DESIGNS, default=CODED,
                        help="coded-flower-of-life (default): beads and teardrop bits that encode the print ID; "
                             "flower-of-life: the legacy floral frame")
    parser.add_argument("--seed", help="any text or number; coded: names the print and derives its print ID, "
                                       "omitted mints a fresh one; flower-of-life: default tatbot-42")
    parser.add_argument("--output", required=True,
                        help="directory for the artwork files; print sheets go into its print/")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="coded design setting, e.g. knot_mm=2.4 or bit=seed; repeatable")
    parser.add_argument("--no-sheets", action="store_true", help="write only the artwork, no print/ sheets")
    parser.add_argument("--fit-area", metavar="WxH",
                        help="also write print/paper-fit.pdf for a printer's printable area in mm (e.g. 203.2x269.4)")
    stencil_frame.add_frame_arguments(parser, defaults=False)


def legacy_given(args):
    """The flower-of-life options given on the command line; an omitted one is None."""
    given = {name: getattr(args, name) for name in LEGACY_OPTIONS if getattr(args, name) is not None}
    if args.unmarked:
        given["unmarked"] = True
    return given


def coded_settings(pairs):
    """`--set KEY=VALUE` pairs as coded generator settings: a number when one parses, else text."""
    settings = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        key = key.replace("-", "_")
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        if key not in CODED_SETTINGS:
            raise ValueError(f"unknown coded setting {key!r}; known: {', '.join(sorted(CODED_SETTINGS))}")
        try:
            settings[key] = float(value)
        except ValueError:
            settings[key] = value
    return settings


def frame_args(args):
    """stencil_frame's namespace: the given options over its own defaults, validated."""
    parser = argparse.ArgumentParser()
    stencil_frame.add_arguments(parser)
    given = legacy_given(args)
    if args.seed is not None:
        given["seed"] = args.seed
    parser.set_defaults(**given)
    frame = parser.parse_args(["--output=" + str(args.output)])
    stencil_frame.validate(frame)
    frame.output = Path(args.output).expanduser()
    return frame


def validate(args):
    """Refuse what the chosen design does not take, before planning or writing anything."""
    if args.design == CODED:
        given = legacy_given(args)
        if given:
            flags = ", ".join("--"+name.replace("_", "-") for name in given)
            raise ValueError(f"{flags}: flower-of-life only (--design flower-of-life); "
                             "the coded design takes --set key=value")
        coded_settings(args.set)
    else:
        if args.set:
            raise ValueError("--set is for the coded design; flower-of-life takes its own options (--frame-mm, ...)")
        frame_args(args)
    if args.fit_area is not None:
        if args.no_sheets:
            raise ValueError("--fit-area shapes the print sheets; drop --no-sheets")
        stencil_print.fit_area(args.fit_area)


def forward(args):
    """Backend argv for parsed options: every given option and nothing defaulted."""
    argv = ["--design", args.design, "--output", args.output]
    argv += [] if args.seed is None else ["--seed", args.seed]
    for pair in args.set or ():
        argv += ["--set", pair]
    for name, value in legacy_given(args).items():
        argv += ["--"+name.replace("_", "-"), *([] if value is True else [str(value)])]
    argv += ["--fit-area", args.fit_area] if args.fit_area else []
    argv += ["--no-sheets"] if args.no_sheets else []
    return argv


def generate(args):
    """Write the artwork and its print sheets; return the summary main prints."""
    output = Path(args.output).expanduser()
    if args.design == CODED:
        import stencil_coded
        seed = secrets.token_hex(6) if args.seed is None else args.seed
        stencil_coded.generate(seed, output, **coded_settings(args.set))
    else:
        frame = frame_args(args)
        stencil_frame.render(frame, stencil_frame.build(frame))
        seed = frame.seed
    reference = json.loads((output/"tracking.json").read_text())
    sheets = None
    if not args.no_sheets:
        fit = stencil_print.fit_area(args.fit_area) if args.fit_area else None
        sheets = stencil_print.write_sheets(output, output/"print", fit_area_mm=fit)
    return {"design": args.design, "seed": str(seed), "output": str(output),
            "pattern_id": reference["pattern_id"], "print_id": reference.get("physical_instance_id"),
            "reference": reference, "sheets": sheets}


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
