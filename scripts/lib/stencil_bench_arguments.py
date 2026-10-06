"""Stdlib argument contract for `tatbot vision stencil bench` and its backend."""

import argparse

BANKS = ("train", "holdout")
CAMERAS = ("mix", "wrist", "overhead")
TRACKERS = ("sift", "coded", "lightglue")
# The learned-matcher baseline's own dependencies, planned only when it is chosen.
LEARNED_REQUIREMENTS = ("torch==2.14.0", "kornia==0.8.3")
CANDIDATES = ("flower-of-life", "coded-flower-of-life")


def add_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate", choices=CANDIDATES, default="flower-of-life",
                        help="stencil generator to bench (default flower-of-life)")
    source.add_argument("--artwork", help="existing generated stencil directory (tracking.json, settings.json, stencil.png)")
    parser.add_argument("--seed", default="tatbot-42", help="generator seed for --candidate")
    parser.add_argument("--marked", action="store_true", help="include the print-ID grid (deterministic ID from the seed)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="generator setting, e.g. frame_mm=10 or stroke_mm=0.3; repeatable")
    parser.add_argument("--tracker", choices=TRACKERS, default="sift")
    parser.add_argument("--bank", choices=BANKS, default="train",
                        help="scene bank; holdout is for final reports, never for tuning")
    parser.add_argument("--scenes", type=int, default=128, help="scenes in the bank (every 8th blank, every 8th another print)")
    parser.add_argument("--camera", choices=CAMERAS, default="mix", help="mix = 60%% wrist D405, 40%% overhead")
    parser.add_argument("--degradation", choices=("full", "clean"), default="full",
                        help="clean = control bank with a crisp dark transfer on the same views")
    parser.add_argument("--workers", type=int, default=0, help="worker processes; 0 = min(8, CPUs)")
    parser.add_argument("--worst", type=int, default=12, help="scenes on the worst-case contact sheet")
    parser.add_argument("--calibration", help="camera calibration bundle; default the fleet golden copy under the log root")
    parser.add_argument("--robot-world", help="world_from_base registration; default the golden copy under the log root")
    parser.add_argument("--output", help="output directory; default the run directory")


def parser(description=None):
    result = argparse.ArgumentParser(description=description)
    add_arguments(result)
    return result


def validate(args):
    if not 1 <= args.scenes <= 5000:
        raise ValueError("scenes must be within 1-5000")
    if not 0 <= args.workers <= 64 or not 0 <= args.worst <= 64:
        raise ValueError("workers and worst must be within 0-64")
    if args.artwork and (args.set or args.marked):
        raise ValueError("--set and --marked apply to --candidate generation, not to --artwork")
    for pair in args.set or ():
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
