"""Candidate stencil generators the bench can score. Each writes stencil.png,
settings.json and tracking.json into a directory and returns it.

Add a generator here to bench a new frame design; `--candidate <name>` selects
it and `--set key=value` passes generator settings.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def flower_of_life(seed, output, *, marked=False, **settings):
    """The current generator (scripts/lib/stencil_frame.py). A marked print gets a
    deterministic print-ID derived from the seed so a bench run is repeatable."""
    import stencil_frame
    parser = argparse.ArgumentParser()
    stencil_frame.add_arguments(parser)
    argv = ["--seed", str(seed), "--output", str(output)]
    for key, value in settings.items():
        argv += ["--"+key.replace("_", "-"), str(value)]
    if marked:
        argv += ["--instance-id", hashlib.sha256(f"bench:{seed}".encode()).hexdigest()[:24]]
    else:
        argv += ["--unmarked"]
    args = parser.parse_args(argv)
    stencil_frame.validate(args)
    args.output = Path(output)
    stencil_frame.render(args, stencil_frame.build(args))
    return Path(output)


def coded_flower_of_life(seed, output, *, marked=False, **settings):
    """The coded flower-of-life (scripts/lib/stencil_coded.py): one petal arc per lattice edge,
    its side a bit, a knot on every junction. The print ID seeds the whole code, so there is no
    separate print-ID grid to mark."""
    import stencil_coded
    if marked:
        raise ValueError("the coded candidate carries its print ID in its bits; --marked does not apply")
    return stencil_coded.generate(seed, output, **settings)


CANDIDATES = {"flower-of-life": flower_of_life, "coded-flower-of-life": coded_flower_of_life}


def parse_settings(pairs):
    settings = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        try:
            settings[key.replace("-", "_")] = float(value)
        except ValueError:
            settings[key.replace("-", "_")] = value
    return settings


def generate(name, seed, output, *, marked=False, settings=None):
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; known: {', '.join(sorted(CANDIDATES))}")
    return CANDIDATES[name](seed, output, marked=marked, **(settings or {}))
