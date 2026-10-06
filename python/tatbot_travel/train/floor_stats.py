"""Keep every normalisation span of an exported flux3 base above a floor.

The flux3 history processors scale each state and command-delta channel by
its q01..q99 span, falling back to 1 only for an exactly zero span. A channel
that barely varies in the data (the v4 wrist roll: its deltas all zero, its
state 1.9 mrad of sensor noise) either turns small model errors into large
motions or clips at the first real deviation. This widens any span under the
floor about its midpoint, identically in the three processor files that must
agree (the history normaliser, the action-target normaliser, the postprocessor).

    python floor_stats.py <exported base dir> [--state-rad 0.05] [--delta-rad 0.005]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PATTERNS = ("policy_preprocessor_step_*_flux3_observation_history_normalizer.safetensors",
            "policy_preprocessor_step_*_flux3_action_target_normalizer.safetensors",
            "policy_postprocessor_step_*_flux3_action_history_unnormalizer.safetensors")


def floored(tensors: dict[str, torch.Tensor], floors: dict[str, float]) -> tuple[dict[str, torch.Tensor], list[str]]:
    out, notes = dict(tensors), []
    for stream, floor in floors.items():
        lo, hi = f"{stream}.q01", f"{stream}.q99"
        if lo not in out:
            continue
        span = out[hi] - out[lo]
        narrow = span < floor
        if narrow.any():
            mid = (out[hi] + out[lo]) / 2
            out[lo] = torch.where(narrow, mid - floor / 2, out[lo])
            out[hi] = torch.where(narrow, mid + floor / 2, out[hi])
            notes.append(f"{stream} joints {narrow.nonzero().flatten().tolist()} widened to {floor}")
    return out, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=Path)
    p.add_argument("--state-rad", type=float, default=0.05)
    p.add_argument("--delta-rad", type=float, default=0.005)
    args = p.parse_args()
    floors = {"state": args.state_rad, "action": args.delta_rad}
    files = [f for pattern in PATTERNS for f in sorted(args.base.glob(pattern))]
    if len(files) != 3:
        raise SystemExit(f"expected the three history processor files in {args.base}, found {len(files)}")
    for f in files:
        tensors, notes = floored(load_file(f), floors)
        save_file(tensors, f)
        print(f"{f.name}: {'; '.join(notes) or 'no span under the floor'}")


if __name__ == "__main__":
    main()
