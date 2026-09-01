#!/usr/bin/env python3
"""Check what a generated dataset — or a night's worth of shards — actually is.

A long generation run is not one dataset but a family of them, and the ways it
goes wrong are quiet. The fitted tool binds at PACKAGE IMPORT
(``tatbot_sim.tools.active_tool``), so a launcher that failed to set
``TATBOT_TOOL_ID`` before the interpreter started produces a dataset that runs
fine, looks fine, and carries the wrong tool's geometry under the right
distribution's name. Nothing else in the repo compares a shard against the
distribution it claims to be.

What it checks, per shard:

  * the tool recorded in run_meta is the one its distribution is defined to run
    (``tatbot_sim.distributions``), which is the import-time binding failure;
  * pigment-field snapshots, when present, number exactly one per episode;
  * the episode table agrees with info.json on how many episodes there are;
  * shards of one distribution agree with each other on tool and substrate.

It also surfaces what a run would otherwise only whisper: batches skipped as
unplannable (``run_meta.skipped_batches``), and episodes that never moved any
pigment. Both leave the episode count looking full.

    cd python/tatbot_sim && uv run python ../../scripts/sim_dataset_audit.py \
        --path ~/tatbot-sim/datasets/overnight-20260827

Exits non-zero if any check fails, so it can gate a training mix.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro


def _distributions() -> dict:
    """The distribution->tool contract, loaded WITHOUT importing the package.

    ``import tatbot_sim`` pulls in the render stack and resolves the fitted tool
    at import time, and the SAPIEN teardown that follows aborts the interpreter
    (exit 134) after main() has already succeeded -- fatal for a script whose
    exit code is supposed to gate a training mix. distributions.py is
    stdlib-only at module scope, so loading the file by path gets the contract
    and none of that. Same reason tatbot_sim.tools loads tool_spec.py by path
    rather than vendoring a copy that would drift.
    """
    path = (Path(__file__).resolve().parent.parent
            / "python/tatbot_sim/src/tatbot_sim/distributions.py")
    spec = importlib.util.spec_from_file_location("_tatbot_distributions", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"distribution registry not importable at {path}")
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: Distribution is a dataclass under postponed
    # annotations, and dataclasses resolves those through sys.modules. Same
    # trap tatbot_sim.tools documents for tool_spec.py.
    sys.modules["_tatbot_distributions"] = mod
    spec.loader.exec_module(mod)
    return mod.DISTRIBUTIONS


DISTRIBUTIONS = _distributions()


@dataclass
class Args:
    path: Path
    """A single dataset, or a directory holding several shards."""
    verbose: bool = False
    """Print every shard, not only the problems and the totals."""
    allow_legacy: bool = False
    """Accept shards from the gripper-held-v1 embodiment (before 2026-08-30).
    Off by default: a new-embodiment split that quietly includes them trains
    on a tool the robot no longer has and cameras it no longer aims that way."""


def _shards(root: Path) -> list[Path]:
    if (root / "meta/info.json").exists():
        return [root]
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith(".")
                  and (p / "meta/info.json").exists())


def _episode_table(ds: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(ds / "meta/episodes/*/*.parquet")))
    return pd.concat([pd.read_parquet(f, columns=["episode_index", "tasks", "length"])
                      for f in files], ignore_index=True)


def main(a: Args) -> int:
    root = a.path.expanduser()
    shards = _shards(root)
    if not shards:
        raise SystemExit(f"no finished datasets under {root} "
                         "(a shard still generating has no meta/info.json)")

    problems: list[str] = []
    by_dist: dict[str, dict] = {}

    for ds in shards:
        with open(ds / "meta/info.json") as fh:
            info = json.load(fh)
        with open(ds / "meta/run_meta.json") as fh:
            rm = json.load(fh)
        cfg = rm.get("config", {})
        dist = cfg.get("distribution")
        tool = rm.get("tool", {}).get("tool_id")
        n_ep = info["total_episodes"]

        # Which follower made it. Shards stamped before 2026-08-30 carry no
        # embodiment at all, which is itself the gripper-held-v1 signature.
        embodiment = rm.get("tool", {}).get("embodiment") or "gripper-held-v1"
        if embodiment != "fixed-mount-v2" and not a.allow_legacy:
            problems.append(
                f"{ds.name}: embodiment {embodiment!r} — generated for the gripper-held "
                "follower (tool in the jaws, upper/lower wrist cameras), not the fixed "
                "mount. Keep for history; pass --allow-legacy to audit it as such.")

        # The import-time binding failure: right name, wrong geometry.
        if dist in DISTRIBUTIONS:
            want = DISTRIBUTIONS[dist].tool_id
            if tool != want:
                problems.append(
                    f"{ds.name}: claims distribution {dist!r} but recorded tool "
                    f"{tool!r}; that distribution runs {want!r}. The tool binds at "
                    "package import — this shard was built with the wrong geometry.")
        elif dist is None:
            problems.append(f"{ds.name}: no distribution recorded — assembled by "
                            "hand from flags, so it cannot claim to be one of the three.")

        fields = sorted((ds / "meta/fields").glob("*.png"))
        if fields and len(fields) != n_ep:
            problems.append(f"{ds.name}: {len(fields)} pigment fields for {n_ep} "
                            "episodes — snapshots and episodes must be 1:1.")

        try:
            ep = _episode_table(ds)
            if len(ep) != n_ep:
                problems.append(f"{ds.name}: episode table has {len(ep)} rows, "
                                f"info.json says {n_ep}.")
            tasks = {t[0] if not isinstance(t, str) else t for t in ep.tasks}
        except Exception as exc:                      # a shard too broken to read
            problems.append(f"{ds.name}: episode table unreadable — {exc}")
            tasks = set()

        skipped = rm.get("skipped_batches") or []
        idle = [e for e in rm.get("episodes", []) if not e.get("engaged", True)]

        d = by_dist.setdefault(dist, {"episodes": 0, "frames": 0, "shards": 0,
                                      "tools": set(), "substrates": set(),
                                      "prompts": set(), "skipped": 0, "idle": 0})
        d["episodes"] += n_ep
        d["frames"] += info["total_frames"]
        d["shards"] += 1
        d["tools"].add(tool)
        d["substrates"].add(rm.get("tool", {}).get("substrate")
                            or cfg.get("distribution"))
        d["prompts"] |= tasks
        d["skipped"] += len(skipped)
        d["idle"] += len(idle)

        if a.verbose or skipped or idle:
            extra = ""
            if skipped:
                extra += f"  SKIPPED {len(skipped)} batch(es)"
            if idle:
                extra += f"  IDLE {len(idle)} episode(s) moved no pigment"
            print(f"{ds.name:24s} eps={n_ep:4d} frames={info['total_frames']:7d} "
                  f"tool={tool}{extra}")

    print()
    for dist in sorted(by_dist, key=lambda k: (k is None, k)):
        d = by_dist[dist]
        if len(d["tools"]) > 1:
            problems.append(f"{dist}: shards disagree on the fitted tool {d['tools']} "
                            "— they cannot be aggregated as one distribution.")
        med = f"{d['frames'] / max(d['episodes'], 1) / 30:.1f}s"
        print(f"  {str(dist):13s} {d['episodes']:5d} episodes  {d['frames']:8d} frames  "
              f"{d['shards']:2d} shards  {len(d['prompts']):4d} prompts  mean {med}")
        if d["skipped"]:
            print(f"  {'':13s} {d['skipped']} batch(es) skipped as unplannable — the "
                  "episode count is full but the scene distribution is narrower")
        if d["idle"]:
            print(f"  {'':13s} {d['idle']} episode(s) moved no pigment — mislabelled "
                  "demonstrations, exclude or regenerate")

    total_e = sum(d["episodes"] for d in by_dist.values())
    total_f = sum(d["frames"] for d in by_dist.values())
    print(f"  {'TOTAL':13s} {total_e:5d} episodes  {total_f:8d} frames  "
          f"{len(shards):2d} shards")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    code = main(tyro.cli(Args))
    # Leave via os._exit, after flushing. Reading the episode parquet loads a
    # native stack that aborts during interpreter finalization in this venv
    # ("terminate called without an active exception", SIGABRT) — long after
    # every check has run and every line has printed. Bisection puts it in the
    # parquet read, not in anything above. A normal return would report 134 on
    # a clean audit, and this exit code is meant to gate a training mix, so it
    # has to mean what it says.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
