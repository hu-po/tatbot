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
  * fixed-mount data carries resolved-tool-v1 geometry and rigid-contact-v1
    interaction, with the visible contact endpoint within 0.5 mm of the TCP;
  * every engaged non-dip episode has contact frames and none fall outside the
    recorded -0.25/+0.50 mm interaction band;
  * pigment-field snapshots, when present, number exactly one per episode;
  * the episode table agrees with info.json on how many episodes there are;
  * every episode's first action is close to its recorded first state, catching
    a stale articulation before it becomes a training label;
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
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

EPISODE_START_MAX_LEAD_RAD = 0.1
"""Gross data-integrity bound, not a robot-motion acceptance threshold.

The broken fixed-EE corpus reached 0.704 rad at simulator frame zero while the
paired real recordings topped out at 0.034 rad.
"""


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
    allow_air_gap: bool = False
    """Accept pre-contact-v1 shards whose texture could change while the
    visible tool was in the air. Off by default; useful only for historical
    inventory or an explicitly named negative control."""
    allow_provisional: bool = False
    """Accept contact-v1 validation shards without a quality-gated pivot TCP.
    Off by default. An axis-inferred body with `pivot-calibrated` contact is
    production-eligible; this flag is for nominal or failed calibration only."""


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


def _episode_start_lead(ds: Path) -> tuple[int, float, int]:
    files = sorted(glob.glob(str(ds / "data/chunk-*/*.parquet")))
    if not files:
        raise ValueError("no data parquet files")
    starts = pd.concat(
        [
            pd.read_parquet(
                file,
                columns=["action", "observation.state", "frame_index"],
                filters=[("frame_index", "==", 0)],
            )
            for file in files
        ],
        ignore_index=True,
    )
    if starts.empty:
        raise ValueError("no frame_index=0 rows")
    actions = np.stack(starts["action"].to_numpy()).astype(np.float64)
    states = np.stack(starts["observation.state"].to_numpy()).astype(np.float64)
    if actions.ndim != 2 or states.ndim != 2 or states.shape[1] < actions.shape[1]:
        raise ValueError(f"incompatible action/state shapes {actions.shape}/{states.shape}")
    lead = np.abs(actions - states[:, : actions.shape[1]])
    flat = int(np.argmax(lead))
    _, joint = np.unravel_index(flat, lead.shape)
    return len(starts), float(lead.max()), int(joint)


def _vector3(value) -> tuple[float, float, float] | None:
    """Return a finite metadata vector, or None for absent/malformed data."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


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
        tool_meta = rm.get("tool", {})
        n_ep = info["total_episodes"]

        if int(rm.get("schema_version", 1)) >= 2:
            software = rm.get("software") or {}
            start = software.get("revision_start")
            end = software.get("revision_end")
            if not (isinstance(start, str) and re.fullmatch(r"[0-9a-f]{40}", start)):
                problems.append(f"{ds.name}: run metadata lacks a full source revision.")
            if end != start:
                problems.append(
                    f"{ds.name}: source revision changed during generation "
                    f"({start or 'unknown'} -> {end or 'unknown'}).")
            if software.get("dirty_start") is not False or software.get("dirty_end") is not False:
                problems.append(
                    f"{ds.name}: source checkout was dirty or unknown during generation.")

        # Which follower made it. Shards stamped before 2026-08-30 carry no
        # embodiment at all, which is itself the gripper-held-v1 signature.
        embodiment = rm.get("tool", {}).get("embodiment") or "gripper-held-v1"
        if embodiment != "fixed-mount-v2" and not a.allow_legacy:
            problems.append(
                f"{ds.name}: embodiment {embodiment!r} — generated for the gripper-held "
                "follower (tool in the jaws, upper/lower wrist cameras), not the fixed "
                "mount. Keep for history; pass --allow-legacy to audit it as such.")

        interaction_model = tool_meta.get("interaction_model")
        geometry_version = tool_meta.get("tool_geometry_version")
        if (interaction_model != "rigid-contact-v1"
                or geometry_version != "resolved-tool-v1"):
            if not a.allow_air_gap:
                problems.append(
                    f"{ds.name}: geometry/interaction is "
                    f"{geometry_version or 'unversioned'}/{interaction_model or 'air-gap-v0'} "
                    "— rigid collision did not prove contact. Keep for history; pass "
                    "--allow-air-gap only for an explicit validation or negative-control "
                    "audit.")
        else:
            geometry_status = tool_meta.get("geometry_status")
            contact_status = tool_meta.get("contact_geometry_status")
            # `qualified` is the backwards-compatible independent-body case.
            # New contact datasets say exactly what was qualified: the pivot
            # TCP, while body_pose_status may honestly remain axis-inferred.
            contact_eligible = (contact_status == "pivot-calibrated"
                                or geometry_status == "qualified")
            if not contact_eligible and not a.allow_provisional:
                problems.append(
                    f"{ds.name}: contact geometry is "
                    f"{contact_status or geometry_status or 'missing'} — production "
                    "contact data requires a quality-gated pivot TCP. Pass "
                    "--allow-provisional only for labelled validation.")
            if tool_meta.get("contact"):
                body_tip = tool_meta.get("body_tip_offset_m")
                tcp = tool_meta.get("tcp_offset_m")
                if not (isinstance(body_tip, list) and len(body_tip) == 3
                        and isinstance(tcp, list) and len(tcp) == 3):
                    problems.append(f"{ds.name}: contact-v1 lacks body-tip/TCP vectors")
                else:
                    error = math.sqrt(sum((float(x) - float(y)) ** 2
                                          for x, y in zip(body_tip, tcp, strict=True)))
                    if error > 0.0005:
                        problems.append(
                            f"{ds.name}: visible body tip is {error * 1000:.3f} mm from "
                            "the contact TCP (maximum 0.500 mm).")
            if abs(float(cfg.get("draw_clearance", 0.0))) > 1e-9:
                problems.append(
                    f"{ds.name}: contact-v1 draw_clearance is "
                    f"{float(cfg['draw_clearance']) * 1000:.3f} mm, must be zero.")

            jitter = bool(cfg.get("tool_calibration_jitter", False))
            raw_delta = tool_meta.get("calibration_delta_m")
            delta = _vector3(raw_delta)
            if raw_delta is not None and delta is None:
                problems.append(
                    f"{ds.name}: calibration_delta_m is not a finite 3-vector.")
            if delta is None:
                delta = (0.0, 0.0, 0.0)
            delta_norm = math.sqrt(sum(value * value for value in delta))
            if jitter:
                uncertainty = tool_meta.get("contact_uncertainty_m")
                try:
                    uncertainty = float(uncertainty)
                    scale = float(cfg.get("tool_calibration_scale", 1.0))
                except (TypeError, ValueError):
                    uncertainty, scale = math.nan, math.nan
                if (not math.isfinite(uncertainty) or uncertainty <= 0
                        or not math.isfinite(scale) or scale < 0):
                    problems.append(
                        f"{ds.name}: calibration jitter lacks a positive finite "
                        "contact uncertainty and non-negative scale.")
                elif delta_norm > uncertainty * scale + 1e-9:
                    problems.append(
                        f"{ds.name}: calibration delta is {delta_norm * 1000:.3f} mm, "
                        f"outside its {uncertainty * scale * 1000:.3f} mm bound.")
                calibrated = _vector3(tool_meta.get("calibrated_tip_offset_m"))
                actual = _vector3(tool_meta.get("tip_offset_m"))
                if calibrated is None or actual is None:
                    problems.append(
                        f"{ds.name}: calibration jitter lacks calibrated/actual tip vectors.")
                else:
                    reconstructed = tuple(
                        actual[i] - calibrated[i] for i in range(3))
                    mismatch = math.sqrt(sum(
                        (reconstructed[i] - delta[i]) ** 2 for i in range(3)))
                    if mismatch > 1e-9:
                        problems.append(
                            f"{ds.name}: actual minus calibrated tip disagrees with "
                            f"calibration_delta_m by {mismatch * 1000:.6f} mm.")
            elif delta_norm > 1e-12:
                problems.append(
                    f"{ds.name}: records a {delta_norm * 1000:.3f} mm calibration "
                    "delta while tool_calibration_jitter is disabled.")

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

        start_lead_max = 0.0
        try:
            start_rows, start_lead_max, start_joint = _episode_start_lead(ds)
            if start_rows != n_ep:
                problems.append(
                    f"{ds.name}: {start_rows} frame-0 rows for {n_ep} episodes — "
                    "every episode must contribute exactly one initial observation."
                )
            if start_lead_max > EPISODE_START_MAX_LEAD_RAD:
                problems.append(
                    f"{ds.name}: episode-start action/state lead reaches "
                    f"{start_lead_max:.3f} rad on joint {start_joint} "
                    f"(limit {EPISODE_START_MAX_LEAD_RAD:.3f}). Pose placement may "
                    "have targeted an articulation replaced during reset; do not train "
                    "on this shard."
                )
        except Exception as exc:
            problems.append(f"{ds.name}: episode-start action/state lead unreadable — {exc}")

        skipped = rm.get("skipped_batches") or []
        idle = [e for e in rm.get("episodes", []) if not e.get("engaged", True)]
        if interaction_model == "rigid-contact-v1":
            for index, episode in enumerate(rm.get("episodes", [])):
                if episode.get("kind") == "dip":
                    continue
                interaction_meta = episode.get("interaction") or {}
                frames = int(interaction_meta.get("frames") or 0)
                if episode.get("engaged", True) and frames == 0:
                    problems.append(
                        f"{ds.name}: episode {index} is engaged with zero contact frames.")
                lo = interaction_meta.get("distance_min_m")
                hi = interaction_meta.get("distance_max_m")
                if lo is not None and float(lo) < -0.00025 - 1e-7:
                    problems.append(
                        f"{ds.name}: episode {index} marks at {float(lo) * 1000:.3f} mm "
                        "penetration (limit 0.250 mm).")
                if hi is not None and float(hi) > 0.0005 + 1e-7:
                    problems.append(
                        f"{ds.name}: episode {index} marks {float(hi) * 1000:.3f} mm "
                        "above the surface (limit 0.500 mm).")

        d = by_dist.setdefault(dist, {"episodes": 0, "frames": 0, "shards": 0,
                                      "tools": set(), "substrates": set(),
                                      "prompts": set(), "skipped": 0, "idle": 0,
                                      "start_lead_max": 0.0})
        d["episodes"] += n_ep
        d["frames"] += info["total_frames"]
        d["shards"] += 1
        d["tools"].add(tool)
        d["substrates"].add(rm.get("tool", {}).get("substrate")
                            or cfg.get("distribution"))
        d["prompts"] |= tasks
        d["skipped"] += len(skipped)
        d["idle"] += len(idle)
        d["start_lead_max"] = max(d["start_lead_max"], start_lead_max)

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
              f"{d['shards']:2d} shards  {len(d['prompts']):4d} prompts  mean {med}  "
              f"start lead max {d['start_lead_max']:.3f} rad")
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
