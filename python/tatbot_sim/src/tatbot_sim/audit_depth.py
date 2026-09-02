"""Phase 1 gate: measure the dataset, don't trust the clamp.

The floor clamp is only right if the written data shows BOTH of these:

1. **Zero commanded steps below the surface.** The instruction to descend
   through the pad is gone.
2. **Touch-and-recover events survive.** Episodes where a noise burst
   presses the achieved needle down to the surface and the trajectory then
   climbs back to the draw plane — the "too deep, come up" demonstrations.
   If the clamp flattened these away too, it is wrong (over-eager), not safe.

Two findings from the first A/B (2026-08-21) shaped criterion 2:

- Every achieved-below-the-*surface* step in unclamped data is downstream of
  a commanded-below step — sim tracking never carries the needle through the
  floor on its own. Once commands are clamped, states below the surface
  disappear from the data entirely, and that is fine: on the real rig the
  follower's z-floor and physical contact make those states unreachable
  anyway. Recovery from *touching* is the region that still exists and still
  matters.
- Recovery is gradual: the burst decays at 0.85/step, so the command climbs
  ~0.5 mm per step. A per-step "commanded up by ≥1 mm" test misses it
  entirely; counting touch events and whether each one returns to the draw
  plane measures what a demonstration actually is.

Runs forward kinematics (the expert's own chain, so there is one source of
geometric truth) on every commanded action and achieved state in the dataset,
compares needle z against the per-episode surface height recorded in
``run_meta.json``, and prints a verdict. Compare a ``--clamp-floor`` run
against a ``--no-clamp-floor`` run of the same seed to see exactly what the
clamp changed.

Usage (tatbot_sim venv, x86_64):
    python -m tatbot_sim.audit_depth <dataset-dir>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

from tatbot_sim.expert import BatchedIK

# FK is float32 and the clamp bisection stops at 2^-12 of a noise burst;
# 0.1 mm absorbs both without hiding a real violation.
EPS_M = 1e-4
# A touch: achieved needle within this height of the surface — well below the
# 4 mm draw plane and beyond the ~0.6 mm tracking jitter.
TOUCH_BAND_M = 1.5e-3
# Recovered: achieved back within one tracking-jitter of the draw plane inside
# this many steps of the touch ending (1.5 s at 30 Hz).
RECOVER_WITHIN_STEPS = 45
RECOVER_SLACK_M = 1e-3


@dataclass
class Args:
    dataset: tyro.conf.Positional[str]
    batch: int = 8192
    """FK batch size (CPU memory bound)."""


def _fk_pos(ik: BatchedIK, q: np.ndarray, batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(q), batch):
        t = torch.as_tensor(q[i : i + batch], dtype=torch.float32)
        out.append(ik.fk(t)[:, :3, 3].numpy())
    return np.concatenate(out)


def main(args: Args):
    base = Path(args.dataset).expanduser()
    meta = json.loads((base / "meta" / "run_meta.json").read_text())
    # Tilt-era datasets record the full plane (point + normal); older clamp-era
    # ones only the height, where the plane is horizontal.
    planes = {}
    for e in meta["episodes"]:
        if "surface_normal" in e:
            planes[e["episode"]] = (np.array(e["surface_point"]), np.array(e["surface_normal"]))
        elif e.get("surface_z") is not None:
            planes[e["episode"]] = (np.array([0.0, 0.0, e["surface_z"]]), np.array([0.0, 0.0, 1.0]))
        else:
            raise SystemExit(
                "run_meta.json has episodes without a surface record — this dataset "
                "predates the clamp work and cannot be audited."
            )
    # "config" since the DR-tree refactor; "args" in older datasets
    gen_cfg = meta.get("config") or meta.get("args") or {}
    # Historical datasets record 0.004 explicitly. Missing means current
    # contact-v1, whose resolved working point is on the surface.
    draw_clearance = float(gen_cfg.get("draw_clearance", 0.0))

    files = sorted((base / "data").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    actions = np.stack(df["action"].to_numpy())[:, :6]
    states = np.stack(df["observation.state"].to_numpy())[:, :6]
    eps = df["episode_index"].to_numpy()
    pts = np.stack([planes[int(e)][0] for e in eps]).astype(np.float32)
    nms = np.stack([planes[int(e)][1] for e in eps]).astype(np.float32)

    ik = BatchedIK(torch.device("cpu"))
    names = ik.chain.get_joint_parameter_names()
    assert names == [f"joint_{i}" for i in range(6)], f"chain order changed: {names}"
    # signed distance from each pose's needle to its episode's pad plane
    z_cmd = (( _fk_pos(ik, actions, args.batch) - pts) * nms).sum(-1)
    z_ach = (( _fk_pos(ik, states, args.batch) - pts) * nms).sum(-1)
    floor = np.zeros_like(z_cmd)  # distances are already plane-relative

    n = len(df)
    cmd_below = z_cmd < floor - EPS_M
    ach_below = z_ach < floor - EPS_M

    # Touch-and-recover events, per episode in frame order.
    touches = 0
    recovered = 0
    order = np.lexsort((df["frame_index"].to_numpy(), df["episode_index"].to_numpy()))
    ep_sorted = df["episode_index"].to_numpy()[order]
    height = (z_ach - floor)[order]
    for ep in np.unique(ep_sorted):
        h = height[ep_sorted == ep]
        near = h < TOUCH_BAND_M
        # event boundaries: runs of consecutive near-surface steps
        starts = np.flatnonzero(near & ~np.roll(near, 1))
        if near[0]:
            starts = np.unique(np.concatenate([[0], starts]))
        ends = np.flatnonzero(near & ~np.roll(near, -1))
        if near[-1]:
            ends = np.unique(np.concatenate([ends, [len(near) - 1]]))
        for _start, e in zip(starts, ends, strict=True):
            touches += 1
            tail = h[e + 1 : e + 1 + RECOVER_WITHIN_STEPS]
            if (tail > draw_clearance - RECOVER_SLACK_M).any():
                recovered += 1

    print(f"dataset: {base}")
    print(f"steps: {n}   episodes: {df['episode_index'].nunique()}")
    print(f"commanded needle below surface : {cmd_below.sum():7d}  ({100 * cmd_below.mean():6.3f}%)")
    if cmd_below.any():
        depth_mm = 1000 * (floor - z_cmd)[cmd_below]
        print(f"  deepest commanded incursion  : {depth_mm.max():6.1f} mm  (median {np.median(depth_mm):5.1f})")
    print(f"achieved needle below surface  : {ach_below.sum():7d}  ({100 * ach_below.mean():6.3f}%)")
    print(f"surface-touch events           : {touches:7d}  ({recovered} recover to the draw plane)")
    print(f"min commanded z above surface  : {1000 * (z_cmd - floor).min():6.1f} mm")

    ok_cmd = not cmd_below.any()
    ok_rec = recovered > 0
    print()
    print(("PASS" if ok_cmd else "FAIL") + " — no commanded step goes below the surface")
    print(("PASS" if ok_rec else "FAIL") + " — touch-and-recover demonstrations survive the clamp")
    if not (ok_cmd and ok_rec):
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
