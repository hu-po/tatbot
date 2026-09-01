#!/usr/bin/env python3
"""Inspect the input/action contract stored with a LeRobot checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SIDECAR = "tatbot_contract.json"


def load_contract(source: str) -> dict[str, Any]:
    sidecar: dict[str, Any] = {}
    if source == "-":
        value = json.load(sys.stdin)
    else:
        path = Path(source)
        if path.is_dir():
            sidecar_path = path / SIDECAR
            path = path / "config.json"
            if not path.is_file():
                path = Path(source) / "train_config.json"
        else:
            sidecar_path = path.parent / SIDECAR
        with path.open() as stream:
            value = json.load(stream)
        if sidecar_path.is_file():
            with sidecar_path.open() as stream:
                sidecar = json.load(stream)

    policy = value.get("policy", value)
    features = policy.get("input_features", {})
    state_shape = features.get("observation.state", {}).get("shape", [0])
    state_size = int(state_shape[0]) if state_shape else 0
    policy_type = str(policy.get("type", "unknown"))
    use_depth = any(str(name).endswith("_depth") for name in features)
    # A 14-wide state carries the external-effort channels on the wire, but it
    # does not say they are meaningful: a checkpoint co-trained on simulation
    # keeps the width while those channels are masked to zero, because sim has
    # no measured force. Feeding such a policy live effort is train/serve skew
    # that no shape check catches, so a training pipeline that masks says so in
    # the sidecar and the serving launchers refuse the checkpoint until a
    # runtime can zero the channels (dropping them is not the same thing — that
    # narrows the state to 7 and breaks the normalizer).
    return {
        "policy_type": policy_type,
        "use_relative_actions": bool(policy.get("use_relative_actions", False)),
        "use_depth": use_depth,
        "depth_encoding": "depth-v1" if policy_type == "groot" and use_depth else "",
        "state_size": state_size,
        "use_external_effort": state_size == 14,
        "mask_external_effort": bool(sidecar.get("mask_external_effort", False)),
    }


def fields(contract: dict[str, Any]) -> str:
    """Pipe-separated contract for the shell launchers. Append only: every
    reader splits positionally, so a new field in the middle silently shifts
    someone's variable."""
    values = (
        contract["policy_type"],
        int(contract["use_depth"]),
        contract["state_size"],
        contract["depth_encoding"],
        int(contract["use_relative_actions"]),
        int(contract["mask_external_effort"]),
    )
    return "|".join(map(str, values))


def declare(root: Path, **facts: bool) -> Path:
    """Record training-side facts the checkpoint config cannot express, for
    every checkpoint under `root`. A training run that masks a channel must
    call this, or the launchers read the config alone and see nothing."""
    written = 0
    for config in sorted(root.rglob("config.json")) or []:
        sidecar = config.parent / SIDECAR
        current = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        current.update(facts)
        sidecar.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        written += 1
    if not written:
        raise SystemExit(f"no checkpoint config.json under {root}")
    print(f"declared {facts} on {written} checkpoint(s) under {root}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="checkpoint directory, config JSON, or - for stdin")
    parser.add_argument("--format", choices=("json", "fields"), default="json")
    parser.add_argument(
        "--declare-masked-effort",
        action="store_true",
        help="record that this run trained with observation.state[7:14] zeroed, "
        "for every checkpoint under the source directory, then exit",
    )
    args = parser.parse_args()
    if args.declare_masked_effort:
        declare(Path(args.source), mask_external_effort=True)
        return 0
    contract = load_contract(args.source)
    if args.format == "fields":
        print(fields(contract))
    else:
        print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
