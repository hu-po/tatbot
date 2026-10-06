"""Read the explicit vision build matrix used by scripts/check rust."""

import json
import os
import re
from pathlib import Path


def profile_rows(root: Path) -> str:
    profiles = json.loads((root / "rust/check-profiles.json").read_text())
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Rust check matrix must be a nonempty object")
    selected = os.environ.get("TATBOT_RUST_PROFILES", "").split(",")
    selected = [name for name in selected if name] or list(profiles)
    required = set(filter(None, os.environ.get("TATBOT_RUST_REQUIRED_PROFILES", "").split(",")))
    if unknown := (set(selected) | required) - profiles.keys():
        raise ValueError(f"unknown Rust check profiles: {sorted(unknown)}")
    if required - set(selected):
        raise ValueError("required Rust profiles must be included in the selection")
    rows = []
    for name in dict.fromkeys(selected):
        profile = profiles[name]
        if set(profile) != {"features", "pkg_config"}:
            raise ValueError(f"invalid Rust profile fields: {name}")
        for values in ([name], profile["features"], profile["pkg_config"]):
            if not isinstance(values, list) or not all(
                isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value) for value in values
            ):
                raise ValueError(f"invalid Rust profile values: {name}")
        rows.append("|".join((name, ",".join(profile["features"]),
                              " ".join(profile["pkg_config"]), str(int(name in required)))))
    return "\n".join(rows)
