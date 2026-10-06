"""Privileged, time-aligned synthetic state kept outside policy features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# /1 existed for 43 minutes on 2026-09-05 and wrote no dataset; only /2 is read.
SCHEMA = "tatbot.sim-privileged-timeline/2"
BASE_FIELDS = {
    "tool_pose_world",
    "contact_distance_m",
    "contact_incidence",
    "pen_down",
    "target_world",
    "target_valid",
    "surface_point_world",
    "surface_normal_world",
    "primitive_index",
    "layer_index",
    "progress",
    "deposited_coverage",
    "remaining_target_fraction",
    "texture_synchronized",
}
VARIANT_FIELDS = {"stencil_visible_fraction", "observation_occlusion_fraction"}


@dataclass
class TemporalRecorder:
    batch_size: int
    values: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def append(self, **values: np.ndarray) -> None:
        expected = BASE_FIELDS | VARIANT_FIELDS
        if set(values) != expected:
            raise ValueError(f"temporal label fields differ: {sorted(set(values) ^ expected)}")
        for name, value in values.items():
            array = np.asarray(value)
            if array.shape[0] != self.batch_size:
                raise ValueError(f"{name} batch size {array.shape[0]} != {self.batch_size}")
            self.values.setdefault(name, []).append(array.copy())

    def write(
        self,
        root: Path,
        *,
        kept: list[int | None],
        lengths: np.ndarray,
        stroke_metadata: list | None,
        scenario_sha256: str | None,
        variant_ids: list[str] | None = None,
        expected_outcomes: list[str] | None = None,
    ) -> list[dict[str, Any] | None]:
        root.mkdir(parents=True, exist_ok=True)
        stacked = {name: np.stack(items, axis=1) for name, items in self.values.items()}
        output: list[dict[str, Any] | None] = [None] * self.batch_size
        for env, episode in enumerate(kept):
            if episode is None:
                continue
            length = int(lengths[env])
            arrays = {name: value[env, :length] for name, value in stacked.items()}
            path = root / f"episode_{episode:06d}.npz"
            np.savez_compressed(path, **arrays)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            meta = {
                "schema": SCHEMA,
                "episode": episode,
                "steps": length,
                "npz": path.name,
                "sha256": digest,
                "scenario_sha256": scenario_sha256,
                "stroke_metadata": stroke_metadata[env] if stroke_metadata is not None else None,
                "episode_variant": variant_ids[env] if variant_ids is not None else "blank-start",
                "expected_outcome": expected_outcomes[env] if expected_outcomes is not None else "nominal",
                "conventions": {
                    "policy_boundary": "privileged metadata sidecar; no key is an observation/action feature",
                    "tool_pose_world": "xyz plus quaternion wxyz after the recorded simulation step",
                    "contact": "declared synthetic interaction model, not measured penetration or force",
                    "primitive_index": "zero-based typed stroke order; -1 while not in contact",
                    "layer_index": "zero-based TattooProgram layer; -1 when unavailable/not in contact",
                    "coverage": "scalar InkField effective coverage after the same step as the RGB frame",
                    "remaining": "soft intended coverage not yet deposited, divided by intended coverage",
                    "texture_synchronized": "true only when the RGB texture was refreshed after this deposition step",
                    "stencil_visible_fraction": "mean of stencil coverage x (1 - deposited pigment) after this step; the guide area itself is stencil_area_fraction in run_meta",
                    "observation_occlusion_fraction": "fraction synthetically hidden in every recorded camera observation",
                },
            }
            manifest_path = root / f"episode_{episode:06d}.json"
            manifest_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            output[env] = {
                "path": f"meta/privileged/{path.name}",
                "manifest": f"meta/privileged/{manifest_path.name}",
                "sha256": digest,
            }
        return output


def audit_timeline(path: Path, manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA:
        return [f"{path}: wrong timeline schema"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        return [f"{path}: missing or digest mismatch"]
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != BASE_FIELDS | VARIANT_FIELDS:
            problems.append(f"{path}: timeline fields differ")
            return problems
        steps = int(manifest.get("steps", -1))
        if any(len(data[name]) != steps for name in data.files):
            problems.append(f"{path}: array length differs from manifest")
        if np.any(data["pen_down"] & ~data["target_valid"]):
            problems.append(f"{path}: pen-down frame has no intended target")
        if np.any(data["pen_down"] & (data["primitive_index"] < 0)) and manifest.get("stroke_metadata") is not None:
            problems.append(f"{path}: typed pen-down frame has no primitive")
        if not np.all(data["texture_synchronized"]):
            problems.append(f"{path}: RGB/deposition texture synchronization failure")
        for name in {"progress", "deposited_coverage", "remaining_target_fraction"} | VARIANT_FIELDS:
            if not np.isfinite(data[name]).all() or np.min(data[name]) < 0 or np.max(data[name]) > 1:
                problems.append(f"{path}: {name} outside finite [0,1]")
        if np.any(np.diff(data["progress"]) < -1e-7):
            problems.append(f"{path}: progress decreases")
        if manifest.get("episode_variant") == "stencil-start" and not np.any(data["stencil_visible_fraction"] > 0):
            problems.append(f"{path}: stencil-start episode has no separate stencil label")
        if manifest.get("episode_variant") == "stencil-start" and np.any(np.diff(data["stencil_visible_fraction"]) > 1e-6):
            problems.append(f"{path}: stencil visibility grows while pigment is only deposited")
        if manifest.get("episode_variant") == "occluded" and not np.any(data["observation_occlusion_fraction"] > 0):
            problems.append(f"{path}: occluded episode has no occlusion label")
    return problems
