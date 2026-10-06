"""Fail-closed audit of Inkmap perception-frame sidecars."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tatbot_sim.inkmap.identities import admitted_identity_ids, identity_contract
from tatbot_sim.inkmap.perception import BACKGROUND_FACE, PerceptionLabels
from tatbot_sim.inkmap.perception_variation import audit_split_records

SHA256 = re.compile(r"[0-9a-f]{64}")
HASH_BINDINGS = (
    "identity_sha256",
    "rest_surface_sha256",
    "posed_surface_sha256",
    "design_sha256",
    "placement_sha256",
    "scenario_sha256",
    "artwork_family_sha256",
)


def _load_labels(path: Path) -> PerceptionLabels:
    with np.load(path, allow_pickle=False) as data:
        expected = set(PerceptionLabels.__dataclass_fields__)
        if set(data.files) != expected:
            raise ValueError(f"labels fields differ: expected {sorted(expected)}, got {sorted(data.files)}")
        return PerceptionLabels(**{name: data[name] for name in expected})


def audit_frame(root: Path) -> tuple[list[str], dict[str, Any] | None]:
    problems: list[str] = []
    manifest_path, array_path = root / "manifest.json", root / "labels.npz"
    if not manifest_path.is_file() or not array_path.is_file():
        return [f"{root}: incomplete frame (manifest.json and labels.npz required)"], None
    try:
        manifest = json.loads(manifest_path.read_text())
        labels = _load_labels(array_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"{root}: unreadable frame: {exc}"], None
    if manifest.get("schema") != "tatbot.inkmap-perception-frame/1":
        problems.append(f"{root}: wrong perception manifest schema")
    if manifest.get("status") != "accepted":
        problems.append(f"{root}: frame status is not accepted")
    bindings = manifest.get("bindings", {})
    for name in HASH_BINDINGS:
        if not isinstance(bindings.get(name), str) or not SHA256.fullmatch(bindings[name]):
            problems.append(f"{root}: missing canonical binding {name}")
    if not isinstance(bindings.get("source_revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", bindings["source_revision"]):
        problems.append(f"{root}: missing full source revision")
    if not isinstance(bindings.get("source_repository"), str) or not bindings["source_repository"]:
        problems.append(f"{root}: missing source repository")
    if not isinstance(bindings.get("asset_provenance"), dict) or not bindings["asset_provenance"]:
        problems.append(f"{root}: missing asset provenance")
    if not isinstance(bindings.get("dependency_versions"), dict) or not bindings["dependency_versions"]:
        problems.append(f"{root}: missing dependency versions")
    if bindings.get("source_dirty") is not False:
        problems.append(f"{root}: source checkout was dirty or unknown")
    admitted = {
        identity_contract(identity_id)["content_sha256"]
        for identity_id in admitted_identity_ids()
    }
    if bindings.get("identity_sha256") not in admitted:
        problems.append(f"{root}: unsupported or unreviewed identity")
    if bindings.get("split") not in {"train", "design-held-out", "identity-held-out", "joint-held-out"}:
        problems.append(f"{root}: invalid split")
    digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
    if manifest.get("arrays_sha256") != digest:
        problems.append(f"{root}: labels.npz byte digest differs")
    if manifest.get("labels_sha256") != labels.canonical_digest():
        problems.append(f"{root}: canonical label digest differs")

    shape = labels.depth_m.shape
    for name in ("depth_clean_m", "depth_clean_valid", "depth_valid", "normal_valid", "body_visible", "face_index", "barycentric_valid", "tattoo_coverage", "tattoo_placement_id", "tattoo_layer_id"):
        if getattr(labels, name).shape != shape:
            problems.append(f"{root}: {name} shape differs from depth")
    if labels.rgb_srgb.shape != (*shape, 3) or labels.normal_camera.shape != (*shape, 3) or labels.barycentric.shape != (*shape, 3):
        problems.append(f"{root}: vector label shape differs from depth")
        return problems, manifest
    if not np.array_equal(labels.body_visible, labels.barycentric_valid):
        problems.append(f"{root}: body and barycentric validity differ")
    if not np.array_equal(labels.body_visible, labels.normal_valid):
        problems.append(f"{root}: body and normal validity differ")
    if np.any(labels.depth_valid != np.isfinite(labels.depth_m)) or np.any(labels.depth_m[labels.depth_valid] <= 0):
        problems.append(f"{root}: invalid depth/validity semantics")
    if np.any(labels.depth_clean_valid != np.isfinite(labels.depth_clean_m)) or np.any(labels.depth_clean_m[labels.depth_clean_valid] <= 0):
        problems.append(f"{root}: invalid clean depth/validity semantics")
    if np.any(labels.face_index[~labels.body_visible] != BACKGROUND_FACE) or np.any(labels.face_index[labels.body_visible] < 0):
        problems.append(f"{root}: face sentinel/validity mismatch")
    if labels.body_visible.any():
        sums = labels.barycentric[labels.body_visible].sum(axis=1)
        if not np.isfinite(labels.barycentric[labels.body_visible]).all() or np.max(np.abs(sums - 1)) > 2e-5 or np.min(labels.barycentric[labels.body_visible]) < -2e-5:
            problems.append(f"{root}: invalid barycentric coordinates")
        lengths = np.linalg.norm(labels.normal_camera[labels.body_visible], axis=1)
        if np.max(np.abs(lengths - 1)) > 2e-5:
            problems.append(f"{root}: camera normals are not unit length")
    if not np.isfinite(labels.rgb_srgb).all() or np.min(labels.rgb_srgb) < 0 or np.max(labels.rgb_srgb) > 1:
        problems.append(f"{root}: RGB is outside finite [0,1]")
    if not np.isfinite(labels.tattoo_coverage).all() or np.min(labels.tattoo_coverage) < 0 or np.max(labels.tattoo_coverage) > 1:
        problems.append(f"{root}: tattoo coverage is outside finite [0,1]")
    no_tattoo = labels.tattoo_coverage <= 0
    if np.any(labels.tattoo_placement_id[no_tattoo] != 0) or np.any(labels.tattoo_layer_id[no_tattoo] != 0):
        problems.append(f"{root}: tattoo IDs set where coverage is zero")
    return problems, manifest


def audit_corpus(root: Path) -> dict[str, Any]:
    frame_roots = sorted(path.parent for path in root.rglob("manifest.json"))
    problems: list[str] = []
    records: list[dict[str, Any]] = []
    accepted = 0
    for frame_root in frame_roots:
        frame_problems, manifest = audit_frame(frame_root)
        problems.extend(frame_problems)
        if manifest is not None:
            bindings = manifest.get("bindings", {})
            records.append(bindings)
            if not frame_problems:
                accepted += 1
    problems.extend(audit_split_records(records))
    request_path = root / "requests.json"
    requested = len(frame_roots)
    rejected = 0
    if request_path.is_file():
        requests = json.loads(request_path.read_text())
        if requests.get("schema") != "tatbot.inkmap-perception-requests/1":
            problems.append(f"{request_path}: wrong request ledger schema")
        entries = requests.get("requests", [])
        requested = len(entries)
        accepted_ids = {item.get("bindings", {}).get("scene_id") for item in (json.loads((path / "manifest.json").read_text()) for path in frame_roots)}
        for index, entry in enumerate(entries):
            status = entry.get("status")
            if status == "accepted" and entry.get("scene_id") not in accepted_ids:
                problems.append(f"request {index}: accepted scene has no complete frame")
            elif status == "rejected":
                rejected += 1
                if not isinstance(entry.get("reason"), str) or not entry["reason"]:
                    problems.append(f"request {index}: rejection has no reason")
            elif status not in {"accepted", "rejected"}:
                problems.append(f"request {index}: unlabeled outcome")
    elif frame_roots:
        problems.append(f"{root}: missing requests.json outcome ledger")
    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {
        "schema": "tatbot.inkmap-perception-audit/1",
        "status": "pass" if not problems else "fail",
        "requested": requested,
        "accepted": accepted,
        "rejected": rejected,
        "problems": problems,
        "bytes": total_bytes,
        "bytes_per_accepted_frame": total_bytes / accepted if accepted else None,
    }


@dataclass
class Args:
    path: Path


def main() -> None:
    import tyro

    report = audit_corpus(tyro.cli(Args).path.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
