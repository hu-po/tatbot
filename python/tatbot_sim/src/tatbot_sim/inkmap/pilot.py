"""Deterministic, fail-closed manifest for the Inkmap synthetic pilot.

This plans and audits work; it deliberately does not launch a renderer or arm.
Pending review, reach, and compute gates remain pending in its ledger. The
matrix size comes from ``config/inkmap/synthetic-pilot.json``; the code checks
the spec's shape and derives every count from it, so a larger or smaller pilot
is a config change, not a code change.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from tatbot_sim.inkmap.identities import admitted_identity_ids
from tatbot_sim.inkmap.perception_variation import stable_seed
from tatbot_sim.repo import repo_root, source_state

SCHEMA = "tatbot.inkmap-synthetic-pilot-plan/1"
SPEC_SCHEMA = "tatbot.inkmap-synthetic-pilot-spec/1"
AUDIT_SCHEMA = "tatbot.inkmap-pilot-audit/1"
SPEC_PATH = repo_root() / "config/inkmap/synthetic-pilot.json"
IDENTITIES_PATH = repo_root() / "config/inkmap/synthetic-identities.json"

# Every variant the generator implements, with the outcome label it writes and
# the implementation status the spec must declare for it. A spec that names a
# different set is refused: the pilot may not silently drop or invent one.
VARIANT_CONTRACT = {
    "blank-start": ("nominal", "implemented"),
    "stencil-start": ("nominal", "implemented_separate_appearance_label"),
    "missed-stroke": ("missed", "implemented_failure_injection"),
    "interrupted": ("interrupted", "implemented_failure_injection"),
    "partial-coverage": ("partial", "implemented_failure_injection"),
    "dry-tool": ("dry", "implemented_supply_model"),
    "occluded": ("occluded", "implemented_visibility_model"),
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _closed(value: dict, keys: set[str], where: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{where} fields differ: {sorted(set(value) ^ keys)}")


def _unique_ids(items: list[dict], where: str) -> None:
    ids = [item.get("id") for item in items]
    if not ids or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError(f"{where} must be a non-empty list with unique string ids")


def load_pilot_spec(path: Path | None = None) -> dict[str, Any]:
    path = SPEC_PATH if path is None else path
    spec = json.loads(path.read_text())
    _closed(spec, {"schema", "views_per_scene", "drawing_episode_count", "identity_ids", "cells", "artworks", "appearances", "episode_variants", "challenges"}, "pilot spec")
    if spec["schema"] != SPEC_SCHEMA:
        raise ValueError("unsupported pilot spec")
    for name in ("cells", "artworks", "appearances", "episode_variants", "challenges"):
        _unique_ids(spec[name], f"pilot spec {name}")
    if not isinstance(spec["views_per_scene"], int) or spec["views_per_scene"] < 1:
        raise ValueError("pilot spec needs at least one view per scene")
    span = max(len(spec["cells"]), len(spec["artworks"]))
    if not isinstance(spec["drawing_episode_count"], int) or spec["drawing_episode_count"] < span:
        raise ValueError(f"pilot spec needs at least {span} drawing episodes to span every cell and artwork")
    identities = spec["identity_ids"]
    if not identities or identities[0] != "reference" or len(set(identities)) != len(identities):
        raise ValueError("pilot spec must name reference first, then distinct candidate identities")
    for index, cell in enumerate(spec["cells"]):
        _closed(cell, {"id", "site", "laterality", "pose", "support"}, f"cell {index}")
    for index, artwork in enumerate(spec["artworks"]):
        _closed(artwork, {"id", "path", "size_mm"}, f"artwork {index}")
        source = repo_root() / artwork["path"]
        if not source.is_file() or not source.resolve().is_relative_to(repo_root().resolve()):
            raise ValueError(f"artwork {index} is not a repository fixture")
        if len(artwork["size_mm"]) != 2 or not all(float(value) > 0 for value in artwork["size_mm"]):
            raise ValueError(f"artwork {index} has invalid physical size")
    for index, appearance in enumerate(spec["appearances"]):
        _closed(appearance, {"id", "skin_tone", "lighting", "occlusion_fraction"}, f"appearance {index}")
        fraction = appearance["occlusion_fraction"]
        if not isinstance(fraction, (int, float)) or not 0.0 <= float(fraction) < 1.0:
            raise ValueError(f"appearance {index} occlusion_fraction must lie in [0, 1)")
    variants = {item["id"] for item in spec["episode_variants"]}
    if variants != set(VARIANT_CONTRACT):
        raise ValueError(f"episode variant coverage differs: {sorted(variants ^ set(VARIANT_CONTRACT))}")
    for index, variant in enumerate(spec["episode_variants"]):
        _closed(variant, {"id", "expected_outcome", "status"}, f"episode variant {index}")
        if (variant["expected_outcome"], variant["status"]) != VARIANT_CONTRACT[variant["id"]]:
            raise ValueError(f"episode variant {variant['id']!r} contract differs")
    if not any(float(a["occlusion_fraction"]) > 0 for a in spec["appearances"]):
        raise ValueError("pilot spec needs one appearance with occlusion_fraction > 0 for the occluded variant")
    return spec


def expected_counts(spec: dict[str, Any], admitted: set[str]) -> dict[str, int]:
    """Every count the ledger must report, derived from the spec alone."""
    templates = len(spec["cells"]) * len(spec["artworks"]) * len(spec["appearances"])
    identities = len(spec["identity_ids"])
    planned_identities = sum(identity in admitted for identity in spec["identity_ids"])
    return {
        "reference_scene_templates": templates,
        "all_identity_scenes": templates * identities,
        "planned_scenes": templates * planned_identities,
        "identity_blocked_scenes": templates * (identities - planned_identities),
        "planned_perception_images": templates * planned_identities * spec["views_per_scene"],
        "full_perception_images_after_identity_review": templates * identities * spec["views_per_scene"],
        "drawing_episodes": spec["drawing_episode_count"],
    }


def _episode_requests(spec: dict[str, Any], artwork_records: list[dict], seed: int) -> list[dict[str, Any]]:
    variants = spec["episode_variants"]
    appearances = spec["appearances"]
    occluding = max(appearances, key=lambda item: float(item["occlusion_fraction"]))
    requests = []
    for index in range(spec["drawing_episode_count"]):
        cell = spec["cells"][index % len(spec["cells"])]
        artwork = artwork_records[index % len(artwork_records)]
        variant = variants[index % len(variants)]
        # Appearance occlusion is only ever exercised through the occluded
        # variant; every other episode records an exact 0.0 so its label and
        # its generator flag agree.
        occluded = variant["id"] == "occluded"
        appearance = occluding if occluded else appearances[index % len(appearances)]
        fraction = float(appearance["occlusion_fraction"]) if occluded else 0.0
        flags = ["--episode-variant", variant["id"]]
        if occluded:
            flags += ["--occlusion-fraction", repr(fraction)]
        requests.append({
            "id": f"episode-{index:02d}-{cell['id']}-{artwork['id']}",
            "seed": stable_seed(seed, f"episode-{index:02d}", variant["id"]),
            "cell_id": cell["id"],
            "artwork_id": artwork["id"],
            "appearance_id": appearance["id"],
            "occlusion_fraction": fraction,
            "variant": variant,
            "generate_flags": flags,
            "status": "blocked_compute_reach_clearance",
            "success_label": None,
        })
    return requests


def build_pilot_plan(output_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise ValueError("pilot artifacts must be outside the repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec = load_pilot_spec()
    catalog_identities = json.loads(IDENTITIES_PATH.read_text())["identities"]
    by_id = {item["id"]: item for item in catalog_identities}
    if any(identity_id not in by_id for identity_id in spec["identity_ids"]):
        raise ValueError("pilot spec names an unknown identity")
    identities = [by_id[identity_id] for identity_id in spec["identity_ids"]]
    admitted = set(admitted_identity_ids())
    artwork_records = []
    for artwork in spec["artworks"]:
        path = repo_root() / artwork["path"]
        artwork_records.append({**artwork, "sha256": _digest(path.read_bytes())})
    scene_templates = []
    for cell, artwork, appearance in itertools.product(spec["cells"], artwork_records, spec["appearances"]):
        key = f"{cell['id']}:{artwork['id']}:{appearance['id']}"
        scene_templates.append({
            "id": key,
            "seed": stable_seed(seed, key, "pilot_scene"),
            "cell": cell,
            "artwork": artwork,
            "appearance": appearance,
            "status": "planned",
        })
    scenes = []
    for identity in identities:
        for template in scene_templates:
            accepted = identity["id"] in admitted
            scenes.append({
                "id": f"{identity['id']}:{template['id']}",
                "identity_id": identity["id"],
                "identity_sha256": identity["identity_sha256"],
                "rest_surface_sha256": identity["rest_surface_sha256"],
                "template_id": template["id"],
                "seed": stable_seed(seed, template["id"], identity["id"]),
                "status": "planned" if accepted else "blocked_identity_review",
                "required_views": spec["views_per_scene"],
            })
    episode_requests = _episode_requests(spec, artwork_records, seed)
    source = source_state()
    plan = {
        "schema": SCHEMA,
        "content_sha256": "",
        "seed": seed,
        "spec_sha256": _digest(SPEC_PATH.read_bytes()),
        "identity_catalog_sha256": _digest(IDENTITIES_PATH.read_bytes()),
        "source": source,
        "counts": expected_counts(spec, admitted),
        "scene_templates": scene_templates,
        "scenes": scenes,
        "drawing_episodes": episode_requests,
        "episode_variants": spec["episode_variants"],
        "challenges": spec["challenges"],
        "gates": {
            "identity_visual_review": "pending",
            "stencil_renderer_contract": "implemented_ungenerated",
            "assigned_compute_host": "pending",
            "full_trajectory_reach_clearance": "pending",
            "gpu_render_and_numeric_audit": "pending",
            "human_visual_review": "pending",
            "powered_motion": "out_of_scope",
        },
    }
    plan["content_sha256"] = _digest(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    problems = audit_pilot_plan(plan)
    if problems:
        raise ValueError("pilot plan audit failed: " + "; ".join(problems))
    (output / "pilot-plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (output / "audit.json").write_text(json.dumps({"schema": AUDIT_SCHEMA, "status": "pass", "problems": []}, indent=2, sort_keys=True) + "\n")
    return plan


def audit_pilot_plan(plan: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if plan.get("schema") != SCHEMA:
        return ["wrong pilot plan schema"]
    copy = dict(plan)
    claimed = copy.pop("content_sha256", None)
    copy["content_sha256"] = ""
    if claimed != _digest(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()):
        problems.append("pilot plan digest mismatch")
    if plan.get("spec_sha256") != _digest(SPEC_PATH.read_bytes()):
        problems.append("pilot specification digest mismatch")
    if plan.get("identity_catalog_sha256") != _digest(IDENTITIES_PATH.read_bytes()):
        problems.append("identity catalog digest mismatch")
    source = plan.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("repository"), str) or not isinstance(source.get("dirty"), bool) or not isinstance(source.get("revision"), str) or len(source["revision"]) != 40:
        problems.append("source provenance is incomplete")
    try:
        spec = load_pilot_spec()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"pilot specification is invalid: {exc}")
        return problems
    admitted = set(admitted_identity_ids())
    expected = expected_counts(spec, admitted)
    counts = plan.get("counts", {})
    if counts != expected:
        problems.append(f"pilot counts differ from the specification: {counts} != {expected}")
    scenes = plan.get("scenes", [])
    if len(scenes) != expected["all_identity_scenes"]:
        problems.append("identity scene count differs from ledger")
    if sum(item.get("status") == "blocked_identity_review" for item in scenes) != expected["identity_blocked_scenes"]:
        problems.append("identity blocked count differs from ledger")
    if len({item.get("id") for item in plan.get("scene_templates", [])}) != expected["reference_scene_templates"]:
        problems.append("scene template IDs are missing or duplicated")
    if len({item.get("seed") for item in scenes}) != len(scenes):
        problems.append("scene seeds are not unique")
    episodes = plan.get("drawing_episodes", [])
    if len(episodes) != expected["drawing_episodes"]:
        problems.append("drawing episode count differs from the specification")
    if {item.get("cell_id") for item in episodes} != {item["id"] for item in spec["cells"]}:
        problems.append("drawing episodes do not span all primary cells")
    if {item.get("artwork_id") for item in episodes} != {item["id"] for item in spec["artworks"]}:
        problems.append("drawing episodes do not span all artwork classes")
    if any(item.get("success_label") is not None for item in episodes):
        problems.append("unrun drawing episode has a success label")
    for item in episodes:
        occluded = (item.get("variant") or {}).get("id") == "occluded"
        fraction = item.get("occlusion_fraction")
        if occluded and not (isinstance(fraction, float) and 0.0 < fraction < 1.0):
            problems.append(f"occluded episode {item.get('id')} has no exact occlusion fraction")
        if not occluded and fraction != 0.0:
            problems.append(f"episode {item.get('id')} declares occlusion outside the occluded variant")
    if any(item.get("status") == "planned" and item.get("identity_id") not in admitted for item in scenes):
        problems.append("unreviewed identity was planned for generation")
    return problems


def audit_pilot_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 20_000_000:
        return {"schema": AUDIT_SCHEMA, "status": "fail", "problems": ["pilot plan is missing or exceeds 20 MB"]}
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"schema": AUDIT_SCHEMA, "status": "fail", "problems": [f"pilot plan is unreadable: {exc}"]}
    problems = audit_pilot_plan(value) if isinstance(value, dict) else ["pilot plan is not an object"]
    return {"schema": AUDIT_SCHEMA, "status": "pass" if not problems else "fail", "problems": problems}
