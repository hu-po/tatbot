"""Deterministic, bounded procedural sampling of posed-body tattoo scenarios."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.compiler import ScenarioCompileError, compile_scenario
from tatbot_sim.inkmap.contracts import ContractError, document_sha256
from tatbot_sim.inkmap.rig import BODY_ASSET_ROOT, CATALOG_PATH, BodyRigError, load_body_rig
from tatbot_sim.inkmap.surface_trace import SurfaceTraceError
from tatbot_sim.inkmap.svg_strokes import SvgCompileError, compile_svg_strokes
from tatbot_sim.repo import repo_root
from tatbot_sim.sites import INKLANG_VERSION, SITES, site_phrase

SUITE_SCHEMA_VERSION = 1
DEFAULT_BODIES = ("hbm-female-stylized", "hbm-male-stylized")
DEFAULT_POSES = (
    "supine",
    "prone",
    "reclined-seated",
    "reclined-left-arm-supported",
    "reclined-right-arm-supported",
)
DEFAULT_SITES = ("forearm", "bicep", "shoulder_cap", "thigh", "calf", "mid_back")
LATERALITY_CODE = {None: 0, "left": 1, "right": 2}


class ScenarioSampleError(ValueError):
    pass


@dataclass(frozen=True)
class DesignChoice:
    id: str
    name: str
    size_mm: tuple[float, float]
    svg: str | None = None


@dataclass(frozen=True)
class SiteChoice:
    id: str
    laterality: str | None


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root(), text=True,
    ).strip()


def _balanced(values, count: int, rng: np.random.Generator) -> list:
    output = []
    while len(output) < count:
        cycle = list(values)
        rng.shuffle(cycle)
        output.extend(cycle)
    return output[:count]


def _designs(
    generated_design_dir: Path | None,
    generated_size_mm: tuple[float, float],
    include_builtin_designs: bool,
) -> tuple[DesignChoice, ...]:
    manifest = json.loads((BODY_ASSET_ROOT / "designs" / "manifest.json").read_text())
    choices = [
        DesignChoice(item["id"], item["name"], tuple(float(v) for v in item["default_size_mm"]))
        for item in manifest["designs"]
    ] if include_builtin_designs else []
    if generated_design_dir is None:
        if not choices:
            raise ScenarioSampleError("generated-only sampling requires --generated-design-dir")
        return tuple(choices)
    if not generated_design_dir.is_dir():
        raise ScenarioSampleError(f"generated design directory does not exist: {generated_design_dir}")
    generated = sorted(generated_design_dir.glob("*.svg"))
    for path in generated:
        svg = path.read_text()
        compile_svg_strokes(svg, generated_size_mm)
        slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "design"
        digest = hashlib.sha256(svg.encode()).hexdigest()[:12]
        choices.append(DesignChoice(f"gen-{slug}-{digest}", path.stem, generated_size_mm, svg))
    if not generated:
        raise ScenarioSampleError(f"generated design directory contains no .svg files: {generated_design_dir}")
    return tuple(choices)


def _site_choices(site_ids: tuple[str, ...]) -> tuple[SiteChoice, ...]:
    choices = []
    for site_id in site_ids:
        try:
            site = SITES[site_id]
        except KeyError as exc:
            raise ScenarioSampleError(f"unknown site {site_id!r}") from exc
        lateralities = ("left", "right") if site["laterality"] == "sided" else (None,)
        choices.extend(SiteChoice(site_id, laterality) for laterality in lateralities)
    return tuple(choices)


def _pose_supports_site(pose_id: str, site: SiteChoice) -> bool:
    """Keep the sampled posture useful for access to the tattooed body part."""
    if pose_id == "reclined-left-arm-supported":
        return site.id in ("forearm", "bicep") and site.laterality == "left"
    if pose_id == "reclined-right-arm-supported":
        return site.id in ("forearm", "bicep") and site.laterality == "right"
    if pose_id == "prone":
        return site.id in ("mid_back", "shoulder_cap", "thigh", "calf")
    if pose_id in ("supine", "reclined-seated"):
        return site.id in ("shoulder_cap", "thigh", "calf")
    return True


def _sites_for_poses(
    pose_draws: list[str],
    site_values: tuple[SiteChoice, ...],
    rng: np.random.Generator,
) -> list[SiteChoice]:
    output: list[SiteChoice | None] = [None] * len(pose_draws)
    for pose_id in dict.fromkeys(pose_draws):
        indices = [index for index, value in enumerate(pose_draws) if value == pose_id]
        compatible = tuple(site for site in site_values if _pose_supports_site(pose_id, site))
        if not compatible:
            raise ScenarioSampleError(f"pose {pose_id!r} has no compatible requested tattoo sites")
        for index, site in zip(indices, _balanced(compatible, len(indices), rng), strict=True):
            output[index] = site
    resolved = [site for site in output if site is not None]
    if len(resolved) >= len({site.id for site in site_values}):
        for missing_id in dict.fromkeys(site.id for site in site_values):
            if any(site.id == missing_id for site in resolved):
                continue
            candidates = [site for site in site_values if site.id == missing_id]
            rng.shuffle(candidates)
            counts = {site.id: sum(value.id == site.id for value in resolved) for site in resolved}
            replacement = next(
                (
                    (index, site)
                    for site in candidates
                    for index, pose_id in enumerate(pose_draws)
                    if _pose_supports_site(pose_id, site) and counts[resolved[index].id] > 1
                ),
                None,
            )
            if replacement is None:
                raise ScenarioSampleError(f"cannot cover requested tattoo site {missing_id!r} with selected poses")
            resolved[replacement[0]] = replacement[1]
    return resolved


@lru_cache(maxsize=4)
def _atlas(body_id: str) -> dict:
    path = BODY_ASSET_ROOT / "bodies" / f"{body_id}.regions.json"
    if not path.is_file():
        raise ScenarioSampleError(f"missing region atlas for {body_id}: {path}")
    return json.loads(path.read_text())


@lru_cache(maxsize=128)
def _candidate_faces(body_id: str, site: SiteChoice) -> tuple[int, ...]:
    atlas = _atlas(body_id)
    body_record = _body_record(body_id)
    if atlas.get("body") != {"id": body_id, "sha256": body_record["asset_sha256"]}:
        raise ScenarioSampleError(f"{body_id}: region atlas body identity is stale")
    if atlas.get("inklang") != INKLANG_VERSION:
        raise ScenarioSampleError(f"{body_id}: region atlas InkLang version is stale")
    try:
        site_index = atlas["sites"].index(site.id)
    except ValueError as exc:
        raise ScenarioSampleError(f"{body_id}: atlas has no site {site.id!r}") from exc
    code = site_index * 4 + LATERALITY_CODE[site.laterality]
    rig = load_body_rig(body_id)
    body_index = rig.part_names.index("Body")
    first = int(rig.part_first_face[body_index])
    stop = first + int(rig.part_face_count[body_index])
    candidates = np.asarray([face for face, value in enumerate(atlas["faces"]) if value == code and first <= face < stop])
    if not len(candidates):
        raise ScenarioSampleError(f"{body_id}: no faces for {site.id}/{site.laterality or 'center'}")
    centers = rig.rest_vertices[candidates].mean(axis=1)
    centroid = np.median(centers, axis=0)
    order = np.argsort(np.linalg.norm(centers - centroid, axis=1), kind="stable")
    # A small central pool lets retries vary the anchor without wandering near
    # the atlas boundary. The trace is checked against the atlas after compile.
    return tuple(int(value) for value in candidates[order[: min(24, len(order))]])


@lru_cache(maxsize=4)
def _body_record(body_id: str) -> dict:
    catalog = json.loads(CATALOG_PATH.read_text())
    try:
        record = catalog["bodies"][body_id]
    except KeyError as exc:
        raise ScenarioSampleError(f"unknown body {body_id!r}") from exc
    return {
        "id": body_id,
        "path": record["source_path"],
        "asset_sha256": record["source_asset_sha256"],
        "surface_sha256": record["surface_sha256"],
    }


def _placement_file(
    body_id: str,
    site: SiteChoice,
    design: DesignChoice,
    *,
    sample_index: int,
    attempt: int,
    rng: np.random.Generator,
) -> dict:
    face_pool = _candidate_faces(body_id, site)
    face = face_pool[min(attempt, len(face_pool) - 1)]
    max_dimension_mm = {
        "forearm": 48.0,
        "bicep": 52.0,
        "shoulder_cap": 38.0,
        "thigh": 62.0,
        "calf": 52.0,
        "mid_back": 46.0,
    }.get(site.id, 42.0)
    # Retry shrinkage is deterministic and bounded. It is applied before the
    # expensive trace compile and makes site-boundary rejects recoverable.
    max_dimension_mm *= max(0.48, 0.82 - 0.07 * attempt)
    scale = min(1.0, max_dimension_mm / max(design.size_mm))
    size = [round(value * scale, 6) for value in design.size_mm]
    phrase = site_phrase(site.id, site.laterality)
    placement_id = f"sample-{sample_index:04d}-{design.id}-{site.id}"
    placement = {
        "id": placement_id,
        "design_id": design.id,
        "anchor": {"face": face, "barycentric": [1 / 3, 1 / 3, 1 / 3]},
        "rotation_rad": float(rng.uniform(-math.pi, math.pi)),
        "size_mm": size,
        "mirror": bool(rng.integers(2)),
        "site": {
            "id": site.id,
            "laterality": site.laterality,
            "aspect": None,
            "level": None,
            "lexicon": INKLANG_VERSION,
        },
        "language": {
            "sentence": f"a {design.name.lower()} on the {phrase}",
            "program": {
                "inklang": INKLANG_VERSION,
                "motif": design.id,
                "style": None,
                "secondary": [],
                "technique": None,
                "color": None,
                "site": {"id": site.id, "laterality": site.laterality, "aspect": None, "level": None},
            },
        },
    }
    document = {
        "schema_version": 4,
        "units": {"length": "m", "tattoo_size": "mm", "up": "+z"},
        "body": dict(_body_record(body_id)),
        "placements": [placement],
    }
    if design.svg is not None:
        document["designs"] = {
            design.id: {"name": design.name, "svg": design.svg, "source": {"kind": "generated-local"}},
        }
    return document


def _trace_stays_on_site(scenario: dict, site: SiteChoice) -> bool:
    atlas = _atlas(scenario["body"]["id"])
    site_index = atlas["sites"].index(site.id)
    code = site_index * 4 + LATERALITY_CODE[site.laterality]
    return all(
        atlas["faces"][anchor["face"]] == code
        for stroke in scenario["trace"]["strokes"]
        for anchor in stroke
    )


def _reason(exc: Exception) -> str:
    if isinstance(exc, SurfaceTraceError):
        return "surface_walk"
    if isinstance(exc, SvgCompileError):
        return "design_compile"
    if isinstance(exc, BodyRigError):
        return "rig_contract"
    if isinstance(exc, ContractError):
        return "schema_contract"
    if isinstance(exc, ScenarioCompileError):
        return "scenario_compile"
    return "invalid_input"


def materialize_scenario_suite(
    output_dir: Path,
    *,
    count: int = 64,
    seed: int = 0,
    bodies: tuple[str, ...] = DEFAULT_BODIES,
    poses: tuple[str, ...] = DEFAULT_POSES,
    sites: tuple[str, ...] = DEFAULT_SITES,
    generated_design_dir: Path | None = None,
    generated_size_mm: tuple[float, float] = (50.0, 50.0),
    include_builtin_designs: bool = True,
    audit_reach: bool = False,
    max_attempts_per_scenario: int = 4,
    created_at: str | None = None,
    git_sha: str | None = None,
) -> dict:
    """Write a self-contained scenario suite and an explicit attempt ledger."""
    output_dir = Path(output_dir)
    if count <= 0:
        raise ScenarioSampleError("count must be positive")
    if max_attempts_per_scenario <= 0:
        raise ScenarioSampleError("max_attempts_per_scenario must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ScenarioSampleError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir = output_dir / "scenarios"
    placement_dir = output_dir / "placements"
    scenario_dir.mkdir()
    placement_dir.mkdir()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    git_sha = git_sha or _git_sha()
    rng = np.random.default_rng(seed)
    designs = _designs(generated_design_dir, generated_size_mm, include_builtin_designs)
    site_values = _site_choices(sites)
    body_draws = _balanced(bodies, count, rng)
    pose_draws = _balanced(poses, count, rng)
    site_draws = _sites_for_poses(pose_draws, site_values, rng)
    design_draws = _balanced(designs, count, rng)
    attempts: list[dict] = []
    accepted: list[dict] = []
    known_errors = (ScenarioCompileError, SurfaceTraceError, SvgCompileError, BodyRigError, ContractError, ValueError)
    for sample_index in range(count):
        body_id = body_draws[sample_index]
        pose_id = pose_draws[sample_index]
        site = site_draws[sample_index]
        design = design_draws[sample_index]
        for retry in range(max_attempts_per_scenario):
            attempt_seed = int(rng.integers(0, 2**31))
            attempt_rng = np.random.default_rng(attempt_seed)
            base = {
                "attempt": len(attempts),
                "sample_index": sample_index,
                "retry": retry,
                "seed": attempt_seed,
                "body": body_id,
                "pose": pose_id,
                "site": site.id,
                "laterality": site.laterality,
                "design": design.id,
            }
            try:
                placement = _placement_file(
                    body_id, site, design, sample_index=sample_index, attempt=retry, rng=attempt_rng,
                )
                target = [
                    float(attempt_rng.uniform(0.30, 0.32)),
                    float(attempt_rng.uniform(-0.035, 0.045)),
                    0.04,
                ]
                scenario = compile_scenario(
                    placement,
                    pose_id=pose_id,
                    seed=attempt_seed,
                    target_world_m=target,
                    created_at=created_at,
                    git_sha=git_sha,
                    generator="tatbot sim sample",
                )
                if not _trace_stays_on_site(scenario, site):
                    attempts.append({**base, "accepted": False, "reason": "site_boundary"})
                    continue
                reach_record = None
                if audit_reach:
                    from tatbot_sim.inkmap.reach import ReachAuditError, select_reachable_patch_yaw

                    try:
                        selection = select_reachable_patch_yaw(scenario, trajectory_seed=attempt_seed)
                    except ReachAuditError as exc:
                        attempts.append({**base, "accepted": False, "reason": "ik_probe", "detail": str(exc)[:240]})
                        continue
                    scenario = selection.scenario
                    reach_record = {
                        "patch_yaw_rad": selection.patch_yaw_rad,
                        "probe_max_residual_m": selection.probe_max_residual_m,
                        "candidates": list(selection.candidates),
                    }
            except known_errors as exc:
                attempts.append({**base, "accepted": False, "reason": _reason(exc), "detail": str(exc)[:240]})
                continue
            placement_name = f"{sample_index:04d}-{body_id}-{pose_id}-{site.id}-{design.id}.placement.json"
            scenario_name = placement_name.replace(".placement.json", ".scenario.json")
            _write_json(placement_dir / placement_name, placement)
            _write_json(scenario_dir / scenario_name, scenario)
            record = {
                **base,
                "accepted": True,
                "placement": f"placements/{placement_name}",
                "scenario": f"scenarios/{scenario_name}",
                "scenario_sha256": document_sha256(scenario),
                "trace_sha256": scenario["trace"]["sha256"],
                **({"reach_probe": reach_record} if reach_record is not None else {}),
            }
            attempts.append(record)
            accepted.append(record)
            break
        else:
            break
    rejected = len(attempts) - len(accepted)
    manifest = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "generator": "tatbot sim sample",
        "seed": seed,
        "created_at": created_at,
        "git_sha": git_sha,
        "requested": count,
        "accepted": len(accepted),
        "rejected_attempts": rejected,
        "rejection_rate": rejected / len(attempts) if attempts else 0.0,
        "complete": len(accepted) == count,
        "max_attempts_per_scenario": max_attempts_per_scenario,
        "reach_audited": audit_reach,
        "coverage": {
            "bodies": sorted({item["body"] for item in accepted}),
            "poses": sorted({item["pose"] for item in accepted}),
            "sites": sorted({item["site"] for item in accepted}),
            "designs": sorted({item["design"] for item in accepted}),
        },
        "scenarios": accepted,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "attempts.jsonl").write_bytes(b"".join(_json_bytes(item) + b"\n" for item in attempts))
    if not manifest["complete"]:
        failed_index = len(accepted)
        raise ScenarioSampleError(
            f"suite stopped at {failed_index}/{count}; inspect {output_dir / 'attempts.jsonl'}",
        )
    return manifest
