"""Typed, deterministic InkLang request to replayable simulation scenario."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.contracts import document_sha256
from tatbot_sim.inkmap.designs import SPIRAL_ID, DesignArtifact, directory_artifacts, spiral_artifact
from tatbot_sim.inkmap.gltf_surface import MODEL_ID
from tatbot_sim.inkmap.inklang_client import InkLangConsumerError, parse_tattoo_request
from tatbot_sim.inkmap.reach import ReachAuditError, optimize_body_placement
from tatbot_sim.inkmap.rig import load_body_rig
from tatbot_sim.inkmap.sampler import (
    DEFAULT_POSES,
    DEFAULT_SITES,
    SiteChoice,
    _body_record,
    _pose_supports_site,
    _resolved_anchor_candidates,
    _trace_stays_on_site,
)
from tatbot_sim.repo import repo_root
from tatbot_sim.site_sampling import INKLANG_VERSION, SITES
from tatbot_sim.tools import active_tool

REQUEST_SCHEMA = "tatbot.scenario-request/1"
RESOLUTION_SCHEMA = "tatbot.scenario-resolution/1"
PARSER_FILES = (
    repo_root() / "web" / "inkmap" / "tools" / "parse_request.ts",
    repo_root() / "web" / "inkmap" / "tools" / "resolve.ts",
    repo_root() / "web" / "inkmap" / "src" / "core" / "lang.ts",
    *sorted((repo_root() / "web" / "inkmap" / "src" / "core" / "inklang").glob("*.ts")),
    repo_root() / "config" / "inkmap" / "sites.json",
    repo_root() / "config" / "inkmap" / "styles.json",
)
DEFAULT_CREATED_AT = "1970-01-01T00:00:00Z"


class ScenarioResolveError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class ScenarioRequest:
    schema: str
    prompt: str
    parser: str
    parser_sha256: str
    design_subject: str
    design_id: str | None
    style_prompt: str | None
    size_mm: tuple[float, float]
    site: str
    laterality: str | None
    aspect: str | None
    level: str | None
    pose: str
    support: str
    seed: int
    program: dict
    intent: dict

    def as_dict(self) -> dict:
        value = asdict(self)
        value["size_mm"] = list(self.size_mm)
        return value


def _digest_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repo_root()).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_scenario_request(
    prompt: str,
    *,
    size_mm: tuple[float, float],
    pose: str,
    support: str,
    seed: int,
    design_id: str | None = None,
) -> ScenarioRequest:
    try:
        parsed = parse_tattoo_request(prompt)
    except InkLangConsumerError as exc:
        raise ScenarioResolveError(exc.code, str(exc)) from exc
    program = parsed["program"]
    intent = parsed["placement_intent"]
    style = parsed.get("style_prompt")
    site = program["site"]
    if site.get("relation") is not None:
        raise ScenarioResolveError("unsupported_relative_site", "relative InkLang sites are not mapped to atlas faces")
    site_id = str(site["id"])
    if site_id not in DEFAULT_SITES:
        raise ScenarioResolveError(
            "unsupported_anatomy",
            f"{site_id!r} is valid InkLang but has no simulation atlas resolver",
        )
    laterality = site.get("laterality")
    if laterality == "center":
        laterality = None
    if SITES[site_id]["laterality"] == "sided" and laterality not in ("left", "right"):
        raise ScenarioResolveError("ambiguous_laterality", f"{site_id!r} requires left or right")
    if seed < 0:
        raise ScenarioResolveError("invalid_seed", "seed must be non-negative")
    if len(size_mm) != 2 or not np.isfinite(size_mm).all() or min(size_mm) <= 0:
        raise ScenarioResolveError("invalid_size", "size_mm must be two positive finite values")
    return ScenarioRequest(
        schema=REQUEST_SCHEMA,
        prompt=prompt,
        parser=INKLANG_VERSION,
        parser_sha256=_digest_files(PARSER_FILES),
        design_subject=str(program["motif"]),
        design_id=design_id,
        style_prompt=style,
        size_mm=(float(size_mm[0]), float(size_mm[1])),
        site=site_id,
        laterality=laterality,
        aspect=site.get("aspect"),
        level=site.get("level"),
        pose=pose,
        support=support,
        seed=int(seed),
        program=program,
        intent=intent,
    )


def _stable_index(seed: int, label: str, count: int) -> int:
    raw = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % count


def _support_matches(preference: str, support_id: str) -> bool:
    if preference == "compatible":
        return True
    if preference == "bed":
        return support_id.startswith("tattoo-bed-")
    if preference == "chair":
        return support_id.startswith("tattoo-chair-") and "armrest" not in support_id
    if preference == "armrest":
        return support_id.endswith("armrest-v1")
    return preference == support_id


def _resolve_pose(request: ScenarioRequest) -> tuple[str, str]:
    rig = load_body_rig()
    site = SiteChoice(request.site, request.laterality)
    if request.pose == "compatible":
        poses = [
            candidate for candidate in DEFAULT_POSES
            if _pose_supports_site(candidate, site)
            and _support_matches(
                request.support, rig.catalog_record["poses"][candidate]["support_id"],
            )
        ]
        if not poses:
            raise ScenarioResolveError(
                "unsupported_support",
                f"no compatible pose for {request.site}/{request.laterality} and {request.support!r}",
            )
        pose_id = poses[_stable_index(request.seed, "pose", len(poses))]
    else:
        if request.pose not in rig.pose_ids:
            raise ScenarioResolveError("unknown_pose", f"unknown pose {request.pose!r}")
        if not _pose_supports_site(request.pose, site):
            raise ScenarioResolveError(
                "incompatible_pose_site",
                f"{request.pose!r} does not support {request.site}/{request.laterality}",
            )
        pose_id = request.pose
    support_id = rig.catalog_record["poses"][pose_id]["support_id"]
    if not _support_matches(request.support, support_id):
        raise ScenarioResolveError(
            "unsupported_support", f"{request.support!r} is incompatible with {pose_id!r}",
        )
    return pose_id, support_id


def _resolve_design(request: ScenarioRequest, design_dir: Path | None) -> DesignArtifact:
    if request.design_id == SPIRAL_ID:
        return spiral_artifact()
    try:
        if design_dir is None:
            from tatbot_sim.inkmap.collection import collection_artifacts
            artifacts = collection_artifacts("all")
        else:
            artifacts = directory_artifacts(design_dir, request.size_mm)
    except ValueError as exc:
        raise ScenarioResolveError("invalid_design_artifact", str(exc)) from exc
    if request.design_id:
        matches = [item for item in artifacts if item.id == request.design_id]
        if not matches:
            raise ScenarioResolveError("unknown_design", f"{request.design_id!r} is not in {design_dir}")
    else:
        subject = " ".join(request.design_subject.lower().split())
        matches = [
            item for item in artifacts
            if " ".join(str(item.source.get("subject", item.name)).lower().split()) == subject
        ]
        if request.style_prompt is not None:
            style = " ".join(request.style_prompt.lower().split())
            matches = [
                item for item in matches
                if " ".join(str(item.source.get("style", "")).lower().split()) == style
            ]
        if not matches:
            raise ScenarioResolveError(
                "design_not_materialized",
                f"no immutable artifact matches subject {request.design_subject!r} and requested style",
            )
    return matches[_stable_index(request.seed, "design", len(matches))]


def _placement(
    request: ScenarioRequest,
    design: DesignArtifact,
    resolution: dict,
    rotation_rad: float,
) -> dict:
    request_hash = document_sha256(request.as_dict())[:16]
    placement = {
        "id": f"request-{request_hash}",
        "design_id": design.id,
        "anchor": resolution["anchor"],
        "rotation_rad": float(rotation_rad),
        "size_mm": list(request.size_mm),
        "mirror": False,
        "site": {
            "id": request.site, "laterality": request.laterality,
            "aspect": request.aspect, "level": request.level,
            "uv": resolution["actual"]["region_uv"], "lexicon": INKLANG_VERSION,
        },
        "language": {
            "sentence": request.prompt,
            "program": request.program,
            "intent": resolution["intent"],
            "resolution": resolution,
            "consumer_policy": "simulation-exposed-grid-v1",
        },
    }
    return {
        "schema_version": 6,
        "units": {"length": "m", "tattoo_size": "mm", "up": "+z"},
        "body": dict(_body_record()),
        "placements": [placement],
        "designs": {design.id: design.embedded()},
    }


def resolve_scenario_request(
    request: ScenarioRequest,
    output_dir: Path,
    *,
    generated_design_dir: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
    git_sha: str | None = None,
    max_attempts: int = 4,
) -> dict:
    """Resolve one typed request, retaining every bounded rejection."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ScenarioResolveError("output_not_empty", f"output directory is not empty: {output_dir}")
    if max_attempts <= 0:
        raise ScenarioResolveError("invalid_attempt_limit", "max_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_id, support_id = _resolve_pose(request)
    design = _resolve_design(request, generated_design_dir)
    site = SiteChoice(request.site, request.laterality)
    resolutions = _resolved_anchor_candidates(
        pose_id,
        SiteChoice(request.site, request.laterality, request.aspect, request.level),
        description=request.prompt,
        seed=request.seed,
    )
    attempts = []
    for attempt in range(max_attempts):
        resolution = resolutions[min(attempt, len(resolutions) - 1)]
        rng = np.random.default_rng(np.random.SeedSequence([request.seed, attempt]))
        placement = _placement(request, design, resolution, rng.uniform(-np.pi, np.pi))
        try:
            if design.id == SPIRAL_ID:
                scenario = compile_scenario(placement, pose_id=pose_id, support_id=support_id,
                    seed=request.seed, target_world_m=[.31, .005, .04], tool_id=active_tool().tool_id,
                    created_at=created_at, git_sha=git_sha, generator="tatbot sim resolve calibration")
            else:
                from tatbot_sim.inkmap.sampler import compile_artwork_placement
                scenario = compile_artwork_placement(placement, design, pose_id=pose_id,
                    seed=request.seed, target=[.31, .005, .04], created_at=created_at, git_sha=git_sha)
            if not _trace_stays_on_site(scenario, site):
                attempts.append({"attempt": attempt, "accepted": False, "reason": "site_boundary"})
                continue
            selection = optimize_body_placement(
                scenario, trajectory_seed=int(rng.integers(0, 2**31)),
            )
            scenario = selection.scenario
        except ReachAuditError as exc:
            attempts.append({
                "attempt": attempt, "accepted": False, "reason": exc.reason,
                "detail": str(exc)[:400],
            })
            continue
        except ValueError as exc:
            attempts.append({
                "attempt": attempt, "accepted": False,
                "reason": "compile_or_trace", "detail": str(exc)[:400],
            })
            continue
        attempts.append({
            "attempt": attempt, "accepted": True,
            "scenario_sha256": document_sha256(scenario),
            "reach_audit": selection.audit,
            "placement_sha256": document_sha256(placement),
        })
        request_path = output_dir / "request.json"
        placement_path = output_dir / "placement.json"
        scenario_path = output_dir / "scenario.json"
        request_path.write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n")
        placement_path.write_text(json.dumps(placement, indent=2, sort_keys=True) + "\n")
        scenario_path.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n")
        manifest = {
            "schema": RESOLUTION_SCHEMA,
            "request": request_path.name,
            "request_sha256": document_sha256(request.as_dict()),
            "placement": placement_path.name,
            "placement_sha256": document_sha256(placement),
            "scenario": scenario_path.name,
            "scenario_sha256": document_sha256(scenario),
            "design": {"id": design.id, "sha256": design.sha256, "source": design.source},
            "body": MODEL_ID,
            "pose": pose_id,
            "support": support_id,
            "attempts": attempts,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (output_dir / "attempts.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in attempts),
        )
        return manifest
    (output_dir / "request.json").write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n")
    (output_dir / "attempts.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in attempts),
    )
    raise ScenarioResolveError("resolution_exhausted", f"all {max_attempts} attempts were rejected")
