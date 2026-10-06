"""Deterministic artwork placement sampling for posed-body tattoo scenarios."""

from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from tatbot_sim.inkmap.compiler import ScenarioCompileError, compile_scenario
from tatbot_sim.inkmap.contracts import ContractError, document_sha256
from tatbot_sim.inkmap.designs import (
    DesignArtifact,
    directory_artifacts,
    spiral_artifact,
)
from tatbot_sim.inkmap.gltf_surface import MODEL_ID
from tatbot_sim.inkmap.inklang_client import resolve_batch
from tatbot_sim.inkmap.rig import BODY_ASSET_ROOT, CATALOG_PATH, BodyRigError, load_body_rig
from tatbot_sim.inkmap.surface_trace import SurfaceTraceError
from tatbot_sim.inkmap.svg_strokes import SvgCompileError
from tatbot_sim.repo import repo_root
from tatbot_sim.site_sampling import INKLANG_VERSION, SiteChoice, site_choices
from tatbot_sim.tools import active_tool

SUITE_SCHEMA_VERSION = 1
DEFAULT_BODY = MODEL_ID
DEFAULT_POSES = (
    "supine",
    "prone",
    "reclined-seated",
    "reclined-left-arm-supported",
    "reclined-right-arm-supported",
)
DEFAULT_SITES = ("forearm", "bicep", "tricep", "thigh", "calf", "shin")
LATERALITY_CODE = {None: 0, "left": 1, "right": 2}
SIM_LOCATION_POLICY = "simulation-exposed-grid-v1"


class ScenarioSampleError(ValueError):
    pass


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
    design_source: str,
    count: int,
    rng: np.random.Generator,
    artwork_split: str = "train",
) -> tuple[DesignArtifact, ...]:
    if design_source == "spiral":
        return (spiral_artifact(),)
    if design_source == "directory":
        if generated_design_dir is None:
            raise ScenarioSampleError("--design-source directory requires --generated-design-dir")
        try:
            return directory_artifacts(generated_design_dir, generated_size_mm)
        except ValueError as exc:
            raise ScenarioSampleError(str(exc)) from exc
    if design_source == "artwork":
        from tatbot_sim.inkmap.collection import collection_artifacts
        return collection_artifacts(artwork_split)
    raise ScenarioSampleError(f"unknown design source {design_source!r}; choose artwork, directory, or spiral")


def compile_artwork_placement(placement, design, *, pose_id, seed, target, created_at, git_sha):
    return compile_scenario(placement, pose_id=pose_id, seed=seed, target_world_m=target,
                            tool_id=active_tool().tool_id, created_at=created_at, git_sha=git_sha)


def _site_choices(site_ids: tuple[str, ...]) -> tuple[SiteChoice, ...]:
    try:
        return site_choices(site_ids)
    except ValueError as exc:
        raise ScenarioSampleError(str(exc)) from exc


def _pose_supports_site(pose_id: str, site: SiteChoice) -> bool:
    """Keep the named posture and support useful for top-down tool access.

    Posterior leg surfaces can face into the bed/chair in supine and seated
    poses. Treating those pairings as valid made the compiler rotate the
    entire supported body to expose the selected face, which preserved joint
    angles but no longer preserved the authored pose relative to gravity.
    """
    if pose_id == "reclined-left-arm-supported":
        return site.id in ("forearm", "bicep") and site.laterality == "left"
    if pose_id == "reclined-right-arm-supported":
        return site.id in ("forearm", "bicep") and site.laterality == "right"
    if pose_id == "prone":
        return site.id in ("tricep", "thigh", "calf")
    if pose_id == "supine":
        return site.id in ("forearm", "bicep", "thigh", "shin")
    if pose_id == "reclined-seated":
        return site.id in ("thigh", "shin")
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


def _retry_sites(
    pose_id: str,
    primary: SiteChoice,
    site_values: tuple[SiteChoice, ...],
    count: int,
    rng: np.random.Generator,
    *,
    deprioritized: frozenset[SiteChoice] = frozenset(),
    priority_ids: frozenset[str] = frozenset(),
) -> tuple[SiteChoice, ...]:
    """Try uncovered and viable exposed sites before a rejected pairing."""
    compatible = [site for site in site_values if _pose_supports_site(pose_id, site)]

    def ordered(values: list[SiteChoice]) -> list[SiteChoice]:
        selected = [site for site in values if site == primary]
        others = [site for site in values if site != primary]
        rng.shuffle(others)
        return [*selected, *others]

    fresh = [site for site in compatible if site not in deprioritized]
    priority = [site for site in fresh if site.id in priority_ids]
    regular = [site for site in fresh if site.id not in priority_ids]
    rejected = [site for site in compatible if site in deprioritized]
    output = [*ordered(priority), *ordered(regular), *ordered(rejected)]
    while len(output) < count:
        cycle = [*ordered(priority), *ordered(regular), *ordered(rejected)]
        output.extend(cycle)
    return tuple(output[:count])


@lru_cache(maxsize=1)
def _atlas() -> dict:
    path = BODY_ASSET_ROOT / "bodies" / f"{MODEL_ID}.regions.json"
    if not path.is_file():
        raise ScenarioSampleError(f"missing region atlas for {MODEL_ID}: {path}")
    return json.loads(path.read_text())


def _resolved_anchor_candidates(
    pose_id: str,
    site: SiteChoice,
    *,
    description: str,
    seed: int,
) -> tuple[dict, ...]:
    """Return only TS-resolved anchors accepted by a named pose-exposure policy."""
    values = (0.18, 0.34, 0.50, 0.66, 0.82)
    coordinates = sorted(
        ((u, v) for u in values for v in values),
        key=lambda uv: ((uv[0] - 0.5) ** 2 + (uv[1] - 0.5) ** 2, uv),
    )
    requests = [
        {
            "site": site.as_inklang_site(region_uv=[u, v]),
            "description": description,
            "policy": "seeded-v1",
            "seed": seed,
        }
        for u, v in coordinates
    ]
    try:
        resolutions = resolve_batch(requests)
    except ValueError as exc:
        raise ScenarioSampleError(str(exc)) from exc
    rig = load_body_rig()
    candidates = np.asarray([resolution["anchor"]["face"] for resolution in resolutions])
    posed_vertices = rig.posed(pose_id).vertices[candidates].astype(np.float64)
    normals = np.cross(
        posed_vertices[:, 1] - posed_vertices[:, 0],
        posed_vertices[:, 2] - posed_vertices[:, 0],
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], 1e-12)
    normal_z = normals[:, 2]
    exposed = normal_z >= 0.5
    if not np.any(exposed):
        raise ScenarioSampleError(
            f"{MODEL_ID}/{pose_id}: {site.id}/{site.laterality or 'center'} "
            f"has no upward-exposed canonical anchors under {SIM_LOCATION_POLICY}",
        )
    selected = [resolution for resolution, keep in zip(resolutions, exposed, strict=True) if keep]
    selected.sort(key=lambda resolution: (
        (resolution["actual"]["region_uv"][0] - 0.5) ** 2
        + (resolution["actual"]["region_uv"][1] - 0.5) ** 2,
        resolution["anchor"]["face"],
    ))
    return tuple(selected)


@lru_cache(maxsize=1)
def _body_record() -> dict:
    catalog = json.loads(CATALOG_PATH.read_text())
    return {
        "model_spec_id": catalog["model_spec_id"],
        "model_spec_sha256": catalog["model_spec_sha256"],
        "identity_sha256": catalog["identity_sha256"],
        "topology_sha256": catalog["topology_sha256"],
        "rest_surface_sha256": catalog["rest_surface_sha256"],
        "asset_path": catalog["rest_asset"]["path"],
        "asset_sha256": catalog["rest_asset"]["sha256"],
    }


COVERAGE_REASON = "fill_coverage"


def _attempt_scale(attempt: int, previous: str | None) -> float:
    """How much of the site's dimension budget this attempt may use.

    Shrinking recovers a placement that fell off the site, which is what the
    retry was written for. It is exactly wrong for a fill-coverage refusal: the
    planner's stroke rows cover a *larger* painted region better, so a coverage
    failure that keeps shrinking walks away from the threshold it needs. On a
    generated daisy chain the four attempts measured 0.92, 0.89, 0.82, 0.61
    against a 0.98 gate — four attempts spent moving in the wrong direction.
    """
    if previous == COVERAGE_REASON:
        return min(1.0, 0.82 + 0.06 * attempt)
    return max(0.48, 0.82 - 0.07 * attempt)


def _placement_file(
    pose_id: str,
    site: SiteChoice,
    design: DesignArtifact,
    *,
    sample_index: int,
    attempt: int,
    rng: np.random.Generator,
    previous_reason: str | None = None,
) -> dict:
    resolutions = _resolved_anchor_candidates(
        pose_id, site,
        description=f"sampled {site.id} placement",
        seed=sample_index,
    )
    resolution = resolutions[min(attempt, len(resolutions) - 1)]
    max_dimension_mm = {
        "forearm": 48.0,
        "bicep": 52.0,
        "tricep": 52.0,
        "thigh": 62.0,
        "calf": 52.0,
        "shin": 48.0,
    }.get(site.id, 42.0)
    # Deterministic, bounded, and pointed at whatever the last attempt actually
    # refused: shrink away from a site boundary, grow into a coverage refusal.
    max_dimension_mm *= _attempt_scale(attempt, previous_reason)
    scale = min(1.0, max_dimension_mm / max(design.size_mm))
    if "size_range_mm" in design.source:
        low, high = design.source["size_range_mm"]
        scale = max(low / max(design.size_mm), min(high / max(design.size_mm), scale))
    size = [round(value * scale, 6) for value in design.size_mm]
    placement_id = f"sample-{sample_index:04d}-{design.id}-{site.id}"
    motif = design.name.lower()
    article = "an" if motif[:1] in "aeiou" else "a"
    sentence = f"{article} {motif} {resolution['intent']['canonical_phrase']}"
    program = {
        "inklang": INKLANG_VERSION,
        "motif": motif,
        "style": None,
        "secondary": [],
        "technique": None,
        "color": None,
        "site": resolution["intent"]["site"],
    }
    placement = {
        "id": placement_id,
        "design_id": design.id,
        "anchor": resolution["anchor"],
        "rotation_rad": float(rng.uniform(-math.pi, math.pi)),
        "size_mm": size,
        "mirror": bool(rng.integers(2)),
        "site": {
            "id": site.id,
            "laterality": site.laterality,
            "aspect": None,
            "level": None,
            "uv": resolution["actual"]["region_uv"],
            "lexicon": INKLANG_VERSION,
        },
        "language": {
            "sentence": sentence,
            "program": program,
            "intent": resolution["intent"],
            "resolution": resolution,
            "consumer_policy": SIM_LOCATION_POLICY,
        },
    }
    document = {
        "schema_version": 6,
        "units": {"length": "m", "tattoo_size": "mm", "up": "+z"},
        "body": dict(_body_record()),
        "placements": [placement],
    }
    document["designs"] = {design.id: design.embedded(tuple(size))}
    return document


def _trace_stays_on_site(scenario: dict, site: SiteChoice) -> bool:
    atlas = _atlas()
    site_index = atlas["sites"].index(site.id)
    code = site_index * 4 + LATERALITY_CODE[site.laterality]
    return all(
        atlas["faces"][anchor["face"]] == code
        for stroke in scenario["trace"]["strokes"]
        for anchor in stroke
    )


MAX_SLOT_DESIGNS = 4


def _slot_designs(drawn, designs, exhausted: set[str]):
    """The artwork this slot may try: the drawn one first, then alternatives.

    A suite used to end the moment one artwork spent its retries, so a single
    piece the fill planner cannot cover took the rest of the run with it. The
    drawn design still goes first — the balanced draw is what keeps a suite's
    artwork distribution honest — and alternatives are only reached when it
    fails, in the draw's own order, skipping artwork already known unusable.
    """
    if drawn.id in exhausted:
        ordered = [item for item in designs if item.id not in exhausted]
    else:
        ordered = [drawn] + [item for item in designs
                             if item.id != drawn.id and item.id not in exhausted]
    return tuple(ordered[:MAX_SLOT_DESIGNS])


def _reason(exc: Exception) -> str:
    # The fill planner's coverage gate is the one refusal a *larger* placement
    # recovers from, so it is named rather than lumped into invalid_input.
    if "loses paint component coverage" in str(exc):
        return COVERAGE_REASON
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
    poses: tuple[str, ...] = DEFAULT_POSES,
    sites: tuple[str, ...] = DEFAULT_SITES,
    generated_design_dir: Path | None = None,
    generated_size_mm: tuple[float, float] = (50.0, 50.0),
    design_source: str = "artwork",
    artwork_split: str = "train",
    audit_reach: bool = False,
    max_attempts_per_scenario: int = 4,
    max_seconds: float | None = None,
    created_at: str | None = None,
    git_sha: str | None = None,
) -> dict:
    """Write a self-contained scenario suite and an explicit attempt ledger."""
    output_dir = Path(output_dir)
    if count <= 0:
        raise ScenarioSampleError("count must be positive")
    if max_attempts_per_scenario <= 0:
        raise ScenarioSampleError("max_attempts_per_scenario must be positive")
    minimum = 2 * len(poses)
    if len(set(sites)) > 1 and len(set(sites)) <= count < minimum:
        # A suite as large as the site list is asked to cover every site, and
        # coverage is repaired by swapping a slot whose pose already holds a
        # duplicated site -- so every pose needs at least two slots. Below
        # that the swap has nothing to give up and the suite used to die
        # mid-run on "cannot cover requested tattoo site" (2026-09-03).
        # Smaller suites make no coverage promise and stay allowed.
        raise ScenarioSampleError(
            f"a suite of {count} must cover all {len(set(sites))} requested sites, which "
            f"needs at least {minimum} scenarios (two per pose). Request {minimum} or more, "
            f"fewer than {len(set(sites))}, or narrow --poses/--sites",
        )
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
    designs = _designs(generated_design_dir, generated_size_mm, design_source, count, rng, artwork_split)
    site_values = _site_choices(sites)
    pose_draws = _balanced(poses, count, rng)
    site_draws = _sites_for_poses(pose_draws, site_values, rng)
    design_draws = _balanced(designs, count, rng)
    attempts: list[dict] = []
    accepted: list[dict] = []
    accepted_site_ids: set[str] = set()
    failed_pairings: set[tuple[str, SiteChoice]] = set()
    known_errors = (ScenarioCompileError, SurfaceTraceError, SvgCompileError, BodyRigError, ContractError, ValueError)
    exhausted_designs: set[str] = set()
    # A suite is a search, and a search needs a wall clock. Raising the episode
    # horizon made every reach audit proportionally longer, and a run that
    # cannot finish is worse than a short one that says so: the budget stops
    # asking for more work and finishes with the honest partial report the
    # incomplete path already produces.
    started = time.monotonic()
    timed_out = False

    def over_budget() -> bool:
        return max_seconds is not None and time.monotonic() - started > max_seconds

    for sample_index in range(count):
        if over_budget():
            timed_out = True
            break
        body_id = MODEL_ID
        pose_id = pose_draws[sample_index]
        # A design that has already spent every retry cannot fill this slot
        # either. Offer the slot the drawn design first, then bounded
        # alternatives, so one unusable artwork costs one slot's worth of
        # attempts rather than the rest of the suite.
        design_choices = _slot_designs(design_draws[sample_index], designs, exhausted_designs)
        for design in design_choices:
            deprioritized = frozenset(
                site for site in site_values
                if (pose_id, site) in failed_pairings
            )
            retry_sites = _retry_sites(
                pose_id, site_draws[sample_index], site_values,
                max_attempts_per_scenario, rng,
                deprioritized=deprioritized,
                priority_ids=frozenset(set(sites) - accepted_site_ids),
            )
            previous_reason: str | None = None
            for retry in range(max_attempts_per_scenario):
                if over_budget():
                    timed_out = True
                    break
                site = retry_sites[retry]
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
                    "artwork_sha256": design.sha256,
                    "artwork_family": design.source.get("family"),
                    "artwork_split": design.source.get("split"),
                }
                try:
                    placement = _placement_file(
                        pose_id, site, design,
                        sample_index=sample_index, attempt=retry, rng=attempt_rng,
                        previous_reason=previous_reason,
                    )
                    target = [
                        float(attempt_rng.uniform(0.30, 0.32)),
                        float(attempt_rng.uniform(-0.035, 0.045)),
                        0.04,
                    ]
                    if design_source == "spiral":
                        scenario = compile_scenario(
                            placement, pose_id=pose_id, seed=attempt_seed, target_world_m=target,
                            tool_id=active_tool().tool_id, created_at=created_at, git_sha=git_sha,
                            generator="tatbot sim sample calibration",
                        )
                    else:
                        scenario = compile_artwork_placement(placement, design, pose_id=pose_id,
                            seed=attempt_seed, target=target, created_at=created_at, git_sha=git_sha)
                    if not _trace_stays_on_site(scenario, site):
                        previous_reason = "site_boundary"
                        attempts.append({**base, "accepted": False, "reason": "site_boundary"})
                        continue
                    reach_record = None
                    if audit_reach:
                        from tatbot_sim.inkmap.reach import ReachAuditError, optimize_body_placement

                        # One candidate may spend at most a quarter of whatever
                        # budget is left, so a single slow audit — and at a
                        # 9,000-step horizon one audit can outlast the whole
                        # budget — cannot eat the run before the between-
                        # candidate check is ever reached.
                        share = None
                        if max_seconds is not None:
                            share = max(30.0, (max_seconds - (time.monotonic() - started)) / 4)
                        try:
                            selection = optimize_body_placement(scenario, trajectory_seed=attempt_seed,
                                                                audit_deadline_s=share)
                        except ReachAuditError as exc:
                            failed_pairings.add((pose_id, site))
                            attempts.append({
                                **base, "accepted": False, "reason": exc.reason,
                                "detail": str(exc)[:240],
                            })
                            continue
                        scenario = selection.scenario
                        reach_record = {
                            "patch_yaw_rad": selection.patch_yaw_rad,
                            "probe_max_residual_m": selection.probe_max_residual_m,
                            "selected_candidate": selection.audit["selected"],
                            "candidate_count": len(selection.candidates),
                            "audit": selection.audit,
                        }
                except known_errors as exc:
                    previous_reason = _reason(exc)
                    attempts.append({**base, "accepted": False, "reason": previous_reason, "detail": str(exc)[:240]})
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
                accepted_site_ids.add(site.id)
                break
            else:
                if timed_out:
                    break
                # Every site this design was offered refused it. Remember that,
                # and let the slot try the next artwork.
                exhausted_designs.add(design.id)
                continue
            break
        if timed_out:
            attempts.append({"sample_index": sample_index, "accepted": False,
                             "reason": "time_budget",
                             "detail": f"suite budget of {max_seconds:.0f} s reached"})
            break
    rejected = len(attempts) - len(accepted)
    coverage = {
        "bodies": sorted({item["body"] for item in accepted}),
        "poses": sorted({item["pose"] for item in accepted}),
        "sites": sorted({item["site"] for item in accepted}),
        "designs": sorted({item["design"] for item in accepted}),
    }

    def covered(requested: tuple[str, ...], actual: list[str]) -> bool:
        required = set(requested)
        return count < len(required) or required <= set(actual)

    complete = (
        len(accepted) == count
        and coverage["bodies"] == [MODEL_ID]
        and covered(poses, coverage["poses"])
        and covered(sites, coverage["sites"])
    )
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
        "complete": complete,
        "max_attempts_per_scenario": max_attempts_per_scenario,
        "max_seconds": max_seconds,
        "elapsed_s": round(time.monotonic() - started, 1),
        "timed_out": timed_out,
        "reach_audited": audit_reach,
        "design_source": design_source,
        "artwork_split": artwork_split if design_source == "artwork" else None,
        "coverage": coverage,
        "deprioritized_pairings": [
            {
                "body": MODEL_ID,
                "pose": pose_id,
                "site": site.id,
                "laterality": site.laterality,
            }
            for pose_id, site in sorted(
                failed_pairings,
                key=lambda item: (item[0], item[1].id, item[1].laterality or ""),
            )
        ],
        "scenarios": accepted,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "attempts.jsonl").write_bytes(b"".join(_json_bytes(item) + b"\n" for item in attempts))
    if not manifest["complete"]:
        missing = {
            key: sorted(set(requested) - set(coverage[key]))
            for key, requested in (("poses", poses), ("sites", sites))
            if count >= len(set(requested)) and set(requested) - set(coverage[key])
        }
        coverage_detail = f"; missing coverage {missing}" if missing else ""
        budget_detail = (f" after the {max_seconds:.0f} s budget" if timed_out else "")
        raise ScenarioSampleError(
            f"suite stopped at {len(accepted)}/{count}{budget_detail}{coverage_detail}; "
            f"inspect {output_dir / 'attempts.jsonl'}",
        )
    return manifest
