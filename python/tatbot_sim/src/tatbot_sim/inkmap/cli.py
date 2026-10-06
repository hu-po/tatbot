"""Command line entry points for placement-to-scenario compilation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.inkgen_materialize import materialize_inkgen_designs
from tatbot_sim.inkmap.resolver import (
    DEFAULT_CREATED_AT,
    ScenarioResolveError,
    build_scenario_request,
    resolve_scenario_request,
)
from tatbot_sim.inkmap.sampler import DEFAULT_POSES, DEFAULT_SITES, materialize_scenario_suite
from tatbot_sim.repo import repo_root


class _CompileOverride(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const if self.nargs == 0 else values)
        namespace.bundle_overrides = True


def _compile(args: argparse.Namespace) -> int:
    from tatbot_sim.human_rep.contracts import parse_json
    from tatbot_sim.inkmap.bundle import MAX_BUNDLE_BYTES
    from tatbot_sim.inkmap.program_scenario import compile_simulation_bundle

    with args.placement.open("rb") as stream:
        data = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("simulation input exceeds 20 MB")
    placement = parse_json(data)
    _require_acquired_input(placement)
    if isinstance(placement, dict) and placement.get("schema") == "tatbot.inkmap-sim-bundle/1":
        # The immutable request owns these choices; no silent CLI-default
        # override. Explicit bundle overrides require a newly authored bundle.
        if getattr(args, "bundle_overrides", False):
            raise ValueError("bundle request is immutable; remove pose/tool/seed/world overrides or author a new bundle")
        scenario = compile_simulation_bundle(placement, placement_id=args.placement_id,
                                             created_at=args.created_at, git_sha=args.git_sha,
                                             operating_budget_s=args.operating_budget_s)
    else:
        scenario = compile_scenario(
            placement,
            placement_id=args.placement_id,
            pose_id=args.pose,
            seed=args.seed,
            target_world_m=args.target_world_m,
            tool_id=args.tool,
            support_id=args.support,
            align_patch_up=not args.preserve_pose_world,
            patch_yaw_rad=args.patch_yaw_rad,
            created_at=args.created_at,
            git_sha=args.git_sha,
            operating_budget_s=args.operating_budget_s,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} trace={scenario['trace']['sha256']}")
    return 0


def _require_acquired_input(document):
    from tatbot_contracts.artwork import require_acquired_artwork
    if not isinstance(document, dict):
        raise ValueError("Generate with DrawingBot V3 and import a current artwork document")
    records = document.get("artworks", {}) if document.get("schema") == "tatbot.inkmap-sim-bundle/1" else document.get("designs", {})
    for record in records.values():
        require_acquired_artwork(record)


def _sample(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("scenario suites are generated data; --output-dir must be outside the repository")
    manifest = materialize_scenario_suite(
        output,
        count=args.count,
        seed=args.seed,
        poses=tuple(args.poses),
        sites=tuple(args.sites),
        generated_design_dir=args.generated_design_dir,
        generated_size_mm=tuple(args.generated_size_mm),
        design_source=args.design_source,
        artwork_split=args.artwork_split,
        audit_reach=not args.no_reach_audit,
        max_attempts_per_scenario=args.max_attempts_per_scenario,
        max_seconds=args.max_seconds or None,
        created_at=args.created_at,
        git_sha=args.git_sha,
    )
    print(
        f"wrote {manifest['accepted']} scenarios to {output} "
        f"rejection_rate={manifest['rejection_rate']:.1%}",
    )
    return 0


def _materialize(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("generated designs are artifacts; --output-dir must be outside the repository")
    subjects = list(args.subject)
    if args.subjects_file:
        subjects.extend(args.subjects_file.read_text().splitlines())
    manifest = materialize_inkgen_designs(
        output,
        subjects,
        count=args.count,
        seed=args.seed,
        api_url=args.api_url,
        style=args.style,
        size_mm=tuple(args.size_mm),
        timeout_s=args.timeout_s,
        autostart=not args.no_autostart,
    )
    print(f"wrote {len(manifest['artifacts'])} generated SVG artifacts to {output}")
    return 0


def _batch(args: argparse.Namespace) -> int:
    """Run or resume one artwork generation job; report every stage honestly."""
    from tatbot_sim.inkmap.inkgen_client import (
        InkgenBackendError,
        InkgenUnreachableError,
        ensure_backend,
        resolve_backend,
    )
    from tatbot_sim.inkmap.inkgen_materialize import (
        MATERIALIZATION_SCHEMA,
        materialize_job,
    )

    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("generated artwork is data; --output-dir must be outside the repository")
    subjects = list(args.subject)
    if args.subjects_file:
        subjects.extend(args.subjects_file.read_text().splitlines())
    try:
        # Bulk work names its generator or refuses. There is no fallback here,
        # and in particular no fallback to the public Space. Naming it contacts
        # nothing.
        backend = resolve_backend(args.api_url, require_configured=True)
        if backend.kind == "space":
            raise InkgenBackendError(
                "a batch will not run against the public Space; configure the `inkgen` role "
                "or give --api-url for a generator you own")
        # A job that already exists carries the weights it was frozen against, so
        # resuming — or retracing cached rasters at a new size — needs no probe.
        # A new job does: its identity binds the revision it will actually use.
        resuming = (output / ".job/request.json").is_file()
        health: dict = {}
        if not resuming:
            backend, health = ensure_backend(backend, autostart=not args.no_autostart)
        revision = args.model_revision or health.get("model_revision")
        if args.require_pinned_model and not revision and not resuming:
            raise InkgenBackendError(
                "the generator did not report a model revision and --model-revision was not given; "
                "a reproducible batch must record the exact weights it used")

        def ensure() -> str:
            live, _ = ensure_backend(backend, autostart=not args.no_autostart)
            return live.url

        result = materialize_job(
            output, subjects, count=args.count, seed=args.seed, api_url=backend.url,
            style=args.style, size_mm=tuple(args.size_mm), timeout_s=args.timeout_s,
            model=args.model or health.get("model"), model_revision=revision,
            replacement_budget=args.replacement_budget, max_attempts=args.max_attempts,
            backend=backend.kind, ensure=ensure)
    except (InkgenBackendError, InkgenUnreachableError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 5
    complete = result.get("schema") == MATERIALIZATION_SCHEMA
    report = result.get("report", result)
    print(json.dumps({"ok": complete, "output_dir": str(output), "complete": complete,
                      "counts": {key: report.get(key) for key in
                                 ("requested", "candidates", "accepted", "refused", "failed",
                                  "duplicate", "pending", "attempts", "generation_seconds")},
                      "manifest": str(output / ("manifest.json" if complete else "selection.json"))},
                     indent=2, sort_keys=True))
    return 0 if complete else 1


def _batch_status(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.inkgen_materialize import batch_module

    batch = batch_module()
    root = args.output_dir.expanduser().resolve()
    request = batch.load_job(root)
    with batch.Job.open(root, request) as job:
        print(json.dumps({"job_id": request["job_id"], "root": str(root),
                          "settings": request["settings"], "report": job.report(),
                          "complete": job.is_complete(),
                          "pending_keys": job.pending_keys()[:20]}, indent=2, sort_keys=True))
    return 0


def _recipes(args: argparse.Namespace) -> int:
    """Expand a frozen artwork library into reproducible recipes. Offline."""
    from tatbot_sim.inkmap.recipes import (
        RecipeAxes,
        RecipeError,
        build_plan,
        materialize_recipes,
        split_audit,
    )
    from tatbot_sim.inkmap.sampler import DEFAULT_POSES, DEFAULT_SITES

    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("recipes are generated data; --output-dir must be outside the repository")
    if args.artwork_dir is not None:
        from tatbot_sim.inkmap.designs import directory_artifacts
        designs = directory_artifacts(args.artwork_dir.expanduser().resolve(),
                                      tuple(args.generated_size_mm))
    else:
        from tatbot_sim.inkmap.collection import collection_artifacts
        designs = collection_artifacts(args.artwork_split)
    axes = RecipeAxes()
    if args.scale is not None or args.rotation_deg is not None:
        from tatbot_sim.inkmap.perception_variation import Range
        axes = RecipeAxes(
            scale=Range(*args.scale) if args.scale else axes.scale,
            rotation_deg=Range(*args.rotation_deg) if args.rotation_deg else axes.rotation_deg,
            mirror_probability=args.mirror_probability)
    try:
        plan = build_plan(designs, count=args.count, seed=args.seed,
                          poses=tuple(args.poses or DEFAULT_POSES),
                          sites=tuple(args.sites or DEFAULT_SITES),
                          axes=axes, holdout_basis_points=args.holdout_basis_points,
                          label=args.label or "")
        ledger = materialize_recipes(output, plan)
    except RecipeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 3
    leaks = split_audit(output)
    print(json.dumps({"ok": not leaks and ledger["complete"], **ledger, "split_leaks": leaks},
                     indent=2, sort_keys=True))
    return 0 if not leaks and ledger["complete"] else 1


def _recipes_status(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.recipes import load_plan, read_ledger, split_audit

    root = args.output_dir.expanduser().resolve()
    plan = load_plan(root)
    print(json.dumps({"plan_id": plan["plan_id"], "seed": plan["seed"],
                      "artworks": len(plan["artworks"]), "axes": plan["axes"],
                      **read_ledger(root), "split_leaks": split_audit(root)},
                     indent=2, sort_keys=True))
    return 0


def _resolve(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("resolved scenarios are artifacts; --output-dir must be outside the repository")
    try:
        request = build_scenario_request(
            args.prompt,
            size_mm=tuple(args.size_mm),
            pose=args.pose,
            support=args.support,
            seed=args.seed,
            design_id=args.design_id,
        )
        manifest = resolve_scenario_request(
            request,
            output,
            generated_design_dir=args.generated_design_dir,
            created_at=args.created_at,
            git_sha=args.git_sha,
            max_attempts=args.max_attempts,
        )
    except ScenarioResolveError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        f"wrote {output / manifest['scenario']} "
        f"body={manifest['body']} pose={manifest['pose']} design={manifest['design']['id']}",
    )
    return 0


def _perception(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.perception_dataset import Args, render_dataset

    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("perception datasets are generated data; --output-dir must be outside the repository")
    report = render_dataset(
        Args(
            scenario=args.scenario,
            output_dir=output,
            seed=args.seed,
            views=args.views,
            width=args.width,
            height=args.height,
            focal_px=args.focal_px,
        )
    )
    print(
        f"wrote {report['accepted']} audited perception frames to {output} "
        f"at {report['performance']['frames_per_s']:.3f} frames/s"
    )
    return 0


def _pilot_plan(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.pilot import build_pilot_plan

    plan = build_pilot_plan(args.output_dir, seed=args.seed)
    print(
        f"wrote {plan['counts']['reference_scene_templates']} reference scene templates, "
        f"{plan['counts']['drawing_episodes']} gated drawing episodes to {args.output_dir}"
    )
    return 0


def _pilot_audit(args: argparse.Namespace) -> int:
    from tatbot_sim.inkmap.pilot import audit_pilot_file

    report = audit_pilot_file(args.plan.expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tatbot_sim.inkmap.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="materialize one Inkmap placement as a posed scenario")
    compile_parser.add_argument("placement", type=Path)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--placement-id")
    compile_parser.add_argument("--pose", default="supine", action=_CompileOverride)
    compile_parser.add_argument("--seed", type=int, default=0, action=_CompileOverride)
    compile_parser.add_argument("--target-world-m", nargs=3, type=float, default=[0.29, 0.0, 0.04], metavar=("X", "Y", "Z"), action=_CompileOverride)
    compile_parser.add_argument("--tool", default="lutin-3rl-bugpin", action=_CompileOverride)
    compile_parser.add_argument("--support", action=_CompileOverride)
    compile_parser.add_argument(
        "--patch-yaw-rad", type=float, default=3.141592653589793, action=_CompileOverride,
        help="robot-world yaw of the patch +u axis after normal alignment",
    )
    compile_parser.add_argument(
        "--preserve-pose-world", action=_CompileOverride, nargs=0, const=True, default=False,
        help="translate only; default rotates the selected patch normal to robot +Z",
    )
    compile_parser.add_argument("--operating-budget-s", type=float,
                                help="legacy cartridge budget metadata retained in the ink program")
    compile_parser.add_argument("--created-at", help="fixed ISO timestamp for reproducible fixtures")
    compile_parser.add_argument("--git-sha", help="fixed source revision for reproducible fixtures")
    compile_parser.set_defaults(func=_compile, bundle_overrides=False)
    sample_parser = sub.add_parser("sample", help="materialize a bounded procedural scenario suite")
    sample_parser.add_argument("--output-dir", type=Path, required=True)
    sample_parser.add_argument("--count", type=int, default=64)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--poses", nargs="+", default=list(DEFAULT_POSES))
    sample_parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    sample_parser.add_argument("--generated-design-dir", type=Path)
    sample_parser.add_argument("--artwork-split", choices=("train", "validation", "test", "all"), default="train")
    sample_parser.add_argument(
        "--design-source", choices=("artwork", "directory", "spiral"), default="artwork",
        help="reviewed artwork (default), materialized artwork directory, or calibration spiral",
    )
    sample_parser.add_argument(
        "--no-reach-audit", action="store_true",
        help="skip CPU IK yaw selection (final generation still enforces exact FK)",
    )
    sample_parser.add_argument("--generated-size-mm", nargs=2, type=float, default=[50.0, 50.0], metavar=("W", "H"))
    sample_parser.add_argument("--max-attempts-per-scenario", type=int, default=4)
    sample_parser.add_argument("--max-seconds", type=float, default=1800.0,
                               help="wall-clock budget; the suite stops and reports honestly (0 for none)")
    sample_parser.add_argument("--created-at", help="fixed ISO timestamp for reproducible suites")
    sample_parser.add_argument("--git-sha", help="fixed source revision for reproducible suites")
    sample_parser.set_defaults(func=_sample)
    materialize_parser = sub.add_parser(
        "materialize-designs",
        help="ask Inkgen for rasters and trace immutable SVGs before simulation",
    )
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument(
        "--subject", action="append", default=[], help="design subject; repeat for a prompt pool"
    )
    materialize_parser.add_argument(
        "--subjects-file", type=Path, help="newline-delimited subject pool"
    )
    materialize_parser.add_argument("--count", type=int, default=1)
    materialize_parser.add_argument("--seed", type=int, default=0)
    materialize_parser.add_argument("--api-url", default="http://127.0.0.1:8600")
    materialize_parser.add_argument("--style")
    materialize_parser.add_argument(
        "--size-mm", nargs=2, type=float, default=[50.0, 50.0], metavar=("W", "H")
    )
    materialize_parser.add_argument("--timeout-s", type=float, default=120.0)
    materialize_parser.add_argument("--no-autostart", action="store_true",
                                    help="refuse instead of starting a stopped fleet generator")
    materialize_parser.set_defaults(func=_materialize)
    batch_parser = sub.add_parser(
        "batch", help="run or resume a resumable artwork generation job",
    )
    batch_parser.add_argument("--output-dir", type=Path, required=True)
    batch_parser.add_argument("--subject", action="append", default=[],
                              help="design subject; repeat for a prompt pool")
    batch_parser.add_argument("--subjects-file", type=Path, help="newline-delimited subject pool")
    batch_parser.add_argument("--count", type=int, required=True, help="artwork slots to fill")
    batch_parser.add_argument("--seed", type=int, default=0)
    batch_parser.add_argument("--api-url", help="generator base URL; default: the fleet generator")
    batch_parser.add_argument("--style")
    batch_parser.add_argument("--size-mm", nargs=2, type=float, default=[50.0, 50.0],
                              metavar=("W", "H"))
    batch_parser.add_argument("--replacement-budget", type=int, default=0,
                              help="extra candidates a refused or duplicate slot may spend")
    batch_parser.add_argument("--max-attempts", type=int, default=3)
    batch_parser.add_argument("--model")
    batch_parser.add_argument("--model-revision", help="exact commit of the weights to record")
    batch_parser.add_argument("--require-pinned-model", action="store_true",
                              help="refuse unless the exact model revision is known")
    batch_parser.add_argument("--timeout-s", type=float, default=300.0)
    batch_parser.add_argument("--no-autostart", action="store_true")
    batch_parser.set_defaults(func=_batch)
    batch_status_parser = sub.add_parser(
        "batch-status", help="read an existing generation job's ledger without any network",
    )
    batch_status_parser.add_argument("--output-dir", type=Path, required=True)
    batch_status_parser.set_defaults(func=_batch_status)
    recipes_parser = sub.add_parser(
        "recipes", help="expand a frozen artwork library into reproducible scenario recipes",
    )
    recipes_parser.add_argument("--output-dir", type=Path, required=True)
    recipes_parser.add_argument("--count", type=int, required=True, help="recipes to plan")
    recipes_parser.add_argument("--seed", type=int, default=0)
    recipes_parser.add_argument("--artwork-dir", type=Path,
                                help="a completed generation job; default: the reviewed collection")
    recipes_parser.add_argument("--artwork-split", choices=("train", "validation", "test", "all"),
                                default="all")
    recipes_parser.add_argument("--generated-size-mm", nargs=2, type=float, default=[50.0, 50.0],
                                metavar=("W", "H"))
    recipes_parser.add_argument("--poses", nargs="+")
    recipes_parser.add_argument("--sites", nargs="+")
    recipes_parser.add_argument("--scale", nargs=2, type=float, metavar=("LOW", "HIGH"))
    recipes_parser.add_argument("--rotation-deg", nargs=2, type=float, metavar=("LOW", "HIGH"))
    recipes_parser.add_argument("--mirror-probability", type=float, default=0.25)
    recipes_parser.add_argument("--holdout-basis-points", type=int, default=1000)
    recipes_parser.add_argument("--label", help="free text recorded in the frozen plan")
    recipes_parser.set_defaults(func=_recipes)
    recipes_status_parser = sub.add_parser(
        "recipes-status", help="read a recipe ledger and audit its splits without any network",
    )
    recipes_status_parser.add_argument("--output-dir", type=Path, required=True)
    recipes_status_parser.set_defaults(func=_recipes_status)
    resolve_parser = sub.add_parser(
        "resolve", help="resolve one typed InkLang request into a replayable posed scenario",
    )
    resolve_parser.add_argument("prompt", help='InkLang sentence, e.g. "a heron on the left forearm"')
    resolve_parser.add_argument("--output-dir", type=Path, required=True)
    resolve_parser.add_argument("--generated-design-dir", type=Path)
    resolve_parser.add_argument(
        "--design-id",
        help="exact materialized id, or explicit spiral-v1 for the sole fixed regression design",
    )
    resolve_parser.add_argument("--size-mm", nargs=2, type=float, required=True, metavar=("W", "H"))
    resolve_parser.add_argument("--pose", default="compatible")
    resolve_parser.add_argument(
        "--support", default="compatible",
        help="compatible, bed, chair, armrest, or an exact support id",
    )
    resolve_parser.add_argument("--seed", type=int, required=True)
    resolve_parser.add_argument("--max-attempts", type=int, default=4)
    resolve_parser.add_argument(
        "--created-at", default=DEFAULT_CREATED_AT,
        help="scenario provenance timestamp; deterministic epoch by default",
    )
    resolve_parser.add_argument("--git-sha")
    resolve_parser.set_defaults(func=_resolve)
    perception_parser = sub.add_parser(
        "perception",
        help="render strict privileged perception labels from one typed v3 scenario",
    )
    perception_parser.add_argument("scenario", type=Path)
    perception_parser.add_argument("--output-dir", type=Path, required=True)
    perception_parser.add_argument("--seed", type=int, default=0)
    perception_parser.add_argument("--views", type=int, default=3)
    perception_parser.add_argument("--width", type=int, default=256)
    perception_parser.add_argument("--height", type=int, default=256)
    perception_parser.add_argument("--focal-px", type=float, default=300.0)
    perception_parser.set_defaults(func=_perception)
    pilot_parser = sub.add_parser(
        "pilot-plan",
        help="write and audit the full Inkmap pilot ledger without launching renderers",
    )
    pilot_parser.add_argument("--output-dir", type=Path, required=True)
    pilot_parser.add_argument("--seed", type=int, default=0)
    pilot_parser.set_defaults(func=_pilot_plan)
    pilot_audit_parser = sub.add_parser(
        "pilot-audit",
        help="audit an existing Inkmap pilot ledger without launching renderers",
    )
    pilot_audit_parser.add_argument("plan", type=Path)
    pilot_audit_parser.set_defaults(func=_pilot_audit)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
