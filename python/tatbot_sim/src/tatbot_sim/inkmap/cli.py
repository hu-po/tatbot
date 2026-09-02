"""Command line entry points for placement-to-scenario compilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tatbot_sim.inkmap.compiler import compile_scenario
from tatbot_sim.inkmap.sampler import DEFAULT_BODIES, DEFAULT_POSES, DEFAULT_SITES, materialize_scenario_suite
from tatbot_sim.repo import repo_root


def _compile(args: argparse.Namespace) -> int:
    placement = json.loads(args.placement.read_text())
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} trace={scenario['trace']['sha256']}")
    return 0


def _sample(args: argparse.Namespace) -> int:
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(repo_root().resolve()):
        raise SystemExit("scenario suites are generated data; --output-dir must be outside the repository")
    manifest = materialize_scenario_suite(
        output,
        count=args.count,
        seed=args.seed,
        bodies=tuple(args.bodies),
        poses=tuple(args.poses),
        sites=tuple(args.sites),
        generated_design_dir=args.generated_design_dir,
        generated_size_mm=tuple(args.generated_size_mm),
        include_builtin_designs=not args.generated_only,
        audit_reach=not args.no_reach_audit,
        max_attempts_per_scenario=args.max_attempts_per_scenario,
        created_at=args.created_at,
        git_sha=args.git_sha,
    )
    print(
        f"wrote {manifest['accepted']} scenarios to {output} "
        f"rejection_rate={manifest['rejection_rate']:.1%}",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tatbot_sim.inkmap.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile", help="materialize one Inkmap placement as a posed scenario")
    compile_parser.add_argument("placement", type=Path)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--placement-id")
    compile_parser.add_argument("--pose", default="supine")
    compile_parser.add_argument("--seed", type=int, default=0)
    compile_parser.add_argument("--target-world-m", nargs=3, type=float, default=[0.29, 0.0, 0.04], metavar=("X", "Y", "Z"))
    compile_parser.add_argument("--tool", default="lutin-3rl-bugpin")
    compile_parser.add_argument("--support")
    compile_parser.add_argument(
        "--patch-yaw-rad", type=float, default=3.141592653589793,
        help="robot-world yaw of the patch +u axis after normal alignment",
    )
    compile_parser.add_argument(
        "--preserve-pose-world", action="store_true",
        help="translate only; default rotates the selected patch normal to robot +Z",
    )
    compile_parser.add_argument("--created-at", help="fixed ISO timestamp for reproducible fixtures")
    compile_parser.add_argument("--git-sha", help="fixed source revision for reproducible fixtures")
    compile_parser.set_defaults(func=_compile)
    sample_parser = sub.add_parser("sample", help="materialize a bounded procedural scenario suite")
    sample_parser.add_argument("--output-dir", type=Path, required=True)
    sample_parser.add_argument("--count", type=int, default=64)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--bodies", nargs="+", default=list(DEFAULT_BODIES))
    sample_parser.add_argument("--poses", nargs="+", default=list(DEFAULT_POSES))
    sample_parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    sample_parser.add_argument("--generated-design-dir", type=Path)
    sample_parser.add_argument("--generated-only", action="store_true", help="exclude checked-in designs")
    sample_parser.add_argument(
        "--no-reach-audit", action="store_true",
        help="skip CPU IK yaw selection (final generation still enforces exact FK)",
    )
    sample_parser.add_argument("--generated-size-mm", nargs=2, type=float, default=[50.0, 50.0], metavar=("W", "H"))
    sample_parser.add_argument("--max-attempts-per-scenario", type=int, default=4)
    sample_parser.add_argument("--created-at", help="fixed ISO timestamp for reproducible suites")
    sample_parser.add_argument("--git-sha", help="fixed source revision for reproducible suites")
    sample_parser.set_defaults(func=_sample)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
