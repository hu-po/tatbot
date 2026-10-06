"""Generate one named distribution.

    python -m tatbot_sim.factory --list
    python -m tatbot_sim.factory paper-draw --out-dir ~/tatbot-sim/datasets/paper-v5
    python -m tatbot_sim.factory skin-erase --out-dir ~/ds/laser-v2 --num-episodes 128

The recipe comes from tatbot_sim.distributions; every leaf of it stays
overridable on the command line, so this is a starting point rather than a
cage. What the launcher adds over calling ``generate`` directly is that the
tool, the substrate, the task and the episode length can no longer be
combined by hand into something that runs happily and means nothing.

Construction inputs are resolved explicitly before creating a world.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import sys

from tatbot_sim import tasks
from tatbot_sim.distributions import DISTRIBUTIONS


def select_tool(dist) -> str:
    """Validate the CLI tool override and return the distribution's tool id.

    TATBOT_TOOL_ID is the preview override (see tatbot_sim.tools) and a
    distribution is another way of saying the same thing, so both being set at
    once is a question rather than a precedence rule: whichever quietly won,
    the other one's user would be reading a run that is not the one they asked
    for. Setting it to the tool the distribution already implies is not a
    conflict — that is just saying it twice.
    """
    prior = os.environ.get("TATBOT_TOOL_ID")
    if prior and prior != dist.tool_id:
        raise SystemExit(
            f"TATBOT_TOOL_ID={prior!r} is already set, but distribution "
            f"{dist.name!r} runs {dist.tool_id!r}. Unset it, or pick the "
            "distribution that matches the tool you meant."
        )
    return dist.tool_id


def _option(rest: list[str], name: str, default: str) -> str:
    """Read one scalar option in either Tyro spelling."""
    value = default
    for index, token in enumerate(rest):
        if token == name:
            if index + 1 >= len(rest):
                raise SystemExit(f"{name} needs a value")
            value = rest[index + 1]
        elif token.startswith(name + "="):
            value = token.split("=", 1)[1]
    return value


def _bool_option(rest: list[str], name: str, default: bool) -> bool:
    """Read a Tyro boolean while parsing the factory arguments."""
    enabled = default
    negative = "--no-" + name.removeprefix("--")
    for token in rest:
        if token == name:
            enabled = True
        elif token == negative:
            enabled = False
    return enabled


def calibration_delta(dist, rest: list[str]) -> tuple[float, float, float]:
    """Resolve the shard-persistent tip perturbation before world construction.

    The fixed-point solve locates one point in the mount frame; it does not
    imply that the pen changes between episodes.  A shard therefore represents
    one plausible fitted session.  Different shard seeds span the retained
    calibration uncertainty with a uniform-volume draw inside that sphere.
    """
    enabled = _bool_option(
        rest, "--tool-calibration-jitter", dist.tip_calibration_jitter)
    try:
        seed = int(_option(rest, "--seed", "0"))
        scale = float(_option(rest, "--tool-calibration-scale", "1.0"))
    except ValueError as exc:
        raise SystemExit(f"invalid calibration jitter option: {exc}") from exc
    if not math.isfinite(scale) or scale < 0:
        raise SystemExit("--tool-calibration-scale must be finite and non-negative")
    if not enabled or scale == 0:
        return (0.0, 0.0, 0.0)

    # Resolve the distribution's tool by id.
    from tatbot_sim import tools

    registry = tools.registry()
    spec = registry.load_tool(dist.tool_id, tools.REPO)
    workspace = tools.workspace()
    geometry = registry.resolved_tool_geometry(spec, workspace, "right", tools.REPO)
    if geometry.contact_status != "pivot-calibrated":
        print(
            f"[factory] WARNING: {dist.name!r} requested calibration jitter, but "
            f"{spec.tool_id!r} uses {tools.geometry_basis(geometry)} geometry; "
            "continuing without calibration jitter",
            file=sys.stderr,
        )
        return (0.0, 0.0, 0.0)
    uncertainty = geometry.contact_uncertainty_m
    if uncertainty is None or not math.isfinite(uncertainty) or uncertainty <= 0:
        print(
            f"[factory] WARNING: {spec.tool_id!r} has no positive measured contact "
            "uncertainty; continuing without calibration jitter",
            file=sys.stderr,
        )
        return (0.0, 0.0, 0.0)

    material = f"tatbot-tip-calibration-v1:{dist.name}:{spec.tool_id}:{seed}"
    stable_seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    rng = random.Random(stable_seed)
    direction = [rng.gauss(0.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(value * value for value in direction))
    radius = uncertainty * scale * rng.random() ** (1.0 / 3.0)
    return tuple(radius * value / norm for value in direction)


def _usage(stream=sys.stdout) -> None:
    print("usage: python -m tatbot_sim.factory <distribution> [generate flags]\n",
          file=stream)
    print("distributions:", file=stream)
    width = max(len(n) for n in DISTRIBUTIONS)
    for name, dist in DISTRIBUTIONS.items():
        note = f"  [BLOCKED: {dist.blockers[0]}]" if dist.blockers else ""
        print(f"  {name:<{width}}  {dist.summary}{note}", file=stream)
    print("\nAny generate flag may follow the distribution name, including every "
          "DR leaf:\n  python -m tatbot_sim.factory skin-erase --out-dir DIR "
          "--dr.laser.clearance 0.05 0.3", file=stream)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--list"):
        _usage()
        return
    name, rest = argv[0], argv[1:]
    dist = DISTRIBUTIONS.get(name)
    if dist is None:
        print(f"unknown distribution {name!r}\n", file=sys.stderr)
        _usage(sys.stderr)
        raise SystemExit(2)
    if dist.blockers:
        blockers = "\n".join(f"  - {b}" for b in dist.blockers)
        raise SystemExit(
            f"{name!r} is not generatable yet:\n{blockers}\n"
            "Its recipe is written and reviewable in tatbot_sim.distributions; "
            "what is missing is measurement, not code."
        )

    import tyro

    from tatbot_sim import generate
    from tatbot_sim.resolved import resolve

    tool_id = select_tool(dist)
    args = tyro.cli(generate.Args, default=dist.build_args(), args=rest)
    if not args.out_dir:
        raise SystemExit(f"{name!r} needs somewhere to write: pass --out-dir")
    if name == "body-tattoo" and not args.scenario:
        raise SystemExit(
            "'body-tattoo' needs a compiled scenario: pass --scenario PATH "
            "(create one with `tatbot sim compile`)"
        )

    if args.tool_id is not None and args.tool_id != tool_id:
        raise SystemExit(f'{name!r} requires tool {tool_id!r}, got --tool-id {args.tool_id!r}')
    args.tool_id = tool_id
    delta = calibration_delta(dist, rest)
    config = resolve(tool_id=tool_id, substrate_name=args.substrate,
        sensor_profile=args.sensor_profile, seed=args.seed, dr=args.dr,
        tip_delta_m=delta, scenario_path=args.scenario, supply=(args.supply, args.supply_ink))
    tool, substrate = config.tool, config.substrate
    for task in tasks.active_tasks(args.task, args.erase_frac, args.squiggle_frac, args.dip_frac):
        try:
            tasks.validate_task(task, tool, substrate)
            tasks.validate_supply(task, tool, config.palette_load)
        except ValueError as exc:
            # a preset is a starting point, so it can be overridden into
            # something invalid; say so before the env spends a minute building
            raise SystemExit(str(exc)) from exc

    # Printed before the scene builds, so a wrong pairing is visible in the
    # first second of a run that may take hours.
    print(f"[factory] {name}: {tool.tool_id} on {substrate.name} — "
          f"task {args.task}, horizon {args.horizon}, "
          f"{args.num_episodes} episodes, tip delta "
          f"[{', '.join(f'{value * 1000:.3f}' for value in delta)}] mm -> "
          f"{args.out_dir}", flush=True)
    generate.main(args, config=config)


if __name__ == "__main__":
    main()
