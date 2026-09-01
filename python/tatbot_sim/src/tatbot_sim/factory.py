"""Generate one named distribution.

    python -m tatbot_sim.factory --list
    python -m tatbot_sim.factory paper-draw --out-dir ~/tatbot-sim/datasets/paper-v5
    python -m tatbot_sim.factory skin-erase --out-dir ~/ds/laser-v2 --num-episodes 128

The recipe comes from tatbot_sim.distributions; every leaf of it stays
overridable on the command line, so this is a starting point rather than a
cage. What the launcher adds over calling ``generate`` directly is that the
tool, the substrate, the task and the episode length can no longer be
combined by hand into something that runs happily and means nothing.

**Why this re-executes itself.** The fitted tool is resolved at IMPORT time:
the agent class body and the URDF build both ask ``tools.active_tool()`` while
they are being defined, ``language`` builds its prompt constants from the
substrate, and the answer is cached for the process. Importing ``tatbot_sim``
alone is enough to fix it — and ``python -m tatbot_sim.factory`` has to import
the package before it can run this module. So by the time any code here could
set TATBOT_TOOL_ID, the URDF has already been derived for whatever
config/workspace.yaml says is fitted, and a paper-draw run on a laser-fitted
bench would build the laser's geometry, write the laser's prompts, and remove
pigment while its dataset said "paper-draw" (measured, 2026-08-27: it does
exactly this, and the task validator is what caught it).

Setting the variable and re-executing is the only thing that actually works,
because the tool has to be in the environment *before the interpreter starts*
— which is what the hand-run incantation `TATBOT_TOOL_ID=... python -m
tatbot_sim.generate` was doing all along. The second pass is a normal run.
"""

from __future__ import annotations

import os
import sys

from tatbot_sim import tasks
from tatbot_sim.distributions import DISTRIBUTIONS

_REEXEC_GUARD = "TATBOT_FACTORY_REEXEC"
"""Set to the distribution name across the re-exec, so the second pass knows
it is the second pass. Any value that does not match means "not yet"."""


def select_tool(dist) -> None:
    """Put the distribution's tool in the gripper, for this process.

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
    os.environ["TATBOT_TOOL_ID"] = dist.tool_id


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

    if os.environ.get(_REEXEC_GUARD) != dist.name:
        select_tool(dist)
        os.environ[_REEXEC_GUARD] = dist.name
        # Replaces this process. Everything above ran against the wrong tool
        # (the package was imported before we got a say); nothing below it did.
        os.execv(sys.executable, [sys.executable, "-m", "tatbot_sim.factory", *argv])

    import tyro

    from tatbot_sim import generate, tools

    # The re-exec is load-bearing and silent when it fails, so check rather
    # than trust: everything downstream -- geometry, prompts, what the tool
    # does to pigment -- was decided by the value read at import.
    tool = tools.active_tool()
    if tool.tool_id != dist.tool_id:
        raise SystemExit(
            f"{name!r} runs {dist.tool_id!r} but this process resolved "
            f"{tool.tool_id!r} at import. The tool has to be set before the "
            "interpreter starts; run this through `python -m tatbot_sim.factory` "
            f"with {_REEXEC_GUARD} unset."
        )

    args = tyro.cli(generate.Args, default=dist.build_args(), args=rest)
    if not args.out_dir:
        raise SystemExit(f"{name!r} needs somewhere to write: pass --out-dir")

    substrate = tools.active_substrate()
    try:
        tools.set_supply(args.supply, args.supply_ink)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for task in tasks.active_tasks(args.task, args.erase_frac, args.squiggle_frac, args.dip_frac):
        try:
            tasks.validate_task(task, tool, substrate)
            tasks.validate_supply(task, tool)
        except ValueError as exc:
            # a preset is a starting point, so it can be overridden into
            # something invalid; say so before the env spends a minute building
            raise SystemExit(str(exc)) from exc

    # Printed before the scene builds, so a wrong pairing is visible in the
    # first second of a run that may take hours.
    print(f"[factory] {name}: {tool.tool_id} on {substrate.name} — "
          f"task {args.task}, horizon {args.horizon}, "
          f"{args.num_episodes} episodes -> {args.out_dir}", flush=True)
    generate.main(args)


if __name__ == "__main__":
    main()
