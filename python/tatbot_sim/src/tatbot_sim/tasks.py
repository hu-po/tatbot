"""What each task family needs from the fitted tool and its substrate.

A task, a tool and a substrate are a triple, and only some triples mean
anything: the laser never draws, the ballpoint never removes, and there is no
printed ruling on a skin for a squiggle to trace. Before this module those
rules lived in three places and covered one direction only — generate refused
an erase task with a pen fitted, but a DRAW task with the laser fitted ran
happily, removing pigment from a blank sheet while shipping prompts that said
"draw". That dataset looks fine until someone trains on it; the only symptom
was the engaged-episode warning printed after the whole run finished.

Kept deliberately tiny and dependency-free so both ``env`` and ``planning``
can import it (planning imports env, so the shared table cannot live in
either) and so the checks are testable without a render device. Tools and
substrates are duck-typed: anything with ``kind``/``tool_id`` and
``ruled``/``display_name`` will do.
"""

from __future__ import annotations

from dataclasses import dataclass

# What each registered tool kind does to the pigment under its tip. The
# registry's `kind` is a free-form string, so this is the one place that
# decides -- fitting a new tool means adding a line here, and the env's
# warning exists to say so rather than letting it quietly mark like a pen.
FIELD_OPS = {
    "ballpoint_pen": "deposit",
    "rotary_pen": "deposit",
    "laser": "remove",
}


@dataclass(frozen=True)
class TaskRequirements:
    """What a task family demands of the bench it runs on."""

    field_op: str
    """"deposit" or "remove" — what the episode does to the pigment field.
    The fitted tool has to be able to do it, or every episode in the run is
    mislabelled in the same direction."""

    needs_ruled: bool
    """Whether the task references the printed ruling. A squiggle traces the
    6 mm grid and its prompt says so, which is a stencil a blank skin does not
    have."""

    needs_ink: bool = True
    """Whether the episode lays pigment down and therefore needs an ink
    SUPPLY: a tool whose ink.mode dips (scripts/lib/ink_spec.py) and, for a
    real needle, a cap that holds something. A removal task needs neither.
    The fourth leg of the validator, added 2026-08-28."""


TASK_REQUIREMENTS = {
    # the squiggle WALKS the printed 6 mm lines; without them it is tracing
    # geometry that is not in the frame, and saying so in its prompt
    "maze": TaskRequirements("deposit", needs_ruled=True),
    # a weaker claim than the maze's -- closed shapes need no ruling to be
    # drawn -- but its task string names the grid paper, so on a blank skin
    # the episode is fine and the label is wrong. Override --task-name and
    # this could be relaxed; it is a legacy repertoire and not worth the
    # loophole.
    "shapes": TaskRequirements("deposit", needs_ruled=True),
    # motifs self-gate on RULED_SURFACE (language.RULED_SURFACE), so scenes are
    # describable on either substrate
    "language": TaskRequirements("deposit", needs_ruled=False),
    "erase": TaskRequirements("remove", needs_ruled=False, needs_ink=False),
    # The dip is its own task family (operator, 2026-08-29): the episode
    # leaves the hover over the sheet, charges the tool at the palette and
    # comes back — no stroke, no mark on the sheet. What it deposits is the
    # charge, so it still needs a supply and a tool that dips.
    "dip": TaskRequirements("deposit", needs_ruled=False, needs_ink=True),
}


def is_dip(task: str) -> bool:
    return task == "dip"


def active_tasks(task: str, erase_frac: float = 0.0, squiggle_frac: float = 0.0,
                 dip_frac: float = 0.0) -> list[str]:
    """The task families a run will actually sample from.

    "mix" is not itself a task, it is a distribution over the ones that are —
    and only over the ones with a non-zero share. A pen-fitted mix run with
    ``erase_frac=0`` never produces an erase episode, so refusing it because a
    pen cannot remove would be refusing a run that is fine.
    """
    if task != "mix":
        return [task]
    out = []
    if erase_frac > 0:
        out.append("erase")
    if squiggle_frac > 0:
        out.append("maze")
    if dip_frac > 0:
        out.append("dip")
    if erase_frac + squiggle_frac + dip_frac < 1:
        out.append("language")
    return out


def _tools_that_can(req: "TaskRequirements") -> list[str]:
    """Registered tool ids that satisfy the whole requirement — a live hint,
    not a hard-coded one that drifts as the bench changes.

    The substrate is part of the answer, not a detail: suggesting the 3RL for
    a squiggle would send someone to a blank skin and a second refusal, since
    a tool drags its substrate along with it.

    Best effort — this only ever runs on the error path, and a missing hint
    must not mask the error it was going to decorate.
    """
    try:
        from tatbot_sim import tools

        registry = tools.registry()
        ids = []
        for tool_id in registry.list_tools(tools.REPO):
            spec = registry.load_tool(tool_id, tools.REPO)
            if FIELD_OPS.get(spec.kind) != req.field_op:
                continue
            if req.needs_ruled and not registry.substrate_for(spec, tools.REPO).ruled:
                continue
            ids.append(tool_id)
        return ids
    except Exception:
        return []


def validate_task(task: str, tool, substrate) -> None:
    """Refuse a (task, tool, substrate, ink-policy) tuple that cannot mean
    what it says.

    Raises ``ValueError`` with the fix in the message. Called before the env
    is built, so a wrong pairing costs a second rather than a run.

    Static: everything here is a property of the task family and the tool's
    datasheet, so a preset is valid or not regardless of what the bench holds
    today. A task that deposits refuses a tool whose ink policy never dips
    (the laser). Whether the caps actually hold ink is a SESSION fact and is
    ``validate_supply``'s question, asked at run start.
    """
    req = TASK_REQUIREMENTS.get(task)
    if req is None:
        known = ", ".join(sorted(TASK_REQUIREMENTS)) or "none"
        raise ValueError(f"unknown task {task!r} (known: {known}, plus the 'mix' of them)")

    op = FIELD_OPS.get(tool.kind)
    if op != req.field_op:
        verb = "remove pigment" if req.field_op == "remove" else "lay pigment down"
        did = {"deposit": "deposits", "remove": "removes"}.get(op, "does nothing registered")
        candidates = _tools_that_can(req)
        fix = (
            f"Fit one of {', '.join(candidates)}, or preview with "
            f"TATBOT_TOOL_ID={candidates[0]}."
            if candidates else
            f"No registered tool kind maps to {req.field_op!r} — see tatbot_sim.tasks.FIELD_OPS."
        )
        raise ValueError(
            f"{task!r} episodes have to {verb}; {tool.tool_id!r} is kind "
            f"{tool.kind!r}, which {did}. {fix}"
        )

    if req.needs_ink:
        _validate_ink_policy(tool)
    if req.needs_ruled and not getattr(substrate, "ruled", False):
        raise ValueError(
            f"{task!r} episodes trace the printed ruling, and {substrate.name!r} "
            f"({substrate.display_name}) is blank — its prompts would name grid "
            "lines that are not in the frame. Use --task language on this "
            "substrate, or fit a tool whose substrate is ruled."
        )


def _validate_ink_policy(tool) -> None:
    """The static ink leg: a depositing task needs a tool that dips."""
    from tatbot_sim import tools

    ink = tools.ink_registry()
    policy = ink.policy_for(tool)
    if not policy.dips:
        raise ValueError(
            f"this task needs an ink supply and {tool.tool_id!r} has ink.mode "
            f"{policy.mode} — fit a tool that dips (lutin-3rl-bugpin, or "
            "lutin-ballpoint-dot to rehearse)")


def validate_supply(task: str, tool, palette_load=None) -> None:
    """The SESSION ink leg: do the caps hold what a depositing task needs?

    Reads config/palette_load.yaml (``palette_load`` overrides it, for
    tests). A real needle refuses dry or under-filled caps; a rehearsal tool
    is content with dry caps by design — that is what the ballpoint is for;
    a removal task never asks. Raises ``ValueError``. Called at run start,
    after ``validate_task``, by generate and the factory; deliberately NOT
    part of the static preset contract, which must not change with what was
    poured this morning.
    """
    req = TASK_REQUIREMENTS.get(task)
    if req is None or not req.needs_ink:
        return
    from tatbot_sim import tools

    ink = tools.ink_registry()
    policy = ink.policy_for(tool)
    pal = ink.load_palette(tools.REPO)
    load = palette_load if palette_load is not None else tools.palette_load()
    try:
        ink.require_supply(policy, pal, load, needs_ink=True, arm="right",
                           tool_id=getattr(tool, "tool_id", "tool"))
    except ink.InkSupplyError as exc:
        raise ValueError(str(exc)) from exc
