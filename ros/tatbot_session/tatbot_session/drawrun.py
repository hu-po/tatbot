"""A draw run's steps around ArmExecutor, shared by the session node and any other driver of the executor (its ports,
tatbot_session.ports): check the program, build an arm's executor and its collision guard, draw and name the
outcome, and write the page and the drawn lines into the run. No ROS.
"""
from __future__ import annotations

import json
import traceback

from tatbot_session import drawn, geometry
from tatbot_session.executor import ArmExecutor, Cancelled, Landed
from tatbot_session.ledger import Ledger
from tatbot_session.ports import Event


def check_program(program: dict, stack: dict, arms: list[str]) -> None:
    """A program fit to execute on `arms` within the stack's page (ValueError otherwise)."""
    from tatbot_contracts.ros_program import validate_for_execution

    validate_for_execution(program)
    if arms != [program['arm']]:
        raise ValueError(f"the program was prepared for the {program['arm']} arm")
    off = geometry.off_print(program, stack["page"])
    if off:
        raise ValueError(off)


def fixed_page(stack: dict, arm: str):
    """page.source fixed: (base_from_page, info) from stack.yaml page.fixed.<arm>."""
    from tatbot_description import names

    fixed = stack["page"]["fixed"][arm]
    return geometry.rpy_matrix(fixed["xyz"], fixed.get("rpy", [0, 0, 0])), {
        "source": "fixed", "pattern_id": names.fixed_pattern_id(arm)}


def build_executor(stack: dict, arm: str, *, io, node, clock, sensing, machine, repo, motion,
                   planner=None) -> ArmExecutor:
    """The arm's executor on the kinematics the description gives (config/workspace.yaml's tool and offsets)."""
    from tatbot_description import robot_description
    from tatbot_motion import Kinematics

    kin = Kinematics(robot_description(repo, arms=(arm,)), arm)
    ex = ArmExecutor(node, io, kin, motion, stack, repo, clock=clock, sensing=sensing, planner=planner)
    ex.machine = machine
    return ex


def wire_guard(repo, stack: dict, motion: dict, ios: dict):
    """The collision guard (tatbot_motion.collision.Guard.from_stack) on the arm's io: each goal's way is checked
    against the arm folding onto itself. (guard or None, a line for the log)."""
    from tatbot_motion.collision import Guard

    guard, text = Guard.from_stack(repo, stack, motion)
    if guard is None:
        return None, text
    for io in ios.values():
        io.clearance = guard.refusal
    return guard, text


def draw(ex: ArmExecutor, program: dict, ledger, *, from_op: str, touch: bool, on_error=None) -> str:
    """The arm's draw and its outcome: complete, landed (why), not landed (why: a landing refused holds the arm over
    the skin), cancelled or error: what. An error is the arm's event, and its traceback goes to `on_error`."""
    try:
        return ex.draw(program, ledger, from_op=from_op, touch=touch)
    except Landed as exc:
        return f"{'landed' if ex.io.flag('landed') else 'not landed, held'} ({exc})"
    except Cancelled:
        return "cancelled"
    except Exception as exc:  # noqa: BLE001 - the arm holds; the operator reads the run
        ex.node.event(Event.KIND_ERROR, arm=ex.arm, text=str(exc))
        if on_error is not None:
            on_error(traceback.format_exc())
        return f"error: {exc}"


def write_page_and_drawn(run_dir, ex: ArmExecutor, program: dict | None, multi: bool) -> None:
    """page[-<arm>].json, and drawn[-<arm>].svg and passes[-<arm>].json from the ledger's measured lines across
    every attempt."""
    tag = f"-{ex.arm}" if multi else ""
    (run_dir / f"page{tag}.json").write_text(json.dumps(ex.page_record, indent=1) + "\n")
    if program is not None:
        ex.drawn = drawn.from_rows(Ledger(run_dir / "ledger.jsonl").rows(), ex.arm)
    if program is not None and ex.drawn:
        page = ex.stack["page"]
        drawn.write(run_dir / f"drawn{tag}.svg", program, ex.drawn, size_m=page["size_m"], clear_m=page["clear_m"])
        write_passes(run_dir / f"passes{tag}.json", ex, program)


def write_passes(path, ex: ArmExecutor, program: dict) -> None:
    """How often the pen passed over each spot of the skin, planned and drawn (tatbot_ink.passes) at the marks the
    program was prepared with, and a line of it. Tracking only: a failure to count is a line, never the run's error."""
    try:
        from tatbot_ink import passes

        record = {"planned": passes.summary(passes.program_lines(program)),
                  "drawn": passes.summary(passes.drawn_lines(program, ex.drawn)), "marks": passes.marks(program)}
        path.write_text(json.dumps(record, indent=1) + "\n")
        text = f"passes over the skin at {passes.at_marks(program)} marks: " + ", ".join(
            f"{name} max {record[name]['max']}, p95 {record[name]['p95']}, {record[name]['mm2_at_least']['3']:.0f} mm2 at 3+"
            for name in ("planned", "drawn")) + f" ({path.name})"
    except Exception as exc:  # noqa: BLE001 - see above
        text = f"passes: not counted ({exc})"
    ex.node.event(Event.KIND_PAGE, arm=ex.arm, text=text)
