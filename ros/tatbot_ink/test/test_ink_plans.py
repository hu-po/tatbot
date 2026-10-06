"""A prepared program plans op after op on the configured page (stack.yaml page.fixed.right), the way
the session sends it: each op from the last one's final joints, a continuing chunk joined pen down.
This is the contract between the program's ops and tatbot_motion.plan_op, on the page the stack uses
with page.source fixed (mock hardware, and the bench's fallback).

The model error is held to the planner's own gate, motion.yaml clik.max_model_error_m, past which
plan_op refuses a plan. clik.model_error_cap_m (0.1 mm) bounds the CLIK's parity with the C++
planner; it is not a drawing bound. The demo table's page lies nearer the base than the old rig's,
and the elbow bends to within 0.2-0.3 rad of its lower limit there. The CLIK trails its reference by
0.1-0.4 mm on pen-up travel and by up to 0.12 mm pen down on the print's lower half, against under
0.06 mm anywhere on the old rig's page: inside every gate. Along the page normal the tip stays
within 0.02 mm, so the pen-down path stays within 0.1 mm of the resource's contact height.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tatbot_ink
import tatbot_motion as tm
from ink_designs import REPO, design, element, placement, program, write
from tatbot_description import load_stack
from tool_spec import load_tool


def _shapes(side=0.04):
    """A closed square, a closed 64-point circle and a zigzag's sharp corners, on one pen."""
    angle = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    circle = np.column_stack([0.72 + 0.2 * np.cos(angle), 0.28 + 0.2 * np.sin(angle)])
    zigzag = [[x, 0.6 if i % 2 == 0 else 0.9] for i, x in enumerate(np.linspace(0.1, 0.9, 13))]
    paths = [element(np.array([[0.1, 0.1], [0.45, 0.1], [0.45, 0.45], [0.1, 0.45]]) * side, closed=True),
             element(circle * side, closed=True), element(np.array(zigzag) * side)]
    return program([("pen-1", paths)], canvas=(side, side))


def test_a_prepared_program_plans_op_after_op_on_the_configured_page(tmp_path):
    pin = pytest.importorskip("pinocchio")
    # the print's lower half, the demo table page's worst tracked; 12 s chunks make continuing ops
    path = write(tmp_path, design([("p", _shapes(), placement((0.04, 0.04), anchor=(0.0, -0.03)))]))
    prepared = tatbot_ink.compile(path, repo=REPO, max_segment_s=12.0)
    ops = [op for op in prepared["ops"] if op["op"] == "stroke"]
    assert any(op["continues"] for op in ops)
    kin = tm.Kinematics.from_repo(REPO, arm="right")
    motion = tm.load_motion(REPO / "ros/tatbot_motion/config/motion.yaml")
    resource = prepared['resources'][0]
    motion['pen']['mode'] = resource['pen_mode']
    pen = tm.pen_down(motion, load_tool(resource['tool']['id'], REPO))
    fixed = load_stack(REPO)["page"]["fixed"]["right"]
    page = np.eye(4)
    page[:3, :3] = pin.rpy.rpyToMatrix(*fixed.get("rpy", [0.0, 0.0, 0.0]))  # URDF roll-pitch-yaw
    page[:3, 3] = fixed["xyz"]
    q = np.array([-0.5, 1.0, 1.0, -0.3, 0.0, 0.0, 0.002])
    hover = np.eye(4)
    hover[:3, :3] = tm.plan.pen_rotation(page, kin.fk(q)[:3, :3])
    hover[:3, 3] = (page @ [0.0, 0.0, 0.03, 1.0])[:3]
    seed = kin.solve(hover, q)
    for index, op in enumerate(ops):
        joined_next = index + 1 < len(ops) and ops[index + 1]["continues"]
        traj = tm.plan_op(op, base_from_page=page, q_seed=seed, kin=kin, motion=motion,
                          speed_m_s=prepared["draw_speed_m_s"], pen_down_at_start=op["continues"],
                          lift_at_end=not joined_next, pen=pen)
        assert traj.info["max_model_error_m"] <= motion["clik"]["max_model_error_m"]
        # a chunk boundary is no lift: the next chunk joins where this one ends, pen down
        assert traj.info["joined"] == op["continues"]
        assert (traj.phase[-1] == tm.PHASE_DRAW) == joined_next
        drawn = traj.tip[traj.phase == tm.PHASE_DRAW]
        heights = (drawn - page[:3, 3]) @ page[:3, 2]
        assert np.abs(heights - pen.height_m).max() < 1e-4
        seed = traj.q[-1]
