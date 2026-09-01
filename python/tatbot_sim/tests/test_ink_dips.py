"""Dips in the plan: the charge model puts the tool in the palette and back.

Three claims, each silent when wrong:

  * with a tool that dips (the ballpoint rehearses; the 3RL is real) and a
    task that deposits, a batch plans dips — spliced at stroke boundaries,
    every dip step flagged so the env withholds deposition there, and each
    credit landing on a step inside its own dip;
  * the trajectory is CLOSED around every dip: the step after a dip segment
    is the step the segment left from, so the stroke that follows is the
    same stroke the canvas plan drew;
  * a tool that never dips (the laser) plans none, and a removal task never
    asks — so erase datasets are byte-for-byte what they were.

Torch + numpy, no render device. The fitted tool is fixed by env var before
the package imports (tools.active_tool is import-bound):

    cd python/tatbot_sim && TATBOT_TOOL_ID=lutin-ballpoint-dot uv run python -m pytest -q tests/test_ink_dips.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from tatbot_sim import dipping, tasks, tools
from tatbot_sim.config import DRConfig
from tatbot_sim.planning import plan_batch
from tatbot_sim.strokes import ShapeConfig, Stroke
from tatbot_sim.surface import PlanarSurface
from tatbot_sim.textures import grid_paper_sheets
from transforms3d.euler import euler2mat

TOOL = tools.active_tool().tool_id
POLICY = tools.active_ink_policy()


def _surface(b=2, seed=0):
    rng = np.random.default_rng(seed)
    rots = np.stack([euler2mat(*rng.uniform(-0.03, 0.03, 3), "sxyz") for _ in range(b)])
    center = torch.as_tensor(np.array([[0.29, 0.0, 0.03]] * b), dtype=torch.float32)
    return PlanarSurface(center, torch.as_tensor(rots, dtype=torch.float32))


def _cap_rims(b=2):
    ink = tools.ink_registry()
    layout = ink.palette_layout_from_urdf(tools.REPO)
    c = np.array(ink.palette_root_in_base(tools.REPO), dtype=np.float32)
    return {s: np.repeat((c + np.array(off, dtype=np.float32))[None], b, 0) for s, off in layout.items()}


@pytest.fixture(autouse=True)
def wet_caps_for_a_real_needle():
    """The repo's palette_load is whatever the bench holds today — dry, at
    the time of writing. A real needle refuses dry caps (correctly), so give
    it a wet rack the way generate does (tools.set_supply); a rehearsal tool
    takes the bench as it is."""
    prior = tools.supply()
    if POLICY.mode == "real":
        tools.set_supply("wet", "nighthawk_black")
    yield
    tools.set_supply(*prior)


def _dr(dips=False, initial=(0.0, 0.0), capacity=(1.0, 1.0)):
    dr = DRConfig()
    dr.ink.dips = dips
    dr.ink.initial_charge_frac = initial
    dr.ink.capacity_scale = capacity
    return dr


def _plan(task="shapes", b=2, seed=3, horizon=900, rims=True, dr=None):
    sheets = grid_paper_sheets(b)
    return plan_batch(
        np.random.default_rng(seed), sheets, _surface(b, seed),
        task=task, horizon=horizon, num_envs=b, dr=dr or _dr(dips=True),
        draw_clearance=0.004, task_name="draw a {shape}", maze_task_name="squiggle",
        cap_rims=_cap_rims(b) if rims else None,
    )


# --- the segment itself ------------------------------------------------------------------

def test_dip_segment_is_closed_and_bottoms_out_in_the_cap():
    geo = dipping.DipGeometry(rim_world=np.array([0.2, -0.05, 0.06]), plunge_m=0.003,
                              cap_depth_m=0.0125, hover_m=0.02, dwell_s=0.4,
                              plunge_speed=0.02, travel_speed=0.12, settle_time=0.2)
    start = np.array([0.3, 0.02, 0.055])
    pos, floor_pts, floor_nms, plunge = dipping.dip_segment(start, geo, 1 / 30)
    assert np.allclose(pos[-1], start, atol=1e-6), "a dip returns to where it left"
    assert np.allclose(pos[plunge], geo.rim_world - [0, 0, geo.plunge_m], atol=1e-6)
    assert pos[:, 2].min() >= min(start[2], geo.rim_world[2] - geo.plunge_m) - 1e-6
    assert pos[:, 2].max() >= geo.rim_world[2] + geo.hover_m - 1e-6, "clears the rim on approach"
    # the floor handed to the expert is the cap floor, world-up
    assert np.allclose(floor_pts[:, 2], geo.rim_world[2] - geo.cap_depth_m)
    assert np.allclose(floor_nms, [0, 0, 1])
    assert dipping.dip_steps(start, geo, 1 / 30) == len(pos)


def test_stroke_needs_count_length_and_time_on_the_sheet():
    cfg = ShapeConfig()
    needs = dipping.stroke_needs([Stroke(np.array([[0, 0], [0.03, 0], [0.03, 0.04]]))], 0.05, cfg)
    assert needs[0].contact_mm == pytest.approx(70.0)
    assert needs[0].contact_s == pytest.approx(0.07 / 0.05 + cfg.settle_time)


# --- plans with the fitted tool ----------------------------------------------------------

@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_drawing_episodes_do_not_dip_unless_asked():
    """A 30 s drawing is not a session (operator, 2026-08-29): with the
    default DR the tool opens full and never leaves the sheet; the charge
    is still accounted, so the env can fade a line if it ever runs out."""
    plan = _plan("shapes", dr=DRConfig())
    assert plan.dips is None and plan.dip_mask is None
    assert np.allclose(plan.ink_initial_ul, POLICY.charge_capacity_ul)
    assert np.allclose(plan.ink_capacity_ul, POLICY.charge_capacity_ul)


@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_a_depositing_batch_dips_and_the_plan_is_consistent():
    plan = _plan("shapes")  # dips on, tool opens empty
    assert plan.dips is not None and all(plan.dips), "every env dips at least once (session start)"
    for i, dips in enumerate(plan.dips):
        assert dips[0]["reason"] == "session_start" and dips[0]["before_stroke"] == 0
        for dip, credit in zip(dips, plan.dip_credits[i], strict=True):
            lo, hi = dip["step"], dip["step"] + dip["steps"]
            assert lo <= credit < hi, "the charge lands inside its own dip"
            assert plan.dip_mask[i, lo:hi].all()
            # the segment ends where the trajectory resumes: closed
            assert np.allclose(plan.targets[i, hi - 1], plan.targets[i, hi], atol=1e-5)
            # away at the palette the floor is the cap floor, world-up
            assert np.allclose(plan.surface_normals[i, lo:hi], [0, 0, 1])
        # nothing outside the dips is flagged
        outside = np.ones(plan.draw_horizon, dtype=bool)
        for dip in dips:
            outside[dip["step"]:dip["step"] + dip["steps"]] = False
        assert not plan.dip_mask[i][outside].any()
    # and it all fits the horizon the planner budgeted against
    assert plan.targets.shape[1] == plan.draw_horizon
    assert (plan.lengths - plan.n_app <= plan.draw_horizon).all()


@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_language_batches_dip_too():
    plan = _plan("language", horizon=1200)
    assert plan.dips is not None and all(plan.dips)
    assert plan.dip_mask.any(axis=1).all()


@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_episodes_can_open_mid_session_and_run_dry():
    """InkDR.initial_charge_frac / capacity_scale: an episode that opens
    charged skips the session_start dip; a needle scaled to a few
    millimetres of range re-dips for low_charge inside the episode — the
    behaviour a policy has to see to learn it."""
    full = _plan("shapes", dr=_dr(dips=True, initial=(1.0, 1.0)))
    # a full needle at the nominal capacity covers any shape: no dip at all
    assert full.dips is None and np.allclose(full.ink_initial_ul, POLICY.charge_capacity_ul)
    tiny = _plan("shapes", seed=5, dr=_dr(dips=True, initial=(1.0, 1.0), capacity=(0.03, 0.03)))
    assert np.allclose(tiny.ink_capacity_ul, 0.03 * POLICY.charge_capacity_ul)
    reasons = [d["reason"] for dips in tiny.dips for d in dips]
    assert "low_charge" in reasons, reasons
    for i, dips in enumerate(tiny.dips):
        for dip in dips:
            assert dip["charge_after_ul"] <= tiny.ink_capacity_ul[i] + 1e-6


@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_the_dip_task_is_one_dip_and_no_stroke():
    """--task dip: hover, palette, back to the same hover. Opens empty
    whatever initial_charge_frac says, one session_start dip, a prompt that
    names the cap, nothing drawn."""
    plan = _plan("dip", horizon=600, dr=_dr(dips=False, initial=(1.0, 1.0)))
    assert plan.kinds == ["dip", "dip"]
    assert np.allclose(plan.ink_initial_ul, 0.0)
    for i in range(2):
        assert len(plan.dips[i]) == 1 and plan.dips[i][0]["reason"] == "session_start"
        assert plan.paths[i] == []
        assert plan.tasks[i].startswith("dip ") and "ink cap" in plan.tasks[i]
        assert plan.programs[i]["slot"] == plan.dips[i][0]["slot"]
        lo, hi = plan.dips[i][0]["step"], plan.dips[i][0]["step"] + plan.dips[i][0]["steps"]
        # away at the palette for the whole middle; hovering before and after
        assert plan.dip_mask[i, lo:hi].all() and not plan.dip_mask[i, :lo].any()
        assert np.allclose(plan.targets[i, hi - 1], plan.targets[i, hi], atol=1e-5)
    assert tasks.active_tasks("mix", dip_frac=0.2) == ["dip", "language"]


def test_the_supply_is_chosen_not_poured(monkeypatch):
    """tatbot_sim.tools.set_supply: a wet rack fills every right-arm cap,
    a dry one empties them, and bench is the yaml — regardless of what the
    bench holds today, the sim run says which it drew from."""
    prior = tools.supply()
    try:
        tools.set_supply("wet", "nighthawk_black")
        wet = tools.palette_load()
        pal = tools.palette()
        assert all(not wet[s].dry and wet[s].ink_id == "nighthawk_black" for s in pal if pal[s].arm == "right")
        assert all(wet[s].dry for s in pal if pal[s].arm != "right")
        tools.set_supply("dry")
        assert all(sl.dry for sl in tools.palette_load().values())
        with pytest.raises(ValueError, match="unknown ink"):
            tools.set_supply("wet", "no_such_ink")
        with pytest.raises(ValueError, match="not one of"):
            tools.set_supply("damp")
        reg = tools.registry()
        real = reg.load_tool("lutin-3rl-bugpin", tools.REPO)
        tools.set_supply("dry")
        with pytest.raises(ValueError, match="no usable right-arm cap"):
            tasks.validate_supply("language", real)
        tools.set_supply("wet", "nighthawk_black")
        tasks.validate_supply("language", real)
    finally:
        tools.set_supply(*prior)


@pytest.mark.skipif(not POLICY.dips, reason=f"{TOOL} never dips")
def test_without_a_palette_nothing_dips():
    plan = _plan("shapes", rims=False)
    assert plan.dips is None and plan.dip_mask is None


@pytest.mark.skipif(POLICY.dips, reason=f"{TOOL} dips")
def test_a_tool_without_ink_never_dips_and_is_refused_for_ink_tasks():
    plan = _plan("erase", horizon=1200) if tools.active_substrate().ruled is False else None
    if plan is not None:
        assert plan.dips is None and plan.dip_mask is None
    # the field-op leg refuses first (a laser removes); the ink leg would too
    with pytest.raises(ValueError, match="removes|ink.mode none"):
        tasks.validate_task("language", tools.active_tool(), tools.active_substrate())
    with pytest.raises(ValueError, match="ink.mode none"):
        tasks._validate_ink_policy(tools.active_tool())
    tasks.validate_task("erase", tools.active_tool(), tools.active_substrate())


def test_validator_reads_the_palette_load():
    """A real needle needs a wet cap; a rehearsal is content with a dry one."""
    ink = tools.ink_registry()
    pal = ink.load_palette(tools.REPO)
    dry = {s: ink.SlotLoad(s, None) for s in pal}
    wet = {s: ink.SlotLoad(s, "nighthawk_black", 500.0) for s in pal}
    reg = tools.registry()
    real = reg.load_tool("lutin-3rl-bugpin", tools.REPO)
    reh = reg.load_tool("lutin-ballpoint-dot", tools.REPO)
    skin = reg.substrate_for(real, tools.REPO)
    paper = reg.substrate_for(reh, tools.REPO)
    laser = reg.load_tool("picosecond-laser-pen", tools.REPO)
    # the static contract does not care what is poured
    tasks.validate_task("language", real, skin)
    tasks.validate_task("language", reh, paper)
    tasks.validate_task("erase", laser, skin)
    # the session check does
    with pytest.raises(ValueError, match="no usable right-arm cap"):
        tasks.validate_supply("language", real, palette_load=dry)
    tasks.validate_supply("language", real, palette_load=wet)
    tasks.validate_supply("language", reh, palette_load=dry)
    tasks.validate_supply("erase", laser, palette_load=dry)  # removal needs no ink


def test_the_palette_sits_where_the_urdf_and_the_measured_hold_agree():
    """palette_root in the arm base frame: between the arms, to the right
    arm's left. The measured hold (poses.yaml palette_center, ROOT frame) put
    the tip within 3 cm of it — the URDF is the rig, not 1.0 folklore."""
    ink = tools.ink_registry()
    c = np.array(ink.palette_root_in_base(tools.REPO))
    assert np.allclose(c, [0.126, 0.2675, 0.085], atol=1e-6)
    measured_tip_base = np.array([0.1541, 0.2826, 0.0965])  # FK of the 2026-08-26 hold
    assert np.linalg.norm(measured_tip_base - c) < 0.04
    assert DRConfig().palette.center_m is None, "the default derives from the URDF"
    assert os.environ.get("TATBOT_TOOL_ID", TOOL) == TOOL
