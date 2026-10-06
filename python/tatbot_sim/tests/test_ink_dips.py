"""Dips in the plan: the charge model puts the tool in the palette and back.

Three claims, each silent when wrong:

  * with a tool that dips (the 3RL is real) and a
    task that deposits, a batch plans dips — spliced at stroke boundaries,
    every dip step flagged so the env withholds deposition there, and each
    credit landing on a step inside its own dip;
  * the trajectory is CLOSED around every dip: the step after a dip segment
    is the step the segment left from, so the stroke that follows is the
    same stroke the canvas plan drew;
  * a tool that never dips (the laser) plans none, and a removal task never
    asks — so erase datasets are byte-for-byte what they were.

Torch + numpy, no render device. The tool is chosen by env var before this
module imports (its skip conditions read the policy at import). The default
suite fits the ballpoint, a cartridge that never dips, so it runs only the
cartridge tests; the dip planning and the no-ink refusals need their own tool:

    cd python/tatbot_sim && TATBOT_TOOL_ID=lutin-3rl-bugpin uv run python -m pytest -q tests/test_ink_dips.py
    cd python/tatbot_sim && TATBOT_TOOL_ID=picosecond-laser-pen uv run python -m pytest -q tests/test_ink_dips.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from tatbot_sim import dipping, tasks, tools
from tatbot_sim import palette as sim_palette
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
    scene = sim_palette.load(tools.REPO)
    _, transform = sim_palette.base_transform(tools.REPO, scene)
    return {slot: np.repeat((transform @ np.array([*off, 1.0]))[:3][None], b, 0).astype(np.float32)
            for slot, off in scene.rims.items()}


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
            # Away at the palette the floor is the cap floor, and its normal is
            # the cap's own — world-up only while the rack is level. The
            # measured pose has a ~5 deg tilt, so a literal [0,0,1] here would
            # pin the assumption that put the simulator down the wrong axis.
            from tatbot_sim.planning import _palette_entry_axis
            assert np.allclose(plan.surface_normals[i, lo:hi],
                               -np.asarray(_palette_entry_axis()), atol=1e-6)
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


@pytest.mark.skipif(POLICY.mode != "cartridge", reason=f"{TOOL} is not a cartridge")
def test_a_cartridge_deposits_from_its_own_supply_and_never_dips():
    """The ballpoint since 3ca4ade7: an ink supply the palette never
    replenishes. A depositing task is admitted, no dip is planned, and the
    session supply check does not look at the caps."""
    plan = _plan("shapes")
    assert plan.dips is None and plan.dip_mask is None
    tasks.validate_task("language", tools.active_tool(), tools.active_substrate())
    ink = tools.ink_registry()
    dry = {s: ink.SlotLoad(s, None) for s in tools.palette()}
    tasks.validate_supply("language", tools.active_tool(), palette_load=dry)


@pytest.mark.skipif(POLICY.mode != "none", reason=f"{TOOL} has an ink supply")
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
    """A real needle needs a wet cap; a cartridge never asks for one."""
    ink = tools.ink_registry()
    pal = tools.palette()
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
    tasks.validate_supply("language", reh, palette_load=dry)  # a cartridge: the caps are not its supply
    tasks.validate_supply("erase", laser, palette_load=dry)  # removal needs no ink


def test_palette_default_is_explicitly_synthetic():
    """Simulation placement never claims to measure the installed gooseneck."""
    scene = sim_palette.load(tools.REPO)
    source, pose = sim_palette.base_transform(tools.REPO, scene)
    assert source == 'synthetic-installed-cad'
    assert np.isfinite(pose).all()
    assert list(scene.rims) == ['inkcap_large_1', 'inkcap_medium_1', 'inkcap_small_1',
                                'inkcap_small_2', 'inkcap_medium_2', 'inkcap_large_2']
    assert DRConfig().palette.center_m is None
    assert os.environ.get("TATBOT_TOOL_ID", TOOL) == TOOL


# --- the gate on the solved reference -----------------------------------------------------

def _fake_expert(tips: np.ndarray, num_envs: int, t_len: int):
    """An expert whose joint reference lands the tip exactly where we say.

    The gate's job is to compare the SOLVED reference against the cap, so the
    solver is the thing to hold still: identity rotations mean the tool points
    exactly where it was commanded, and every remaining error is the lateral
    miss under test.
    """
    from types import SimpleNamespace

    n = num_envs * t_len
    poses = torch.eye(4).repeat(n, 1, 1)
    poses[:, :3, 3] = torch.as_tensor(tips.reshape(n, 3), dtype=torch.float32)
    ik = SimpleNamespace(n_joints=7, fk=lambda q: poses)
    return SimpleNamespace(
        q_ref=torch.zeros((num_envs, t_len, 7)), ik=ik,
        target_rotations=lambda normals, b: torch.eye(3).repeat(b, 1, 1),
    )


def _dip_plan(tips, *, slot="inkcap_small_1", num_envs=2, t_len=3, step=1):
    from types import SimpleNamespace

    targets = np.zeros((num_envs, t_len, 3), dtype=np.float32)
    up = np.zeros((num_envs, t_len, 3), dtype=np.float32)
    up[..., 2] = 1.0
    return SimpleNamespace(
        n_app=0, q_raised=None, targets=targets, surface_normals=up, pen_normals=up,
        dip_credits=[[step]] + [[] for _ in range(num_envs - 1)],
        dips=[[{"slot": slot}]] + [[] for _ in range(num_envs - 1)],
    ), tips


def test_a_dip_that_misses_the_cap_is_caught_even_though_the_point_is_1mm_true():
    """The residual gate cannot see this: the commanded point sits inside the
    cap, so a reference 10 mm to the side still satisfies it. A small cap is
    8 mm across, and a dip that never entered one was still credited a full
    charge by step index (measured 2026-09-09)."""
    from tatbot_sim.reference import _dip_reference_error

    tips = np.zeros((2, 3, 3), dtype=np.float32)
    tips[0, 1] = [0.010, 0.0, 0.0]          # 10 mm sideways, well outside a 4 mm radius
    plan, tips = _dip_plan(tips)
    expert = _fake_expert(tips, 2, 3)
    lateral, axis, missed = _dip_reference_error(expert, plan, {"inkcap_small_1": np.zeros((2, 3))}, 2)
    assert missed[0] and not missed[1], "only the env that dipped, and missed, is flagged"
    assert lateral[0] == pytest.approx(10.0, abs=1e-3)
    assert axis[0] == pytest.approx(0.0, abs=1e-6), "identity rotation holds the commanded axis"
    assert lateral[1] == 0.0 and axis[1] == 0.0, "an env with no dips scores zero"


def test_a_dip_inside_the_cap_passes():
    """1 mm off centre in an 8 mm cap is a dip. The gate must not condemn the
    ordinary case, or every dip episode is dropped and the corpus is empty."""
    from tatbot_sim.reference import _dip_reference_error

    tips = np.zeros((2, 3, 3), dtype=np.float32)
    tips[0, 1] = [0.001, 0.0, 0.0]
    plan, tips = _dip_plan(tips)
    expert = _fake_expert(tips, 2, 3)
    lateral, _, missed = _dip_reference_error(expert, plan, {"inkcap_small_1": np.zeros((2, 3))}, 2)
    assert not missed.any()
    assert lateral[0] == pytest.approx(1.0, abs=1e-3)


def test_the_gate_is_silent_when_a_batch_plans_no_dips():
    """Every non-dipping distribution goes through this call. It must cost
    nothing and flag nothing."""
    from types import SimpleNamespace

    from tatbot_sim.reference import _dip_reference_error

    plan = SimpleNamespace(n_app=0, q_raised=None, dip_credits=None, dips=None)
    lateral, axis, missed = _dip_reference_error(None, plan, None, 3)
    assert not missed.any() and not lateral.any() and not axis.any()


def test_the_segment_enters_a_tilted_cap_along_the_shared_axis():
    """The simulator's dip and the arm's now come from one module
    (scripts/lib/dip_motion.py). On a level rack both stacks agreed anyway;
    tilt the rack and the old world-Z assumption would put the hover and the
    plunge somewhere the arm would never go."""
    from tatbot_sim import tools

    axis = np.array([0.3, 0.0, -1.0])
    axis /= np.linalg.norm(axis)
    rim = np.array([0.2, -0.05, 0.06])
    geo = dipping.DipGeometry(rim_world=rim, plunge_m=0.003, cap_depth_m=0.0125,
                              hover_m=0.02, dwell_s=0.4, plunge_speed=0.02,
                              travel_speed=0.12, settle_time=0.2,
                              entry_axis=tuple(axis))
    pos, floor_pts, floor_nms, plunge = dipping.dip_segment(np.array([0.3, 0.02, 0.09]), geo, 1 / 30)
    shared = tools.dip_motion().dip_poses(rim, axis, hover_m=geo.hover_m, plunge_m=geo.plunge_m)

    assert np.allclose(pos[plunge], shared.bottom, atol=1e-6), "bottoms out where the arm would"
    assert np.allclose(floor_nms, shared.outward_normal), "and holds the cap's own axis"
    assert np.allclose(floor_pts, rim + axis * geo.cap_depth_m)
    # the tilt is real: a world-Z segment would have put both on the rim's x
    assert abs(shared.bottom[0] - rim[0]) > 1e-4


@pytest.mark.parametrize('control_hz', [30, 400])
def test_contact_time_and_bleed_follow_control_period(control_hz):
    from types import SimpleNamespace

    import torch
    from tatbot_sim.env import TatbotDrawEnv

    # A stationary tip spends one second touching: time-dependent ink use
    # must be independent of how many controller samples describe that second.
    world = SimpleNamespace(
        control_freq=control_hz, ink_policy=SimpleNamespace(dips=True, touches_stock=False,
            deposit_ul_per_mm=2.0, bleed_ul_per_s=3.0),
        _dip_credit=None, _prev_tcp=None, ink_charge_ul=torch.tensor([100.0]),
        ink_used_ul=torch.zeros(1), ink_contact_mm=torch.zeros(1), ink_contact_s=torch.zeros(1))
    for step in range(control_hz):
        TatbotDrawEnv._ink_step(world, torch.zeros((1, 3)), torch.tensor([True]), step)
    assert float(world.ink_contact_s[0]) == pytest.approx(1.0, abs=2e-6)
    assert float(world.ink_used_ul[0]) == pytest.approx(3.0, abs=2e-5)
