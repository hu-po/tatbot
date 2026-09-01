"""How a canvas-frame plan becomes world motion on a shaped surface.

Two claims live here, and both are silent when wrong:

  * the canvas->world mapping must reduce to the old affine one on a plane,
    or every planar episode ever generated quietly changes;
  * the tool must be held along the LOCAL normal, with lean as a perturbation
    measured from that normal rather than from a fixed plane's. That is the
    behaviour the curved-surface work exists to demonstrate, and a plan that
    aimed the tool at a constant normal would look perfectly reasonable in a
    video while teaching the wrong thing.

Torch + numpy, no render device (imports the env module but builds no scene):

    cd python/tatbot_sim && uv run python tests/test_planning.py
"""

from __future__ import annotations

import numpy as np
import torch
from tatbot_sim.expert import ReachMask
from tatbot_sim.language import sample_scene
from tatbot_sim.planning import (
    SceneTooLongError,
    canvas_to_world,
    cap_lean,
    lean_normals,
    plan_batch,
    sample_lean_profile,
)
from tatbot_sim.surface import DisplacedSurface, PlanarSurface, PlaneChart
from tatbot_sim.textures import SHEET_H_M, SHEET_W_M, grid_paper_sheets
from transforms3d.euler import euler2mat

HR, HC = 84, 65


def _frames(b=3, seed=0):
    rng = np.random.default_rng(seed)
    rots = np.stack([euler2mat(*rng.uniform(-0.09, 0.09, 3), "sxyz") for _ in range(b)])
    center = torch.as_tensor(rng.uniform(-0.05, 0.05, (b, 3)), dtype=torch.float32)
    return center, torch.as_tensor(rots, dtype=torch.float32)


def _curved(b, amp=0.010, k=26.0):
    """A doubly-curved field, so the local normal actually varies."""
    us = np.linspace(-SHEET_W_M / 2, SHEET_W_M / 2, HC)
    vs = np.linspace(-SHEET_H_M / 2, SHEET_H_M / 2, HR)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    h = amp * np.sin(k * uu) * np.cos(0.7 * k * vv)
    return torch.as_tensor(np.repeat(h[None], b, 0), dtype=torch.float32)


def _traj(t=40, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(-0.06, 0.06, t),
        rng.uniform(-0.08, 0.08, t),
        rng.uniform(0.0, 0.02, t),
    ], axis=1).astype(np.float32)


def test_on_a_plane_the_mapping_is_the_affine_one_it_replaced():
    """The planar regression: same world targets as top_center + lifted @ rot.T."""
    b, clearance = 3, 0.004
    center, rot = _frames(b)
    surf = PlanarSurface(center, rot)
    traj = _traj()
    for i in range(b):
        got, pts, normals = canvas_to_world(traj, surf, i, clearance)
        lifted = traj.copy()
        lifted[:, 2] += clearance
        want = center[i].numpy()[None, :] + lifted @ rot[i].numpy().T
        assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()
        assert np.allclose(normals, rot[i].numpy()[:, 2][None, :], atol=1e-6)
        # the surface points are where the tool would touch: canvas z = 0
        flat = traj.copy()
        flat[:, 2] = 0.0
        want_pts = center[i].numpy()[None, :] + flat @ rot[i].numpy().T
        assert np.allclose(pts, want_pts, atol=1e-6)


def test_a_plan_keeps_its_clearance_over_a_shape_instead_of_cutting_across_it():
    """Canvas z is height above the surface AT THAT POINT. Travel over a rise
    stays the same distance off the skin rather than flying a plane through
    it — checked by projecting the planned world points back down, which is a
    different code path from the one that placed them."""
    b = 1
    center, rot = _frames(b)
    surf = DisplacedSurface(PlaneChart(center, rot), _curved(b))
    traj = _traj()
    world, _, _ = canvas_to_world(traj, surf, 0, 0.004)
    _, dist, _ = surf.env_view(0, len(world)).project(
        torch.as_tensor(world, dtype=torch.float32)
    )
    assert np.allclose(dist.numpy(), traj[:, 2] + 0.004, atol=2e-5), dist


def test_lean_is_unchanged_when_the_normal_is_constant():
    """The generalisation must not move the flat case: a single normal and the
    same normal repeated per step have to give identical pen axes."""
    rng = np.random.default_rng(5)
    profile = sample_lean_profile(rng, 60, 0.12, 4)
    n = np.array([0.03, -0.05, 0.998])
    n /= np.linalg.norm(n)
    one = lean_normals(n, profile)
    many = lean_normals(np.repeat(n[None], len(profile), 0), profile)
    assert np.allclose(one, many, atol=1e-12)


def test_with_no_lean_the_tool_points_straight_down_the_local_normal():
    b = 1
    center, rot = _frames(b)
    surf = DisplacedSurface(PlaneChart(center, rot), _curved(b))
    traj = _traj()
    _, _, normals = canvas_to_world(traj, surf, 0, 0.004)
    axes = lean_normals(normals, np.zeros((len(traj), 2)))
    assert np.allclose(axes, normals, atol=1e-6)
    # and the normal really does vary along a slope, or this proves nothing
    spread = np.degrees(
        np.arccos(np.clip((normals[:-1] * normals[1:]).sum(-1), -1, 1))
    )
    assert spread.max() > 1.0, spread.max()


def test_lean_is_an_angle_off_the_local_normal_not_off_a_fixed_plane():
    """The claim in one assertion: whatever the surface does underneath, the
    angle between the tool and the skin's own normal is exactly the lean the
    profile asked for."""
    b = 1
    center, rot = _frames(b)
    surf = DisplacedSurface(PlaneChart(center, rot), _curved(b))
    traj = _traj()
    _, _, normals = canvas_to_world(traj, surf, 0, 0.004)
    profile = sample_lean_profile(np.random.default_rng(7), len(traj), 0.12, 4)
    axes = lean_normals(normals, profile)
    want = np.linalg.norm(profile, axis=1)
    # compared as cosines: arccos is ill-conditioned near zero lean, where it
    # would report 1e-4 of error on an angle that is right to 1e-7
    assert np.allclose((axes * normals).sum(-1), np.cos(want), atol=1e-6)
    # against the pad's plane normal it would NOT be the requested lean
    flat = np.arccos(np.clip(axes @ rot[0].numpy()[:, 2], -1, 1))
    assert np.abs(flat - want).max() > 0.05


def test_a_reach_mask_reads_canvas_metres():
    """Nearest-node lookup over the canvas, so placement can ask about a point
    in the same units it places in."""
    m = np.zeros((5, 5), dtype=bool)
    m[:, :2] = True                      # only the left of the canvas is workable
    rm = ReachMask(m, 0.10, 0.10)
    assert rm.fraction == 0.4
    assert rm.node_ok(-0.05, 0.0) and rm.node_ok(-0.03, 0.0)
    assert not rm.node_ok(0.05, 0.0)
    assert rm.ok([[-0.05, 0.0], [-0.03, 0.02]])
    assert not rm.ok([[-0.05, 0.0], [0.05, 0.0]])   # ALL points, not any
    # outside the canvas clamps to the border rather than throwing
    assert rm.node_ok(-10.0, -10.0)


def test_a_scene_is_placed_only_where_the_tool_can_work():
    """The whole point of the mask: on a mound the flanks ask the wrist for a
    lean it cannot make, and a scene laid across them is a label the arm
    quietly misses by centimetres."""
    sheet = grid_paper_sheets(1, seed=1)[0]
    m = np.zeros((21, 21), dtype=bool)
    m[:, :8] = True                       # workable only on the -x side
    rm = ReachMask(m, 0.20, 0.20)
    placed = 0
    for seed in range(8):
        strokes, _ = sample_scene(np.random.default_rng(seed), sheet, 14.0,
                                  verb="erase", reachable=rm)
        for st in strokes:
            placed += 1
            assert rm.ok(st.points), st.points.mean(0)
    assert placed > 0, "the mask left nowhere to place anything"


def test_the_tool_leans_as_far_as_the_wrist_can_and_no_further():
    """A tool need not be exactly perpendicular to skin to work it, and on a
    mound's flanks exactly perpendicular is a pose the arm cannot make — it
    returns a best effort tens of millimetres away and the labels never say so.
    The surface still decides WHICH WAY the tool tilts; the arm decides how
    far."""
    base = np.array([0.0, 0.0, 1.0])
    cap = np.radians(20.0)
    for asked in (0.0, 10.0, 20.0, 35.0, 60.0):
        axis = np.array([[np.sin(np.radians(asked)), 0.0, np.cos(np.radians(asked))]])
        got = cap_lean(axis, base, cap)
        held = np.degrees(np.arccos(np.clip(got @ base, -1, 1)))[0]
        assert held <= np.degrees(cap) + 1e-6
        assert abs(held - min(asked, 20.0)) < 1e-6, (asked, held)
        assert abs(np.linalg.norm(got) - 1.0) < 1e-9        # still a direction
        assert got[0, 0] >= -1e-9                            # still tilts the same way


def test_capping_does_nothing_where_nothing_is_too_steep():
    """The paper pad never asks for more than the lean DR allows, so this must
    be invisible on every planar run."""
    rng = np.random.default_rng(3)
    tilt = np.radians(rng.uniform(0.0, 12.0, 40))            # all well under the cap
    az = rng.uniform(0, 2 * np.pi, 40)
    axes = np.stack([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)], 1)
    base = np.array([0.0, 0.0, 1.0])
    assert np.degrees(np.arccos(np.clip(axes @ base, -1, 1))).max() < 20.0
    assert np.allclose(cap_lean(axes, base, np.radians(20.0)), axes)
    assert np.allclose(cap_lean(axes, base, 0.0), axes)      # disabled is a no-op


def test_an_unplannable_scene_is_typed_so_one_batch_cannot_kill_a_run():
    """A scene that will not fit the horizon is a BATCH's problem, not a run's.

    This used to be a bare RuntimeError out of plan_batch, which propagated
    through generate and ended the process — taking every episode already
    written with it. On 2026-08-27 that cost 96 episodes of a 144-episode shard,
    and a second shard the same night. generate now catches this specific type,
    skips the batch and redraws, so the TYPE is the contract: widen it back to a
    bare RuntimeError and the run-killing behaviour returns silently.

    Forced flower of life at the training horizon is the honest trigger — about
    110 s of stroke against a budget deliberately skewed short, which is exactly
    the draw that refused in production. It refuses on most seeds but not all,
    so this sweeps a fixed seed range rather than pinning one lucky seed.
    """
    from tatbot_sim.config import DRConfig
    from tatbot_sim.language import SceneStyle

    center, rot = _frames(1)
    surf = PlanarSurface(center, rot)
    sheets = grid_paper_sheets(1, seed=1)
    style = SceneStyle(motifs=("flower_of_life",), max_motifs=1)

    caught = []
    for seed in range(24):
        try:
            plan_batch(
                np.random.default_rng(seed), sheets, surf,
                task="language", horizon=900, num_envs=1, dr=DRConfig(),
                draw_clearance=0.004, task_name="draw a {shape}",
                maze_task_name="squiggle", style=style,
            )
        except SceneTooLongError as e:
            caught.append(e)

    assert caught, "a forced flower of life at horizon 900 should refuse somewhere"
    e = caught[0]
    assert isinstance(e, RuntimeError), "old callers must still catch it"
    # It reports the sizes, so a skip can be explained rather than merely
    # survived — a run that skipped half its batches must not look full.
    # needed is None when the sampler could place nothing at any budget --
    # a different cause that used to wear the same message (and crashed the
    # error path itself, since traj was never bound).
    assert e.needed is None or e.needed > e.horizon, (e.needed, e.horizon)
    assert e.task == "language"
    assert "horizon" in str(e)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
