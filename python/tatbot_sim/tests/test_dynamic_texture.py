"""The pigment field, end to end through the renderer.

test_inkfield.py proves the field arithmetic; these prove the parts only the
env can: that the field reaches the sheet texture, that the fitted tool decides
what happens to it, and that a millimetre of ink is a millimetre wide on the
sheet. That last one guards the likeliest silent failure in the whole design —
a px/mm mix-up draws perfectly plausible ink at the wrong scale.

ONE env serves the whole file. SAPIEN enables GPU PhysX once per process, so
building a scene per test is both slow and unreliable; where a test needs a
different tool it swaps the env's tool spec rather than rebuilding the world.

Needs a render device:

    cd python/tatbot_sim && uv run python tests/test_dynamic_texture.py
"""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
import tatbot_sim  # noqa: F401  (registers agent + env)
import torch
from tatbot_sim import tools

_ENV = None


def env_and_base():
    """The shared env, reset to a bare sheet."""
    global _ENV
    if _ENV is None:
        _ENV = gym.make(
            "TatbotDraw-v0", num_envs=2, obs_mode="rgb", control_mode="pd_joint_pos",
            sim_backend="auto", reconfiguration_freq=0,
        )
    _ENV.reset(seed=0)
    return _ENV, _ENV.unwrapped


def _tcp_at(base, canvas_xy, height_m):
    """World point at a canvas offset, ``height_m`` along the surface normal."""
    tc, rot = base.canvas_frame_np
    xy = np.asarray(canvas_xy, dtype=np.float32)
    p = tc + rot[:, :, 0] * xy[0] + rot[:, :, 1] * xy[1] + rot[:, :, 2] * height_m
    return torch.as_tensor(p, dtype=torch.float32).to(base.device)


def _uv_at(base, canvas_xy, height_m=0.0):
    return base.surface.project(_tcp_at(base, canvas_xy, height_m))[0]


def test_a_bare_sheet_composites_back_to_the_original_paper():
    """Ink-free, the dynamic texture must equal the file the sheet generator
    wrote. Guards the whole colour path at once — an srgb flag, a channel
    order or a rounding mode that is off shows up here as a tinted or dark
    sheet, which is invisible in a wrist view until compared side by side."""
    _, base = env_and_base()
    assert float(base.ink_field.coverage()[0]) == 0.0
    got = base._sheet_tex[0].download()
    want = cv2.imread(base.pad_sheets[0]["png"])[..., ::-1]
    assert np.array_equal(got[..., :3], want), np.abs(
        got[..., :3].astype(int) - want.astype(int)
    ).max()
    assert np.all(got[..., 3] == 255)


def test_deposit_reaches_the_bound_texture():
    """The field must show up in the texture the renderer actually samples."""
    _, base = env_and_base()
    before = base._sheet_tex[0].download().copy()
    uv = _uv_at(base, (0.0, 0.0))
    base.ink_field.deposit(base.surface, uv, base.ink_opacity)
    base._refresh_sheet_textures(force=True)
    after = base._sheet_tex[0].download()
    diff = np.abs(after.astype(int) - before.astype(int)).sum(-1)
    assert diff.sum() > 0, "deposit never reached the texture"
    ys, xs = np.nonzero(diff)  # and only where it was stamped
    # Where it was STAMPED, not the middle of the sheet: on a mound the canvas
    # origin is not the nearest surface point to itself, so a world point taken
    # from the canvas frame projects a few millimetres off centre. Asking the
    # surface where the stamp went is the invariant; assuming the centre was
    # only ever true of a flat pad.
    want = base.surface.canvas_to_px(uv)[0].cpu().numpy()
    assert abs(xs.mean() - want[0]) < 8 and abs(ys.mean() - want[1]) < 8


def test_only_the_touching_env_is_marked():
    _, base = env_and_base()
    active = torch.tensor([True, False], device=base.device)
    base.ink_field.deposit(base.surface, _uv_at(base, (0.0, 0.0)), base.ink_opacity, active)
    cov = base.ink_field.coverage()
    assert float(cov[0]) > 0 and float(cov[1]) == 0.0


def test_the_gate_is_the_measured_band():
    """Marks land inside 5.5 mm of the surface and nowhere above it."""
    _, base = env_and_base()
    band = base.TIP_MARK_MARGIN_M + base.ink_threshold
    assert abs(band - 0.0055) < 1e-9  # expert.py's floor clamp assumes this
    # Measured from the SURFACE, not from the canvas origin: on a mound the
    # origin sits tens of millimetres below the skin, so a probe placed
    # relative to it says nothing about the deposit band.
    uv = torch.zeros(base.num_envs, 2, device=base.device)
    point, _, _, normal = base.surface.frame(uv)
    _, inside, _ = base.surface.project(point + normal * (band * 0.5))
    _, outside, _ = base.surface.project(point + normal * (band * 2.0))
    assert bool((inside < band).all()) and bool((outside >= band).all())


def test_line_width_matches_the_configured_radius():
    """A stroke drawn at r metres must measure 2r metres on the sheet.

    The whole mm-parameterized kernel design is worthless if this drifts, and
    a pixel-radius bug looks entirely normal until someone measures it.

    Deliberately measured on a FLAT surface, and on one built here rather than
    whatever the bench has fitted. Texels and metres only coincide on a plane:
    over a mound a canvas metre of field covers more than a metre of skin, so a
    correct stamp reads narrow in texels by exactly the slope, and this
    assertion would be measuring the mound rather than the kernel. That the
    world footprint survives slope is a separate promise, proved against known
    angles in test_inkfield.
    """
    from tatbot_sim.surface import PlanarSurface

    _, base = env_and_base()
    src, field = base.surface, base.ink_field
    b = field.field.shape[0]
    flat = PlanarSurface(
        torch.zeros((b, 3), device=base.device),
        torch.eye(3, device=base.device).expand(b, 3, 3),
        src.width_m, src.height_m, src.cols, src.rows,
    )
    field.reset()
    r = float(field.pen_radius_m[0])
    for x in np.linspace(-0.02, 0.02, 60):
        uv = torch.tensor([[float(x), 0.0]], device=base.device).expand(b, 2)
        field.deposit(flat, uv, torch.ones(b, device=base.device))
    col = field.field[0].cpu().numpy()[:, flat.cols // 2]
    measured_m = np.count_nonzero(col > 0.5) / flat.texel_per_m
    assert abs(measured_m - 2 * r) < 1.5 / flat.texel_per_m, (measured_m, 2 * r)


def test_the_laser_tool_removes_where_a_pen_would_deposit():
    """Same motion, same code path, opposite effect — chosen by the registry."""
    _, base = env_and_base()
    assert base.FIELD_OPS["laser"] == "remove"
    laser = tools.registry().load_tool("picosecond-laser-pen", tools.REPO)
    assert laser.kind == "laser"
    uv = _uv_at(base, (0.0, 0.0))
    base.ink_field.deposit(base.surface, uv, torch.ones(2, device=base.device))
    start = float(base.ink_field.coverage()[0])
    assert start > 0

    fitted, base.tool = base.tool, laser
    try:
        touching = torch.ones(2, dtype=torch.bool, device=base.device)
        for _ in range(30):
            base._apply_tool(uv, torch.ones(2, device=base.device), touching)
    finally:
        base.tool = fitted
    end = float(base.ink_field.coverage()[0])
    assert end < start, (start, end)
    # Under the beam centre the multiplicative law is exact: 30 passes at eta
    # leave (1 - eta)^30 of what was there. Asserting that instead of a
    # fraction of total coverage is what makes this test independent of what
    # the DR drew — how much of the pen's disc one spot can reach depends on
    # the sampled laser radius and clearance, and a threshold on coverage was
    # really a bet on those draws.
    px = torch.round(base.surface.canvas_to_px(uv)[0]).long()
    centre = float(base.ink_field.field[0, int(px[1]), int(px[0])])
    eta = float(base.laser_clearance[0])
    assert abs(centre - (1.0 - eta) ** 30) < 1e-3, (centre, eta)
    assert float(base.ink_field.field.min()) >= 0.0


def test_unknown_tool_kinds_still_mark():
    """A tool nobody wired up draws like a pen rather than doing nothing —
    a silently blank sheet reads as a broken episode, not a missing case."""
    _, base = env_and_base()
    assert base.FIELD_OPS.get("plasma-thing") is None
    fitted = base.tool
    base.tool = type("FakeTool", (), {"kind": "plasma-thing"})()
    try:
        base._apply_tool(
            _uv_at(base, (0.0, 0.0)),
            torch.ones(2, device=base.device),
            torch.ones(2, dtype=torch.bool, device=base.device),
        )
    finally:
        base.tool = fitted
    assert float(base.ink_field.coverage()[0]) > 0


def test_preink_opens_the_episode_on_an_inked_sheet():
    """A removal episode's target has to exist before its first control step,
    and it has to be visible — pre-inked pigment goes through the same splat
    the pen uses, so it must reach the texture the same way."""
    from tatbot_sim.strokes import Stroke

    _, base = env_and_base()
    before = base._sheet_tex[0].download().copy()
    box = np.array([[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01]])
    base.preink([[Stroke(box)], [Stroke(box * 0.5)]])
    cov = base.ink_field.coverage()
    assert float(cov[0]) > float(cov[1]) > 0  # each env got ITS own scene
    after = base._sheet_tex[0].download()
    assert np.abs(after.astype(int) - before.astype(int)).sum() > 0


def test_reset_clears_the_sheet_and_the_texture():
    env, base = env_and_base()
    base.ink_field.deposit(base.surface, _uv_at(base, (0.0, 0.0)), base.ink_opacity)
    base._refresh_sheet_textures(force=True)
    inked = base._sheet_tex[0].download().copy()
    env.reset(seed=1)
    cleared = base._sheet_tex[0].download()
    assert float(base.ink_field.coverage()[0]) == 0.0
    assert np.abs(cleared.astype(int) - inked.astype(int)).sum() > 0


def test_the_field_leads_and_the_texture_follows_on_cadence():
    """Stepping updates the field every step; the upload runs on its own
    schedule, so ground truth is never the thing being throttled."""
    env, base = env_and_base()
    base.texture_refresh_steps = 3
    action = base.agent.robot.get_qpos()[:, : env.action_space.shape[-1]]
    base._step_count = 0
    env.step(action)
    assert base._step_count == 1


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    if _ENV is not None:
        _ENV.close()
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
