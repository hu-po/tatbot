"""Can the fitted tool reach everywhere the randomization sends it?

The generator's reach check probes four corners — the pad-height endpoints at
the canvas reach limit, tool perpendicular, pad level and centred. That is
enough to catch a tool that cannot work at all, and it is NOT enough to
approve a randomization: the sampled distribution also slides the pad +-20 mm
in xy, tilts it +-0.09 rad, yaws it, leans the tool +-0.12 rad off the surface
normal, and visits three heights (draw plane, hover, episode start), and each
of those pushes the wrist further than the corners do.

This audit samples the ACTUAL distribution and reports where it lands. Ink
that misses by a few millimetres does not look like a failure — the sheet
still gets marked, the video still looks like drawing — so the only way to
know a DR range is safe for a tool is to measure it before generating a
dataset, not after training on one.

Headless: IK comes from the URDF, so no renderer and no GPU are needed.

    cd python/tatbot_sim && uv run python -m tatbot_sim.audit_reach
    TATBOT_TOOL_ID=picosecond-laser-pen uv run python -m tatbot_sim.audit_reach

Seeded from the rest pose for every sample, which is pessimistic: the expert
seeds each timestep from the episode's start pose and sweeps neighbours into
agreement, so it finds branches this does not. Read a failure here as "this
region is hard", and the generator's own check as the gate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
import tyro
from transforms3d.euler import euler2mat

from tatbot_sim import tools
from tatbot_sim.config import MAX_TOOL_Z_CENTER, PAD_CENTER, DRConfig
from tatbot_sim.expert import StrokeExpert
from tatbot_sim.language import REACH
from tatbot_sim.planning import lean_normals
from tatbot_sim.strokes import ShapeConfig


@dataclass
class Args:
    samples: int = 3000
    """Random (pad pose, canvas point, height, lean) draws."""
    tolerance_mm: float = 1.0
    """Residual above which a pose counts as unreachable. Sub-line-width on
    purpose: a bigger miss still marks the sheet somewhere plausible."""
    buckets: int = 6
    """Pad-height buckets to report the pass rate across."""
    max_tool_z: float = MAX_TOOL_Z_CENTER
    """World-z ceiling on the tool, which caps the episode's start height.
    An arg rather than the class constant so it can be ablated: 0.105 was
    measured holding a TATTOO PEN perpendicular, and a longer tool has its
    own ceiling."""
    start_height_range: tuple[float, float] = ShapeConfig().start_height_range
    """Height above the pad an episode opens at, before the descent."""
    reach: float = REACH
    """Radius of the canvas area motifs are placed in, m. language.REACH is
    0.06 — the flat-reach envelope measured for a tattoo pen."""
    pad_center_x: float = float(PAD_CENTER[0])
    """How far in front of the base the pad sits, m. Where the operator puts
    the pad, not a free parameter — but it is a lever, because reaching far
    forward is what a long tool cannot do."""
    palette: bool = True
    """Also audit the DIP approach at every ink cap, for a tool that dips.

    The pad audit above says nothing about the palette: it samples a disc
    around the pad centre, and the rack is somewhere else entirely. A dip is
    the one motion that leaves the sheet, and on a large design it is most of
    the episode -- 20 dips of ~350 steps against 4,500 steps of drawing, in a
    flower-of-life take measured 2026-09-09.
    """
    palette_samples: int = 200
    """Palette-jitter draws per cap and height (config.PaletteDR)."""
    axis_tolerance_deg: float = 5.0
    """How far the tool axis may sit off the cap's entry axis. The dip plan
    enters along the palette normal (scripts/lib/dip_motion.py); the simulator
    drives the tip down and leaves the axis to IK, so this is the number that
    tells them apart."""
    ablate: bool = False
    """Zero one randomization at a time and report what each is costing."""
    recommend: bool = False
    """Search pad-height x start-height for the widest envelope that passes."""
    target_pass: float = 0.995
    """Pass fraction an envelope must clear to be recommended."""
    seed: int = 0
    iters: int = 400
    dr: DRConfig = field(default_factory=DRConfig)


def sample_poses(rng: np.random.Generator, args: Args, n: int):
    """Draw n targets the way an episode would, and their required tool axes."""
    pad, shape = args.dr.pad, ShapeConfig()
    shape = dataclasses.replace(shape, start_height_range=tuple(args.start_height_range))
    z_range = pad.z_range if pad.z_range is not None else tuple(tools.active_substrate().rest_z_m)
    lo, hi = z_range
    top_z = rng.uniform(lo, hi, n)
    xy = rng.uniform(-pad.xy_range, pad.xy_range, (n, 2))
    yaw = rng.uniform(-pad.yaw_range, pad.yaw_range, n)
    roll = rng.uniform(-pad.tilt_range, pad.tilt_range, n)
    pitch = rng.uniform(-pad.tilt_range, pad.tilt_range, n)

    # canvas point: uniform over the reach DISC, not a line out from centre
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = args.reach * np.sqrt(rng.uniform(0, 1, n))
    cx, cy = rad * np.cos(ang), rad * np.sin(ang)

    # the three heights an episode actually visits, in canvas z above the pad
    kind = rng.integers(0, 3, n)
    start_hi = np.minimum(shape.start_height_range[1], args.max_tool_z - top_z)
    start_z = rng.uniform(
        np.minimum(shape.start_height_range[0], start_hi), np.maximum(start_hi, 0.0)
    )
    cz = np.where(kind == 0, 0.0, np.where(kind == 1, shape.hover_height, start_z))
    cz = cz + 0.0  # contact-v1 puts the resolved working point on the surface

    targets = np.zeros((n, 3), dtype=np.float32)
    axes = np.zeros((n, 3), dtype=np.float32)
    seeds = np.zeros((n, 3), dtype=np.float32)
    seed_axes = np.zeros((n, 3), dtype=np.float32)
    lean_mag = rng.uniform(0, args.dr.pen_lean.max_rad, n)
    lean_dir = rng.uniform(0, 2 * np.pi, n)
    for i in range(n):
        rot = euler2mat(roll[i], pitch[i], yaw[i], "sxyz")
        top = np.array([
            args.pad_center_x + xy[i, 0],
            PAD_CENTER[1] + xy[i, 1],
            top_z[i],
        ])
        targets[i] = top + rot @ np.array([cx[i], cy[i], cz[i]])
        profile = np.array([[lean_mag[i] * np.cos(lean_dir[i]),
                             lean_mag[i] * np.sin(lean_dir[i])]])
        axes[i] = lean_normals(rot[:, 2], profile)[0]
        seeds[i] = top + rot @ np.array([0.0, 0.0, 0.004])
        seed_axes[i] = rot[:, 2]
    return targets, axes, seeds, seed_axes, top_z, kind


def pass_rate(expert, rng: np.random.Generator, args: Args, tol: float):
    """Fraction of the sampled envelope within tolerance, plus the residuals.

    IK is WARM-STARTED the way the expert starts an episode: solve the pad's
    centre at the draw plane from rest, then seed the real target from that.
    Seeding every sample from rest instead scored the ballpoint — which
    demonstrably works — at 93%, so a cold seed measures the solver's basin of
    attraction rather than the arm's reach.
    """
    targets, axes, seeds, seed_axes, top_z, kind = sample_poses(rng, args, args.samples)
    n = len(targets)
    # Build the seed in the IK chain's OWN joint order. The carriage became a
    # seventh solved axis on 2026-09-08, so a fixed six-value slice of the rest
    # keyframe silently stopped matching the chain and this audit could not run.
    names = expert.ik.chain.get_joint_parameter_names()
    staged = dict(zip((f"joint_{i}" for i in range(6)), tools.staged_pose()[:6], strict=True))
    q_rest = torch.tensor(
        [[staged.get(name, tools.carriage_rest_m()) for name in names]], dtype=torch.float32
    ).repeat(n, 1)
    q_seed = expert.ik.step(
        q_rest, torch.as_tensor(seeds), expert.target_rotations(seed_axes, n),
        iters=args.iters,
    )
    t = torch.as_tensor(targets)
    q = expert.ik.step(q_seed, t, expert.target_rotations(axes, n), iters=args.iters)
    res = torch.linalg.norm(expert.ik.fk(q)[:, :3, 3] - t, dim=-1).numpy()
    return res, res <= tol, top_z, kind


def _warm_seed(expert, n: int, args: Args):
    """The pose a dip departs from: the pad centre at the draw plane.

    pass_rate warm-starts the same way, and for the same reason — seeding from
    the rest keyframe measures the solver's basin of attraction rather than
    the arm's reach.
    """
    names = expert.ik.chain.get_joint_parameter_names()
    staged = dict(zip((f"joint_{i}" for i in range(6)), tools.staged_pose()[:6], strict=True))
    q_rest = torch.tensor(
        [[staged.get(name, tools.carriage_rest_m()) for name in names]], dtype=torch.float32
    ).repeat(n, 1)
    pad = np.repeat(
        np.array([[args.pad_center_x, float(PAD_CENTER[1]), 0.034]],
                 dtype=np.float32), n, axis=0)
    up = np.repeat(np.array([[0.0, 0.0, 1.0]]), n, axis=0)
    return expert.ik.step(q_rest, torch.as_tensor(pad),
                          expert.target_rotations(up, n), iters=args.iters)


def _solve_and_score(expert, args: Args, targets: np.ndarray, axes: np.ndarray, q_seed):
    """(position residual m, tool-axis error rad, achieved tip m) at each target."""
    n = len(targets)
    t = torch.as_tensor(targets.astype(np.float32))
    rot = expert.target_rotations(axes, n)
    q = expert.ik.step(q_seed, t, rot, iters=args.iters)
    pose = expert.ik.fk(q)
    tip = pose[:, :3, 3].numpy().astype(np.float64)
    achieved = pose[:, :3, :3].numpy().astype(np.float64)
    # The local direction the commanded rotation maps onto the entry axis; the
    # same vector carried through the achieved rotation is where the tool
    # actually points.
    commanded = rot.numpy().astype(np.float64)
    local = np.einsum("bji,bj->bi", commanded, axes)
    pointing = np.einsum("bij,bj->bi", achieved, local)
    pointing /= np.linalg.norm(pointing, axis=1, keepdims=True)
    angle = np.arccos(np.clip((pointing * axes).sum(1), -1.0, 1.0))
    return np.linalg.norm(tip - targets, axis=1), angle, tip


def run_palette(expert, args: Args, tol: float):
    """Can this tool actually enter every cap it may be sent to?

    Reports the two things a dip has to get right and the pad audit never
    looks at: the tip inside the cap's own radius, and the tool along the
    cap's entry axis. Containment is the harder gate — a small cap is 8 mm
    across, so a 5 mm miss is already outside it, while the same miss on the
    sheet would only move a stroke.
    """
    ink = tools.ink_registry()
    policy = tools.active_ink_policy()
    if not policy.dips:
        print(f"\npalette: {tools.active_tool().tool_id} never dips — nothing to audit.")
        return
    from tatbot_sim import palette as sim_palette

    scene = sim_palette.load(tools.REPO)
    layout = scene.rims
    slots = scene.palette
    source, transform = sim_palette.base_transform(tools.REPO, scene)
    # One definition of which way a cap faces, shared with the arm. dip_motion
    # names the axis INTO the cap; this audit measures against the approach,
    # which is the sign the simulator's surface normals carry.
    entry = -tools.dip_motion().palette_entry_axis(transform)
    root = np.asarray(transform[:3, 3], dtype=np.float64)

    print(f"\npalette dip approach — pose {source.upper()}"
          f", root {np.round(root, 4).tolist()} in the arm base frame")
    if source != "measured":
        print("  this is config/palette_geometry.json's synthetic scene placement,"
              " which is not where the rack is.")
    print(f"  entry axis {np.round(entry, 4).tolist()}, "
          f"hover {args.dr.palette.hover_m * 1000:.0f} mm, "
          f"axis tolerance {args.axis_tolerance_deg:.1f} deg")

    rng = np.random.default_rng(args.seed)
    n = args.palette_samples
    q_seed = _warm_seed(expert, n, args)
    axes = np.repeat(entry[None, :], n, axis=0)
    axis_tol = np.radians(args.axis_tolerance_deg)
    # the supply the planner would see, so the plunge depth matches the dip it plans
    load = tools.palette_load()

    print(f"\n{'slot':>18} {'height':>7} {'pos med mm':>11} {'pos max mm':>11} "
          f"{'axis med d':>11} {'axis max d':>11} {'in cap':>7} {'axis ok':>8}")
    worst_contained, worst_axis = 1.0, 1.0
    for name in sorted(layout):
        slot = slots[name]
        radius = slot.size.bore_diameter_m / 2.0   # the tool enters the bore, not the envelope
        fill = load[name].fill_ul if name in load else 0.0
        plunge = ink.dip_plunge_m(policy, slot, fill)
        # cap offsets are in the palette body frame, so they turn with the rack
        nominal = root + transform[:3, :3] @ np.array(layout[name], dtype=np.float64)
        # PaletteDR moves the whole rack, so every cap on it moves together.
        jitter = np.zeros((n, 3))
        if args.dr.palette.enabled:
            jitter[:, :2] = rng.uniform(-args.dr.palette.xy_jitter_m,
                                        args.dr.palette.xy_jitter_m, (n, 2))
            jitter[:, 2] = rng.uniform(-args.dr.palette.z_jitter_m,
                                       args.dr.palette.z_jitter_m, n)
        rims = nominal[None, :] + jitter
        for label, offset in (("hover", args.dr.palette.hover_m), ("plunge", -plunge)):
            targets = rims + entry[None, :] * offset
            res, ang, tip = _solve_and_score(expert, args, targets, axes, q_seed)
            # lateral miss about the cap axis is what decides containment
            delta = tip - targets
            lateral = np.linalg.norm(delta - np.outer(delta @ entry, entry), axis=1)
            inside = float((lateral < radius).mean())
            axis_ok = float((ang <= axis_tol).mean())
            if label == "plunge":
                worst_contained = min(worst_contained, inside)
                worst_axis = min(worst_axis, axis_ok)
            print(f"{name:>18} {label:>7} {np.median(res) * 1000:11.2f} "
                  f"{res.max() * 1000:11.2f} {np.degrees(np.median(ang)):11.2f} "
                  f"{np.degrees(ang.max()):11.2f} {100 * inside:6.1f}% {100 * axis_ok:7.1f}%")

    print(f"\n  worst cap, at full plunge: {100 * worst_contained:.1f}% of draws put the tip "
          f"inside the cap, {100 * worst_axis:.1f}% hold the axis within "
          f"{args.axis_tolerance_deg:.1f} deg")
    if worst_contained < args.target_pass or worst_axis < args.target_pass:
        print("  this palette placement is NOT dip-qualified for this tool: a scheduled dip"
              "\n  can be credited with the tip outside the cap. Fix the placement or the"
              "\n  tool before generating dip episodes; tatbot_sim.env credits by step index.")


def run_ablation(expert, args: Args, tol: float):
    """What is each randomization costing this tool?

    One knob relaxed to nothing at a time, against the same sample count. The
    knob whose removal recovers the most is the one to tighten first — which
    is not guessable from the ranges alone, because reach couples height,
    distance and orientation.
    """
    base = dataclasses.replace(args, ablate=False)
    variants = [("(nothing ablated)", base)]
    d = base.dr

    def with_dr(**kw):
        return dataclasses.replace(base, dr=dataclasses.replace(d, **kw))

    variants += [
        ("pad tilt -> 0", with_dr(pad=dataclasses.replace(d.pad, tilt_range=0.0))),
        ("pad xy -> 0", with_dr(pad=dataclasses.replace(d.pad, xy_range=0.0))),
        ("pad yaw -> 0", with_dr(pad=dataclasses.replace(d.pad, yaw_range=0.0))),
        ("pen lean -> 0", with_dr(pen_lean=dataclasses.replace(d.pen_lean, max_rad=0.0))),
        ("pad z -> (0, 0.01)",
         with_dr(pad=dataclasses.replace(d.pad, z_range=(0.0, 0.01)))),
        ("start height -> (0.02, 0.03)",
         dataclasses.replace(base, start_height_range=(0.02, 0.03))),
        ("max_tool_z -> 0.07", dataclasses.replace(base, max_tool_z=0.07)),
        ("canvas reach -> 0.03", dataclasses.replace(base, reach=0.03)),
        ("pad centre x -> 0.25", dataclasses.replace(base, pad_center_x=0.25)),
    ]
    print("\nablation — one randomization relaxed at a time:")
    for name, variant in variants:
        _, ok, _, kind = pass_rate(expert, np.random.default_rng(args.seed), variant, tol)
        print(f"  {name:30s} draw {100 * ok[kind == 0].mean():5.1f}%   "
              f"all {100 * ok.mean():5.1f}%")


def run_recommend(expert, args: Args, tol: float):
    """Widest pad placement this tool can actually work, height x distance.

    Those two are what move the number. Canvas reach, pad xy jitter, tilt,
    yaw and pen lean each cost about a percent, so they keep their full
    ranges and the distribution stays broad where breadth is free. Start
    height is excluded on purpose: it moves the overall figure a lot and the
    draw-plane figure not at all, because nothing is marked up there.
    """
    base = dataclasses.replace(args, ablate=False, recommend=False)
    print(f"\nrecommendation search — gated on the draw plane "
          f"(>= {100 * args.target_pass:.1f}% within {args.tolerance_mm:.1f} mm):")
    print("  pad z ceiling   pad centre x   draw-plane    overall")
    best = None
    for z_cap in (0.055, 0.04, 0.03, 0.02, 0.01):
        for cx in (0.29, 0.27, 0.25, 0.23):
            variant = dataclasses.replace(
                base,
                dr=dataclasses.replace(
                    base.dr, pad=dataclasses.replace(base.dr.pad, z_range=(0.0, z_cap))
                ),
                pad_center_x=cx,
            )
            _, ok, _, kind = pass_rate(
                expert, np.random.default_rng(args.seed), variant, tol)
            # Gate on the DRAW PLANE. That is the only height where the tool
            # marks: hover sits ~29 mm up and the episode start higher still,
            # both far outside the 5.5 mm band, so a miss up there costs
            # approach realism and cannot put pigment anywhere.
            draw = float(ok[kind == 0].mean())
            rate = float(ok.mean())
            flag = ""
            if draw >= args.target_pass:
                # prefer the widest pad-height band, then the pad kept far out
                key = (z_cap, cx)
                if best is None or key > best[0]:
                    best, flag = (key, z_cap, cx), "  <- widest so far"
            print(f"  0..{z_cap:.3f}       {cx:.3f}        "
                  f"draw {100 * draw:5.1f}%   all {100 * rate:5.1f}%{flag}")
    if best is None:
        print("  nothing in the search grid passes — the tool's geometry is the limit")
    else:
        _, z_cap, cx = best
        print(f"\n  use: --dr.pad.z-range 0.0 {z_cap:.3f}  with the pad "
              f"{cx:.3f} m in front of the base "
              f"(PAD_CENTER x is {PAD_CENTER[0]:.3f})")


def main(args: Args):
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")
    expert = StrokeExpert(1, device)
    tol = args.tolerance_mm / 1000.0

    # This audit never builds an env, so nothing else fills in the ranges the
    # fitted substrate owns -- and auditing reach over heights no run samples
    # is worse than not auditing it at all.
    args.dr.resolve_for(tools.active_substrate())

    tool = tools.active_tool()
    print(f"tool: {tool.tool_id}  kind={tool.kind}  "
          f"protrusion={tool.protrusion_m * 1000:.0f} mm")
    print(f"pad z_range={args.dr.pad.z_range} xy=+-{args.dr.pad.xy_range} "
          f"tilt=+-{args.dr.pad.tilt_range} lean=+-{args.dr.pen_lean.max_rad}")

    res, ok, top_z, kind = pass_rate(expert, rng, args, tol)
    print(f"\noverall: {100 * ok.mean():.1f}% within {args.tolerance_mm:.1f} mm "
          f"| median {np.median(res) * 1000:.2f} mm  p95 {np.percentile(res, 95) * 1000:.2f} mm"
          f"  max {res.max() * 1000:.1f} mm")

    print("\nby pad top height:")
    pad_z_range = args.dr.pad.z_range if args.dr.pad.z_range is not None else tuple(tools.active_substrate().rest_z_m)
    lo, hi = pad_z_range
    edges = np.linspace(lo, hi, args.buckets + 1)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (top_z >= a) & (top_z <= b if b == edges[-1] else top_z < b)
        if not m.any():
            continue
        print(f"  {a:.3f}-{b:.3f}: {100 * ok[m].mean():5.1f}% ok   "
              f"median {np.median(res[m]) * 1000:6.2f} mm   max {res[m].max() * 1000:6.1f} mm")

    print("\nby height the tool is at:")
    for k, name in ((0, "draw plane"), (1, "hover"), (2, "episode start")):
        m = kind == k
        print(f"  {name:14s}: {100 * ok[m].mean():5.1f}% ok   "
              f"median {np.median(res[m]) * 1000:6.2f} mm   max {res[m].max() * 1000:6.1f} mm")

    # the ceiling that would make the whole distribution pass
    bad = top_z[~ok]
    if len(bad) == 0:
        print("\nthe full sampled envelope is reachable — no z ceiling needed.")
    else:
        print(f"\nlowest pad top that FAILS: {bad.min():.4f} m "
              f"({len(bad)} of {args.samples} samples fail, worst at z={top_z[np.argmax(res)]:.3f})")
        best = None
        for cap in np.linspace(lo, hi, 23):
            m = top_z <= cap
            if m.sum() >= 30 and ok[m].all():
                best = cap
        print("highest all-pass ceiling over these samples: "
              + (f"{best:.3f} m" if best is not None else "none — even the floor fails"))

    if args.palette:
        run_palette(expert, args, tol)
    if args.ablate:
        run_ablation(expert, args, tol)
    if args.recommend:
        run_recommend(expert, args, tol)


if __name__ == "__main__":
    main(tyro.cli(Args))
