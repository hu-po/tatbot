#!/usr/bin/env python3
"""Path-traced takes of a distribution, for showing the work outside the lab.

The training renders are 640x480, rasterized, matched to the real D405s, and lit
by a randomized rig that deliberately includes ugly draws — all correct for a
policy and all wrong for a video. This runs one episode of a named distribution
(tatbot_sim.distributions, the same recipe the factory generates from) through
SAPIEN's path tracer instead, at whatever resolution and aspect the destination
wants, from cameras chosen to be looked at rather than learned from, under a
fixed studio rig.

Nothing here can affect training data: it writes video and nothing else.

Run on an x86_64 sim host in the tatbot_sim venv:

    .venv/bin/python ../../scripts/sim_cinematic.py skin-tattoo \\
        --out ~/renders/tattoo --seed 4 --shots hero pov --aspect vertical

**It plans its own take rather than replaying a recorded one**, and that is a
constraint rather than a preference. Ray tracing runs ONE environment per
process (ManiSkill refuses more), while datasets are generated many envs at a
time — and the per-env surface is drawn from the RNG in a way a single-env
reset does not reproduce, so the recorded trajectory lands somewhere the skin
is not. Measured on 2026-08-27: replaying episode 2 of a 4-env batch put the
pad 20 mm from where that episode was recorded against, and the tool drew
beside the skin for the whole take. The height field that would be needed to
re-stage it faithfully is not recorded, so the honest options were "plan a
fresh take" or "record the surface too", and a film does not need the second.

A take is reproducible by ``--seed``: same seed, same scene, same drawing.

Cost scales with pixels x samples x cameras. `rt-med` (4 samples per pixel,
OptiX-denoised) is the default and is what the 2026-08-27 renders were made
at: clean at 1080x1920, about 4-5 control steps a second with three cameras on
a 3090. `rt` raises that to 32 samples — reach for it on a still that will be
scrutinized, not on a clip, and expect it to be several times slower. The two
have not been compared side by side.

Long episodes want `--speed`, which renders fewer steps for a faster playback:
a laser removal that really takes a minute is both better television and four
times cheaper to render at 4x.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import sapien
import tyro
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from tatbot_sim.distributions import DISTRIBUTIONS
from tatbot_sim.env import TatbotDrawEnv

NOMINAL_WRIST_FOV = 0.96

# Where the work happens: the canvas centre, a little above the table. Every
# staged camera aims here, so a shot stays framed when the pad moves.
LOOK_AT = np.array([0.29, 0.0, 0.03])

ASPECTS = {
    "vertical": (1080, 1920),   # reels / shorts / tiktok
    "square": (1440, 1440),     # feed
    "wide": (1920, 1080),       # youtube / landing pages
}


@dataclass
class Shot:
    """A camera to be looked at. ``track`` moves it during the episode."""

    fov: float
    eye: tuple[float, float, float] | None = None
    mounted: bool = False
    """Mounted shots ride the robot — the POV is the wrist camera itself,
    at render resolution instead of the D405's."""
    track: str | None = None
    """None = locked off. "orbit" = swing around the work over the episode.
    "tool" = ride the tool tip at a fixed offset, which is the only way a
    close shot survives a tool that wanders the whole canvas."""
    offset: tuple[float, float, float] = (0.10, -0.09, 0.055)
    """Where a tool-tracking camera sits relative to the tip."""
    smooth: float = 0.12
    """Low-pass on a tracking camera's aim, per step. The tip's own motion is
    jittery at 30 Hz and a camera welded to it reads as a shaky handheld;
    gliding after it reads as a camera operator."""
    orbit_deg: float = 95.0
    up: tuple[float, float, float] = (0, 0, 1)
    mount: str = "upper"
    """Which wrist camera a mounted shot rides."""
    at: tuple[float, float, float] | None = None
    """Where a locked-off shot looks. None = LOOK_AT, the canvas centre."""


def mounted_fov(width: int, height: int) -> float:
    """Vertical FOV that keeps the real wrist camera's HORIZONTAL view.

    ``fov`` is vertical, and the D405's 0.96 rad was chosen against a 4:3
    frame. Render that same number at 9:16 and the horizontal view collapses:
    measured, the POV shot became the tool's own body with the drawing out of
    frame on both sides. Widening the vertical angle instead keeps every
    object the real camera can see and adds sky and table above and below,
    which is what a taller frame should do.
    """
    fov_x = 2 * np.arctan(np.tan(NOMINAL_WRIST_FOV / 2) * 640 / 480)
    return float(2 * np.arctan(np.tan(fov_x / 2) * height / width))


# Framing note: `fov` is the VERTICAL field of view, so the same number frames
# very differently at 9:16 than at 16:9 — these are staged for vertical, where
# the subject is the tool meeting the surface and the arm is context above it.
# A long lens (small fov) from a normal distance flatters hard-surface
# geometry; backing off with a wide one puts the whole bench in shot.
SHOTS: dict[str, Shot] = {
    # The money shot: the tool on the surface, arm entering from the top.
    "hero": Shot(fov=0.52, eye=(0.60, -0.34, 0.33)),
    # The whole machine at work, for context and for the first second of a cut.
    "wide": Shot(fov=0.95, eye=(0.88, -0.58, 0.46)),
    # A camera move rather than a robot move, which is what makes a scene
    # where only a pen tip moves watchable for more than a few seconds.
    "orbit": Shot(fov=0.62, eye=(0.62, -0.36, 0.34), track="orbit"),
    # What the robot sees. The upper wrist camera, rendered at social
    # resolution instead of the D405's 640x480 — see mounted_fov for why the
    # angle has to change when the frame stops being 4:3.
    "pov": Shot(fov=NOMINAL_WRIST_FOV, mounted=True),
    # The lower wrist camera: the same view from under the tool, which on a
    # mound sees the contact point the upper one is looking over.
    "pov_low": Shot(fov=NOMINAL_WRIST_FOV, mounted=True, mount="lower"),
    # Close on the contact point, riding the tip. The best shot of the three
    # for a feed: it is the needle drawing, filling the frame, for as long as
    # the episode lasts.
    "macro": Shot(fov=0.60, track="tool", offset=(0.082, -0.078, 0.032)),
    # The rack. A locked-off camera on the ink caps: the tool arrives, dips,
    # and leaves, and between dips the frame is the palette waiting. Framed
    # from the arm's right-front so the caps are seen past the tool, not
    # through it. PALETTE_AT is the rack's centre in the arm base frame.
    "palette": Shot(fov=0.42, eye=(0.46, 0.05, 0.30), at=(0.126, 0.2675, 0.085)),
    # Close on the dip itself, riding the tip from the pad-side so the cap
    # and the needle entering it fill the frame; the same tracking as macro
    # with the camera on the other side of the tool.
    "dip": Shot(fov=0.55, track="tool", offset=(0.070, -0.060, 0.045)),
}


@dataclass
class Args:
    out: str
    """Output directory; one mp4 per shot plus a poster frame."""
    sensor_profile: str = "deployment"
    seed: int = 0
    """Picks the take: the scene, the drawing, and the tool's path. Same seed,
    same film."""
    task: str = ""
    """Override the distribution's task ("language", "maze", "erase"). Empty
    keeps the recipe's own — except for a recipe whose task is "mix", which
    is a per-batch coin flip and has to be resolved to one here."""
    horizon: int = 0
    """Override the recipe's episode length. 0 keeps it."""
    shots: tuple[str, ...] = ("hero", "pov")
    shader: str = "rt-med"
    """rt (32 samples), rt-med (4), rt-fast (2), or default for rasterized."""
    aspect: str = "vertical"
    width: int = 0
    height: int = 0
    """Explicit size; overrides --aspect when both are non-zero."""
    fps: int = 30
    """Frame rate of the file."""
    speed: float = 1.0
    """Playback speed against the real thing. 1 is real time; 4 turns a
    minute of laser work into fifteen seconds. Control runs at 30 Hz, so this
    and --fps together decide how many steps get rendered — which is where
    the render time goes, so a time-lapse is also four times cheaper."""
    max_frames: int = 1200
    crf: int = 17
    """x264 quality. 17 is visually lossless; raise it for smaller files."""
    clutter: bool = True
    """Legacy and inert: the scene no longer builds table distractors
    (config.ClutterDR is kept only so old configs load); accepted so existing
    command lines still parse."""
    studio_light: bool = True
    """Replace the randomized training rig with a fixed key/fill/rim setup."""
    exposure: float = 1.5
    """Scales the whole studio rig. Path tracing responds to light very
    differently from the rasterizer the training levels were tuned under."""
    look: str = "clean"
    """clean = studio surfaces and a softbox environment. bench = the training
    draw's dark, patterned, deliberately noisy textures."""
    design: str = "clean"
    """What gets drawn. "clean" = deliberate geometry only, no wavy or dashed
    treatment, no nesting (language.CLEAN_STYLE). "full" = the training draw,
    scribble and all. Any motif key ("flower_of_life", "seed_of_life",
    "circle", ...) forces that one figure — see language.MOTIFS."""
    portable_design: str | None = None
    """A tatbot.inkmap-design/1 file to draw instead of the shared collection
    — generate's --design, for the artwork task (tatbot_sim.design_scene).
    A cylinder design needs TATBOT_SUBSTRATE=paper_cylinder; the take's
    --horizon has to hold the whole design, and the run says what it needs."""
    design_placement: str = "authored"
    """authored keeps Inkmap's placement; sampled recentres and offsets it."""
    render_skipped_steps: bool = False
    """Render the sensors on every control step, captured or not. Off, a
    time-lapse only renders the steps it keeps and physics runs the rest,
    which is where the speed-up in --speed actually comes from."""
    steady: bool = True
    """Drop the DART perturbation bursts and most of the pen lean. They are
    deployment realism the policy needs to have seen; on camera they read as
    a shaky hand, and the drawing wobbles off its own geometry."""
    surface_profile: str = ""
    """Override the recipe's surface profile: "flat", "cylinder" or
    "balanced". Empty keeps the recipe's. paper-draw is flat; the skin
    recipes are balanced, which is a per-batch coin flip a single-env take
    cannot be asked to win. Naming the profile is how a film chooses."""
    cylinder_radius: tuple[float, float] = (0.0, 0.0)
    """Override the sampled cylinder radius range, metres. (0, 0) keeps the
    recipe's 75-110 mm. A tighter curl is a harder reach and not merely a
    different picture — check the reachable fraction the run prints."""
    dips: bool = False
    """Splice planned dips into the take. Drawing recipes leave them off
    (config.InkDR.dips), so the palette and dip shots have nothing to film
    without this; --task dip is the other way in. Needs a palette in the
    scene and a tool whose ink.mode dips."""
    charge: tuple[float, float] = (0.05, 0.30)
    """Per-env multiplier on the tool's datasheet charge capacity, with
    --dips. At the nominal 2 uL and 0.004 uL/mm a charge covers ~500 mm of
    line, far more than one take draws, so at 1.0 the tool never runs dry and
    no dip is ever planned."""
    ink_radius: tuple[float, float] = (0.0, 0.0)
    """Override the deposited line's half-width, metres. (0, 0) keeps the
    recipe's. The skin-tattoo recipe borrows the BALLPOINT's 1.1-2.0 mm, which
    draws a 2.2-4 mm line — wider than the interior spacing of an intricate
    figure, so a flower of life comes out a solid blob (measured, 2026-08-27).
    A real 3RL is a 0.30 mm needle; 0.0004-0.0006 reads as a liner and is
    about as thin as the pigment field's ~2.4 px/mm can carry."""
    clearance: tuple[float, float] = (0.0, 0.0)
    """Override the laser's per-pass clearance (fraction of remaining pigment
    a pass removes). (0, 0) keeps the recipe's. The recipe's 0.08-0.25 is
    tuned so a dataset spans partly-faded through nearly-clean over a full
    episode; a clip wants the ink visibly going."""
    wet: str = ""
    """Fill every right-arm cap with this ink for the take (an ink_id from
    config/inks.yaml). Shorthand for --supply wet --supply-ink <id>; writes
    nothing, the override lives in this process."""
    supply: str = "bench"
    """Which palette load the take sees (tatbot_sim.tools.set_supply): "bench"
    is config/palette_load.yaml as it is today — a film shows the rack as
    poured; "wet" fills every right-arm cap with --supply-ink; "dry" empties
    them. A real needle refuses dry caps, and a film of the 3RL should show
    ink in them; the ballpoint rehearses dry unless told otherwise."""
    supply_ink: str = "nighthawk_black"



def make_studio_rig(exposure: float):
    """A deliberate three-light setup, in place of the training draw.

    The training rig randomizes count, direction, colour and intensity because
    a policy has to survive a desk lamp and a window. A render wants the
    opposite: the same flattering light every time, so successive takes cut
    together and the tool reads against the surface.

    ``exposure`` scales the whole rig at once. Path tracing responds to light
    very differently from the rasterizer these levels were tuned under — the
    training rig's intensities blow a path-traced skin to paper white — so the
    number that matters is set here and swept from the command line, not
    buried per light.
    """
    def rig(env, options):
        scene = env.scene
        scene.set_ambient_light([0.02 * exposure] * 3)
        key = (LOOK_AT + np.array([0.45, -0.55, 0.75])).tolist()
        # Key: warm, from the operator's shoulder, the only shadow caster — one
        # crisp shadow reads as a lit bench; three read as a bug.
        scene.add_point_light(key, [c * exposure for c in (1.5, 1.42, 1.30)],
                              shadow=True, shadow_map_size=2048)
        # Fill: cool, opposite side, no shadow, a third of the key so the
        # shadow side keeps detail without going flat.
        scene.add_point_light((LOOK_AT + np.array([-0.30, 0.55, 0.45])).tolist(),
                              [c * exposure for c in (0.42, 0.46, 0.55)], shadow=False)
        # Rim: from behind and above, to lift the black arm off the background.
        scene.add_directional_light([-0.35, -0.45, -0.85],
                                    [c * exposure for c in (0.30, 0.30, 0.33)],
                                    shadow=False)
    return rig


def clean_surfaces(seed: int = 0):
    """Swap the bench's procedural clutter of textures for studio surfaces.

    The floor and tabletop textures are drawn dark, tinted and patterned on
    purpose: what a wrist camera mostly sees is a black self-healing cutting
    mat, and the training set has to contain it. Under a path tracer at social
    sizes that same speckle reads as colour noise, and the eye goes to it
    instead of the tool.

    Return instance-local texture suppliers for explicit world construction.
    """
    from tatbot_sim.textures import TEX_DIR

    out_dir = TEX_DIR / "cinematic"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = out_dir / f"table_matte-{seed}.png"
    if not table.exists():
        n = 1024
        rng = np.random.default_rng(seed)
        # Near-uniform warm grey with grain far too fine to read as pattern:
        # a surface, not a subject.
        img = np.ones((n, n, 3)) * np.array([0.20, 0.195, 0.185])
        img += rng.normal(0, 0.006, (n, n, 3))
        _cv2().imwrite(str(table), np.clip(img[..., ::-1] * 255, 0, 255).astype(np.uint8))

    faces = []
    for name in ("px", "nx", "py", "ny", "pz", "nz"):
        f = out_dir / f"studio_{name}.png"
        faces.append(str(f))
        if f.exists():
            continue
        n = 256
        # A softbox room: bright above, falling off to a dark floor. This is
        # the scene's ambience, and on a path tracer it does most of the work
        # that a fill light does on a rasterizer.
        if name == "pz":       # up
            img = np.ones((n, n, 3)) * 0.62
        elif name == "nz":     # down
            img = np.ones((n, n, 3)) * 0.05
        else:
            ramp = np.linspace(0.52, 0.07, n)[:, None, None]
            img = np.ones((n, n, 3)) * ramp
        _cv2().imwrite(str(f), np.clip(img[..., ::-1] * 255, 0, 255).astype(np.uint8))

    return (lambda count, seed=0: [str(table)] * count,
            lambda count, seed=0: [tuple(faces)] * count)


def _cv2():
    import cv2
    return cv2


def tool_pose(shot: Shot, tip: np.ndarray) -> sapien.Pose:
    """Camera pose for a tool-tracking shot, aimed at ``tip``."""
    return sapien_utils.look_at(eye=(tip + np.array(shot.offset)).tolist(),
                                target=tip.tolist(), up=list(shot.up)).sp


def build_cameras(names, width, height):
    """The staged cameras, as ManiSkill sensor configs.

    Mounted presentation shots are added separately from the resolved wrist
    pose, with their own image size and feature name.
    """
    out = []
    for name in names:
        shot = SHOTS[name]
        if shot.mounted:
            continue
        # A tracking camera is placed every step; its config only needs a
        # valid starting pose, so it begins looking at the work like the rest.
        at = np.array(shot.at) if shot.at else LOOK_AT
        eye = list(shot.eye) if shot.eye else (at + np.array(shot.offset)).tolist()
        out.append(CameraConfig(
            uid=f"cine_{name}",
            pose=sapien_utils.look_at(eye=eye, target=at.tolist(), up=list(shot.up)),
            width=width, height=height, fov=shot.fov, near=0.005, far=100,
        ))
    return out


def orbit_pose(shot: Shot, phase: float, look_at=LOOK_AT) -> sapien.Pose:
    """Camera pose at ``phase`` in [0, 1] through the arc, about ``look_at``.

    Swings symmetrically about the shot's own eye so the middle of the episode
    is the framing the shot was staged for, and the ends are equal departures
    from it.
    """
    look_at = np.asarray(look_at, dtype=np.float64)
    eye = np.array(shot.eye) - LOOK_AT
    a = np.radians(shot.orbit_deg) * (phase - 0.5)
    c, s = np.cos(a), np.sin(a)
    rotated = np.array([c * eye[0] - s * eye[1], s * eye[0] + c * eye[1], eye[2]])
    # `.sp` unwraps ManiSkill's batched Pose into the plain sapien one the
    # render component takes; handing it the wrapper is a TypeError at runtime.
    return sapien_utils.look_at(eye=(look_at + rotated).tolist(),
                                target=look_at.tolist(), up=list(shot.up)).sp


def encode(frames, path: Path, fps: float, crf: int):
    """h264 in an mp4, faststart, yuv420p — what every platform ingests."""
    import av

    h, w = frames[0].shape[:2]
    # h264 requires even dimensions; a stray odd pixel is not worth a failure
    w, h = w - (w % 2), h - (h % 2)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "slow", "movflags": "+faststart"}
    for f in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()



def presentation(args, config, width, height):
    """Separate display cameras and appearance from the policy sensor profile."""
    lighting = make_studio_rig(args.exposure) if args.studio_light else None
    surfaces = None
    if args.look == 'clean':
        surfaces = clean_surfaces(seed=args.seed)
    elif args.look != 'bench':
        raise SystemExit(f'unknown look {args.look!r}; have clean, bench')
    mounted = [n for n in args.shots if SHOTS[n].mounted]
    staged = [n for n in args.shots if not SHOTS[n].mounted]
    available = {camera.role for camera in config.cameras}
    for name in mounted:
        if f'wrist_{SHOTS[name].mount}' not in available:
            raise SystemExit(f'{name} requires its wrist view in --sensor-profile')

    def cameras(world):
        views = build_cameras(staged, width, height)
        policy_views = {camera.uid: camera for camera in world.agent._sensor_configs}
        for name in mounted:
            view = policy_views[f'wrist_{SHOTS[name].mount}']
            views.append(replace(view, uid=f'cine_{name}', width=width, height=height,
                                 intrinsic=None, fov=mounted_fov(width, height)))
        return views

    return cameras, lighting, surfaces, staged


def main(args: Args, dist):
    for name in args.shots:
        if name not in SHOTS:
            raise SystemExit(f"unknown shot {name!r}; have {', '.join(SHOTS)}")
    if args.aspect not in ASPECTS and not (args.width and args.height):
        raise SystemExit(f"unknown aspect {args.aspect!r}; have {', '.join(ASPECTS)}")
    width, height = ((args.width, args.height) if args.width and args.height
                     else ASPECTS[args.aspect])
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from tatbot_sim import tasks
    from tatbot_sim.backends.maniskill import ManiSkillWorld
    from tatbot_sim.episode import Episode
    from tatbot_sim.expert import (
        StrokeExpert,
        reachable_canvas_masks,
        reachable_height_ceiling,
    )
    from tatbot_sim.judge import strokes_from_plan_paths
    from tatbot_sim.observations import ObservationBuilder
    from tatbot_sim.planning import plan_batch

    recipe = dist.build_args()
    task = args.task or recipe.task
    if args.portable_design and task not in ("artwork", "erase"):
        raise SystemExit(f"--portable-design draws the artwork task; pass --task artwork (got {task!r})")
    if task == "mix":
        # A mix is a per-batch coin flip over language and squiggles. A film
        # is one take, so the flip has to be made here rather than by an RNG
        # whose result nobody chose.
        task = "language"
    horizon = args.horizon or recipe.horizon
    from tatbot_sim.resolved import resolve
    supply = ('wet' if args.wet else args.supply, args.wet or args.supply_ink)
    config = resolve(tool_id=dist.tool_id, sensor_profile=args.sensor_profile,
                     seed=args.seed, dr=recipe.dr, supply=supply)
    tool, substrate = config.tool, config.substrate
    tasks.validate_task(task, tool, substrate)
    tasks.validate_supply(task, tool, config.palette_load)

    from tatbot_sim.language import CLEAN_STYLE, DEFAULT_STYLE, MOTIFS, SceneStyle
    if args.design == "clean":
        style = CLEAN_STYLE
    elif args.design == "full":
        style = DEFAULT_STYLE
    elif args.design in MOTIFS:
        # One figure, drawn deliberately: the scene is that motif and nothing
        # else, at its largest allowed size.
        # One figure, and the whole episode to draw it in: a forced motif is
        # the point of the take, so it should not lose a coin flip for budget.
        style = SceneStyle(style_prob=0.0, nest_prob=0.0, max_motifs=1,
                           motifs=(args.design,), fill_budget=True)
    else:
        raise SystemExit(
            f"unknown design {args.design!r}; use clean, full, or a motif key: "
            + ", ".join(sorted(MOTIFS)))
    # planning.plan_batch only consults ``style`` for the language task; artwork,
    # spiral and erase scenes come from the shared collection and ignore it. Say
    # so rather than printing a design the take never drew.
    if args.design != "clean" and task in ("artwork", "spiral", "erase"):
        raise SystemExit(
            f"--design {args.design!r} does nothing for task {task!r}: that scene comes "
            "from the shared artwork collection, not language motifs. Add --task language "
            "to draw a motif.")

    dr = recipe.dr.resolve_for(substrate)
    if args.surface_profile:
        if args.surface_profile not in ("flat", "cylinder", "balanced"):
            raise SystemExit(f"unknown surface profile {args.surface_profile!r}; "
                             "use flat, cylinder or balanced")
        dr.surface.profile = args.surface_profile
    if any(args.cylinder_radius):
        dr.surface.cylinder_radius_m = tuple(args.cylinder_radius)
    if args.dips:
        # A take draws far less line than one charge covers, so dipping has to
        # be asked for AND the reservoir shrunk, or nothing is ever planned.
        dr.ink.dips = True
        dr.ink.capacity_scale = tuple(args.charge)
    dr.clutter.enabled = args.clutter
    dr.rgb.enabled = False        # sensor response modelling, not a look
    dr.corrupt_depth = False
    dr.camera.mount_jitter_mm = 0.0
    dr.camera.mount_jitter_deg = 0.0
    if args.steady:
        dr.noise.prob = (0.0, 0.0)          # no knock-off-the-stroke bursts
        dr.pen_lean.max_rad = 0.02          # the tool stays where it is aimed
        dr.approach.pose_jitter_rad = 0.0
    if args.clearance != (0.0, 0.0):
        dr.laser.clearance = tuple(args.clearance)
    if args.ink_radius != (0.0, 0.0):
        dr.ink.radius_m = tuple(args.ink_radius)
    portable = None
    if args.portable_design:
        from tatbot_sim import design_scene

        try:
            portable = design_scene.load(Path(args.portable_design), substrate, args.design_placement)
        except (design_scene.DesignSceneError, OSError, ValueError) as exc:
            raise SystemExit(f"--portable-design {args.portable_design}: {exc}") from exc
        # the design's own footprint, as generate draws it
        dr.ink.radius_m = (portable.width_mm / 2000,) * 2
        need = portable.est_cost_s(config)
        budget = ((horizon - int(dr.approach.duration_s[1] * 30)) / 30.0 - 0.5) * 0.96
        if need > budget:
            raise SystemExit(f"design {portable.name!r} needs ~{need:.0f} s and --horizon {horizon} "
                             f"allows {budget:.0f} s; pass --horizon {int((need / 0.96 + 0.5) * 30) + 70} or more")
    cameras, lighting, surfaces, staged = presentation(args, config, width, height)

    config = config.with_dr(dr)
    rng = np.random.default_rng(args.seed)
    env = TatbotDrawEnv(config=config, presentation_cameras=cameras,
        presentation_lighting=lighting, presentation_surfaces=surfaces, num_envs=1, obs_mode="rgb", control_mode="pd_joint_pos",
        sim_backend="auto", reconfiguration_freq=1, dr=dr,
        sensor_configs={"shader_pack": args.shader},
        # artwork is drawn on the factory's fine ink field, not the sheet's own
        # 2.4 px/mm texels: a macro shot of a 0.3 mm line is made of them
        ink_pixels_per_m=recipe.artwork_pixels_per_m if task in ("artwork", "spiral", "erase") else None,
    )
    base = env.unwrapped
    expert = StrokeExpert(1, base.device, config=config, noise=dr.noise, seed=config.seed_for("noise"))
    world = ManiSkillWorld(env, config, expert)
    episode = Episode(world, ObservationBuilder(config, 1, base.device))
    episode.reset(seed=args.seed)
    q_now = world.positions()
    slack = dr.pen_lean.max_off_base_rad
    masks = reachable_canvas_masks(expert, q_now, base.surface, recipe.draw_clearance,
                                   1, max_off_base_rad=slack)
    ceiling = reachable_height_ceiling(expert, q_now, base.surface, 1,
                                       max_off_base_rad=slack)

    plan = plan_batch(
        rng, base.pad_sheets, base.surface, config=config, task=task, horizon=horizon, num_envs=1,
        dr=dr, draw_clearance=recipe.draw_clearance, task_name=recipe.task_name,
        maze_task_name=recipe.maze_task_name, erase_passes=recipe.erase_passes,
        erase_seconds=recipe.erase_seconds, reachable=masks, tool_ceiling=ceiling,
        style=style, cap_rims=base.cap_rims_np(),
        artwork_sampler=None if portable is None else portable.sample,
    )
    episode.install(plan)
    # Aim the staged cameras at the work itself: the centre of what this take
    # draws, on the surface it is drawn on. LOOK_AT is the flat pad's centre;
    # on the paper cylinder the crest sits a hand higher, and a design placed
    # off the crest is drawn on the flank.
    strokes = strokes_from_plan_paths(plan.paths[0])
    drawn = np.concatenate([stroke.points for stroke in strokes]) if strokes else np.zeros((1, 2))
    centre_uv = (drawn.min(axis=0) + drawn.max(axis=0)) / 2
    work = base.surface.frame_np(0, centre_uv[None, :].astype(np.float32))[0][0]
    look_at = np.asarray(work, dtype=np.float64)
    print(f"[cinematic] work centre {np.round(look_at, 3).tolist()} (canvas {np.round(centre_uv, 3).tolist()})")
    for name in staged:
        shot = SHOTS[name]
        if shot.track is None and shot.at is None:
            eye = np.array(shot.eye) - LOOK_AT + look_at
            base.scene.sensors[f"cine_{name}"].camera.set_local_pose(
                sapien_utils.look_at(eye=eye.tolist(), target=look_at.tolist(), up=list(shot.up)).sp)
    if plan.dips:
        for d in plan.dips[0]:
            print(f"[cinematic] dip before stroke {d['before_stroke']} ({d['reason']}) "
                  f"into {d['slot']} at step {plan.n_app + d['step']}, {d['steps']} steps")

    # Control is 30 Hz. To play back at `speed` and land `fps` frames a
    # second, take every Nth step — never less than every step, since the sim
    # has no frames between them to give.
    stride = max(1, int(round(30.0 * args.speed / args.fps)))
    actual_speed = stride * args.fps / 30.0
    if abs(actual_speed - args.speed) > 1e-6:
        print(f"[cinematic] speed {args.speed}x is not reachable at {args.fps} fps "
              f"from 30 Hz control; rendering {actual_speed:g}x")
    n_steps = min(episode.horizon, args.max_frames * stride)
    cams = {n: [] for n in args.shots}
    sensor_uid = {n: f"cine_{n}"
                  for n in args.shots}
    orbiting = [n for n in args.shots if SHOTS[n].track == "orbit"]
    tracking = [n for n in args.shots if SHOTS[n].track == "tool"]
    aim: dict[str, np.ndarray] = {}

    print(f"[cinematic] {dist.name}: {tool.tool_id} on {substrate.name}, "
          f"task {task}, design {args.design}")
    print(f"[cinematic] \"{plan.tasks[0]}\"")
    print(f"[cinematic] {n_steps} steps -> {n_steps // stride} frames at "
          f"{width}x{height}, shader {args.shader}, shots {', '.join(args.shots)}")

    t0 = time.time()
    for t in range(n_steps):
        if episode.done:
            break
        for name in orbiting:
            base.scene.sensors[f"cine_{name}"].camera.set_local_pose(
                orbit_pose(SHOTS[name], t / max(1, n_steps - 1), look_at))
        if tracking:
            tip = base.agent.tcp.pose.p[0].cpu().numpy()
            for name in tracking:
                shot = SHOTS[name]
                prev = aim.get(name)
                aim[name] = tip if prev is None else prev + shot.smooth * (tip - prev)
                base.scene.sensors[f"cine_{name}"].camera.set_local_pose(
                    tool_pose(shot, aim[name]))
        # A step nobody keeps needs no picture: the path tracer is the whole
        # cost of a take, and physics without it runs ~forty times faster.
        keep = t % stride == 0
        episode.step(capture=keep or args.render_skipped_steps)
        obs = episode.last_raw
        if not keep:
            continue
        for name in args.shots:
            cams[name].append(
                obs["sensor_data"][sensor_uid[name]]["rgb"][0].cpu().numpy().astype(np.uint8))
        if t and t % (50 * stride) == 0:
            print(f"[cinematic] {t / n_steps:5.0%}  {t / (time.time() - t0):.1f} steps/s",
                  flush=True)

    fps = float(args.fps)
    from PIL import Image
    for name, frames in cams.items():
        if not frames:
            continue
        path = out / f"{dist.name}-{task}-s{args.seed}-{name}.mp4"
        encode(frames, path, fps, args.crf)
        Image.fromarray(frames[len(frames) // 2]).save(
            path.with_suffix(".poster.jpg"), quality=92)
        print(f"[cinematic] {path.name}  {len(frames)} frames  {fps:.1f} fps  "
              f"{path.stat().st_size / 1024 / 1024:.1f} MB")

    (out / f"{dist.name}-{task}-s{args.seed}.json").write_text(json.dumps({
        "distribution": dist.name, "tool": tool.tool_id, "substrate": substrate.name,
        "resolved_config": config.metadata(), "runtime": episode.metadata(),
        "task": task, "prompt": plan.tasks[0], "seed": args.seed,
        "design": args.design, "steady": args.steady,
        "portable_design": None if portable is None else portable.summary(),
        "laser_clearance": list(dr.laser.clearance),
        "ink_radius_m": list(dr.ink.radius_m),
        "shader": args.shader, "exposure": args.exposure, "look": args.look,
        "size": [width, height], "fps": fps, "shots": list(args.shots),
        "frames": max((len(frames) for frames in cams.values()), default=0),
        "speed": actual_speed,
        "ink_coverage_end": float(base.ink_field.coverage()[0]),
        "ink": {"mode": base.ink_policy.mode, "wet": args.wet or None,
                "supply": {"kind": config.supply[0], "ink": config.supply[1]},
                "dips": plan.dips[0] if plan.dips else [],
                **{("n_dips" if k == "dips" else k): float(v[0])
                   for k, v in base.ink_episode_stats().items() if k != "mode"}},
    }, indent=2) + "\n")
    print(f"[cinematic] {time.time() - t0:.0f}s total -> {out}")
    episode.close()


def cli():
    """Select an explicit distribution configuration and render one take."""
    import sys

    argv = sys.argv[1:]
    names = list(DISTRIBUTIONS)
    if not argv or argv[0] in ("-h", "--help", "--list"):
        print("usage: sim_cinematic.py <distribution> --out DIR [options]\n")
        print("distributions: " + ", ".join(names))
        print("shots:         " + ", ".join(SHOTS))
        print("aspects:       " + ", ".join(ASPECTS))
        return
    name, rest = argv[0], argv[1:]
    dist = DISTRIBUTIONS.get(name)
    if dist is None:
        raise SystemExit(f"unknown distribution {name!r}; have {', '.join(names)}")
    from tatbot_sim.factory import select_tool
    select_tool(dist)
    main(tyro.cli(Args, args=rest), dist)


if __name__ == "__main__":
    cli()
