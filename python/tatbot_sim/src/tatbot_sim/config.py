"""Every tunable of the data factory, in one tree.

Before this module, the ~65 knobs lived in three tiers: CLI-exposed
(generate's Args), dataclass-but-not-CLI (DepthNoiseConfig, ShapeConfig),
and hardcoded inside env methods — the newest, broadest DR axes (lighting,
backgrounds, the table) were the least controllable. Now `DRConfig` is the
single home: tyro exposes every leaf on the command line
(``--dr.lighting.ambient 0.02 0.3``), the environment and generator read
ranges from it instead of carrying literals, and the generator dumps the
FULL resolved tree into ``run_meta.json`` — every dataset records exactly
which distribution produced it.

Defaults reproduce the tuned production values (probe-calibrated depth
noise, exposure-calibrated lighting, measured approach poses). Broadening a
run is a CLI override, not a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tatbot_sim.depth_noise import DepthNoiseConfig, RGBJitterConfig


@dataclass
class PadDR:
    """Pose of the substrate, re-sampled every episode."""

    xy_range: float = 0.02
    yaw_range: float = 0.6
    tilt_range: float = 0.09  # roll/pitch, rad — the surface is a plane
    # None = however this substrate rests, per config/substrates.yaml. A pad
    # put down on a bench and a skin draped over a wrist pad do not sit at the
    # same heights, and one default cannot describe both; set a range here only
    # to override the substrate deliberately.
    z_range: tuple[float, float] | None = None
    # None = the scene's stock placement (TatbotDrawEnv.PAD_CENTER). The
    # 2026-08-31 measured ballpoint tip seats ~9 deg crooked, and the region it
    # can be held tip-axis-vertical over is low, near, and biased +y — a place
    # the stock (0.29, 0) pad is not. Recipes for a tool with a measured seat
    # set this to put the substrate where that tool can actually work.
    center_xy: tuple[float, float] | None = None


@dataclass
class SurfaceDR:
    """The SHAPE of the skin, re-sampled when the scene is rebuilt.

    Off by default. The ranges that are safe to draw from are the ones the
    reach audit says the arm can hold the tool perpendicular over, and that
    measurement has not been made for the laser yet -- it is only at 63% within
    a millimetre on a flat pad over the default pad heights, and curvature
    costs both height and lean. Turning this on is a deliberate CLI override
    made against an audit, not a default anybody inherits.

    Sampled at scene build rather than per episode because the shape is baked
    into the sheet's mesh: the renderer has to show the surface the ink model
    believes in, and a mesh cannot be re-cut between episodes. Generation runs
    with reconfiguration_freq=1, so this is per batch.
    """

    # A DRAPED substrate is shaped whether anyone asks or not — the skin has
    # its mound in the room. This switch is for the near-flat ones, where a
    # millimetre of paper lift is genuinely optional.
    enabled: bool = False
    chart: str = "plane"                             # "plane"; "cylinder" awaits a pad body
    # Randomisation AROUND the substrate's measured mound, not instead of it:
    # the summit varies a little between drapes, and the footprint and where it
    # sits under the skin vary more. None = the substrate's own range, which is
    # the only place that knows whether its shape is the training signal or an
    # incidental millimetre of lift.
    peak_scale: tuple[float, float] | None = None
    radius_u_m: tuple[float, float] = (0.050, 0.065)
    radius_v_m: tuple[float, float] = (0.065, 0.085)
    center_jitter_m: float = 0.012
    # How far the surface ripples, how steeply, and over what wavelength.
    # None = the substrate's own ranges, per config/substrates.yaml. These
    # described the PAPER PAD for every substrate until 2026-08-27; the numbers
    # did not change, they moved to the object they were always about. Read
    # only on the near_flat path: a draped substrate is shaped by its mound
    # (see _sample_skin_shape), so setting these does nothing to the skin.
    # Set one here only to override the substrate deliberately.
    feature_m: tuple[float, float] | None = None     # wavelength of the undulation
    max_slope_rad: tuple[float, float] | None = None
    amplitude_m: tuple[float, float] | None = None
    components: int = 6
    grid_cols: int = 65                              # mesh and height resolution
    taper_frac: float = 0.15                         # flat where the skin meets the pad


@dataclass
class LightingDR:
    """Per-env light rigs, redrawn at every scene build.

    Intensity ranges sit on top of the environment map's image-based ambience.
    An earlier cut ran bright lights over a uniformly bright map and blew the
    paper to white, which leaves a ruled sheet with no stencil to trace; the
    response was to pull the lights down, and the result was a stream with no
    dynamic range at all -- measured against the bench on 2026-08-26, no sim
    pixel fell below 0.05 or rose above 0.95 where the real wrist stream puts
    26% and 7.6% of its pixels there.

    The floor was the ambience, not the lights. With the map's level now drawn
    across dim rooms as well as bright ones (see textures.environment_face_sets)
    and the bench's own albedo reaching near-black (floor_textures), the lights
    can be strong enough to clip a real highlight: measured over both
    substrates, sigma went 0.205 -> 0.26-0.31 against the bench's 0.304, and
    the blown tail 0.004 -> 0.075 against its 0.076.

    STILL OPEN: the dark tail. The bench puts 26% of its pixels below 0.05 and
    this rig reaches about 0.5%. Deep shadow needs occlusion the scene does not
    contain -- one lamp and a cluttered bench throwing real shadows -- and the
    intensities that darken the skin's periphery blow the pad's ruling instead.
    Closing it means more scene, not more knob.
    """

    enabled: bool = True
    ambient: tuple[float, float] = (0.0, 0.06)
    # Leaning harder on point/spot (which fall off with distance) and away from
    # directional (which does not) buys a darker periphery on the skin, but at
    # spot levels that reach the pad it blew 35% of a paper frame to white --
    # measured, 2026-08-26 -- and a ruling that is not there is not traceable.
    # These are the levels where both substrates measured well at once.
    max_directional: int = 2
    max_point: int = 2
    max_spot: int = 1
    directional_level: tuple[float, float] = (0.2, 1.2)
    point_level: tuple[float, float] = (0.1, 0.8)
    spot_level: tuple[float, float] = (0.6, 2.2)
    warmth: float = 0.2          # +- colour-temperature skew on any light
    shadow_prob: float = 0.9     # chance an env gets its one shadow caster


@dataclass
class BackgroundDR:
    """Environment cube maps, floors, and the table under the pad."""

    enabled: bool = True
    table_half_x: tuple[float, float] = (0.25, 0.45)
    table_half_y: tuple[float, float] = (0.30, 0.55)
    table_half_z: tuple[float, float] = (0.01, 0.02)


@dataclass
class SheetWearDR:
    """Reused-sheet look: ghost strokes on the ruling, smudges and uneven
    yellowing, baked as cached texture variants of each base sheet. Real
    sheets are drawn on more than once; a pool of always-pristine paper is a
    sim tell (and this is half the groundwork for the "finish the drawing"
    task). The base sheets stay in the pool, so clean paper remains in the
    distribution; the ruling geometry is shared, so strokes still trace the
    printed lines on worn sheets."""

    enabled: bool | None = None
    """None = whether this substrate has a ruling to ghost, per
    config/substrates.yaml. The wear IS ghost strokes over printed lines, so a
    blank skin has nothing to reuse the look of, and baking variants of a blank
    sheet spends texture memory on noise."""

    variants: int = 3  # wear variants generated per base sheet


@dataclass
class ClutterDR:
    """Distractor objects strewn on the table around the pad, redrawn per
    scene build and repositioned per episode. Real tables carry pens, tape
    and tools; a policy should key on the pad, not on an empty table. The
    objects are visual-only (no collision — they can never foul the arm)
    and keep a margin off the pad, but they appear in RGB and depth."""

    enabled: bool = True
    max_objects: int = 4          # per env; the count draws 0..max per build
    half_size: tuple[float, float] = (0.008, 0.045)


@dataclass
class InkDR:
    """Appearance of deposited ink, redrawn per scene build.

    A constant 3 mm pure-black line everywhere is a sim tell: real line
    weight moves with pressure and speed, and real ink is not one grey.
    Since ink became a per-env pigment field (tatbot_sim.inkfield) rather
    than a batch-shared actor pool, width and colour draw PER ENV.
    """

    radius_m: tuple[float, float] = (0.0011, 0.0020)  # line width 2.2-4 mm
    level: tuple[float, float] = (0.02, 0.12)         # ink darkness (blue-black)
    opacity: tuple[float, float] = (0.55, 0.95)
    """Pigment a single pass lays down, 0-1. Passes accumulate and saturate
    at fully opaque, so a traced-over stroke goes solid the way real ink
    does; below 1 the first pass lets a little paper through."""
    dry_floor: float = 0.12
    """What fraction of ``opacity`` a needle with NO charge still lays down.
    The tool carries a charge of ink between dips (scripts/lib/ink_spec.py);
    deposition scales from this floor at empty to full opacity at a fresh
    dip, so a policy can see the line fading and learn that a dip fixes it.
    Not zero: a dry needle still scratches a faint mark, and a line that
    vanished outright would be a sim tell in the other direction. Ignored
    for tools whose ink.mode is none."""
    dips: bool = False
    """Splice planned dips (scripts/lib/ink_spec.plan_dips) into DRAWING
    episodes. Off by default (operator, 2026-08-29): a 30 s drawing episode
    is not a session, and forcing every one to open at the palette taught
    "dip, then draw" and nothing else. Dipping is its own task family
    (``--task dip``); turn this on for episodes that are windows into a
    longer session, together with ``initial_charge_frac`` / ``capacity_scale``
    below so the low-charge dips actually land inside the episode."""
    initial_charge_frac: tuple[float, float] = (1.0, 1.0)
    """What fraction of the tool's capacity it carries when the episode
    opens, drawn per env. (1, 1) is a freshly dipped needle — the right
    default when ``dips`` is off, so a drawing episode draws at full
    opacity. With ``dips`` on, (0, 1) makes episodes open anywhere in a
    session: some dip first (session_start), some run dry mid-way
    (low_charge), some never need to. A ``--task dip`` episode always opens
    empty, whatever this says — that is what it is for."""
    capacity_scale: tuple[float, float] = (1.0, 1.0)
    """Per-env multiplier on the datasheet's charge_capacity_ul (and its
    uptake_ul). At the nominal 2 uL / 0.004 uL per mm a charge covers ~500
    mm of line, far more than any episode draws, so a low-charge dip never
    occurs inside one; (0.05, 0.3) puts the needle's range at 25-150 mm and
    the re-dip in the training distribution. Only meaningful with ``dips``."""


@dataclass
class PaletteDR:
    """Where the ink-cap rack sits, and how much it wanders.

    The rack is a fixed fixture on the real bench, so the ranges are the
    millimetres it is re-placed by between sessions, not a pad's freedom.
    ``center_m`` is ``palette_root`` in the FOLLOWER BASE frame; None (the
    default) derives it from the URDF (ink_spec.palette_root_in_base), where
    the rack and the arm mount are both fixed joints off the rig root. That
    puts it at about (0.126, 0.268, 0.085): between the two arms, to the
    right arm's left. The measured palette hold (config/poses.yaml
    palette_center, 2026-08-26, ROOT frame — not base) lands within 3 cm of
    it, which the jitter below covers.
    """

    enabled: bool = True
    center_m: tuple[float, float, float] | None = None
    xy_jitter_m: float = 0.006
    z_jitter_m: float = 0.003
    yaw_jitter_rad: float = 0.05
    rim_above_tag_m: float = 0.0
    """Cap rims sit this far above the tag face. The caps stand in holes in
    the rack so their rims are near flush; measure and set if not."""
    hover_m: float = 0.02
    """Clearance above a cap rim for the approach and the retract."""
    plunge_speed: float = 0.02
    """m/s into and out of the cap — a dip is a slow, deliberate motion."""


@dataclass
class LaserDR:
    """The removal tool's dose, redrawn per scene build.

    Deliberately NOT read from the tool datasheet: that file states outright
    that every optical figure on the consumer "picosecond" pen — wavelength,
    pulse width, power — is marketing copy, and no seller publishes a spot
    size or fluence. So removal strength is a tunable of the sim, drawn per
    env and logged into run_meta like every other axis, rather than a number
    pretending to be measured.

    ``clearance`` is the fraction of REMAINING pigment one pass clears, so
    removal is asymptotic: at 0.15, ten passes leave ~20% of the original
    and twenty leave ~4%. That spread across a few passes is exactly the
    partial-removal data the task family wants.
    """

    spot_radius_m: tuple[float, float] = (0.0010, 0.0025)
    clearance: tuple[float, float] = (0.08, 0.25)


@dataclass
class CameraDR:
    """Mounting tolerance on the wrist cameras, redrawn per scene build.

    Real brackets are not exact: a millimetre of seating and a degree of
    twist are within tolerance, and training across that slop regularizes
    the policy's dependence on one calibration.
    """

    mount_jitter_mm: float = 1.5   # +- per axis, uniform
    mount_jitter_deg: float = 1.0  # +- per axis (rpy), uniform


@dataclass
class PenLeanDR:
    # Furthest the tool is held from the SUBSTRATE's own normal, whatever the
    # local surface asks for. A tool does not have to be exactly perpendicular
    # to skin to work it, and on a mound's flanks exactly perpendicular is a
    # pose the arm cannot make -- it returns a best effort tens of millimetres
    # away and the labels never say so. Measured: the laser reaches 71% of a
    # 25 mm mound held exactly normal, 82% allowed 20 degrees of slack.
    max_off_base_rad: float = 0.35  # 20 degrees
    """Continuous lean of the pen off the surface normal along each path."""

    max_rad: float = 0.12
    keypoints: int = 4


@dataclass
class ApproachDR:
    """Episodes that open with the staged-pose descent (measured from the
    real squiggle-grid-draw eps 0/10)."""

    prob: float = 0.25
    duration_s: tuple[float, float] = (1.2, 2.2)
    pose_jitter_rad: float = 0.03


@dataclass
class NoiseDR:
    """DART perturbation bursts on the expert's commanded joints — the
    knock-off-the-stroke-and-recover data. Per-env prob/scale draw from
    these ranges each batch, so the dataset spans near-clean episodes
    through moderately perturbed ones instead of being uniformly messy.

    Ranges are sized to the errors deployment actually produces (tracking
    jitter ~0.6 mm, latency offsets of a few mm): EE deviations land in the
    ~2-8 mm band. The original constants (prob 0.02, scale 0.03) perturbed
    33% of steps with p99 = 24 mm / max 64 mm hops — on a +-60 mm canvas —
    which read as chaos in the wrist views and spent most of the noise
    budget on camera viewpoints the robot will never see (operator call,
    2026-08-24)."""

    prob: tuple[float, float] = (0.002, 0.008)   # burst probability per step
    scale: tuple[float, float] = (0.004, 0.014)  # rad, joint-space burst std
    decay: float = 0.85


@dataclass
class LatencyDR:
    """Deployment-timing realism: the async stack feeds the policy STALE
    observations (~250 ms upload cadence, queue latency) while naive sim
    data pairs fresh obs_t with action_t. Each episode samples a per-env
    delay k and the dataset stores obs from t-k against action_t, so the
    policy trains on the pairing it will actually face. 0 disables."""

    obs_delay_steps: tuple[int, int] = (0, 3)


@dataclass
class DRConfig:
    pad: PadDR = field(default_factory=PadDR)
    surface: SurfaceDR = field(default_factory=SurfaceDR)
    sheet: SheetWearDR = field(default_factory=SheetWearDR)
    ink: InkDR = field(default_factory=InkDR)
    laser: LaserDR = field(default_factory=LaserDR)
    palette: PaletteDR = field(default_factory=PaletteDR)
    clutter: ClutterDR = field(default_factory=ClutterDR)
    lighting: LightingDR = field(default_factory=LightingDR)
    background: BackgroundDR = field(default_factory=BackgroundDR)
    camera: CameraDR = field(default_factory=CameraDR)
    rgb: RGBJitterConfig = field(default_factory=RGBJitterConfig)
    depth_noise: DepthNoiseConfig = field(default_factory=DepthNoiseConfig)
    corrupt_depth: bool = True
    pen_lean: PenLeanDR = field(default_factory=PenLeanDR)
    noise: NoiseDR = field(default_factory=NoiseDR)
    approach: ApproachDR = field(default_factory=ApproachDR)
    latency: LatencyDR = field(default_factory=LatencyDR)

    def resolve_for(self, substrate) -> "DRConfig":
        """Fill the ranges that default to "whatever this substrate does".

        Resolved once, as early as the substrate is known, so every later
        reader sees numbers rather than None -- including the run_meta dump,
        which has to record what a batch actually drew from to be worth
        keeping. Idempotent, and an explicit override survives it untouched.
        """
        if self.pad.z_range is None:
            self.pad.z_range = tuple(substrate.rest_z_m)
        if self.surface.peak_scale is None:
            self.surface.peak_scale = tuple(substrate.peak_scale)
        if self.surface.amplitude_m is None:
            self.surface.amplitude_m = tuple(substrate.surface_amplitude_m)
        if self.surface.max_slope_rad is None:
            self.surface.max_slope_rad = tuple(substrate.surface_max_slope_rad)
        if self.surface.feature_m is None:
            self.surface.feature_m = tuple(substrate.surface_feature_m)
        if self.sheet.enabled is None:
            self.sheet.enabled = bool(substrate.ruled)
        return self
