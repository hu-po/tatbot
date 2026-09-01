#!/usr/bin/env python3
"""The tool registry: one datasheet per physical tattoo tool, loaded once.

A tool is described by ``config/tools/<tool_id>.yaml`` and nothing else. Every
consumer — the sim URDF builder, the touch-off solver, the dataset writers —
derives what it needs from that file instead of carrying its own copy of the
pen's dimensions. Adding a second pen is adding a second file.

Three rules keep this from rotting:

1. **One file per PHYSICAL UNIT, not per product.** Two Lutins off the same
   shelf get two files. Nominally identical pens differ after wear and
   cartridge choice, the per-file cost is a copy-paste, and an inheritance
   hierarchy would cost far more than it saves.
2. **Geometry is a profile of revolution, not a mesh.** ``profile`` is a list
   of ``[z_m, radius_m]`` samples along the tool axis, tip last. Calipers
   produce one in a minute; a 3D scan is *reduced* to one (plus an optional
   display mesh). The schema never depends on having scanned anything.
3. **Datasets snapshot the spec, they never reference it.** This file will be
   edited and this pen will be retired; a dataset recorded today has to stay
   readable anyway. See :func:`dataset_tool_metadata`.

Frame (schema 2, since 2026-08-30): ``z = 0`` at the MOUNT BORE FACE — the
face of the printed mount block the tool leaves through, which is the origin
of the ``*/tool_mount`` link in ``urdf/tatbot.urdf`` — and ``+z`` along the
tool axis toward the tip, so ``z_max`` of the profile is the tip and the
protrusion past the bore face. Negative ``z`` is the part of the body that
sits inside the block. Where the mount itself sits on the arm (the printed
chain on the left finger carriage; the bore runs 45 deg between the
carriage's -y and +x) is ARM geometry and lives in the URDF, not here.

Schema 1 put ``z = 0`` at the gripper fingertips and carried five ``grip_*``
fields, because the tool was pinched between two fingers. It is not any more:
the fingers no longer touch the tool, so a schema-1 datasheet is refused
rather than reinterpreted — its numbers are in a frame that no longer
describes the hardware.

Stdlib only, on purpose: the calibration and analysis scripts run under bare
interpreters that have numpy and nothing else, so this parses its own YAML
subset (two-space indent, scalars, and inline JSON arrays) exactly like
``config/workspace.yaml`` is already parsed elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIRNAME = "config/tools"
SUBSTRATES_FILENAME = "config/substrates.yaml"
WORKSPACE_RELPATH = "config/workspace.yaml"
SCHEMA_VERSION = 2
# Which physical follower this code describes. Stamped into every dataset's
# meta/tool.json; a training manifest that mixes embodiments has to say so.
# gripper-held-v1: tool pinched between two fingers, wrist cameras an
#   upper/lower pair (every recording before 2026-08-30).
# fixed-mount-v2: tool in a bore on the left-carriage mount, wrist rolled 90
#   deg so the cameras are a left/right pair.
EMBODIMENT = "fixed-mount-v2"
LEGACY_EMBODIMENT = "gripper-held-v1"
# The frame every measured tip offset is expressed in. workspace.yaml names
# it (`tip_frame`), and a file that names any other frame — or none, as every
# gripper-era file did — is treated as having no measured tip at all.
TIP_FRAME = "{arm}/tool_mount"
TIP_LINK = "{arm}/tattoo_needle"

# How many cylinders approximate one tapered profile segment when the tool has
# no display mesh. URDF has no cone primitive, so a taper is either a mesh or a
# stack. 16 steps: 8 read as a cone on a pen's short tip, but the laser's 64 mm
# nose is one long taper and stepped visibly in renders.
TAPER_STEPS = 16

# How far the solved tool axis may lean from the mount's nominal +z before
# the touch-off refuses to file the tip under this tool. This is the DEFAULT,
# for a mount whose bore actually locates the tool on its axis; a datasheet
# whose seat has real freedom overrides it with `seat_tolerance_deg`.
#
# The printed mount's bore is not a precision locator: ee_pen_mount.stl has a
# ~33 mm bore with only ~20 mm of wall around the Lutin's segmented 29 mm
# body, so the clamp — not the bore — fixes where the tool points, and the
# clearance grants it ~10 deg of legitimate seat freedom (measured 11.6 deg
# on sweep-20260831_082526, with the solve's cond at 5.6 and 67.7 deg of
# spread — a well-determined, stable seat, not a fault). The default still
# catches a mount screwed on askew for tools whose datasheet claims a snug
# seat.
AXIS_TOLERANCE_DEG = 5.0


# --- the YAML subset --------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, respecting double quotes."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def _scalar(text: str):
    text = text.strip()
    if text in ("null", "~", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    if text[0] in "[{":
        return json.loads(text)
    try:
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except ValueError:
        return text.strip('"')


def parse_simple_yaml(text: str) -> dict:
    """Nested maps by two-space indent, scalars, JSON arrays, block scalars.

    An array may span lines (accumulated until its brackets balance), which is
    the only reason a `profile:` stays readable, and ``>``/``|`` blocks carry
    the multi-line provenance notes that are half the point of a datasheet.
    Deliberately not PyYAML: see the module docstring.

    Anything outside that subset raises. Silently misreading a datasheet is
    the one outcome worse than refusing to read it.
    """
    lines = text.splitlines()
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-2, root)]
    pending_key: str | None = None
    pending_text = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if pending_key is not None:
            pending_text += " " + _strip_comment(line).strip()
            if pending_text.count("[") - pending_text.count("]") <= 0:
                stack[-1][1][pending_key] = json.loads(pending_text)
                pending_key, pending_text = None, ""
            continue
        raw = _strip_comment(line).rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key, sep, value = raw.strip().partition(":")
        if not sep:
            raise ValueError(f"line {index}: not a `key: value` pair: {raw.strip()!r}")
        key, value = key.strip(), value.strip()
        if value in (">", ">-", "|", "|-"):
            block, index = _read_block(lines, index, indent)
            joiner = "\n" if value.startswith("|") else " "
            parent[key] = joiner.join(block).strip()
        elif not value:
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        elif value.startswith("[") and value.count("[") > value.count("]"):
            pending_key, pending_text = key, value
        else:
            parent[key] = _scalar(value)
    if pending_key is not None:
        raise ValueError(f"unterminated array for key {pending_key!r}")
    return root


def _read_block(lines: list[str], index: int, indent: int) -> tuple[list[str], int]:
    """Consume a ``>``/``|`` block: every following line indented past its key.

    Comments are NOT stripped inside a block — a note is prose, and a '#' in it
    is a '#', not the start of a comment.
    """
    out: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line.strip())
        index += 1
    return out, index


# --- the spec ---------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One physical tool, as loaded from its datasheet."""

    tool_id: str
    kind: str
    display_name: str
    prompt_phrase: str
    profile: tuple[tuple[float, float], ...]
    # What this tool works on. A tool and a substrate are a pair on this bench:
    # the ballpoint only ever draws on the paper pad, the laser and the 3RL
    # only ever work on the silicone skin. See config/substrates.yaml.
    substrate: str = "paper_pad"
    rings: tuple[tuple[float, float, float], ...] = ()
    tip_mesh: str | None = None
    # Scale for tip_mesh, when the mesh was modelled at a different size than
    # the tool measures (ours is a 40 mm cone; the cartridge is 35 mm).
    tip_mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # One colour per profile SEGMENT, when a tool changes material along its
    # length. Null entries fall back to the body/tip default. Without this the
    # renderer guesses from shape — constant radius is body, a taper is tip —
    # which is right for a pen and wrong for anything built out of sections.
    tip_detail: dict = field(default_factory=dict)
    """What the last millimetres of the tool actually look like, beyond the
    profile of revolution. A profile can only make a cone, and the three tools
    end in things a cone cannot describe: a ballpoint's rounded steel ball, a
    3RL's three separate needles, a laser's emitter window. Keys: ``kind``
    (ball | needles | emitter) plus that kind's own dimensions. Absent = the
    profile is the whole story, which is what every tool said until
    2026-08-27."""
    segment_colors: tuple = ()
    body_color: str = "0.04 0.04 0.045 1"
    tip_color: str = "0.93 0.93 0.92 1"
    # The URDF link this tool's frame IS, per arm ("{arm}" is substituted).
    # Every fitted tool sits in the same printed mount on the left finger
    # carriage, so this is the one value — a tool with `mount: none` has no
    # mount on this arm yet (the laser pen, until its adapter exists) and is
    # refused by everything that would fly it.
    mount: str | None = "tool_mount"
    mass_kg: float | None = None
    com_z_m: float | None = None
    tip_tolerance_m: float = 0.015
    # How the tool actually sits in that mount. A bore with real clearance
    # (the printed mount: ~33 mm bore, ~20 mm wall, 29 mm body) locates the
    # tool with the clamp, not the bore axis, so the measured tip may
    # legitimately lean this far off the mount's nominal +z (default: the
    # snug-seat AXIS_TOLERANCE_DEG) and a planted-tip solve carries the
    # seat's own contact migration on top of the tip-face term (metres,
    # default 0 = no seat play claimed).
    seat_tolerance_deg: float = AXIS_TOLERANCE_DEG
    seat_residual_m: float = 0.0
    # Does the working point touch the work? A needle does; a laser's focus
    # hangs in free space in front of the aperture. False turns off everything
    # that assumes contact — the touch-off above all, which cannot plant a
    # working point that never lands on anything.
    contact: bool = True
    # Where the working point sits along the tool axis. Defaults to the end of
    # the profile, which is right for anything that works by touching. A
    # non-contact tool sets it past the profile: aperture plus working
    # distance, a point with no material at it.
    tcp_z_m: float | None = None
    # Descriptive only — what this configuration is made of. Two cartridges in
    # one machine are two tools (different tip, different calibration), and
    # these fields are how a human sees they share a body.
    body: str | None = None
    cartridge: str | None = None
    measured: dict = field(default_factory=dict)
    source: Path | None = None
    sha256: str = ""
    raw: dict = field(default_factory=dict)

    # --- derived geometry ---

    @property
    def back_m(self) -> float:
        """Rear end of the body. Negative means it sits inside the mount block."""
        return self.profile[0][0]

    @property
    def mounted(self) -> bool:
        """Does this tool have a mount on the arm at all?"""
        return bool(self.mount)

    def mount_frame(self, arm: str = "right") -> str:
        if not self.mount:
            raise ToolMountError(
                f"{self.tool_id} has no mount on the arm (mount: none in its "
                f"datasheet) — it cannot be fitted, calibrated or flown until one exists")
        return f"{arm}/{self.mount}"

    @property
    def body_tip_z_m(self) -> float:
        """The far end of the physical body."""
        return self.profile[-1][0]

    @property
    def protrusion_m(self) -> float:
        """Working point distance past the mount bore face — the nominal TCP offset.

        The same number for a needle tip and a laser focus; only one of them
        has material at it.
        """
        return self.body_tip_z_m if self.tcp_z_m is None else self.tcp_z_m

    @property
    def standoff_m(self) -> float:
        """How far the working point floats past the physical body. Zero for
        contact tools."""
        return self.protrusion_m - self.body_tip_z_m

    @property
    def body_radius_m(self) -> float:
        return max(r for _, r in self.profile)

    @property
    def tip_radius_m(self) -> float:
        return self.profile[-1][1]

    @property
    def nominal_tip_offset_m(self) -> tuple[float, float, float]:
        """Where the working point should be, in the mount frame: along +z."""
        return (0.0, 0.0, self.protrusion_m)

    @property
    def touchoff_nominal_m(self) -> tuple[float, float, float]:
        """What a touch-off should measure — which is not always the TCP.

        A touch-off works by planting something solid on a surface, so what it
        finds is the end of the BODY. For a needle that is also the working
        point. For a non-contact tool it is the aperture, and the working point
        is a standoff further on; checking the solve against the TCP there
        would refuse every good calibration by exactly the standoff.
        """
        return (0.0, 0.0, self.body_tip_z_m)

    def segments(self):
        """Consecutive profile samples as ``(z0, z1, r0, r1)``."""
        return [
            (z0, z1, r0, r1)
            for (z0, r0), (z1, r1) in zip(self.profile, self.profile[1:], strict=False)
        ]

    def meshes(self) -> tuple[str, ...]:
        return (self.tip_mesh,) if self.tip_mesh else ()

    def geometry_parts(self, taper_steps: int = TAPER_STEPS) -> list[dict]:
        """The profile as renderable parts along the tool axis.

        One definition, two consumers: the sim's derived URDF and the real
        rig's. Each turns these into its own XML; neither owns the geometry.
        A part is either a cylinder (``length``/``radius``) or the tool's
        display mesh, both positioned by their own ``z`` convention — a
        cylinder by its centre, a mesh by its origin.
        """
        parts: list[dict] = []
        for index, (z0, z1, r0, r1) in enumerate(self.segments()):
            override = (self.segment_colors[index]
                        if index < len(self.segment_colors) else None)
            if r0 == r1:
                parts.append({"kind": "cylinder", "z": (z0 + z1) / 2,
                              "length": z1 - z0, "radius": r0,
                              "color": override or self.body_color})
            elif self.tip_mesh:
                parts.append({"kind": "mesh", "z": z0, "mesh": self.tip_mesh,
                              "scale": self.tip_mesh_scale,
                              "color": override or self.tip_color})
            else:
                # URDF has no cone primitive, so a taper is either a mesh or
                # a stack of cylinders. 8 steps reads as a cone at wrist-
                # camera resolution, and costs nothing to render.
                step = (z1 - z0) / taper_steps
                for k in range(taper_steps):
                    frac = (k + 0.5) / taper_steps
                    parts.append({"kind": "cylinder", "z": z0 + step * (k + 0.5),
                                  "length": step, "radius": r0 + (r1 - r0) * frac,
                                  "color": override or self.tip_color})
        parts += self.tip_detail_parts()
        for center, half_len, extra_r in self.rings:
            parts.append({"kind": "cylinder", "z": center, "length": 2 * half_len,
                          "radius": self.body_radius_m + extra_r,
                          "color": self.body_color})
        return parts

    def tip_detail_parts(self) -> list[dict]:
        """The tip's own geometry, in the same part vocabulary as the profile.

        Positioned against ``protrusion_m``, which is where the TCP is: a ball
        sits tangent to it, needles run back from it, an emitter window sits
        just behind it. Parts may carry ``x``/``y`` offsets — the profile
        cannot, being a solid of revolution, and three needles in a cluster
        are exactly the case that needs them.
        """
        d = self.tip_detail
        if not d:
            return []
        kind = d.get("kind")
        tip_z = self.protrusion_m
        if kind == "ball":
            # A ballpoint writes with a rotating steel ball seated in the
            # socket; its underside IS the contact point, so it is tangent to
            # the TCP rather than centred on it.
            r = float(d.get("radius_m", 0.0005))
            return [{"kind": "sphere", "z": tip_z - r, "radius": r,
                     "color": d.get("color", "0.78 0.79 0.82 1")}]
        if kind == "needles":
            # Three needles in a tight cluster, which is what "3RL" means: 3
            # Round Liner, grouped so they act as one point. Centres sit on a
            # circle of radius s/sqrt(3) for three touching needles of
            # diameter s.
            n = int(d.get("count", 3))
            r = float(d.get("radius_m", 0.00015))
            length = float(d.get("length_m", 0.004))
            spread = float(d.get("spread_m", 2 * r / math.sqrt(3)))
            colour = d.get("color", "0.80 0.81 0.84 1")
            out = []
            for i in range(n):
                a = 2 * math.pi * i / n
                out.append({"kind": "cylinder", "z": tip_z - length / 2,
                            "x": spread * math.cos(a), "y": spread * math.sin(a),
                            "length": length, "radius": r, "color": colour})
            return out
        if kind == "emitter":
            # The window the beam leaves through. Emissive, and the env pulses
            # it: under the path tracer an emissive material is an area light,
            # so the flash actually lights the skin rather than just looking
            # bright.
            r = float(d.get("radius_m", 0.0035))
            return [{"kind": "sphere", "z": tip_z - r * float(d.get("inset", 0.35)),
                     "radius": r, "color": d.get("color", "0.10 0.35 1.00 1"),
                     "emissive": True}]
        raise ValueError(f"{self.tool_id}: unknown tip_detail kind {kind!r}")

    @property
    def verified(self) -> bool:
        """Has anyone actually measured this tool, or is it vendor copy?"""
        return self.measured.get("status") == "measured"

    def summary(self) -> str:
        standoff = ("" if self.contact
                    else f" ({self.standoff_m * 1000:.0f} mm standoff, non-contact)")
        unverified = "" if self.verified else "  [UNVERIFIED]"
        return (f"{self.tool_id} ({self.display_name}): "
                f"{self.protrusion_m * 1000:.0f} mm protrusion, "
                f"{self.body_radius_m * 2000:.0f} mm body{standoff}{unverified}")


def _require(data: dict, key: str, source: Path):
    if data.get(key) in (None, ""):
        raise ValueError(f"{source}: missing required field {key!r}")
    return data[key]


def _validate(spec: ToolSpec) -> ToolSpec:
    source = spec.source
    if len(spec.profile) < 2:
        raise ValueError(f"{source}: profile needs at least two [z, radius] samples")
    zs = [z for z, _ in spec.profile]
    if any(b <= a for a, b in zip(zs, zs[1:], strict=False)):
        raise ValueError(f"{source}: profile z values must strictly increase (tip last), got {zs}")
    if any(r <= 0 for _, r in spec.profile):
        raise ValueError(f"{source}: profile radii must be positive")
    if spec.protrusion_m <= 0:
        raise ValueError(f"{source}: the tip must protrude past the mount bore face (z > 0)")
    if spec.standoff_m < 0:
        raise ValueError(
            f"{source}: tcp_z_m {spec.tcp_z_m} is inside the body (ends at "
            f"{spec.body_tip_z_m}) — a working point cannot be buried in the tool")
    if spec.contact and spec.standoff_m > 0:
        raise ValueError(
            f"{source}: a contact tool's working point cannot float "
            f"{spec.standoff_m * 1000:.0f} mm off the end of it; set contact: false")
    if spec.segment_colors and len(spec.segment_colors) != len(spec.profile) - 1:
        raise ValueError(
            f"{source}: segment_colors has {len(spec.segment_colors)} entries for "
            f"{len(spec.profile) - 1} profile segments")
    for key in ("grip_diameter_m", "grip_rest_m", "grip_force_n",
                "grip_rest_tolerance_m", "grip_slop_m"):
        if key in spec.raw:
            raise ValueError(
                f"{source}: {key} is a schema-1 field. Nothing grips this tool any "
                "more — it sits in a bore on the mount — so the field has no "
                "referent; delete it rather than carrying a number nobody reads")
    if spec.mass_kg is not None and spec.mass_kg <= 0:
        raise ValueError(f"{source}: mass_kg must be positive, got {spec.mass_kg}")
    if not 0 < spec.seat_tolerance_deg < 45:
        raise ValueError(
            f"{source}: seat_tolerance_deg must be in (0, 45), got "
            f"{spec.seat_tolerance_deg} — a seat freer than 45 deg is not a mount")
    if not 0 <= spec.seat_residual_m < 0.02:
        raise ValueError(
            f"{source}: seat_residual_m must be in [0, 0.02), got "
            f"{spec.seat_residual_m} — 2 cm of seat play is a broken clamp, not a budget")
    return spec


def tools_dir(repo: Path | str = REPO) -> Path:
    return Path(repo) / TOOLS_DIRNAME


def list_tools(repo: Path | str = REPO) -> list[str]:
    return sorted(p.stem for p in tools_dir(repo).glob("*.yaml"))


@dataclass(frozen=True)
class Substrate:
    """What a tool works on: the pad or the skin, as measured."""

    name: str
    display_name: str
    width_m: float
    height_m: float
    thickness_m: float
    texel_cols: int
    texel_rows: int
    shape: str
    surface_phrase: str
    # How far the substrate's high point stands above its flat edges. A
    # measured fact about the object, so the sim starts from the real shape and
    # randomises around it rather than inventing an amplitude.
    mound_peak_m: float = 0.0
    ruled: bool = False
    base_color: str | None = None
    # How this substrate SITS, and how much its shape varies between setups.
    # They belong to the object rather than to a run: a letter pad lies wherever
    # it is put on a table, a skin draped over a wrist pad rests on the pad. The
    # two also trade against each other, because the arm can only reach so high
    # and a tall mound therefore has to sit low -- and only the substrate knows
    # which of the two matters for the work done on it.
    rest_z_m: tuple[float, float] = (0.0, 0.055)
    peak_scale: tuple[float, float] = (0.85, 1.10)
    # The UNDULATION a NEAR_FLAT substrate carries: how far it ripples, how
    # steeply, and over what wavelength. A sheet of paper on a pad lifts by a
    # millimetre over a hand's width; that is the shape these describe, and
    # until 2026-08-27 they were the sim's defaults for every substrate.
    #
    # A `draped` substrate does NOT read them -- its shape comes from
    # mound_peak_m and peak_scale through the drape height field -- so the
    # defaults below stand unused on the skin rather than describing it.
    surface_amplitude_m: tuple[float, float] = (0.0004, 0.0018)
    surface_max_slope_rad: tuple[float, float] = (0.005, 0.04)
    surface_feature_m: tuple[float, float] = (0.05, 0.12)

    @property
    def texel_per_m(self) -> float:
        """Texels per metre, averaged over both axes as the kernels see it."""
        return 0.5 * (self.texel_cols / self.width_m + self.texel_rows / self.height_m)


def _pair(entry: dict, key: str, default: tuple[float, float], path: Path) -> tuple[float, float]:
    """A two-number range, ordered low-high.

    A reversed or single-ended range would sample an empty interval and quietly
    pin every draw to one value, which reads in the data as "the randomiser is
    off" rather than as a typo.
    """
    raw = entry.get(key)
    if raw is None:
        return default
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{path}: {key} must be a two-element array, got {raw!r}")
    lo, hi = float(raw[0]), float(raw[1])
    if hi < lo:
        raise ValueError(f"{path}: {key} is reversed: [{lo}, {hi}]")
    return (lo, hi)


def substrates_path(repo: Path | str = REPO) -> Path:
    return Path(repo) / SUBSTRATES_FILENAME


def load_substrate(name: str, repo: Path | str = REPO) -> Substrate:
    """One substrate from config/substrates.yaml."""
    path = substrates_path(repo)
    if not path.is_file():
        raise FileNotFoundError(f"no substrate registry at {path}")
    data = parse_simple_yaml(path.read_text())
    version = data.get("schema_version", 1)
    if version != 1:
        raise ValueError(f"{path}: schema_version {version}, this code reads 1")
    entry = data.get(name)
    if not isinstance(entry, dict):
        known = ", ".join(k for k, v in data.items() if isinstance(v, dict)) or "none"
        raise ValueError(f"{path}: unknown substrate {name!r} (known: {known})")
    return Substrate(
        name=name,
        display_name=_require(entry, "display_name", path),
        width_m=float(_require(entry, "width_m", path)),
        height_m=float(_require(entry, "height_m", path)),
        thickness_m=float(_require(entry, "thickness_m", path)),
        texel_cols=int(_require(entry, "texel_cols", path)),
        texel_rows=int(_require(entry, "texel_rows", path)),
        shape=str(_require(entry, "shape", path)),
        surface_phrase=str(_require(entry, "surface_phrase", path)),
        mound_peak_m=float(entry.get("mound_peak_m", 0.0)),
        ruled=bool(entry.get("ruled", False)),
        base_color=entry.get("base_color"),
        rest_z_m=_pair(entry, "rest_z_m", (0.0, 0.055), path),
        peak_scale=_pair(entry, "peak_scale", (0.85, 1.10), path),
        surface_amplitude_m=_pair(entry, "surface_amplitude_m", (0.0004, 0.0018), path),
        surface_max_slope_rad=_pair(entry, "surface_max_slope_rad", (0.005, 0.04), path),
        surface_feature_m=_pair(entry, "surface_feature_m", (0.05, 0.12), path),
    )


def substrate_for(spec: "ToolSpec", repo: Path | str = REPO) -> Substrate:
    """The substrate this tool works on. A tool and its substrate are a pair."""
    return load_substrate(spec.substrate, repo)


def load_tool(tool_id: str, repo: Path | str = REPO) -> ToolSpec:
    """Load and validate one tool datasheet."""
    path = tools_dir(repo) / f"{tool_id}.yaml"
    if not path.is_file():
        known = ", ".join(list_tools(repo)) or "none"
        raise FileNotFoundError(f"unknown tool {tool_id!r}: no {path} (known tools: {known})")
    text = path.read_text()
    data = parse_simple_yaml(text)
    version = data.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {version}, this code reads {SCHEMA_VERSION}. "
            + ("A schema-1 datasheet puts z = 0 at the gripper fingertips; the tool "
               "is in a mount now, so re-measure its profile from the bore face and "
               "drop the grip_* fields." if version == 1 else ""))
    if data.get("tool_id") != tool_id:
        raise ValueError(f"{path}: tool_id {data.get('tool_id')!r} does not match its filename")
    spec = ToolSpec(
        tool_id=tool_id,
        kind=_require(data, "kind", path),
        substrate=data.get("substrate", "paper_pad"),
        display_name=_require(data, "display_name", path),
        prompt_phrase=_require(data, "prompt_phrase", path),
        profile=tuple((float(z), float(r)) for z, r in _require(data, "profile", path)),
        rings=tuple((float(c), float(h), float(e)) for c, h, e in data.get("rings") or ()),
        tip_mesh=data.get("tip_mesh"),
        tip_mesh_scale=(
            (float(s[0]), float(s[1]), float(s[2]))
            if (s := data.get("tip_mesh_scale")) and len(s) == 3
            else (1.0, 1.0, 1.0)
        ),
        tip_detail=data.get("tip_detail") or {},
        segment_colors=tuple(data.get("segment_colors") or ()),
        body_color=data.get("body_color") or ToolSpec.body_color,
        tip_color=data.get("tip_color") or ToolSpec.tip_color,
        mount=(None if str(data.get("mount", "tool_mount")).lower() in ("none", "null", "")
               else str(data.get("mount", "tool_mount"))),
        mass_kg=data.get("mass_kg"),
        com_z_m=data.get("com_z_m"),
        tip_tolerance_m=data.get("tip_tolerance_m") or ToolSpec.tip_tolerance_m,
        seat_tolerance_deg=data.get("seat_tolerance_deg") or ToolSpec.seat_tolerance_deg,
        seat_residual_m=data.get("seat_residual_m") or 0.0,
        contact=data.get("contact", True),
        tcp_z_m=data.get("tcp_z_m"),
        body=data.get("body"),
        cartridge=data.get("cartridge"),
        measured=data.get("measured") or {},
        source=path,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        raw=data,
    )
    return _validate(spec)


# --- the arm golden the scripts share ---------------------------------------

ARM_GOLDEN_RELPATH = "config/trossen/tatbot.yaml"


def arm_golden(repo: Path | str = REPO) -> dict:
    """The ``follower:`` section of config/trossen/tatbot.yaml.

    One source for the staged pose and the carriage constants. Scripts used
    to carry literal copies of the staged pose (three of them by 2026-08-30);
    they read this instead, and scripts/tests/test_staged_pose_single_source.py
    fails if a copy comes back.
    """
    data = parse_simple_yaml((Path(repo) / ARM_GOLDEN_RELPATH).read_text())
    return data["follower"]


def staged_positions(repo: Path | str = REPO) -> list[float]:
    """The follower's 7-value staged/idle pose: six joints, then the carriage."""
    pose = [float(v) for v in arm_golden(repo)["staged_positions"]]
    if len(pose) != 7:
        raise ValueError(f"{ARM_GOLDEN_RELPATH}: staged_positions has {len(pose)} values, need 7")
    return pose


# --- which tool is fitted right now -----------------------------------------

def read_workspace(repo: Path | str = REPO) -> dict:
    path = Path(repo) / WORKSPACE_RELPATH
    return parse_simple_yaml(path.read_text()) if path.is_file() else {}


def active_tool_id(repo: Path | str = REPO, arm: str = "right",
                   workspace: dict | None = None) -> str | None:
    """The tool the touch-off says is fitted. One pointer, set during
    calibration — so a recording can never silently disagree with the
    calibration it is using."""
    ws = read_workspace(repo) if workspace is None else workspace
    return (ws.get(arm) or {}).get("tool_id")


def load_active_tool(repo: Path | str = REPO, arm: str = "right",
                     workspace: dict | None = None) -> ToolSpec:
    tool_id = active_tool_id(repo, arm, workspace)
    if not tool_id:
        known = ", ".join(list_tools(repo)) or "none"
        raise RuntimeError(
            f"no tool_id under `{arm}:` in {WORKSPACE_RELPATH} — the fitted tool is "
            f"unknown. Set it with `il_touchoff.py ... --tool-id <id> --write` after a "
            f"touch-off, or add the line by hand (known tools: {known})."
        )
    return load_tool(tool_id, repo)


class ToolMismatchError(RuntimeError):
    """The stated tool is not the one the live calibration was measured with."""


def require_stated_tool(tool_id: str | None, repo: Path | str = REPO,
                        arm: str = "right", workspace: dict | None = None,
                        context: str = "this run") -> ToolSpec:
    """The fitted tool, STATED by the caller and cross-checked against the
    calibration in workspace.yaml. Never inferred.

    Two failures are possible and both are refusals, because each one produces
    a plausible number that is wrong:

    - Nothing stated. Reading workspace.yaml instead is how you silently
      inherit the PREVIOUS tool. On 2026-08-26 that gated a laser-pen
      touch-off against the ballpoint's datasheet, and the same inheritance
      would have handed a 130 mm tool a 63.7 mm tip offset and a 33 N grip
      chosen for a machined pen body.
    - Stated, but not what the calibration was measured with. Everything else
      under `right:` — the tip offset, the paper plane, the pivot — belongs to
      the tool named there. Loading THIS tool's datasheet while using THAT
      tool's geometry mixes two tools into one arm, which is worse than
      either alone.

    So the arg is what the mount holds, and workspace.yaml is what the
    constants were measured with; they have to agree, and the fix for a real
    swap is a new touch-off, not a louder assertion.
    """
    if not tool_id:
        known = ", ".join(list_tools(repo)) or "none"
        calibrated = active_tool_id(repo, arm, workspace)
        raise ToolMismatchError(
            f"{context}: no end-effector tool stated. Name what is in the "
            f"mount — tip offset, prompt phrase and ink policy all come from it, "
            f"and guessing means using the last tool's numbers on this one. "
            f"Known tools: {known}."
            + (f" The live calibration was measured with {calibrated!r}."
               if calibrated else ""))
    spec = load_tool(tool_id, repo)
    spec.mount_frame(arm)  # raises ToolMountError for a tool with no mount
    calibrated = active_tool_id(repo, arm, workspace)
    if calibrated and calibrated != tool_id:
        raise ToolMismatchError(
            f"{context}: {tool_id!r} is fitted but {WORKSPACE_RELPATH} was "
            f"measured with {calibrated!r}. Every constant under `{arm}:` "
            f"(tip offset, paper plane, pivot) belongs to {calibrated!r}, so "
            f"running {tool_id!r} against them mixes two tools. Re-run the "
            f"touch-off for the fitted tool:\n"
            f"  tatbot --ee-tool {tool_id} vision calib sweep --phases tip   (= scripts/vision/calib_sweep.sh)")
    return spec


def tip_frame(arm: str = "right") -> str:
    return TIP_FRAME.format(arm=arm)


def tip_link(arm: str = "right") -> str:
    return TIP_LINK.format(arm=arm)


def tip_offset_m(workspace: dict, arm: str = "right") -> tuple[float, float, float] | None:
    """The MEASURED tip offset in the mount frame, or None before a touch-off.

    None also for a workspace whose `tip_frame` is not the mount frame: every
    file written before 2026-08-30 solved the tip in `right/ee_gripper_link`,
    a frame the tool no longer has any fixed relation to. Those numbers are
    not stale-but-close, they are in the wrong frame, so they read as absent
    and every consumer behaves as if no touch-off had been run.
    """
    side = workspace.get(arm) or {}
    if side.get("tip_frame") != tip_frame(arm):
        return None
    x = side.get("pen_tip_offset_x")
    y = side.get("pen_tip_offset_y")
    z = side.get("pen_tip_offset_z")
    if x is None or y is None or z is None:
        return None
    return (float(x), float(y), float(z))


def tip_offset_error_m(spec: ToolSpec, measured: tuple[float, float, float]) -> float:
    """Distance between what the touch-off measured and what the datasheet says
    it should have measured.

    Large means the fitted tool is not the one named in workspace.yaml (pens
    differ by tens of millimetres), or the datasheet is wrong. Either way the
    calibration should not be written under that name.
    """
    nominal = spec.touchoff_nominal_m
    return sum((a - b) ** 2 for a, b in zip(measured, nominal, strict=True)) ** 0.5


def tcp_from_touchoff_m(spec: ToolSpec, measured: tuple[float, float, float]
                        ) -> tuple[float, float, float]:
    """The working point, given what the touch-off planted on the surface.

    Identical to the measurement for a contact tool. For a non-contact one the
    working point lies the standoff further along the same axis — the tool
    points where its body points, so the measured direction is the one to
    extend.
    """
    if spec.standoff_m == 0:
        return measured
    norm = sum(v * v for v in measured) ** 0.5
    if norm < 1e-9:
        raise ValueError(f"{spec.tool_id}: measured offset has no direction to extend")
    mx, my, mz = measured
    scale = 1.0 + spec.standoff_m / norm
    return (mx * scale, my * scale, mz * scale)


def axis_lean_deg(measured: tuple[float, float, float]) -> float:
    """How far the measured tip direction leans from the mount's +z.

    The mount origin sits on the bore axis, so the line from it to the solved
    tip is the tool axis; this is that axis's angle from where the URDF says
    the bore points. Large means the tool is seated crooked, the mount is on
    askew, or the mount transform in the URDF is wrong — all things to fix
    physically or in the URDF, never by accepting the number.
    """
    norm = sum(v * v for v in measured) ** 0.5
    if norm < 1e-9:
        raise ValueError("measured offset has no direction")
    return math.degrees(math.acos(max(-1.0, min(1.0, measured[2] / norm))))


def axis_rpy(direction) -> tuple[float, float, float]:
    """Roll/pitch/yaw putting local +z along ``direction``.

    URDF composes rpy as Rz(yaw)·Ry(pitch)·Rx(roll), so mapping +z onto a
    direction needs no roll at all: pitch tips it off vertical, yaw swings it
    round. Leaving roll at zero also keeps the tool's own spin out of the
    model, which is right — a body of revolution has no measurable roll.
    Shared by the real rig's URDF generator and the sim's, so the two never
    disagree about which way the tool points.
    """
    x, y, z = direction
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        raise ValueError("tool direction is degenerate")
    return (0.0, math.acos(max(-1.0, min(1.0, z / norm))), math.atan2(y, x))


class ToolMountError(RuntimeError):
    """The tool has no mount on the arm, so nothing can fit or fly it."""


# --- dataset metadata -------------------------------------------------------

def dataset_tool_metadata(spec: ToolSpec, workspace: dict | None = None,
                          arm: str = "right", extra: dict | None = None) -> dict:
    """The ``meta/tool.json`` payload: the full spec INLINED, not referenced.

    A dataset outlives the datasheet that made it. Anyone re-reading this in a
    year gets the geometry itself, plus the hash of the file it came from so a
    match against a still-living datasheet is checkable.
    """
    side = (workspace or {}).get(arm) or {}
    touchoff = side.get("touchoff") or {}
    measured = tip_offset_m(workspace or {}, arm)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_id": spec.tool_id,
        "arm": arm,
        "spec": dict(spec.raw),
        "spec_sha256": spec.sha256,
        "protrusion_m": spec.protrusion_m,
        "contact": spec.contact,
        # what the touch-off measured (the body's end) and where the tool
        # actually works — the same point unless the tool works at a distance
        "tip_offset_m": list(measured) if measured else None,
        "tcp_offset_m": (list(tcp_from_touchoff_m(spec, measured)) if measured else None),
        # the frame tip_offset_m is expressed in, and the URDF link that sits
        # at the working point once scripts/gen_tool_urdf.py has run
        "tip_frame": tip_frame(arm),
        "tip_link": tip_link(arm),
        "mount": spec.mount,
        "embodiment": EMBODIMENT,
        "verified": spec.verified,
        "touchoff": {
            "utc": touchoff.get("utc"),
            "session": touchoff.get("session"),
            "residual_mm": touchoff.get("residual_mm"),
            "cond": touchoff.get("cond"),
        } if touchoff else None,
        "paper_plane_z_m": side.get("paper_plane_z"),
    }
    if extra:
        payload.update(extra)
    return payload


def write_dataset_tool_metadata(dataset_root: Path | str, spec: ToolSpec,
                                workspace: dict | None = None, arm: str = "right",
                                extra: dict | None = None) -> Path:
    """Write ``<dataset>/meta/tool.json``. Returns the path written."""
    meta_dir = Path(dataset_root).expanduser() / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "tool.json"
    payload = dataset_tool_metadata(spec, workspace, arm, extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_dataset_tool_metadata(dataset_root: Path | str) -> dict | None:
    path = Path(dataset_root).expanduser() / "meta" / "tool.json"
    return json.loads(path.read_text()) if path.is_file() else None


# --- derivations that used to be hand-picked constants ----------------------

def derive_z_floor_m(spec: ToolSpec, workspace: dict, arm: str = "right",
                     margin_m: float = 0.010) -> dict:
    """Where a floor on the mount origin WOULD sit, given the measured
    surface and the tool's own length. Advisory: the follower bounds the TIP
    through the URDF, this only reports what the geometry implies.

    Returns the derived value together with whether it can be trusted, because
    usually it cannot yet: ``paper_plane_z`` is only the paper pad when the
    touch-off actually touched the pad (``n_pad > 0``). After a palette-only
    tip session it is the palette plane, which is a different surface at a
    different height — deriving a safety floor from it would be confidently
    wrong. Report, never silently apply.
    """
    side = workspace.get(arm) or {}
    plane = side.get("paper_plane_z")
    measured = tip_offset_m(workspace, arm)
    touchoff = side.get("touchoff") or {}
    reasons = []
    if plane is None:
        reasons.append("paper_plane_z is null — no touch-off has been written")
    if measured is None:
        reasons.append("no measured tip offset")
    if not touchoff.get("n_pad"):
        reasons.append(
            "the touch-off recorded no pad touches (n_pad = 0), so paper_plane_z is the "
            "palette plane, not the paper the pen draws on")
    if plane is None or measured is None or reasons:
        return {"z_floor_m": None, "trustworthy": False, "reasons": reasons}
    reach = sum(v * v for v in measured) ** 0.5
    plane_z = float(plane)
    return {
        "z_floor_m": round(plane_z - reach - margin_m, 6),
        "trustworthy": True,
        "reasons": [],
        "note": (f"paper_plane_z {plane_z:.6f} - tool reach {reach:.6f} - margin {margin_m:.3f}; "
                 f"tool {spec.tool_id}"),
    }


if __name__ == "__main__":  # a datasheet reader, for eyeballing
    import sys

    for name in (sys.argv[1:] or list_tools()):
        tool = load_tool(name)
        print(tool.summary())
        print(f"  profile   {[list(p) for p in tool.profile]}")
        print(f"  mount     {tool.mount or 'NONE — cannot be fitted'}")
        print(f"  mass      {tool.mass_kg} kg")
        print(f"  sha256    {tool.sha256[:16]}")
