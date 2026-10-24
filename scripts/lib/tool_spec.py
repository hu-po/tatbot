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
TOOL_GEOMETRY_VERSION = "resolved-tool-v1"
# Contact tools put material at the working point. Half a millimetre is below
# the narrowest modelled ink line and is the largest visual/FK mismatch a
# contact dataset may claim as aligned.
CONTACT_ALIGNMENT_TOLERANCE_M = 0.0005
# A fixed-point/pivot touch-off identifies the contact point in the mount
# frame when its wrist poses span enough orientations.  These are calibration
# observability gates, not claims of sub-millimetre body metrology.
CONTACT_TOUCH_MIN_SAMPLES = 4
CONTACT_TOUCH_COND_MAX = 50.0
CONTACT_TOUCH_SPREAD_MIN_DEG = 30.0
CONTACT_TOUCH_RESIDUAL_FLOOR_M = 0.0015
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
# The printed mount's bore is not a precision locator: the original printed
# mount had a ~33 mm bore with only ~20 mm of wall around the Lutin's segmented 29 mm
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
        _unique_key(parent, key, index)
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


def _unique_key(parent, key, line):
    if key in parent:
        raise ValueError(f"line {line}: duplicate key {key!r}")


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
    # the ballpoint draws on paper, the needle cartridges tattoo silicone or
    # human skin, the laser works on the silicone skin. See config/substrates.yaml.
    substrate: str = "paper_pad"
    # Every substrate this tool may work on; the default above is first.
    substrates: tuple[str, ...] = ("paper_pad",)
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
    # Radius of the physical contact element, for collision/contact modelling.
    # This is separate from display geometry: the ballpoint's visible cap is
    # oversized so it reads at wrist-camera resolution, while the steel ball
    # that touches paper is only 0.5 mm in diameter.
    contact_radius_m: float | None = None
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
    # The line this tool lays on its substrate, as drawn: a deposition
    # measurement on one substrate at one speed, recorded like `measured:`
    # with ``{width_mm, status: measured|assumed, utc, method, substrate}``.
    # The fill planner's planning width and the stroke schedule's dedup
    # footprint come from here. It is not a contact model and changes no
    # motion limit. Absent = nothing is known about the drawn line.
    line: dict = field(default_factory=dict)
    # Where a rotary machine's working tip travels: its stroke, and how far the tip stands out of the
    # cartridge's tube at the top of it, where a top-down touch jams it (negative: inside the tube). The
    # collar sets the second, so a change needs a new touch-off. A touch meets the tip at the top of its
    # stroke, and motion.yaml pen.mode ride rides a fraction of the stroke over that (tatbot_motion.pen_down).
    # None: not recorded; such a tool cannot ride.
    stroke_m: float | None = None
    tip_out_at_top_m: float | None = None
    source: Path | None = None
    sha256: str = ""
    raw: dict = field(default_factory=dict)

    # --- derived geometry ---

    @property
    def line_width_m(self) -> float | None:
        """The recorded drawn-line width in metres, or None when unrecorded."""
        return float(self.line["width_mm"]) / 1000.0 if self.line else None

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
            elif self.tip_mesh and index == len(self.profile) - 2:
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
                                  "color": override or (self.tip_color if index == len(self.profile) - 2 else self.body_color)})
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


@dataclass(frozen=True)
class ResolvedToolGeometry:
    """One runtime answer for body pose, planted point, and working TCP.

    A datasheet describes the body in axial coordinates while touch-off
    measures a point in the mount frame. Treating that point as an axis but
    leaving the nominal body at the mount origin created the fixed-mount
    ballpoint's invisible extension. URDF and metadata consumers share this
    object so the two representations cannot drift again.

    ``touch-axis-inferred`` is deliberately honest about the BODY: pivot
    calibration observes the tip, and the axisymmetric profile plus the known
    bore-face origin make the mount-to-tip vector its contact-relevant axis.
    Roll is unobservable but irrelevant for a profile of revolution.  This is
    sufficient to qualify TCP/contact geometry without pretending it is an
    independent six-DOF body measurement.
    """

    version: str
    source: str
    status: str
    measured: bool
    contact_status: str
    body_pose_status: str
    contact_qualification_error: str | None
    contact_uncertainty_m: float | None
    calibration_delta_m: tuple[float, float, float]
    body_origin_m: tuple[float, float, float]
    body_rpy_rad: tuple[float, float, float]
    body_tip_offset_m: tuple[float, float, float]
    touch_offset_m: tuple[float, float, float]
    tcp_offset_m: tuple[float, float, float]
    tcp_in_body_m: tuple[float, float, float]
    alignment_error_m: float


def _rpy_matrix(rpy: tuple[float, float, float]) -> tuple[tuple[float, float, float], ...]:
    """URDF Rz(yaw) * Ry(pitch) * Rx(roll), as a stdlib matrix."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matvec(matrix, vector) -> tuple[float, float, float]:
    res = tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)
    return (float(res[0]), float(res[1]), float(res[2]))


def _transpose_matvec(matrix, vector) -> tuple[float, float, float]:
    values = tuple(sum(matrix[row][col] * vector[row] for row in range(3))
                   for col in range(3))
    # libm differs by a few ulps across the Python 3.11/3.12 interpreters used
    # by the generators. Do not let numerical zero make the checked-in URDF
    # permanently stale on one of them.
    v0 = 0.0 if abs(values[0]) < 1e-15 else float(values[0])
    v1 = 0.0 if abs(values[1]) < 1e-15 else float(values[1])
    v2 = 0.0 if abs(values[2]) < 1e-15 else float(values[2])
    return (v0, v1, v2)


def contact_pose_qualification_error(spec: ToolSpec, workspace: dict,
                                     arm: str = "right") -> str | None:
    """Why a planted-point calibration cannot qualify contact geometry.

    The fixed-point solve identifies the mount-to-tip vector directly.  It
    does not need an external body-axis instrument for an axisymmetric contact
    tool; it does need enough varied poses and a residual compatible with the
    tool's own contact/seat budget.  Held-out disagreement is retained as an
    uncertainty diagnostic rather than a second, stricter physical model.
    """
    side = workspace.get(arm) or {}
    measured = tip_offset_m(workspace, arm)
    if measured is None:
        return "no mount-frame tip calibration"
    receipt = side.get("touchoff") or {}
    count = int(receipt.get("n_plate") or 0) + int(receipt.get("n_pad") or 0)
    if count < CONTACT_TOUCH_MIN_SAMPLES:
        return (f"touch-off has {count} planted poses; need at least "
                f"{CONTACT_TOUCH_MIN_SAMPLES}")
    required = ("cond", "residual_mm", "spread_deg")
    missing = [key for key in required if receipt.get(key) is None]
    if missing:
        return f"touch-off lacks {', '.join(missing)}"
    cond = float(receipt["cond"])
    residual_m = float(receipt["residual_mm"]) / 1000.0
    spread = float(receipt["spread_deg"])
    residual_limit = max(CONTACT_TOUCH_RESIDUAL_FLOOR_M,
                         spec.tip_radius_m / math.sqrt(2.0),
                         spec.seat_residual_m)
    if cond > CONTACT_TOUCH_COND_MAX:
        return (f"touch-off condition number {cond:.1f} exceeds "
                f"{CONTACT_TOUCH_COND_MAX:.1f}")
    if spread < CONTACT_TOUCH_SPREAD_MIN_DEG:
        return (f"touch-off rotation spread {spread:.1f} deg is below "
                f"{CONTACT_TOUCH_SPREAD_MIN_DEG:.1f} deg")
    if residual_m > residual_limit:
        return (f"touch-off residual {residual_m * 1000:.3f} mm exceeds "
                f"the tool budget {residual_limit * 1000:.3f} mm")
    offset_error = tip_offset_error_m(spec, measured)
    if offset_error > spec.tip_tolerance_m:
        return (f"tip offset differs from the datasheet by {offset_error * 1000:.3f} mm "
                f"(limit {spec.tip_tolerance_m * 1000:.3f} mm)")
    lean = axis_lean_deg(measured)
    if lean > spec.seat_tolerance_deg:
        return (f"tip axis leans {lean:.3f} deg from the bore "
                f"(limit {spec.seat_tolerance_deg:.3f} deg)")
    return None


def contact_uncertainty_m(workspace: dict, arm: str = "right") -> float | None:
    """Conservative scalar from the touch-off's retained diagnostics.

    It is metadata for domain randomisation and comparison, not a clearance
    that lets pigment appear away from collision.  Missing older diagnostics
    do not erase the ones that are present.
    """
    receipt = (workspace.get(arm) or {}).get("touchoff") or {}
    values = [receipt.get(key) for key in
              ("residual_mm", "holdout_mm", "tip_loo_max_mm")]
    finite = [float(value) for value in values
              if value is not None and math.isfinite(float(value))]
    return max(finite) / 1000.0 if finite else None


def resolved_tool_geometry(spec: ToolSpec, workspace: dict | None = None,
                           arm: str = "right", repo: Path | str = REPO,
                           tip_delta_m: tuple[float, float, float] | None = None,
                           ) -> ResolvedToolGeometry:
    """Resolve the exact geometry used by URDF, sim, and metadata consumers.

    A workspace measurement belongs only to the tool named beside it. With no
    matching touch-off this returns complete nominal geometry, including the
    concrete nominal TCP, so metadata records what actually ran rather than a
    null that callers have to guess about.
    """
    ws = workspace or {}
    measured_tip = (tip_offset_m(ws, arm)
                    if active_tool_id(repo, arm, ws) == spec.tool_id else None)
    measured = measured_tip is not None
    delta = tuple(float(value) for value in (tip_delta_m or (0.0, 0.0, 0.0)))
    if len(delta) != 3 or not all(math.isfinite(value) for value in delta):
        raise ValueError(f"tip_delta_m must be three finite metres, got {tip_delta_m!r}")
    if any(delta) and not measured:
        raise ValueError("tip_delta_m requires a measured mount-frame touch-off")
    base_touch = measured_tip or spec.touchoff_nominal_m
    touch = (base_touch[0] + delta[0], base_touch[1] + delta[1], base_touch[2] + delta[2])
    tcp = tcp_from_touchoff_m(spec, touch)

    contact_error = (contact_pose_qualification_error(spec, ws, arm)
                     if measured and spec.contact else None)
    contact_status = ("pivot-calibrated" if measured and contact_error is None
                      else "unqualified" if spec.contact else "not-applicable")
    uncertainty = contact_uncertainty_m(ws, arm) if measured else None
    if measured:
        body_rpy = axis_rpy(touch)
        rotation = _rpy_matrix(body_rpy)
        rotated_tip = _matvec(rotation, (0.0, 0.0, spec.body_tip_z_m))
        body_origin = (touch[0] - rotated_tip[0], touch[1] - rotated_tip[1], touch[2] - rotated_tip[2])
        source = "touch-axis-inferred"
        status = "contact-qualified" if contact_status == "pivot-calibrated" else "provisional"
        body_pose_status = "axis-inferred"
    else:
        body_origin = (0.0, 0.0, 0.0)
        body_rpy = (0.0, 0.0, 0.0)
        source = "datasheet-nominal"
        status = "nominal"
        body_pose_status = "nominal"

    rotation = _rpy_matrix(body_rpy)
    rotated_tip = _matvec(rotation, (0.0, 0.0, spec.body_tip_z_m))
    body_tip = (body_origin[0] + rotated_tip[0], body_origin[1] + rotated_tip[1], body_origin[2] + rotated_tip[2])
    tcp_delta = (tcp[0] - body_origin[0], tcp[1] - body_origin[1], tcp[2] - body_origin[2])
    tcp_in_body = _transpose_matvec(rotation, tcp_delta)
    separation = math.sqrt(sum((a - b) ** 2 for a, b in zip(body_tip, tcp, strict=True)))
    # A non-contact tool intentionally separates material and working point by
    # its standoff. Alignment error is only the unexplained part.
    alignment_error = abs(separation - spec.standoff_m)
    return ResolvedToolGeometry(
        version=TOOL_GEOMETRY_VERSION,
        source=source,
        status=status,
        measured=measured,
        contact_status=contact_status,
        body_pose_status=body_pose_status,
        contact_qualification_error=contact_error,
        contact_uncertainty_m=uncertainty,
        calibration_delta_m=delta,
        body_origin_m=body_origin,
        body_rpy_rad=body_rpy,
        body_tip_offset_m=body_tip,
        touch_offset_m=touch,
        tcp_offset_m=tcp,
        tcp_in_body_m=tcp_in_body,
        alignment_error_m=alignment_error,
    )


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
    if spec.contact_radius_m is not None and (
            not spec.contact or not 0 < spec.contact_radius_m < 0.01):
        raise ValueError(
            f"{source}: contact_radius_m requires a contact tool and must be in "
            f"(0, 0.01), got {spec.contact_radius_m}")
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
    _validate_line(spec.line, source)
    _validate_stroke(spec, source)
    return spec


STROKE_MM_MAX = 6.0   # the rotary machines in use: 2.5-4.5 mm cams
TIP_OUT_AT_TOP_MM_MAX = 6.0


def _validate_stroke(spec: ToolSpec, source) -> None:
    if spec.stroke_m is not None and not 0.0 < spec.stroke_m * 1000.0 <= STROKE_MM_MAX:
        raise ValueError(f"{source}: stroke_mm must be in (0, {STROKE_MM_MAX}], got {spec.stroke_m * 1000.0:g}")
    if spec.tip_out_at_top_m is not None and not abs(spec.tip_out_at_top_m * 1000.0) <= TIP_OUT_AT_TOP_MM_MAX:
        raise ValueError(f"{source}: tip_out_at_top_mm must be within +-{TIP_OUT_AT_TOP_MM_MAX} mm, got "
                         f"{spec.tip_out_at_top_m * 1000.0:g}")


LINE_WIDTH_MM_RANGE = (0.05, 2.0)
LINE_FIELDS = ("width_mm", "status", "utc", "method", "substrate")


def _validate_line(line: dict, source) -> None:
    """A `line:` block is a measurement record, complete or absent."""
    if not line:
        return
    if not isinstance(line, dict):
        raise ValueError(f"{source}: line must be a mapping {{width_mm, status, utc, method, substrate}}")
    missing = [key for key in LINE_FIELDS if line.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{source}: line is missing {', '.join(missing)}")
    width = line["width_mm"]
    low, high = LINE_WIDTH_MM_RANGE
    if isinstance(width, bool) or not isinstance(width, (int, float)) or not low < float(width) <= high:
        raise ValueError(f"{source}: line.width_mm must be in ({low}, {high}] mm, got {width!r}")
    if line["status"] not in ("measured", "assumed"):
        raise ValueError(f"{source}: line.status must be measured or assumed, got {line['status']!r}")
    if not isinstance(line["substrate"], str) or not isinstance(line["method"], str):
        raise ValueError(f"{source}: line.method and line.substrate must be text")


def tools_dir(repo: Path | str = REPO) -> Path:
    return Path(repo) / TOOLS_DIRNAME


def _substrates(data: dict, path: Path) -> tuple[str, ...]:
    """The datasheet's admitted substrates, default first and without repeats."""
    default = str(data.get("substrate", "paper_pad"))
    raw = data.get("substrates")
    if raw is None:
        return (default,)
    if not isinstance(raw, list) or not all(isinstance(v, str) and v for v in raw):
        raise ValueError(f"{path}: substrates must be a list of substrate names, got {raw!r}")
    if default not in raw:
        raise ValueError(f"{path}: substrates {raw!r} must include the default substrate {default!r}")
    return (default, *(v for v in dict.fromkeys(raw) if v != default))


def list_tools(repo: Path | str = REPO) -> list[str]:
    return sorted(p.stem for p in tools_dir(repo).glob("*.yaml"))


SUBSTRATE_SHAPES = ("pad", "cylinder")


@dataclass(frozen=True)
class Substrate:
    """What a tool works on: the paper or silicone material, as measured.

    ``shape`` says how the material is presented. A ``pad`` is a flat sheet:
    its surface profile is deliberately NOT part of this record, because the
    same sheet may be laid flat or wrapped around a support in simulation, and
    that choice belongs to the scenario distribution. A ``cylinder`` is a
    rigid fixture — paper glued around a tube — with one fixed ``radius_m``
    that the scenario cannot bend or flatten; ``thickness_m`` is then its
    diameter, so the actor origin is the axis, ``height_m`` is its length
    along the axis and ``width_m`` is the arc the drawable band spans around
    the crest.
    """

    name: str
    display_name: str
    width_m: float
    height_m: float
    thickness_m: float
    ruled: bool = False
    base_color: str | None = None
    shape: str = "pad"
    radius_m: float | None = None
    # Printed square grid pitch; None when nothing is printed.
    grid_pitch_m: float | None = None
    rule_color: str | None = None

    def rgb(self, field: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
        """A ``"r g b"`` colour field as floats in [0, 1]."""
        raw = getattr(self, field)
        if raw is None:
            return default
        parts = tuple(float(v) for v in str(raw).split())
        if len(parts) != 3 or not all(0.0 <= v <= 1.0 for v in parts):
            raise ValueError(f"substrate {self.name}: {field} must be three numbers in [0, 1], got {raw!r}")
        return parts  # type: ignore[return-value]


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
    shape = str(entry.get("shape", "pad"))
    if shape not in SUBSTRATE_SHAPES:
        raise ValueError(f"{path}: {name}: shape must be one of {SUBSTRATE_SHAPES}, got {shape!r}")
    ruled = bool(entry.get("ruled", False))
    pitch = entry.get("grid_pitch_m")
    if ruled and pitch is None:
        raise ValueError(f"{path}: {name}: a ruled substrate needs grid_pitch_m")
    if pitch is not None and not float(pitch) > 0:
        raise ValueError(f"{path}: {name}: grid_pitch_m must be positive, got {pitch!r}")
    radius = None
    thickness = float(_require(entry, "thickness_m", path))
    width = float(_require(entry, "width_m", path))
    if shape == "cylinder":
        diameter = float(_require(entry, "diameter_m", path))
        if not diameter > 0:
            raise ValueError(f"{path}: {name}: diameter_m must be positive, got {diameter!r}")
        if abs(thickness - diameter) > 1e-9:
            raise ValueError(f"{path}: {name}: a cylinder's thickness_m is its diameter "
                             f"({diameter}), got {thickness}")
        radius = diameter / 2.0
        # A band that closes on itself is no longer a chart: the arc must stay
        # short of the full circumference.
        if width >= 2 * math.pi * radius - 1e-9:
            raise ValueError(f"{path}: {name}: width_m {width} wraps the full "
                             f"circumference {2 * math.pi * radius:.5f} of a {radius} m radius")
    elif "diameter_m" in entry:
        raise ValueError(f"{path}: {name}: diameter_m belongs to a cylinder-shaped substrate")
    sub = Substrate(
        name=name,
        display_name=_require(entry, "display_name", path),
        width_m=width,
        height_m=float(_require(entry, "height_m", path)),
        thickness_m=thickness,
        ruled=ruled,
        base_color=entry.get("base_color"),
        shape=shape,
        radius_m=radius,
        grid_pitch_m=None if pitch is None else float(pitch),
        rule_color=entry.get("rule_color"),
    )
    sub.rgb("base_color", (1.0, 1.0, 1.0))
    sub.rgb("rule_color", (0.62, 0.74, 0.90))
    return sub


def list_substrates(repo: Path | str = REPO) -> list[str]:
    path = substrates_path(repo)
    data = parse_simple_yaml(path.read_text())
    return sorted(k for k, v in data.items() if isinstance(v, dict))


def substrate_for(spec: "ToolSpec", repo: Path | str = REPO, name: str | None = None) -> Substrate:
    """The substrate this tool works on: its default, or one it admits.

    A tool and its substrates are a pair on this bench. Naming one the
    datasheet does not admit (a laser on paper) is refused here rather than
    letting the scene follow a claim the bench cannot make.
    """
    chosen = name or spec.substrate
    if chosen not in spec.substrates:
        raise ValueError(
            f"tool {spec.tool_id!r} does not work on substrate {chosen!r} "
            f"(its datasheet admits: {', '.join(spec.substrates)})")
    return load_substrate(chosen, repo)


def _millimetres(data: dict, key: str, path: Path) -> float | None:
    """An optional millimetre field in metres."""
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path}: {key} must be a number of millimetres or null, got {value!r}")
    return float(value) / 1000.0


def _tool_text(path, repo, snapshot):
    if snapshot is not None:
        return snapshot.decode('utf-8')
    if not path.is_file():
        known = ", ".join(list_tools(repo)) or "none"
        raise FileNotFoundError(f"unknown tool {path.stem!r}: no {path} (known tools: {known})")
    return path.read_text()


def load_tool(tool_id: str, repo: Path | str = REPO, *, snapshot: bytes | None = None) -> ToolSpec:
    """Load and validate one tool datasheet."""
    path = tools_dir(repo) / f"{tool_id}.yaml"
    text = _tool_text(path, repo, snapshot)
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
        substrates=_substrates(data, path),
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
        contact_radius_m=data.get("contact_radius_m"),
        tcp_z_m=data.get("tcp_z_m"),
        body=data.get("body"),
        cartridge=data.get("cartridge"),
        measured=data.get("measured") or {},
        line=data.get("line") or {},
        stroke_m=_millimetres(data, "stroke_mm", path),
        tip_out_at_top_m=_millimetres(data, "tip_out_at_top_mm", path),
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
            f"unknown. Add the line by hand (known tools: {known})."
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
            f"running {tool_id!r} against them mixes two tools. Record "
            f"{tool_id!r} under `{arm}:` there with its datasheet's nominal "
            f"tip, deploy the ROS stack, and calibrate it on the station probe:\n"
            f"  tatbot ros calib run --arm {arm}")
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
    if side.get("tip_frame") != tip_frame(arm) or (side.get('tool_fit') or {}).get('status') == 'nominal':
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
    geometry = resolved_tool_geometry(spec, workspace, arm)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_geometry_version": geometry.version,
        "tool_id": spec.tool_id,
        "arm": arm,
        "spec": dict(spec.raw),
        "spec_sha256": spec.sha256,
        "protrusion_m": spec.protrusion_m,
        "contact": spec.contact,
        "contact_radius_m": spec.contact_radius_m,
        # What this run actually modelled. Nominal geometry is concrete too;
        # source/status say whether it came from measurement, inference, or
        # the datasheet instead of forcing readers to reconstruct nulls.
        "geometry_source": geometry.source,
        "geometry_status": geometry.status,
        "geometry_measured": geometry.measured,
        # Contact calibration and physical-body provenance are deliberately
        # separate.  A pivot solve qualifies the TCP used to make marks; an
        # axisymmetric profile may still place its body by inference.
        "contact_geometry_status": geometry.contact_status,
        "contact_geometry_error": geometry.contact_qualification_error,
        "contact_uncertainty_m": geometry.contact_uncertainty_m,
        "calibration_delta_m": list(geometry.calibration_delta_m),
        "body_pose_status": geometry.body_pose_status,
        "body_origin_m": list(geometry.body_origin_m),
        "body_rpy_rad": list(geometry.body_rpy_rad),
        "body_tip_offset_m": list(geometry.body_tip_offset_m),
        "tip_offset_m": list(geometry.touch_offset_m),
        "tcp_offset_m": list(geometry.tcp_offset_m),
        "tcp_in_body_m": list(geometry.tcp_in_body_m),
        "alignment_error_m": geometry.alignment_error_m,
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
            "holdout_mm": touchoff.get("holdout_mm"),
            "tip_loo_max_mm": touchoff.get("tip_loo_max_mm"),
            "cond": touchoff.get("cond"),
            "spread_deg": touchoff.get("spread_deg"),
            "n_plate": touchoff.get("n_plate"),
            "n_pad": touchoff.get("n_pad"),
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


if __name__ == "__main__":  # a datasheet reader, for eyeballing
    import sys

    for name in (sys.argv[1:] or list_tools()):
        tool = load_tool(name)
        print(tool.summary())
        print(f"  profile   {[list(p) for p in tool.profile]}")
        print(f"  mount     {tool.mount or 'NONE — cannot be fitted'}")
        print(f"  mass      {tool.mass_kg} kg")
        print(f"  sha256    {tool.sha256[:16]}")
