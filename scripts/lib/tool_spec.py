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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
BODY_POSE_REPORT_SCHEMA_VERSION = 1
BODY_POSE_MIN_RESEATS = 5
BODY_POSE_REPORT_DIR = "internal/calibration/tool-body"
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
    independent six-DOF body measurement.  Optional external body evidence can
    still replace the inferred visual/collision envelope when one exists.
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
    qualification_error: str | None


@dataclass(frozen=True)
class BodyPoseQualification:
    """Validated independent body-axis evidence for the currently seated tool.

    The tool bodies are profiles of revolution, so spin about the axis is not
    observable or geometrically relevant.  A report therefore measures the
    body origin and +z axis independently from the planted-tip solve; the
    deterministic zero-roll RPY is derived from that axis.
    """

    method: str
    measurement_source: str
    selected_cycle: int
    selected_utc: str
    selected_session: str
    sample_count: int
    body_origin_m: tuple[float, float, float]
    body_axis_unit: tuple[float, float, float]
    body_rpy_rad: tuple[float, float, float]
    tip_offset_m: tuple[float, float, float]
    endpoint_alignment_max_m: float
    tip_repeatability_max_m: float
    origin_repeatability_max_m: float
    axis_repeatability_max_deg: float


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


def _workspace_triplet(side: dict, prefix: str) -> tuple[float, float, float] | None:
    vx = side.get(f"{prefix}_x")
    vy = side.get(f"{prefix}_y")
    vz = side.get(f"{prefix}_z")
    if vx is None or vy is None or vz is None:
        return None
    return (float(vx), float(vy), float(vz))


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
    side = ws.get(arm) or {}
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

    triplet_origin = _workspace_triplet(side, "tool_body_origin") if measured else None
    triplet_rpy = _workspace_triplet(side, "tool_body_rpy") if measured else None
    explicit_frame = side.get("tool_body_frame")
    qualification_error = None
    contact_error = (contact_pose_qualification_error(spec, ws, arm)
                     if measured and spec.contact else None)
    contact_status = ("pivot-calibrated" if measured and contact_error is None
                      else "unqualified" if spec.contact else "not-applicable")
    uncertainty = contact_uncertainty_m(ws, arm) if measured else None
    qualification = None
    if triplet_origin is not None or triplet_rpy is not None or side.get("tool_body_status"):
        try:
            qualification = body_pose_qualification(spec, ws, arm, repo)
        except ValueError as exc:
            qualification_error = str(exc)
    if qualification is not None and explicit_frame == tip_frame(arm) and not any(delta):
        body_origin = qualification.body_origin_m
        body_rpy = qualification.body_rpy_rad
        source = "workspace-body-pose"
        status = "qualified"
        body_pose_status = "independent-qualified"
    elif measured:
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
        qualification_error=qualification_error,
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
        contact_radius_m=data.get("contact_radius_m"),
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


def _vector3(value, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"body report {label} must be a three-value JSON array")
    try:
        vector = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"body report {label} contains a non-number") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"body report {label} contains a non-finite number")
    return vector


def _distance3(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _axis_angle_deg(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _utc_timestamp(value, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"body report {label} must be an ISO-8601 UTC timestamp ending Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"body report {label} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"body report {label} must be UTC")
    return parsed


def validate_body_pose_report(spec: ToolSpec, report: dict, arm: str = "right",
                              expected_tip: tuple[float, float, float] | None = None,
                              expected_session: str | None = None
                              ) -> BodyPoseQualification:
    """Validate an independent remove/reseat body-axis study.

    Each cycle supplies two independent observations in the mount frame: the
    body profile's origin/+z axis, and the planted working-tip offset.  The
    report is useful only if those agree for every cycle and the selected last
    cycle is the touch-off currently recorded in workspace.yaml.
    """
    if not isinstance(report, dict):
        raise ValueError("body report root must be a JSON object")
    if report.get("schema_version") != BODY_POSE_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"body report schema_version must be {BODY_POSE_REPORT_SCHEMA_VERSION}")
    if report.get("tool_id") != spec.tool_id:
        raise ValueError(
            f"body report tool_id {report.get('tool_id')!r} != fitted {spec.tool_id!r}")
    if report.get("arm") != arm:
        raise ValueError(f"body report arm {report.get('arm')!r} != {arm!r}")
    if report.get("frame") != tip_frame(arm):
        raise ValueError(
            f"body report frame {report.get('frame')!r} != {tip_frame(arm)!r}")
    method = report.get("method")
    if (not isinstance(method, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", method) is None
            or method == "touch-axis-inferred"):
        raise ValueError(
            "body report method must be a stable lowercase method id, not the tip fit")
    source = report.get("measurement_source")
    if (not isinstance(source, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", source) is None):
        raise ValueError(
            "body report measurement_source must be a stable lowercase instrument id")
    if report.get("independent_of_tip_fit") is not True:
        raise ValueError(
            "body report must assert independent_of_tip_fit: true; a planted tip alone "
            "does not observe the body axis/origin")

    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) < BODY_POSE_MIN_RESEATS:
        count = len(samples) if isinstance(samples, list) else 0
        raise ValueError(
            f"body report has {count} reseat samples; need at least {BODY_POSE_MIN_RESEATS}")
    selected_cycle = report.get("selected_cycle")
    if isinstance(selected_cycle, bool) or not isinstance(selected_cycle, int):
        raise ValueError("body report selected_cycle must be an integer")

    parsed = []
    seen_cycles: set[int] = set()
    last_timestamp: datetime | None = None
    for index, sample in enumerate(samples):
        label = f"samples[{index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"body report {label} must be an object")
        cycle = sample.get("cycle")
        if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 1:
            raise ValueError(f"body report {label}.cycle must be a positive integer")
        if cycle in seen_cycles:
            raise ValueError(f"body report cycle {cycle} is duplicated")
        seen_cycles.add(cycle)
        timestamp = _utc_timestamp(sample.get("utc"), f"{label}.utc")
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValueError("body report sample UTC timestamps must be strictly increasing")
        last_timestamp = timestamp
        session = sample.get("touchoff_session")
        if not isinstance(session, str) or not session.strip():
            raise ValueError(f"body report {label}.touchoff_session must be non-empty")
        origin = _vector3(sample.get("body_origin_m"), f"{label}.body_origin_m")
        axis_raw = _vector3(sample.get("body_axis_unit"), f"{label}.body_axis_unit")
        axis_norm = math.sqrt(sum(value * value for value in axis_raw))
        if abs(axis_norm - 1.0) > 0.001:
            raise ValueError(
                f"body report {label}.body_axis_unit norm is {axis_norm:.6f}, need 1 +/- 0.001")
        axis = tuple(value / axis_norm for value in axis_raw)
        tip = _vector3(sample.get("tip_offset_m"), f"{label}.tip_offset_m")
        endpoint = tuple(origin[i] + axis[i] * spec.body_tip_z_m for i in range(3))
        alignment = _distance3(endpoint, tip)
        if alignment > CONTACT_ALIGNMENT_TOLERANCE_M:
            raise ValueError(
                f"body report cycle {cycle} endpoint is {alignment * 1000:.3f} mm "
                f"from its independently planted tip; maximum is "
                f"{CONTACT_ALIGNMENT_TOLERANCE_M * 1000:.3f} mm")
        nominal_error = tip_offset_error_m(spec, tip)
        if nominal_error > spec.tip_tolerance_m:
            raise ValueError(
                f"body report cycle {cycle} tip is {nominal_error * 1000:.1f} mm "
                f"from datasheet nominal; tolerance is {spec.tip_tolerance_m * 1000:.1f} mm")
        lean = _axis_angle_deg(axis, (0.0, 0.0, 1.0))
        if lean > spec.seat_tolerance_deg:
            raise ValueError(
                f"body report cycle {cycle} axis leans {lean:.2f} deg; seat tolerance "
                f"is {spec.seat_tolerance_deg:.2f} deg")
        parsed.append({
            "cycle": cycle,
            "utc": sample["utc"],
            "session": session,
            "origin": origin,
            "axis": axis,
            "tip": tip,
            "endpoint": endpoint,
            "alignment": alignment,
        })

    selected = next((sample for sample in parsed
                     if sample["cycle"] == selected_cycle), None)
    if selected is None:
        raise ValueError(f"body report selected_cycle {selected_cycle} is not in samples")
    if selected is not parsed[-1]:
        raise ValueError(
            "body report selected_cycle must be the final reseat/current physical seat")
    if expected_tip is not None:
        delta = _distance3(selected["tip"], expected_tip)
        if delta > CONTACT_ALIGNMENT_TOLERANCE_M:
            raise ValueError(
                f"body report selected tip differs from current workspace touch-off by "
                f"{delta * 1000:.3f} mm; rerun the final touch-off or use its report")
    if expected_session is not None and selected["session"] != expected_session:
        raise ValueError(
            f"body report selected touch-off session {selected['session']!r} != current "
            f"workspace session {expected_session!r}")

    endpoint_alignment = max(sample["alignment"] for sample in parsed)
    tip_repeatability = max(_distance3(sample["tip"], selected["tip"])
                            for sample in parsed)
    origin_repeatability = max(_distance3(sample["origin"], selected["origin"])
                               for sample in parsed)
    axis_repeatability = max(_axis_angle_deg(sample["axis"], selected["axis"])
                             for sample in parsed)
    seat_repeat_limit = max(spec.seat_residual_m, CONTACT_ALIGNMENT_TOLERANCE_M)
    if tip_repeatability > seat_repeat_limit:
        raise ValueError(
            f"body report reseat tip spread is {tip_repeatability * 1000:.3f} mm; "
            f"seat repeatability limit is {seat_repeat_limit * 1000:.3f} mm")
    if origin_repeatability > seat_repeat_limit:
        raise ValueError(
            f"body report reseat origin spread is {origin_repeatability * 1000:.3f} mm; "
            f"seat repeatability limit is {seat_repeat_limit * 1000:.3f} mm")
    if axis_repeatability > spec.seat_tolerance_deg:
        raise ValueError(
            f"body report reseat axis spread is {axis_repeatability:.3f} deg; "
            f"seat tolerance is {spec.seat_tolerance_deg:.3f} deg")

    return BodyPoseQualification(
        method=method,
        measurement_source=source,
        selected_cycle=selected_cycle,
        selected_utc=selected["utc"],
        selected_session=selected["session"],
        sample_count=len(parsed),
        body_origin_m=selected["origin"],
        body_axis_unit=selected["axis"],
        body_rpy_rad=axis_rpy(selected["axis"]),
        tip_offset_m=selected["tip"],
        endpoint_alignment_max_m=endpoint_alignment,
        tip_repeatability_max_m=tip_repeatability,
        origin_repeatability_max_m=origin_repeatability,
        axis_repeatability_max_deg=axis_repeatability,
    )


def body_pose_qualification(spec: ToolSpec, workspace: dict, arm: str = "right",
                            repo: Path | str = REPO) -> BodyPoseQualification:
    """Load and revalidate the report bound to workspace.yaml.

    A `qualified` word and six coordinates are not evidence.  The report must
    live in the repository, match its recorded digest, pass every physical
    gate again, and agree byte-for-byte/numerically with the workspace summary.
    """
    side = workspace.get(arm) or {}
    if side.get("tool_body_status") != "qualified":
        raise ValueError("tool body status is not qualified")
    if side.get("tool_body_frame") != tip_frame(arm):
        raise ValueError(f"tool body frame must be {tip_frame(arm)}")
    report_rel = side.get("tool_body_report")
    if not isinstance(report_rel, str) or not report_rel:
        raise ValueError("qualified tool body has no report path")
    relpath = Path(report_rel)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise ValueError("tool body report path must stay inside the repository")
    root = Path(repo).resolve()
    report_path = (root / relpath).resolve()
    try:
        report_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("tool body report resolves outside the repository") from exc
    if not report_path.is_file():
        raise ValueError(f"tool body report is missing: {report_rel}")
    report_bytes = report_path.read_bytes()
    digest = hashlib.sha256(report_bytes).hexdigest()
    if side.get("tool_body_report_sha256") != digest:
        raise ValueError("tool body report SHA-256 does not match workspace")
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("tool body report is not valid UTF-8 JSON") from exc
    current_tip = tip_offset_m(workspace, arm)
    if current_tip is None:
        raise ValueError("qualified tool body has no current touch-off")
    current_session = (side.get("touchoff") or {}).get("session")
    if not current_session:
        raise ValueError("qualified tool body touch-off has no session id")
    qualification = validate_body_pose_report(
        spec, report, arm, expected_tip=current_tip, expected_session=current_session)

    string_fields = {
        "tool_body_utc": qualification.selected_utc,
        "tool_body_method": qualification.method,
        "tool_body_measurement_source": qualification.measurement_source,
    }
    for field_name, expected in string_fields.items():
        if side.get(field_name) != expected:
            raise ValueError(f"{field_name} does not match the body report")
    integer_fields = {
        "tool_body_samples": qualification.sample_count,
        "tool_body_selected_cycle": qualification.selected_cycle,
    }
    for field_name, expected in integer_fields.items():
        if side.get(field_name) != expected:
            raise ValueError(f"{field_name} does not match the body report")
    vector_fields = {
        "tool_body_origin": qualification.body_origin_m,
        "tool_body_rpy": qualification.body_rpy_rad,
    }
    for prefix, expected in vector_fields.items():
        actual = _workspace_triplet(side, prefix)
        # workspace.yaml serializes metres/radians to six decimals; allow only
        # that sub-micrometre/sub-microradian representation loss.
        if actual is None or _distance3(actual, expected) > 1e-6:
            raise ValueError(f"{prefix} does not match the body report")
    metric_fields = {
        "tool_body_alignment_max_mm": qualification.endpoint_alignment_max_m * 1000,
        "tool_body_tip_repeatability_mm": qualification.tip_repeatability_max_m * 1000,
        "tool_body_origin_repeatability_mm": qualification.origin_repeatability_max_m * 1000,
        "tool_body_axis_repeatability_deg": qualification.axis_repeatability_max_deg,
    }
    for field_name, expected in metric_fields.items():
        val = side.get(field_name)
        if val is None:
            raise ValueError(f"{field_name} is missing from workspace")
        try:
            actual = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} is invalid in workspace") from exc
        if not math.isclose(actual, expected, abs_tol=5e-7):
            raise ValueError(f"{field_name} does not match the body report")
    return qualification


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
        "geometry_qualification_error": geometry.qualification_error,
        "body_pose_qualification": {
            "status": side.get("tool_body_status"),
            "utc": side.get("tool_body_utc"),
            "method": side.get("tool_body_method"),
            "measurement_source": side.get("tool_body_measurement_source"),
            "report": side.get("tool_body_report"),
            "report_sha256": side.get("tool_body_report_sha256"),
            "samples": side.get("tool_body_samples"),
            "selected_cycle": side.get("tool_body_selected_cycle"),
            "alignment_max_mm": side.get("tool_body_alignment_max_mm"),
            "tip_repeatability_mm": side.get("tool_body_tip_repeatability_mm"),
            "origin_repeatability_mm": side.get("tool_body_origin_repeatability_mm"),
            "axis_repeatability_deg": side.get("tool_body_axis_repeatability_deg"),
        },
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
