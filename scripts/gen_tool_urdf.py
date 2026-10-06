#!/usr/bin/env python3
"""Put the fitted tool into urdf/tatbot.urdf, where FK can see it.

    gen_tool_urdf.py [--check] [--arm right]

The real rig's model used to stop at an empty ``<arm>/ee_gripper_link`` while
the tool hung 60 mm past it, unmodelled; everything downstream carried the
tool's length itself. Since 2026-08-30 the tool sits in a mount on the left
finger carriage — ``<arm>/tool_mount``, a hand-placed link in the URDF whose
origin is the bore face and whose +z is the bore axis — and this regenerates
a marked block at the end of the URDF hanging the datasheet's body off it:

  <arm>/tattoo_pen      the tool body, from the datasheet profile
  <arm>/tattoo_needle   the TCP, at the MEASURED tip when a touch-off in the
                        mount frame exists, else at the datasheet's nominal

Because the mount is downstream of ``left_carriage_joint``, the needle's FK
now depends on the carriage reading — which is the point: opening the
carriage retracts the pen, and the model says so.

The block is purely additive — existing links, joints and their transforms are
untouched, so no existing FK answer changes. Re-run it after a touch-off or a
tool swap; ``--check`` verifies the file is current without writing (exit 1 if
it is stale), which is what `scripts/check tools` runs (with `tool sync` and
`export_wrist_tags.py --check`).
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

import tool_spec  # noqa: E402

URDF_PATH = REPO / "urdf" / "tatbot.urdf"
MESH_DIR = "meshes/ee"
BEGIN = "  <!-- BEGIN generated tool block — scripts/gen_tool_urdf.py, do not edit -->"
END = "  <!-- END generated tool block -->"


axis_rpy = tool_spec.axis_rpy  # one definition, shared with the sim's URDF builder


def _fmt(values) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def render_block(arm: str, spec, tip_offset, measured: bool,
                 workspace: dict | None = None) -> str:
    """The generated XML for one arm's tool.

    ``tip_offset`` is where the tip is in the mount frame: what the touch-off
    planted on the surface (the needle for a contact tool, the aperture for one
    that works at a distance), or the datasheet's nominal when no touch-off in
    this frame exists yet. The TCP link goes at the working point, which for a
    non-contact tool is the standoff further on.
    """
    if workspace is None:
        workspace = ({arm: {
            "tool_id": spec.tool_id,
            "tip_frame": tool_spec.tip_frame(arm),
            "pen_tip_offset_x": tip_offset[0],
            "pen_tip_offset_y": tip_offset[1],
            "pen_tip_offset_z": tip_offset[2],
        }} if measured else {})
    geometry = tool_spec.resolved_tool_geometry(spec, workspace, arm)
    tcp = geometry.tcp_offset_m
    reach = math.sqrt(sum(v * v for v in tcp))
    provenance = (
        ["       Body profile and TCP resolve from the same measured touch-off,",
         f"       {reach * 1000:.2f} mm out along an axis {tool_spec.axis_lean_deg(tcp):.1f} deg",
         f"       off the mount's nominal +z (datasheet nominal {spec.protrusion_m * 1000:.0f} mm;",
         f"       source {geometry.source}; visual/TCP error "
         f"{geometry.alignment_error_m * 1000:.3f} mm)."]
        if measured else
        [f"       NOMINAL: no touch-off in {spec.mount_frame(arm)} yet, so the needle",
         f"       sits at the datasheet's {reach * 1000:.0f} mm along the bore axis. FK",
         "       answers about the tip are good to the datasheet, not to a measurement;",
         "       require_z_floor refuses to fly on this. Run the tip phase."])
    lines = [
        BEGIN,
        f"  <!-- {spec.display_name} ({spec.tool_id}), datasheet "
        f"config/tools/{spec.tool_id}.yaml sha256 {spec.sha256[:12]}.",
        *provenance,
        "       Regenerate with scripts/gen_tool_urdf.py. -->",
        f'  <link name="{arm}/tattoo_pen">',
    ]
    for part in spec.geometry_parts():
        lines.append("    <visual>")
        lines.append(f'      <origin rpy="0 0 0" xyz="{part.get("x", 0.0):.9g} '
                     f'{part.get("y", 0.0):.9g} {part["z"]:.9g}"/>')
        lines.append("      <geometry>")
        if part["kind"] == "mesh":
            lines.append(f'        <mesh filename="{MESH_DIR}/{part["mesh"]}" '
                         f'scale="{_fmt(part["scale"])}"/>')
        elif part["kind"] == "sphere":
            lines.append(f'        <sphere radius="{part["radius"]:.9g}"/>')
        else:
            lines.append(f'        <cylinder length="{part["length"]:.9g}" '
                         f'radius="{part["radius"]:.9g}"/>')
        lines.append("      </geometry>")
        lines.append("    </visual>")
    lines += [
        "  </link>",
        f'  <joint name="{arm}/tattoo_pen_joint" type="fixed">',
        f'    <origin rpy="{_fmt(geometry.body_rpy_rad)}" '
        f'xyz="{_fmt(geometry.body_origin_m)}"/>',
        f'    <parent link="{spec.mount_frame(arm)}"/>',
        f'    <child link="{arm}/tattoo_pen"/>',
        "  </joint>",
        f'  <link name="{arm}/tattoo_needle">',
    ]
    if spec.contact_radius_m is not None:
        radius = float(spec.contact_radius_m)
        lines += [
            "    <collision>",
            f'      <origin rpy="0 0 0" xyz="0 0 {-radius:.9g}"/>',
            "      <geometry>",
            f'        <sphere radius="{radius:.9g}"/>',
            "      </geometry>",
            "    </collision>",
        ]
    lines += [
        "  </link>",
        f'  <joint name="{arm}/tattoo_needle_joint" type="fixed">',
        f'    <origin rpy="0 0 0" xyz="{_fmt(geometry.tcp_in_body_m)}"/>',
        f'    <parent link="{arm}/tattoo_pen"/>',
        f'    <child link="{arm}/tattoo_needle"/>',
        "  </joint>",
        END,
    ]
    return "\n".join(lines)


def strip_block(text: str) -> str:
    """Remove any previously generated block, leaving the rest byte-identical."""
    while BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        # also eat the newline the block sits on
        while end < len(text) and text[end] == "\n":
            end += 1
        text = text[:start] + text[end:]
    return text


def mount_links(text: str) -> set[str]:
    """Every link the hand-written URDF defines, to check the mount exists."""
    return {link.get("name") for link in ET.fromstring(text).iter("link") if link.get("name") is not None}


ARM_IDS = ("right", "left")


def fitted_arms(workspace: dict) -> list[str]:
    """Every arm whose workspace section names a fitted tool, follower first."""
    return [arm for arm in ARM_IDS if (workspace.get(arm) or {}).get("tool_id")]


def _require_fitted(workspace: dict, arms: list[str] | None) -> list[str]:
    """The fitted arms, after checking they are exactly the ones asked for."""
    fitted = fitted_arms(workspace)
    requested = list(fitted) if arms is None else list(dict.fromkeys(arms))
    unfitted = [arm for arm in requested if arm not in fitted]
    if unfitted:
        raise SystemExit(
            f"{unfitted[0]}: no tool_id under `{unfitted[0]}:` in config/workspace.yaml — "
            f"nothing to model for that arm (fitted: {', '.join(fitted) or 'none'})")
    unasked = [arm for arm in fitted if arm not in requested]
    if unasked:
        raise SystemExit(
            f"{unasked[0]} carries {tool_spec.active_tool_id(REPO, unasked[0], workspace)} but "
            f"was not asked for — the URDF models every fitted tool, so drop --arm or name it too")
    return fitted


def build(arms: list[str] | None = None) -> str:
    """Render one block per fitted arm; ``arms`` names the ones the caller
    expects to be fitted and is checked against the workspace.

    The generated blocks are stripped and re-rendered together, so the URDF
    always models every fitted tool: an arm asked for without a fitted tool,
    or a fitted arm left out of the request, is a refusal rather than a skip.
    A lone ``--arm left`` used to strip the right arm's block silently."""
    workspace = tool_spec.read_workspace(REPO)
    fitted = _require_fitted(workspace, arms)
    text = strip_block(URDF_PATH.read_text())
    links = mount_links(text)
    blocks = []
    for arm in fitted:
        tool_id = tool_spec.active_tool_id(REPO, arm, workspace)
        spec = tool_spec.load_tool(tool_id, REPO)
        mount = spec.mount_frame(arm)  # ToolMountError for a tool with no mount
        if mount not in links:
            raise SystemExit(
                f"{arm}: {tool_id} mounts on {mount}, which urdf/tatbot.urdf does not "
                f"define — add the mount link (hand-placed, see the 2026-08-30 block).")
        tip = tool_spec.tip_offset_m(workspace, arm)
        measured = tip is not None
        if measured:
            error_m = tool_spec.tip_offset_error_m(spec, tip)
            if error_m > spec.tip_tolerance_m:
                raise SystemExit(
                    f"{arm}: the measured tip is {error_m * 1000:.1f} mm from {tool_id}'s "
                    f"nominal tip — refusing to model a tool the calibration disagrees with. "
                    f"Re-run il_touchoff.py with the right --tool-id.")
            lean = tool_spec.axis_lean_deg(tip)
            if lean > spec.seat_tolerance_deg:
                raise SystemExit(
                    f"{arm}: the measured tip leans {lean:.1f} deg off the mount's bore axis "
                    f"(tolerance {spec.seat_tolerance_deg:.0f}) — the tool is seated "
                    "crooked or the mount transform in the URDF is wrong. Fix that, not this.")
        else:
            tip = spec.touchoff_nominal_m
            print(f"  {arm}: no touch-off in {mount} yet — modelling {tool_id} at its "
                  "datasheet nominal", file=sys.stderr)
        blocks.append(render_block(arm, spec, tip, measured, workspace))
        print(f"  {arm}: {spec.summary()}")
    if not blocks:
        return text
    closing = text.rindex("</robot>")
    return text[:closing] + "\n".join(blocks) + "\n\n" + text[closing:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None, choices=ARM_IDS,
                    help="arm expected to carry a fitted tool (repeatable); the URDF always "
                         "models every arm with a tool_id in config/workspace.yaml, and this "
                         "refuses when that set differs")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the URDF is not what this would generate")
    args = ap.parse_args()

    rendered = build(args.arm)
    current = URDF_PATH.read_text()
    if args.check:
        if rendered != current:
            print("urdf/tatbot.urdf is stale — run scripts/gen_tool_urdf.py", file=sys.stderr)
            return 1
        print("urdf/tatbot.urdf is current")
        return 0
    if rendered == current:
        print("urdf/tatbot.urdf already current")
        return 0
    URDF_PATH.write_text(rendered)
    print(f"wrote {URDF_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
