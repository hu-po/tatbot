"""Each drawing arm's fitted tool in its model's URDF (urdf/tatbot.urdf, urdf/xarm7.urdf), where FK sees it.

The real rig's model used to stop at an empty ``<arm>/ee_gripper_link`` while
the tool hung 60 mm past it, unmodelled; everything downstream carried the
tool's length itself. Since 2026-08-30 the tool sits in a mount on the left
finger carriage — ``<arm>/tool_mount``, a hand-placed link in the URDF whose
origin is the bore face and whose +z is the bore axis — and a marked block at
the end of the URDF hangs the datasheet's body off it:

  <arm>/tattoo_pen      the tool body, from the datasheet profile
  <arm>/tattoo_needle   the TCP, at the MEASURED tip when a touch-off in the
                        mount frame exists, else at the datasheet's nominal

Because the mount is downstream of ``left_carriage_joint``, the needle's FK
now depends on the carriage reading — which is the point: opening the
carriage retracts the pen, and the model says so.

The block is purely additive — existing links, joints and their transforms are
untouched, so no existing FK answer changes. scripts/gen_tool_urdf.py writes
it into the URDFs in git from config/workspace.yaml; the robot description
renders it again from the workspace as the ROS owner reads it (`rendered`), so
the cartridge fitted there and its tip are modelled without a commit.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import arm_model
import tool_spec
from tatbot_cli import arms as arm_registry

MESH_DIR = "meshes/ee"
BEGIN = "  <!-- BEGIN generated tool block — scripts/gen_tool_urdf.py, do not edit -->"
END = "  <!-- END generated tool block -->"


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
         "       answers about the tip are good to the datasheet, not to a measurement.",
         "       Run the tip phase."])
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


def drawing_arms(repo: Path, workspace: dict) -> dict[Path, list[str]]:
    """{a model's URDF: its arms with a tool fitted}, for every configured arm its model's URDF carries (its base link
    there: the left, the teleop leader alone, is not modelled), every such URDF listed."""
    out: dict[Path, list[str]] = {}
    for arm, physical in arm_registry.load(repo).items():
        path = Path(repo) / arm_model.for_arm(arm, repo).urdf
        if f"{physical.urdf_prefix}/base_link" not in mount_links(path.read_text()):
            continue
        out.setdefault(path, [])
        if (workspace.get(physical.workspace_section) or {}).get("tool_id"):
            out[path].append(arm)
    return out


def tip(repo: Path, arm: str, workspace: dict):
    """(the fitted tool's datasheet, the tip its block hangs at, whether that tip was measured), or ValueError for a
    tool the model cannot carry: no mount link for it, or a measured tip too far from the datasheet's or leaning off
    the bore."""
    tool_id = tool_spec.active_tool_id(repo, arm, workspace)
    spec = tool_spec.load_tool(tool_id, repo)
    spec.mount_frame(arm)  # ToolMountError for a tool with no mount
    measured = tool_spec.tip_offset_m(workspace, arm)
    if measured is None:
        return spec, spec.touchoff_nominal_m, False
    error_m = tool_spec.tip_offset_error_m(spec, measured)
    if error_m > spec.tip_tolerance_m:
        raise ValueError(f"{arm}: the measured tip is {error_m * 1000:.1f} mm from {tool_id}'s nominal tip — refusing "
                         "to model a tool the calibration disagrees with. Measure its tip (`tatbot ros calib sweep`) "
                         "or fit the cartridge in the mount (`tatbot ros cartridge fit`).")
    lean = tool_spec.axis_lean_deg(measured)
    if lean > spec.seat_tolerance_deg:
        raise ValueError(f"{arm}: the measured tip leans {lean:.1f} deg off the mount's bore axis (tolerance "
                         f"{spec.seat_tolerance_deg:.0f}) — the tool is seated crooked or the mount transform in the "
                         "URDF is wrong. Fix that, not this.")
    return spec, measured, True


def build(repo: Path, path: Path, fitted: list[str], workspace: dict) -> str:
    """One URDF with its arms' blocks, stripped and re-rendered whole; an arm with no fitted tool gets none."""
    text = strip_block(path.read_text())
    links = mount_links(text)
    blocks = []
    for arm in fitted:
        spec, at, measured = tip(repo, arm, workspace)
        if spec.mount_frame(arm) not in links:
            raise ValueError(f"{arm}: {spec.tool_id} mounts on {spec.mount_frame(arm)}, which {path.name} does not "
                             "define — add the mount link (hand-placed, see the 2026-08-30 block).")
        blocks.append(render_block(arm, spec, at, measured, workspace))
    if not blocks:
        return text
    closing = text.rindex("</robot>")
    return text[:closing] + "\n".join(blocks) + "\n\n" + text[closing:]


def rendered(repo: Path, path: Path) -> str:
    """The model's URDF with each of its arms' tools as the workspace reads here (tool_spec.read_workspace): the
    committed file, byte for byte, in a checkout whose URDF is current; the fitted cartridge's on its ROS owner."""
    workspace = tool_spec.read_workspace(repo)
    fitted = drawing_arms(repo, workspace).get(Path(path), [])
    return build(repo, Path(path), fitted, workspace)
