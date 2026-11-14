#!/usr/bin/env python3
"""urdf/xarm7.urdf, the UFACTORY xArm 7 the stack drives as the physical arm `xarm`, from UFACTORY's description.

    gen_arm_urdf.py XARM_ROS2_CHECKOUT

XARM_ROS2_CHECKOUT is a clone of xArm-Developer/xarm_ros2 at PIN (its jazzy branch on 2026-10-20). xacro (ROS 2
Jazzy) expands its xarm_device.urdf.xacro for a 1305-series xArm 7 under the prefix `xarm/`, and the result is put in
the stack's shape, as urdf/tatbot.urdf is for the WXAI:

- `xarm/base_link` (UFACTORY's `link_base`) hangs from `root` on `xarm/mount_joint`, at the WXAI follower's nominal
  mount until the merged rig places it; an arm registration places it on the rig.
- `xarm/link1`..`xarm/link7` and `xarm/joint1`..`xarm/joint7` keep UFACTORY's limits and inertials, and this arm's own
  kinematics: config/xarm-kinematics.yaml, its factory calibration as its control box reports it, in place of the
  model's nominal ones (xarm_ros2's `kinematics_suffix`). The model's 1305 meshes are copied to
  urdf/meshes/xarm7_1305/ (collision OBJ, visual STL) and named relative to urdf/.
- `xarm/tool_mount` is the nominal end effector, R7 (a printed mount): the pen on link 7's axis, its
  bore face where the 3RL's tube end measured 168.3 mm from the flange (2026-11-05) less the datasheet's 73 mm
  protrusion. `xarm/end_effector` is R7's core as one collision cylinder, 80 mm across from the flange to that bore
  face (its 72 mm wrist adapter and the pen's cradle inside it), and with it the dual-D405 bracket as boxes, one to a
  slab of its mesh along link 7's x (bracket_collision). The fiducial cube is the collision scene's, from its tags.
  Its visuals are what the arm carries for the merged rig: R7's wrist adapter and pen cradle
  (urdf/meshes/ee/r7/) with the dual-D405 bracket in place of R7's D435i bracket, holding R7's re-posed fiducial cube
  (urdf/meshes/ee/xarm-dual-d405-v5/, its part poses in provenance.json there). R7's camera cable and bracket keep
  `xarm/joint7` to -80..+85 deg, its limits here, so every planner keeps to them.
- Both wrist D405s ride link 7 in the bracket's carriers, each chained as the WXAI's are (`xarm/realsense2_*`, the
  session camera, and `xarm/realsense1_*`, by their vision.toml names): their case, camera link and colour and depth
  optical frames. Each case is its own collision, the stock mesh as on the WXAI, and its camera link carries the
  camera's body as a box and a cylinder where the cable leaves the case (camera_collision). The bracket's boxes and
  the cables' cylinders are COLLISION_MARGIN_M larger each way than what they enclose; a case and a body are as CAD
  has them.
- UFACTORY's world link, `link_eef`, ros2_control, transmissions and Gazebo plugins go: the stack adds its own.

The fitted tool's block is scripts/gen_tool_urdf.py's, as for the WXAI: run it after this. The fiducial blocks
scripts/vision/export_wrist_tags.py writes (the cube's tags) carry over from the URDF this replaces.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import stl_extent  # noqa: E402
from urdf_wrist_blocks import urdf_blocks  # noqa: E402

OUT = REPO / "urdf" / "xarm7.urdf"
MESHES = Path("meshes") / "xarm7_1305"   # relative to urdf/, as tatbot.urdf names its meshes
PIN = "d837f2e17f7f36043103b367fca3af8172626245"
PREFIX = "xarm/"
# The xacro reads the model from an argument in a serial's format: XS = xArm 7, 1305 = its series. This is a model
# code padded to that format, no unit's serial; a non-digit at index 8 leaves the date unparsed, so the default
# inertials apply.
MODEL_CODE = "XS1305" + "00D00000"
KINEMATICS = REPO / "config" / "xarm-kinematics.yaml"
KINEMATICS_SUFFIX = "tatbot"   # names it in the package the xacro reads: config/kinematics/user/xarm7_kinematics_<it>.yaml
MOUNT_FROM = "right/mount_joint"   # urdf/tatbot.urdf's WXAI follower mount: the xArm's nominal place on the rig
TOOL_MOUNT_Z_M = 0.1683 - 0.073    # R7: tube end from the flange, less the 3RL datasheet's protrusion
CORE_RADIUS_M = 0.040              # R7's core envelope: its 72 mm wrist adapter and the pen's cradle within
JOINT7_WINDOW_RAD = (-1.3962634, 1.4835299)   # R7's cable and bracket, -80..+85 deg (2026-11-04)
R7_MESHES = Path("meshes") / "ee" / "r7"   # relative to urdf/: R7's printed parts in its CAD frame (mm)
DUAL_D405 = Path("meshes") / "ee" / "xarm-dual-d405-v5"   # the bracket carrying both D405s, and its part poses
R7_RETAINED = ("wrist-adapter", "pen-bracket", "pen-cap")   # R7's parts the dual-D405 bracket keeps
LINK7_FROM_CAD = ((1, 0, 0), (0, 0, 1), (0, -1, 0))   # rows: CAD X, Y, Z -> link 7 X, -Z, Y about the flange centre
R7_RGBA = "0.13 0.36 0.78 1"        # R7's printed parts and its cube, printed blue
BRACKET_RGBA = "0.93 0.93 0.91 1"   # the dual-D405 bracket, printed white
CAMERA_RGBA = "0.12 0.12 0.13 1"    # the D405 cases and bodies, as on the WXAI
# Inside a Trossen D405 case (urdf/meshes/wxai/camera_mount_d405.stl), the camera body's mesh, as urdf/tatbot.urdf
# chains it on the WXAI follower: case -> bottom screw frame -> camera link -> its visual.
CASE_TO_SCREW = ((0.02927207801, 0.0, 0.03824951197), (0.0, 0.3490658503988659, 0.0))
SCREW_TO_CAMERA = ((0.01085, 0.009, 0.021), (0.0, 0.0, 0.0))
CAMERA_TO_BODY = ((0.0038, -0.009, 0.0), (1.5707963267948966, 0.0, 1.5707963267948966))
OPTICAL = ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, -math.pi / 2))   # a camera frame -> its optical frame (z forward)
# The D405 in each of the bracket's carrier poses, by its frames' prefix (xarm/<prefix>_color_optical_frame, ...),
# the camera's vision.toml name: RealSense 2 (wrist_2, the session's) rides carrier 1 and RealSense 1 (wrist_1)
# carrier 2. Placed so, each camera's depth lay level on the surface under the arm, 7 and 12 mm over the overhead's
# reading of it through the xArm's registration; swapped, both put it 250 mm up (2026-11-08).
CAMERA_CARRIERS = ("realsense2", "realsense1")
# What the bracket's boxes and the cables' cylinders reach past what they enclose: the bracket is a print whose datums
# are nominal and which moves a camera about 1 mm with the arm's turn, and no cable is measured. The cases get none:
# they are stock parts, their hulls exact, and a box 5 mm past one refused every dip the arm has made (the session
# camera's case passes 16 mm from the palette camera's envelope, which the station's own 10 mm margin allows). The
# hand-eye of 2026-11-11 put the session camera 0.7 mm and the other 4.5 mm from its CAD pose.
COLLISION_MARGIN_M = 0.005
BRACKET_SLABS = 6                  # the bracket's boxes, along link 7's x: 20 mm of its 120 mm reach each
# The USB-C cable leaves each case at its connector shroud, the case's reach past the camera on the camera link's +y,
# and turns back along the bracket. Neither the plug nor the turn is in CAD or measured: a plug's overmold with the
# start of its bend, read off an overhead frame of the arm (2026-11-10), until someone puts a rule to it.
CABLE_REACH_M = 0.040
CABLE_RADIUS_M = 0.008
SHROUD_END_M = 0.001               # the case's vertices this near its +y end are the shroud's end face
DROP_TAGS = ("ros2_control", "transmission", "gazebo")


def package(checkout: Path, share: Path) -> None:
    """The checkout's xarm_description at `share`, linked entry by entry, with KINEMATICS as its user kinematics file:
    the xacro reads that file only from inside the package, and the checkout stays as pinned."""
    user = ("config", "kinematics", "user")
    src = checkout / "xarm_description"
    for depth in range(len(user)):
        here = share.joinpath(*user[:depth])
        here.mkdir(parents=True, exist_ok=True)
        for child in src.joinpath(*user[:depth]).iterdir():
            if child.name != user[depth]:
                (here / child.name).symlink_to(child)
    share.joinpath(*user).mkdir()
    shutil.copyfile(KINEMATICS, share.joinpath(*user, f"xarm7_kinematics_{KINEMATICS_SUFFIX}.yaml"))


def expand(checkout: Path) -> ET.Element:
    """UFACTORY's xArm 7 (1305) URDF with this arm's kinematics, through a throwaway ament index naming `package`."""
    with tempfile.TemporaryDirectory() as prefix:
        index = Path(prefix) / "share" / "ament_index" / "resource_index" / "packages"
        index.mkdir(parents=True)
        (index / "xarm_description").touch()
        package(checkout, Path(prefix) / "share" / "xarm_description")
        env = {**os.environ, "AMENT_PREFIX_PATH": os.pathsep.join(filter(None, (prefix, os.environ.get("AMENT_PREFIX_PATH"))))}
        out = subprocess.run(
            ["xacro", str(checkout / "xarm_description" / "urdf" / "xarm_device.urdf.xacro"), f"prefix:={PREFIX}",
             "dof:=7", "robot_type:=xarm", f"robot_sn:={MODEL_CODE}", "limited:=true", "add_gripper:=false",
             f"kinematics_suffix:={KINEMATICS_SUFFIX}"],
            env=env, capture_output=True, text=True, check=True).stdout
    return ET.fromstring(out)


def _rename(name: str) -> str:
    return f"{PREFIX}base_link" if name == f"{PREFIX}link_base" else name


def _joint(joint: ET.Element, dropped: set) -> bool:
    """A UFACTORY joint put in the stack's shape (its links renamed, joint7 limited to R7's window); False when it
    touches a dropped link and goes."""
    if {joint.find(end).get("link") for end in ("parent", "child")} & dropped:
        return False
    for end in ("parent", "child"):
        joint.find(end).set("link", _rename(joint.find(end).get("link")))
    if joint.get("name") == f"{PREFIX}joint7":
        joint.find("limit").set("lower", repr(JOINT7_WINDOW_RAD[0]))
        joint.find("limit").set("upper", repr(JOINT7_WINDOW_RAD[1]))
    return True


def _frame(xyz_rpy, rotation=None) -> list:
    """A 4x4 (nested lists) from an origin's (xyz, rpy) (URDF: R = Rz(y) Ry(p) Rx(r)), or from a 3x3 rotation."""
    (x, y, z), (r, p, w) = xyz_rpy
    if rotation is None:
        cr, sr, cp, sp, cw, sw = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(w), math.sin(w)
        rotation = ((cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr),
                    (sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr),
                    (-sp, cp * sr, cp * cr))
    return [[*rotation[i], (x, y, z)[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _matmul(*frames) -> list:
    out = frames[0]
    for b in frames[1:]:
        out = [[sum(out[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
    return out


def _xyz_rpy(m) -> tuple[tuple, tuple]:
    """A 4x4's URDF origin (xyz, rpy)."""
    pitch = math.atan2(-m[2][0], math.hypot(m[0][0], m[1][0]))
    if math.hypot(m[0][0], m[1][0]) < 1e-9:   # pitch +-90 deg: roll folds into yaw
        roll, yaw = 0.0, math.atan2(-m[0][1], m[1][1])
    else:
        roll, yaw = math.atan2(m[2][1], m[2][2]), math.atan2(m[1][0], m[0][0])
    return (m[0][3], m[1][3], m[2][3]), (roll, pitch, yaw)


def _inverse(m) -> list:
    """A rigid 4x4's inverse."""
    back = [[m[j][i] for j in range(3)] for i in range(3)]
    return [[*back[i], -sum(back[i][k] * m[k][3] for k in range(3))] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _metres(pose) -> list:
    """A CAD-frame pose in mm (4x4) as one in metres."""
    return [[*row[:3], row[3] / 1000.0] for row in pose[:3]] + [[0.0, 0.0, 0.0, 1.0]]


def _origin(element: ET.Element, frame) -> None:
    xyz, rpy = _xyz_rpy(frame)
    ET.SubElement(element, "origin", {"rpy": " ".join(f"{v:.9g}" for v in rpy),
                                      "xyz": " ".join(f"{v:.9g}" for v in xyz)})


def _visual(link: ET.Element, name: str, mesh: Path, frame, scale: str, rgba: str) -> None:
    visual = ET.SubElement(link, "visual")
    _origin(visual, frame)
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", {"filename": str(mesh), "scale": scale})
    ET.SubElement(ET.SubElement(visual, "material", {"name": name}), "color", {"rgba": rgba})


def _collision(link: ET.Element, centre, rpy) -> ET.Element:
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"rpy": " ".join(f"{v:.6g}" for v in rpy),
                                        "xyz": " ".join(f"{v:.6g}" for v in centre)})
    return collision


def _shape(link: ET.Element, centre, rpy, shape: str, size: dict) -> None:
    ET.SubElement(ET.SubElement(_collision(link, centre, rpy), "geometry"), shape, size)


def _box(link: ET.Element, extent, margin: float) -> None:
    """A collision box in the link's frame about `extent` (its least and greatest corner), `margin` larger each
    way."""
    low, high = extent
    _shape(link, [(a + b) / 2 for a, b in zip(low, high, strict=True)], (0, 0, 0), "box",
           {"size": " ".join(f"{b - a + 2 * margin:.6g}" for a, b in zip(low, high, strict=True))})


def bracket_collision(end_effector: ET.Element) -> None:
    """The dual-D405 bracket as boxes on the end effector: its mesh cut into BRACKET_SLABS along link 7's x, the
    way it reaches out from the wrist adapter to the cameras, each slab's box holding all of the mesh in it."""
    link7_from_cad = _frame(((0, 0, 0), (0, 0, 0)), LINK7_FROM_CAD)
    mesh = [stl_extent.moved(link7_from_cad, triangle)
            for triangle in stl_extent.triangles(REPO / "urdf" / DUAL_D405 / "camera-cube-bracket.stl", 0.001)]
    for extent in stl_extent.slabs(mesh, 0, BRACKET_SLABS):
        _box(end_effector, extent, COLLISION_MARGIN_M)


def camera_collision(case: ET.Element, camera: ET.Element) -> None:
    """A D405's collision, the same for each: its Trossen case's mesh on the case link, as urdf/tatbot.urdf has the
    WXAI's (the collision scene takes its convex hull); on the camera link the body's mesh as a box, and a cylinder
    CABLE_RADIUS_M about the cable for CABLE_REACH_M out of the case's connector shroud, along +y from the middle of
    the shroud's end face."""
    mesh = Path("meshes") / "wxai" / "camera_mount_d405.stl"
    ET.SubElement(ET.SubElement(_collision(case, (0, 0, 0), (0, 0, 0)), "geometry"), "mesh", {"filename": str(mesh)})
    _box(camera, stl_extent.bounds(stl_extent.moved(_frame(CAMERA_TO_BODY), [p for triangle in stl_extent.triangles(
        REPO / "urdf" / "meshes" / "cameras" / "d405.stl", 0.001) for p in triangle])), 0.0)
    camera_from_case = _inverse(_matmul(_frame(CASE_TO_SCREW), _frame(SCREW_TO_CAMERA)))
    shell = stl_extent.moved(camera_from_case, [p for triangle in stl_extent.triangles(REPO / "urdf" / mesh)
                                                for p in triangle])
    end = stl_extent.bounds(shell)[1][1]
    end_low, end_high = stl_extent.bounds([p for p in shell if p[1] > end - SHROUD_END_M])
    _shape(camera, ((end_low[0] + end_high[0]) / 2, end + CABLE_REACH_M / 2, (end_low[2] + end_high[2]) / 2),
           (math.pi / 2, 0, 0), "cylinder", {"radius": f"{CABLE_RADIUS_M + COLLISION_MARGIN_M:.6g}",
                                             "length": f"{CABLE_REACH_M + 2 * COLLISION_MARGIN_M:.6g}"})


def _fixed(robot: ET.Element, name: str, parent: str, child: str, frame) -> ET.Element:
    """A fixed joint and its child link, which it returns."""
    link = ET.SubElement(robot, "link", {"name": child})
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    _origin(joint, frame)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return link


def end_effector_parts() -> list[tuple[str, Path, list, str, str]]:
    """(material name, mesh relative to urdf/, CAD <- part 4x4 in metres, mesh scale, rgba) for every printed part the
    xArm carries: R7's retained adapter and pen cradle, the dual-D405 bracket, and R7's cube where the bracket holds it
    (its provenance.json). The D405s are wrist_cameras'."""
    record = json.loads((REPO / "urdf" / DUAL_D405 / "provenance.json").read_text())
    mm, identity = "0.001 0.001 0.001", _frame(((0, 0, 0), (0, 0, 0)))
    parts = [(f"r7_{part}", R7_MESHES / f"{part}.stl", identity, mm, R7_RGBA) for part in R7_RETAINED]
    parts.append(("dual_d405_bracket", DUAL_D405 / "camera-cube-bracket.stl", identity, mm, BRACKET_RGBA))
    parts.append(("r7_fiducial-cube", R7_MESHES / "fiducial-cube.stl", _metres(record["cube_mesh_to_assembly"]), mm,
                  R7_RGBA))
    return parts


def wrist_cameras(robot: ET.Element) -> None:
    """Each D405 the bracket carries, chained as urdf/tatbot.urdf chains the WXAI's: its stock Trossen case on link 7
    at its carrier pose (CAMERA_CARRIERS), the bottom screw frame, the camera link with the body's visual, and its
    colour and depth frames with their optical frames. CAD's poses, nominal until the cameras are calibrated. The
    case and the camera link carry the collision (camera_collision)."""
    record = json.loads((REPO / "urdf" / DUAL_D405 / "provenance.json").read_text())
    identity = _frame(((0, 0, 0), (0, 0, 0)))
    link7_from_cad = _frame(((0, 0, 0), (0, 0, 0)), LINK7_FROM_CAD)
    for prefix, pose in zip(CAMERA_CARRIERS, record["camera_carrier_poses"], strict=True):
        name = f"{PREFIX}{prefix}"
        case = _fixed(robot, f"{name}_mount_joint", f"{PREFIX}link7", f"{name}_mount_d405",
                      _matmul(link7_from_cad, _metres(pose)))
        _visual(case, f"{prefix}_case", Path("meshes") / "wxai" / "camera_mount_d405.stl", identity, "1 1 1",
                CAMERA_RGBA)
        _fixed(robot, f"{name}_joint", f"{name}_mount_d405", f"{name}_bottom_screw_frame", _frame(CASE_TO_SCREW))
        camera = _fixed(robot, f"{name}_link_joint", f"{name}_bottom_screw_frame", f"{name}_link",
                        _frame(SCREW_TO_CAMERA))
        _visual(camera, f"{prefix}_body", Path("meshes") / "cameras" / "d405.stl", _frame(CAMERA_TO_BODY),
                "0.001 0.001 0.001", CAMERA_RGBA)
        camera_collision(case, camera)
        for stream in ("color", "depth"):
            _fixed(robot, f"{name}_{stream}_joint", f"{name}_link", f"{name}_{stream}_frame", identity)
            _fixed(robot, f"{name}_{stream}_optical_joint", f"{name}_{stream}_frame",
                   f"{name}_{stream}_optical_frame", _frame(OPTICAL))


def shape(source: ET.Element, mount_origin: dict, checkout: Path) -> tuple[ET.Element, list[tuple[Path, Path]]]:
    """The stack's xArm 7 from UFACTORY's: (the robot element, the meshes to copy as (source, destination))."""
    robot = ET.Element("robot", {"name": "tatbot_xarm7"})
    robot.extend(source.findall("material"))
    ET.SubElement(robot, "link", {"name": "root"})
    mount = ET.SubElement(robot, "joint", {"name": f"{PREFIX}mount_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", mount_origin)
    ET.SubElement(mount, "parent", {"link": "root"})
    ET.SubElement(mount, "child", {"link": f"{PREFIX}base_link"})
    dropped = {"world", f"{PREFIX}link_eef"}
    meshes = []
    for element in source:
        if element.tag in DROP_TAGS or element.tag == "material":
            continue
        if element.tag == "link":
            if element.get("name") in dropped:
                continue
            element.set("name", _rename(element.get("name")))
            for mesh in element.iter("mesh"):
                rel = mesh.get("filename").removeprefix("package://xarm_description/meshes/xarm7_1305/")
                meshes.append((checkout / "xarm_description" / "meshes" / "xarm7_1305" / rel, MESHES / rel))
                mesh.set("filename", str(MESHES / rel))
        elif element.tag == "joint" and not _joint(element, dropped):
            continue
        robot.append(element)
    end_effector = ET.SubElement(robot, "link", {"name": f"{PREFIX}end_effector"})
    for name, mesh, cad_from_part, scale, rgba in end_effector_parts():
        _visual(end_effector, name, mesh, _matmul(_frame(((0, 0, 0), (0, 0, 0)), LINK7_FROM_CAD), cad_from_part),
                scale, rgba)
    _shape(end_effector, (0, 0, TOOL_MOUNT_Z_M / 2), (0, 0, 0), "cylinder",
           {"radius": f"{CORE_RADIUS_M:.6g}", "length": f"{TOOL_MOUNT_Z_M:.6g}"})
    bracket_collision(end_effector)
    joint = ET.SubElement(robot, "joint", {"name": f"{PREFIX}end_effector_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": f"{PREFIX}link7"})
    ET.SubElement(joint, "child", {"link": f"{PREFIX}end_effector"})
    ET.SubElement(robot, "link", {"name": f"{PREFIX}tool_mount"})
    joint = ET.SubElement(robot, "joint", {"name": f"{PREFIX}tool_mount_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"rpy": "0 0 0", "xyz": f"0 0 {TOOL_MOUNT_Z_M:.6g}"})
    ET.SubElement(joint, "parent", {"link": f"{PREFIX}link7"})
    ET.SubElement(joint, "child", {"link": f"{PREFIX}tool_mount"})
    wrist_cameras(robot)
    return robot, meshes


HEADER = f"""<?xml version="1.0" ?>
<!-- Generated by scripts/gen_arm_urdf.py from xArm-Developer/xarm_ros2 at {PIN[:12]} (BSD-3-Clause,
     THIRD_PARTY.md): the xArm 7 (1305 series) as the arm `xarm`, with its own kinematics
     (config/xarm-kinematics.yaml). Do not edit; regenerate, then run
     scripts/gen_tool_urdf.py for the tool block (scripts/vision/export_wrist_tags.py's fiducial blocks carry
     over). xarm/tool_mount, xarm/end_effector and joint7's limits are R7's (see the script). -->
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path, help=f"an xarm_ros2 clone at {PIN[:12]}")
    args = ap.parse_args()
    checkout = args.checkout.expanduser().resolve()
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != PIN:
        sys.exit(f"{checkout} is at {head[:12]}, not {PIN[:12]}: check out the pinned commit")
    tatbot = ET.parse(REPO / "urdf" / "tatbot.urdf").getroot()
    mount_origin = next(j for j in tatbot.findall("joint") if j.get("name") == MOUNT_FROM).find("origin").attrib
    robot, meshes = shape(expand(checkout), dict(mount_origin), checkout)
    ET.indent(robot, space="  ")
    text = HEADER + ET.tostring(robot, encoding="unicode") + "\n"
    kept = urdf_blocks(OUT.read_text()) if OUT.exists() else ""
    close = text.rfind("</robot>")
    OUT.write_text(text[:close] + kept + text[close:])
    for src, rel in meshes:
        dest = REPO / "urdf" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    print(f"wrote {OUT.relative_to(REPO)} and {len(meshes)} meshes under urdf/{MESHES}; "
          "now run scripts/gen_tool_urdf.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
