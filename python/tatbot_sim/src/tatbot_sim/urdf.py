"""Derive a tatbot-specific WidowX AI URDF from the stock ManiSkill asset.

The stock ``wxai_follower.urdf`` already carries the UPPER D405 with exactly
the transforms the real rig uses (verified against ``urdf/tatbot.urdf``:
identical mount, bracket pitch, and link offsets). Two things are missing, and
both are added here:

1. **The lower D405.** On the real arm it is the same bracket rolled 180 deg
   about the mount, so the chain is duplicated with ``rpy="3.14159 0 0"`` on
   the mount joint.
2. **The fitted tattoo tool**, built from its datasheet in ``config/tools/``
   (which one is fitted comes from ``config/workspace.yaml``; see
   :mod:`tatbot_sim.tools`). The profile of revolution in that file becomes
   cylinders and, at the taper, the tool's display mesh. Its last point is the
   TCP.

   Since 2026-08-30 the tool sits in the bore of a printed three-part chain
   (cube plate, EE base, angled pen mount) bolted to the LEFT finger
   carriage's front face; the bore runs 45 deg between the carriage's -y
   and +x, so with the wrist rolled 90 deg (cube up, cameras a left/right
   pair) the tip points forward-and-down. The mount transform is
   GRAFTED from ``urdf/tatbot.urdf`` (``right/tool_mount_joint``, hand-placed
   there from calipers) rather than re-derived here, so the real rig and the
   sim cannot disagree about where the tool is. The right finger is physically
   removed and so is the left fingertip: both finger links stay (the upstream
   agent looks them up by name, and the mount chain hangs off the left one)
   but carry no geometry; the right mimic joint is fixed, so the actuator has
   one carriage.

   The tool body remains visual-only with a token mass; a contact tool may add
   one small physical tip collision at its resolved TCP. Its measured mass
   belongs to the real arm's gravity compensation.

The tool is welded to **link_6 at the mount's rest position**, not to the
carriage link it rides on the real arm. The carriage joint is kept and
position-held at ``carriage_rest_m`` (it is a real, safety-owned DOF now and
the 7th action channel), but putting the tool downstream of it would make the
IK chain seven-dimensional and let the solver "reach" with the carriage. In
sim the carriage never leaves rest, so the two placements coincide; the sim
does not model the retract.

Rather than vendoring the arm meshes, the derived URDF is written next to the
downloaded asset so its relative mesh paths keep resolving; our two adapter
STLs are copied in alongside.

The companion SRDF is copied too, and that is load-bearing: the loader picks up
``<urdf stem>.srdf`` by name, so a derived URDF without one silently loses all
36 disabled self-collision pairs. The arm then fights its own contacts and
holds roughly 0.1 rad off any commanded pose no matter how stiff the servo.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from mani_skill import ASSET_DIR

from tatbot_sim.repo import repo_root
from tatbot_sim.tools import active_tool, tool_source_paths

REPO = repo_root()


def rig_from_follower_base():
    """Fixed transform from the dual-arm rig root to the follower base.

    ``robot_world.json`` retains the historical ``world_from_base`` field
    name, but its FK solve uses the complete Tatbot URDF rooted at ``root``.
    ManiSkill loads only the follower and therefore roots the simulated robot
    at ``base_link``. Derive the bridge from the canonical URDF instead of
    copying the mount offset into the camera and benchmark code.
    """
    import numpy as np
    from transforms3d.euler import euler2mat

    root = ET.parse(REPO / "urdf/tatbot.urdf").getroot()
    matches = []
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") == "right/base_link":
            matches.append(joint)
    if len(matches) != 1 or matches[0].get("type") != "fixed":
        raise ValueError("canonical URDF must have one fixed mount for right/base_link")
    origin = matches[0].find("origin")
    origin_xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    origin_rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    xyz = [float(value) for value in origin_xyz.split()]
    rpy = [float(value) for value in origin_rpy.split()]
    transform = np.eye(4)
    transform[:3, :3] = euler2mat(*rpy)
    transform[:3, 3] = xyz
    return transform


def _wrist_inventory() -> tuple[tuple[int, ...], float, str, str]:
    """Wrist ids, edge, parent link, and content hash from the inventory.

    Keep the simulator package independent of repository script imports while
    preserving the same raw-file hash contract used by the live tracker.
    """
    import hashlib
    import json

    raw = (REPO / "config" / "fiducials.json").read_bytes()
    data = json.loads(raw)
    wrist = data["targets"]["wrist"]
    return (
        tuple(int(i) for i in wrist["ids"]),
        float(wrist["edge_m"]),
        str(wrist["parent_frame"]),
        hashlib.sha256(raw).hexdigest(),
    )



STOCK_URDF = Path(ASSET_DIR) / "robots/widowxai/wxai_follower.urdf"
STOCK_SRDF = Path(ASSET_DIR) / "robots/widowxai/wxai_follower.srdf"
def derived_paths(tool_id: str, calibration_delta=None) -> tuple[Path, Path]:
    """Where a tool's derived URDF (and its companion SRDF) live.

    Scoped by tool id so previewing one tool cannot hand a stale robot to the
    next: two tools are two different robots, not two states of one file.
    """
    delta = tuple(calibration_delta or (0.0, 0.0, 0.0))
    suffix = ""
    if any(delta):
        import hashlib
        token = ",".join(f"{float(value):.9g}" for value in delta)
        suffix = "-cal" + hashlib.sha256(token.encode()).hexdigest()[:10]
    stem = Path(ASSET_DIR) / "robots/widowxai" / f"wxai_tatbot_{tool_id}{suffix}"
    return stem.with_suffix(".urdf"), stem.with_suffix(".srdf")
MESH_SUBDIR = "meshes/tatbot_ee"
REPO_MESHES = REPO / "urdf/meshes/ee"

# --- where the tool hangs (ARM geometry, not the tool's) ---------------------
# Read from the canonical URDF: right/tool_mount_joint (gripper_left ->
# tool_mount) composed with the carriage joint's origin at carriage_rest_m,
# giving the mount pose in link_6. One source; the real rig's FK and the
# sim's cannot disagree about where the tool is.
RIGHT_FINGER_LINKS = ("carriage_right", "gripper_right")
RIGHT_FINGER_JOINTS = ("right_carriage_joint", "right_gripper_joint")


def _joint_origin(root, name):
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            origin = joint.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            axis_el = joint.find("axis")
            axis = ([float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [1.0, 0.0, 0.0])
            return joint, xyz, rpy, axis
    raise ValueError(f"urdf/tatbot.urdf has no joint {name!r}")


def mount_in_link6(carriage_m: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(xyz, rpy) of right/tool_mount in link_6 with the carriage at ``carriage_m``."""
    import numpy as np
    from transforms3d.euler import euler2mat, mat2euler

    root = ET.parse(REPO / "urdf/tatbot.urdf").getroot()
    _, c_xyz, c_rpy, c_axis = _joint_origin(root, "right/left_carriage_joint")
    _, g_xyz, g_rpy, _ = _joint_origin(root, "right/left_gripper_joint")
    _, m_xyz, m_rpy, _ = _joint_origin(root, "right/tool_mount_joint")

    def tf(xyz, rpy):
        t = np.eye(4)
        t[:3, :3] = euler2mat(*rpy)
        t[:3, 3] = xyz
        return t

    slide = np.eye(4)
    slide[:3, 3] = np.asarray(c_axis, float) * carriage_m
    pose = tf(c_xyz, c_rpy) @ slide @ tf(g_xyz, g_rpy) @ tf(m_xyz, m_rpy)
    xyz_vec = pose[:3, 3]
    rpy_vec = mat2euler(pose[:3, :3])
    return (float(xyz_vec[0]), float(xyz_vec[1]), float(xyz_vec[2])), (float(rpy_vec[0]), float(rpy_vec[1]), float(rpy_vec[2]))

def _el(tag, **attrs):
    return ET.Element(tag, dict(attrs))


def _fmt(v) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def _add_link(robot, name, visuals=(), collisions=(), mass=1e-4):
    """Add a link. Masses are token values: these bodies are welded to link_6
    and only need to render and provide a frame, not perturb the arm's
    dynamics."""
    link = ET.SubElement(robot, "link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-6", "ixy": "0", "ixz": "0", "iyy": "1e-6", "iyz": "0", "izz": "1e-6"},
    )
    for i, (origin_xyz, origin_rpy, geom, color) in enumerate(visuals):
        vis = ET.SubElement(link, "visual")
        ET.SubElement(vis, "origin", {"xyz": _fmt(origin_xyz), "rpy": _fmt(origin_rpy)})
        g = ET.SubElement(vis, "geometry")
        g.append(geom)
        if color is None:
            # no URDF material: the mesh's own MTL (e.g. a tag texture)
            # must win, and a URDF color would override it
            continue
        if color.startswith("@"):
            # reference a material the stock URDF defines (e.g. @trossen_black)
            ET.SubElement(vis, "material", {"name": color[1:]})
        else:
            # material names must be unique PER VISUAL: URDF materials are
            # global by name, so visuals of one link sharing "<link>_mat"
            # all silently render whichever color was registered first
            # (that bug kept the white pen tip dark through 8/22).
            mat = ET.SubElement(vis, "material", {"name": f"{name}_mat{i}"})
            ET.SubElement(mat, "color", {"rgba": color})
    for origin_xyz, origin_rpy, geom in collisions:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"xyz": _fmt(origin_xyz), "rpy": _fmt(origin_rpy)})
        g = ET.SubElement(collision, "geometry")
        g.append(geom)
    return link


def _add_joint(robot, name, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    j = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    ET.SubElement(j, "origin", {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
    return j


def _add_lower_camera(robot):
    """Duplicate the stock camera chain, rolled 180 deg like the real bracket.

    With the bracket and camera-body visuals the stock chain gives the upper
    camera: the real assembly hangs under the wrist in the upper camera's view
    (and its bulk shows in real recordings), so a sim that renders only the
    upper one teaches the wrong end-effector silhouette.
    """
    # Same meshes the stock URDF renders for the upper chain, at the same
    # link-local origins — the mirrored mount joint carries them under the arm.
    mount_visuals = [
        ((0, 0, 0), (0, 0, 0),
         _el("mesh", filename="meshes/camera_mount_d405.stl"), "@trossen_black"),
    ]
    d405_visuals = [
        ((0.0038, -0.009, 0), (1.5707963267948966, 0, 1.5707963267948966),
         _el("mesh", filename="meshes/d405.stl", scale="0.001 0.001 0.001"), "0.5 0.5 0.5 1"),
    ]
    chain = [
        ("camera_lower_mount_joint", "link_6", "camera_lower_mount_d405",
         (0.012, 0, 0), (3.141592653589793, 0, 0), mount_visuals),
        ("camera_lower_joint", "camera_lower_mount_d405", "camera_lower_bottom_screw_frame",
         (0.02927207801, 0, 0.03824951197), (0, 0.3490658503988659, 0), ()),
        ("camera_lower_link_joint", "camera_lower_bottom_screw_frame", "camera_lower_link",
         (0.01085, 0.009, 0.021), (0, 0, 0), d405_visuals),
        ("camera_lower_color_joint", "camera_lower_link", "camera_lower_color_frame",
         (0, 0, 0), (0, 0, 0), ()),
        ("camera_lower_color_optical_joint", "camera_lower_color_frame",
         "camera_lower_color_optical_frame", (0, 0, 0),
         (-1.5707963267948966, 0, -1.5707963267948966), ()),
    ]
    for _, _, child, _, _, visuals in chain:
        _add_link(robot, child, visuals=visuals)
    for name, parent, child, xyz, rpy, _ in chain:
        _add_joint(robot, name, parent, child, xyz, rpy)


def tool_visuals(spec):
    """The fitted tool's profile as URDF visuals, in tool-local z.

    The geometry itself comes from the datasheet (ToolSpec.geometry_parts), so
    this and the real rig's URDF cannot disagree about the tool's shape; only
    the XML around it differs. Colors come from the datasheet too — for the
    current machine, black glossy metal body and white glossy plastic tip,
    checked against it by the operator (2026-08-22). Gloss itself cannot ride
    through URDF color tags: the env sets metallic/roughness after load
    (TatbotDrawEnv._polish_ee_materials).
    """
    visuals = []
    for part in spec.geometry_parts():
        if part["kind"] == "mesh":
            geom = _el("mesh", filename=f"{MESH_SUBDIR}/{part['mesh']}",
                       scale=_fmt(part["scale"]))
        elif part["kind"] == "sphere":
            geom = _el("sphere", radius=str(part["radius"]))
        else:
            geom = _el("cylinder", length=str(part["length"]), radius=str(part["radius"]))
        # x/y are how three needles sit beside each other: a profile of
        # revolution cannot express a cluster, so tip detail may offset.
        visuals.append(((part.get("x", 0.0), part.get("y", 0.0), part["z"]),
                        (0, 0, 0), geom, part["color"]))
    return visuals


def _tool_tip_m(reg, ws, spec) -> tuple[float, float, float]:
    """The tip to weld at: the workspace touch-off when it belongs to this
    tool, the datasheet nominal otherwise. See _add_tool's docstring."""
    if reg.active_tool_id(REPO, "right", ws) == spec.tool_id:
        tip = reg.tip_offset_m(ws, "right")
        if tip is not None:
            return tip
    return spec.touchoff_nominal_m


def tool_tcp_m() -> tuple[float, float, float]:
    """The fitted tool's TCP vector in the mount frame — what _add_tool welds.

    Exposed so the expert can derive the pen-down orientation from the same
    numbers the geometry is built from (bore axis vs tip vector); see
    expert._pen_down_matrix."""
    tool_module = __import__("tatbot_sim.tools", fromlist=["resolved_geometry"])
    return tool_module.resolved_geometry().tcp_offset_m


def _add_tool(robot, spec, carriage_m: float):
    """Weld the fitted tool to link_6 where the mount sits at carriage rest.

    ``tool_mount`` is the same frame as the real URDF's ``right/tool_mount``
    (bore face, +z along the bore); the tool hangs off it exactly as
    scripts/gen_tool_urdf.py hangs it on the real rig — nominal along the
    bore until a mount-frame touch-off exists, then along the measured tip.

    The measured tip is used ONLY when workspace.yaml was calibrated with
    THIS tool. The real rig enforces that with require_stated_tool; the sim
    has to make the same check itself because TATBOT_TOOL_ID deliberately
    overrides the fitted tool for preview and factory runs. Without it, the
    2026-08-31 ballpoint touch-off (68.5 mm) was welded onto every simulated
    tool — a laser losing 61 mm of protrusion, silently, in geometry that
    every dataset then records (the same cross-tool inheritance e61193e
    fixed in the metadata).
    """
    tool_module = __import__("tatbot_sim.tools", fromlist=["resolved_geometry"])
    geometry = tool_module.resolved_geometry(spec)
    mount_xyz, mount_rpy = mount_in_link6(carriage_m)
    _add_link(robot, "tool_mount")
    _add_joint(robot, "tool_mount_joint", "link_6", "tool_mount", mount_xyz, mount_rpy)
    _add_link(robot, "tattoo_pen", visuals=tool_visuals(spec))
    # The point itself: the TCP for IK and for ink deposition. The link names
    # stay generic across tools — they are in every trained policy's IK chain.
    tip_collisions = ()
    if spec.contact_radius_m is not None:
        radius = float(spec.contact_radius_m)
        # The TCP is the forward tangent of the physical contact element.
        tip_collisions = (((0, 0, -radius), (0, 0, 0),
                           _el("sphere", radius=f"{radius:.9g}")),)
    _add_link(robot, "tattoo_needle", collisions=tip_collisions)
    _add_joint(robot, "tattoo_pen_joint", "tool_mount", "tattoo_pen",
               geometry.body_origin_m, geometry.body_rpy_rad)
    _add_joint(robot, "tattoo_needle_joint", "tattoo_pen", "tattoo_needle",
               geometry.tcp_in_body_m)


EE_MOUNT_LINK = "right/ee_mount"


def _graft_ee_mount(robot, mesh_dir: Path):
    """Copy the printed EE mount chain (cube plate, EE base, pen mount) from
    the real URDF onto gripper_left, and strip the (removed) fingertip.

    The real URDF is the one place the chain is described (hole patterns from
    the STLs); the sim takes the visuals verbatim so the wrist cameras see the
    same silhouette the real ones do. Mesh files are copied beside the derived
    URDF under MESH_SUBDIR so its relative paths keep resolving.
    """
    real = ET.parse(REPO / "urdf/tatbot.urdf").getroot()
    src = next((link for link in real.findall("link") if link.get("name") == EE_MOUNT_LINK), None)
    if src is None:
        return
    visuals = []
    for vis in src.findall("visual"):
        origin = vis.find("origin")
        origin_xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        origin_rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = tuple(float(v) for v in origin_xyz.split())
        rpy = tuple(float(v) for v in origin_rpy.split())
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        rel = mesh.get("filename")  # meshes/ee/<file>.stl in the real URDF
        if rel is None:
            continue
        stl = REPO / "urdf" / rel
        rel_path = Path(rel)
        dst = mesh_dir / rel_path.name
        if not dst.exists() or dst.stat().st_mtime < stl.stat().st_mtime:
            shutil.copy2(stl, dst)
        color = vis.find("material/color")
        rgba = color.get("rgba") if color is not None else "0.12 0.12 0.13 1"
        scale = mesh.get("scale", "1 1 1")
        visuals.append((xyz, rpy, _el("mesh", filename=f"{MESH_SUBDIR}/{rel_path.name}",
                                       scale=scale if scale is not None else "1 1 1"), rgba))
    _add_link(robot, "ee_mount", visuals=visuals)
    _add_joint(robot, "ee_mount_joint", "gripper_left", "ee_mount")
    # neither fingertip is fitted any more: gripper_left stays as the frame
    # the chain hangs from, with no geometry (the right one is handled by
    # _drop_right_finger)
    for link in robot.findall("link"):
        if link.get("name") == "gripper_left":
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)


def _drop_right_finger(robot, srdf_path: Path | None):
    """The right finger is physically removed (2026-08-30).

    The links stay in the derived URDF — ManiSkill's WidowXAI agent looks
    them up by name at init — but lose every visual and collision, so the
    wrist cameras see what the real ones see, and the mimic prismatic joint
    becomes a fixed one so the actuator has ONE carriage (qpos 7, matching
    the real driver vector). The SRDF pairs naming them are scrubbed too.
    """
    for joint in robot.findall("joint"):
        if joint.get("name") in RIGHT_FINGER_JOINTS and joint.get("type") != "fixed":
            joint.set("type", "fixed")
            for tag in ("mimic", "axis", "limit", "dynamics", "safety_controller"):
                for el in joint.findall(tag):
                    joint.remove(el)
    for link in robot.findall("link"):
        if link.get("name") in RIGHT_FINGER_LINKS:
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
    if srdf_path is not None and srdf_path.exists():
        tree = ET.parse(srdf_path)
        root = tree.getroot()
        for pair in list(root.findall("disable_collisions")):
            if pair.get("link1") in RIGHT_FINGER_LINKS or pair.get("link2") in RIGHT_FINGER_LINKS:
                root.remove(pair)
        tree.write(srdf_path, encoding="utf-8", xml_declaration=True)


# --- fiducial plates (2026-08-22) -------------------------------------------
# The rig carries FOUR wrist fiducials (follower EE, ids 3/6/7/8): 16h5 patterns,
# 56 mm black square on 4 mm white foamboard with ~10 mm margin (76 mm
# plates). Tag identities and their caliper edge come from the canonical
# config/fiducials.json inventory.
# Jaw mounting poses come from config/wrist_tags_measured.json (see
# _add_wrist_plates). The base-ring placeholders are only the fallback when
# that file is absent; pending layouts may render but never benchmark.
PLATE_SIZE = 0.076
PLATE_THICK = 0.004


def _write_plate_assets(mesh_dir: Path, ids: tuple[int, ...], tag_edge_m: float) -> None:
    """Per tag id: a texture PNG (white plate, centred 16h5 marker) and a UV'd
    OBJ quad the size of the plate face. Textured URDF visuals need real UVs,
    which primitive boxes don't carry — hence the tiny mesh."""
    import cv2
    import numpy as np

    px_per_m = 5000  # 5 px/mm keeps the 4x4 tag modules crisp
    plate_px = round(PLATE_SIZE * px_per_m)
    tag_px = round(tag_edge_m * px_per_m)
    margin = (plate_px - tag_px) // 2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
    half = PLATE_SIZE / 2
    for tag_id in ids:
        png = mesh_dir / f"tag16h5_{tag_id}.png"
        # 16h5 = 6 modules across; render at an exact multiple, resize NEAREST
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 6 * 100)
        marker = cv2.resize(marker, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)
        sheet = np.full((plate_px, plate_px), 255, dtype=np.uint8)
        sheet[margin : margin + tag_px, margin : margin + tag_px] = marker
        cv2.imwrite(str(png), sheet)
        (mesh_dir / f"plate_{tag_id}.mtl").write_text(
            f"newmtl tagface\nKd 1 1 1\nmap_Kd tag16h5_{tag_id}.png\n"
        )
        (mesh_dir / f"plate_{tag_id}.obj").write_text(
            f"mtllib plate_{tag_id}.mtl\n"
            f"v {-half} {-half} 0\nv {half} {-half} 0\nv {half} {half} 0\nv {-half} {half} 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "vn 0 0 1\n"
            "usemtl tagface\n"
            "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n"
        )


def _plate_link(robot, tag_id: int) -> str:
    """One fiducial plate, tag plane at the link's z=0 facing +z, board behind."""
    name = f"fiducial_plate_{tag_id}"
    _add_link(robot, name, visuals=[
        ((0, 0, -PLATE_THICK / 2), (0, 0, 0),
         _el("box", size=f"{PLATE_SIZE} {PLATE_SIZE} {PLATE_THICK}"),
         "0.96 0.96 0.95 1"),
        ((0, 0, 0.0003), (0, 0, 0),
         _el("mesh", filename=f"{MESH_SUBDIR}/plate_{tag_id}.obj", scale="1 1 1"),
         None),
    ])
    return name


def _wrist_tag_poses() -> tuple[str, dict[int, tuple[list[float], list[float]]]] | None:
    """Current tag poses relative to their configured rigid parent link.

    Calibrated files come from ``export_wrist_tags.py``. A pending file may
    retain provisional visualization poses, but emits a warning before those
    poses are placed into the derived simulator URDF.
    """
    import json

    path = REPO_MESHES.parents[2] / "config" / "wrist_tags_measured.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    ids, edge_m, expected_parent, inventory_hash = _wrist_inventory()
    if data.get("inventory_hash") != inventory_hash:
        raise ValueError("wrist layout inventory hash is stale; regenerate it")
    if tuple(int(tag_id) for tag_id in data.get("target_ids", ())) != ids:
        raise ValueError("wrist layout ids differ from config/fiducials.json")
    if abs(float(data.get("edge_m", 0)) - edge_m) > 1e-9:
        raise ValueError("wrist layout edge differs from config/fiducials.json")
    if data.get("parent_frame") != expected_parent:
        raise ValueError(f"sim expects wrist layout in {expected_parent}")
    if data.get("calibration_status") != "calibrated":
        import warnings

        warnings.warn(
            "rendering provisional wrist geometry; do not use it for tracking benchmarks",
            RuntimeWarning,
            stacklevel=2,
        )
    import math

    poses = {}
    for key, entry in data["tags"].items():
        transform = entry["ee_from_tag"]
        rotation = [row[:3] for row in transform[:3]]
        xyz = [float(row[3]) for row in transform[:3]]
        rpy = [
            math.atan2(rotation[2][1], rotation[2][2]),
            math.atan2(-rotation[2][0], math.hypot(rotation[2][1], rotation[2][2])),
            math.atan2(rotation[1][0], rotation[0][0]),
        ]
        poses[int(key)] = (xyz, rpy)
    # The derived single-arm URDF strips the real robot's `right/` namespace.
    return expected_parent.rsplit("/", 1)[-1], poses


def _add_wrist_plates(
    robot, parent_link: str, poses: dict[int, tuple[list[float], list[float]]]
):
    for tag_id, (xyz, rpy) in sorted(poses.items()):
        name = _plate_link(robot, tag_id)
        _add_joint(robot, f"{name}_joint", parent_link, name, tuple(xyz), tuple(rpy))


def _add_base_plates(robot, ids: tuple[int, ...]):
    """Fallback when no measured poses exist: park the plates at the base."""
    import math

    for index, tag_id in enumerate(ids):
        a = math.radians(120) + 2 * math.pi * index / max(1, len(ids))
        name = _plate_link(robot, tag_id)
        _add_joint(robot, f"{name}_joint", "base_link", name,
                   (0.15 * math.cos(a), 0.15 * math.sin(a), PLATE_THICK), (0, 0, a))


def build_tatbot_urdf(force: bool = False) -> str:
    """Write (if needed) and return the path to the derived URDF."""
    if not STOCK_URDF.exists():
        raise FileNotFoundError(
            f"{STOCK_URDF} missing — run `python -m mani_skill.utils.download_asset widowxai -y`"
        )
    tool = active_tool()
    from tatbot_sim.tools import calibration_delta_m
    derived_urdf, derived_srdf = derived_paths(tool.tool_id, calibration_delta_m())
    mesh_dir = STOCK_URDF.parent / MESH_SUBDIR
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for stl in tool.meshes():
        src = REPO_MESHES / stl
        if not src.exists():
            raise FileNotFoundError(
                f"tool mesh {src} missing from the repo (tip_mesh of {tool.tool_id})")
        dst = mesh_dir / stl
        if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)

    wrist_layout_json = REPO_MESHES.parents[2] / "config" / "wrist_tags_measured.json"
    fiducial_inventory = REPO / "config" / "fiducials.json"
    # Editing the datasheet (or swapping which tool is fitted) has to rebuild
    # the derived URDF, the same way editing this module does.
    inputs = [STOCK_URDF, Path(__file__), fiducial_inventory, REPO / "urdf/tatbot.urdf",
              REPO / "config/trossen/tatbot.yaml"] + tool_source_paths() \
        + ([wrist_layout_json] if wrist_layout_json.exists() else [])
    fresh = derived_urdf.exists() and derived_srdf.exists() and derived_urdf.stat().st_mtime >= max(
        p.stat().st_mtime for p in inputs
    )
    if fresh and not force:
        return str(derived_urdf)

    if STOCK_SRDF.exists():
        shutil.copy2(STOCK_SRDF, derived_srdf)

    tree = ET.parse(STOCK_URDF)
    robot = tree.getroot()
    existing = {link.get("name") for link in robot.iter("link")}
    if "tattoo_needle" not in existing:
        from tatbot_sim.tools import carriage_rest_m

        tag_ids, edge_m, _parent_frame, _inventory_hash = _wrist_inventory()
        _write_plate_assets(mesh_dir, tag_ids, edge_m)
        _drop_right_finger(robot, derived_srdf)
        _add_lower_camera(robot)
        _graft_ee_mount(robot, mesh_dir)
        _add_tool(robot, tool, carriage_rest_m())
        wrist_poses = _wrist_tag_poses()
        if wrist_poses:
            parent_link, poses = wrist_poses
            _add_wrist_plates(robot, parent_link, poses)
        else:
            _add_base_plates(robot, tag_ids)
    ET.indent(tree, space="  ")
    tree.write(derived_urdf, encoding="utf-8", xml_declaration=True)
    return str(derived_urdf)
