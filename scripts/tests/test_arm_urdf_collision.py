"""urdf/xarm7.urdf's end-effector collision (scripts/gen_arm_urdf.py): the dual-D405 bracket as boxes that hold its
mesh with the generator's margin to spare, each camera's case as its mesh, its body as a box and its cable as a
cylinder out of the case's connector shroud."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import gen_arm_urdf as gen  # noqa: E402
import stl_extent  # noqa: E402

URDF = ET.parse(REPO / "urdf" / "xarm7.urdf").getroot()


def _collisions(link: str, shape: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(centre, sizes) of each of a link's collision primitives of one shape: a box's sides, a cylinder's (radius,
    length)."""
    out = []
    for collision in URDF.find(f"link[@name='{link}']").findall("collision"):
        geometry = collision.find(f"geometry/{shape}")
        if geometry is not None:
            sizes = geometry.get("size", f"{geometry.get('radius')} {geometry.get('length')}")
            out.append((np.array(collision.find("origin").get("xyz").split(), float), np.array(sizes.split(), float)))
    return out


def _vertices(mesh: Path, scale: float, frame) -> np.ndarray:
    return np.array(stl_extent.moved(frame, [p for triangle in stl_extent.triangles(mesh, scale) for p in triangle]))


def _spare(points: np.ndarray, boxes) -> np.ndarray:
    """How far inside its best box each point lies (m; negative outside every one)."""
    return np.max([np.min(size / 2 - np.abs(points - centre), axis=1) for centre, size in boxes], axis=0)


def test_the_brackets_boxes_hold_its_mesh_and_reach_no_farther_than_the_margin():
    boxes = _collisions("xarm/end_effector", "box")
    assert len(boxes) == gen.BRACKET_SLABS
    link7_from_cad = gen._frame(((0, 0, 0), (0, 0, 0)), gen.LINK7_FROM_CAD)
    bracket = _vertices(REPO / "urdf" / gen.DUAL_D405 / "camera-cube-bracket.stl", 0.001, link7_from_cad)
    assert _spare(bracket, boxes).min() > gen.COLLISION_MARGIN_M - 1e-5
    low, high = bracket.min(axis=0), bracket.max(axis=0)
    reach = np.array([[centre - size / 2, centre + size / 2] for centre, size in boxes])
    assert np.allclose(reach[:, 0].min(axis=0), low - gen.COLLISION_MARGIN_M, atol=1e-5)
    assert np.allclose(reach[:, 1].max(axis=0), high + gen.COLLISION_MARGIN_M, atol=1e-5)
    # not one box about the whole bracket: its arm by the wrist is far thinner than its plate by the cameras
    assert boxes[0][1][2] < 0.6 * max(size[2] for _, size in boxes)


@pytest.mark.parametrize("camera", gen.CAMERA_CARRIERS)
def test_a_cameras_case_is_its_mesh_its_body_a_box_and_its_cable_leaves_the_shroud(camera):
    """The case as the WXAI's URDF has it, no margin: a box about it refused dips the arm makes every draw."""
    case_mesh = URDF.find(f"link[@name='xarm/{camera}_mount_d405']/collision/geometry/mesh")
    assert case_mesh.get("filename") == "meshes/wxai/camera_mount_d405.stl" and case_mesh.get("scale") is None
    (centre, size), = _collisions(f"xarm/{camera}_link", "box")
    body = _vertices(REPO / "urdf" / "meshes" / "cameras" / "d405.stl", 0.001, gen._frame(gen.CAMERA_TO_BODY))
    assert _spare(body, [(centre, size)]).min() > -1e-6
    assert sorted(size) == pytest.approx([0.023, 0.042, 0.042], abs=0.0005)   # the D405's published case
    camera_from_case = gen._inverse(gen._matmul(gen._frame(gen.CASE_TO_SCREW), gen._frame(gen.SCREW_TO_CAMERA)))
    case = _vertices(REPO / "urdf" / "meshes" / "wxai" / "camera_mount_d405.stl", 1.0, camera_from_case)
    (at, (radius, length)), = _collisions(f"xarm/{camera}_link", "cylinder")
    assert radius == pytest.approx(gen.CABLE_RADIUS_M + gen.COLLISION_MARGIN_M)
    assert at[1] - length / 2 == pytest.approx(case[:, 1].max() - gen.COLLISION_MARGIN_M, abs=1e-5)   # from the case's +y end
    assert at[1] + length / 2 == pytest.approx(case[:, 1].max() + gen.CABLE_REACH_M + gen.COLLISION_MARGIN_M, abs=1e-5)
    shroud = case[case[:, 1] > body[:, 1].max() + 0.005]   # the case past the camera's own +y face
    assert shroud[:, 0].min() < at[0] < shroud[:, 0].max() and shroud[:, 2].min() < at[2] < shroud[:, 2].max()


def test_the_cables_leave_toward_the_flange_on_one_camera_and_outward_on_the_other():
    """The cases mirror each other in the bracket, so the connector shroud of the session camera points in toward
    the pen and back to the flange, and the other's out and on toward the tip: the one that hangs lowest."""
    from urdf_kinematics import UrdfChain

    chain = UrdfChain(REPO / "urdf" / "xarm7.urdf")
    link7 = np.linalg.inv(chain.link_pose("xarm/link7", {}))
    session, other = ((link7 @ chain.link_pose(f"xarm/{camera}_link", {}))[:3, 1] for camera in gen.CAMERA_CARRIERS)
    assert session[0] < -0.5 and session[2] < -0.5 and other[0] > 0.5 and other[2] > 0.5
