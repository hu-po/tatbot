"""scripts/lib/stl_extent.py: a binary STL's reach, and a mesh enclosed slab by slab."""
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
import stl_extent  # noqa: E402

# An L in the x-z plane, 2 thick in y: a foot 10 x 1 and a post 1 x 10 at its far end, as two triangles each face.
FOOT = [((0, 0, 0), (10, 0, 0), (10, 2, 1)), ((0, 0, 0), (10, 2, 1), (0, 2, 1))]
POST = [((9, 0, 0), (10, 0, 10), (10, 2, 10)), ((9, 0, 0), (10, 2, 10), (9, 2, 0))]


def _write(path, mesh):
    body = b"".join(struct.pack("<12fH", 0, 0, 0, *(v for point in triangle for v in point), 0) for triangle in mesh)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", len(mesh)) + body)


def test_a_binary_stl_reads_back_scaled_and_anything_else_is_refused(tmp_path):
    _write(tmp_path / "l.stl", FOOT + POST)
    mesh = stl_extent.triangles(tmp_path / "l.stl", 0.001)
    assert len(mesh) == 4 and mesh[0][1] == pytest.approx((0.010, 0.0, 0.0))
    (tmp_path / "ascii.stl").write_text("solid l\nendsolid l\n" + " " * 80)
    with pytest.raises(ValueError, match="not a binary STL"):
        stl_extent.triangles(tmp_path / "ascii.stl")


def test_points_move_through_a_frame_and_have_bounds():
    quarter_turn = [[0, -1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert stl_extent.moved(quarter_turn, [(1, 0, 0), (0, 2, 3)]) == [(1, 1, 0), (-1, 0, 3)]
    assert stl_extent.bounds([(1, 5, 0), (-1, 2, 3)]) == ((-1, 2, 0), (1, 5, 3))


def test_slabs_hold_the_surface_cut_at_their_faces():
    """The L's foot is low until its post: a triangle that spans slabs counts in each only where it lies in it, so
    the first slab stays as low as the foot there, and only the last reaches the post's top."""
    first, *_, last = stl_extent.slabs(FOOT + POST, 0, 5)
    assert first[0] == pytest.approx((0, 0, 0)) and first[1] == pytest.approx((2, 2, 1))
    assert last[0] == pytest.approx((8, 0, 0)) and last[1] == pytest.approx((10, 2, 10))
    assert stl_extent.slabs(FOOT, 2, 4)[-1][1][2] == pytest.approx(1.0)
    whole = stl_extent.bounds([p for triangle in FOOT + POST for p in triangle])
    assert len(stl_extent.slabs(FOOT + POST, 0, 1)) == 1 and stl_extent.slabs(FOOT + POST, 0, 1)[0] == whole
