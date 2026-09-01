"""Board-instance routing tests.

    uvx --with pytest --with numpy --with opencv-python-headless \
      pytest -q scripts/tests/test_calibrate_board_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import calibrate_board_session as board  # noqa: E402


def test_reused_id_must_match_the_root_defined_board_plane():
    root = board.BOARD_ROOT
    root_plane = np.array(
        [[-0.022, 0.022, 0.0], [0.022, 0.022, 0.0],
         [0.022, -0.022, 0.0], [-0.022, -0.022, 0.0]]
    )
    tag6_plane = root_plane + [0.0, 0.065, 0.0]
    tag7_plane = root_plane + [0.065, 0.065, 0.0]
    layout = {root: root_plane, 6: tag6_plane, 7: tag7_plane}

    def pixels(points):
        return points[:, :2] * 4000.0 + [1500.0, 800.0]

    tags = {
        root: pixels(root_plane),
        6: pixels(tag6_plane),
        # Same numeric ID, but this detection is a different physical tag.
        7: pixels(tag7_plane) + [500.0, -350.0],
    }
    intrinsics = np.array(
        [[1500.0, 0.0, 1480.0], [0.0, 1500.0, 834.0], [0.0, 0.0, 1.0]]
    )

    assert board.board_instance_ids(tags, layout, intrinsics, None) == [6, root]

