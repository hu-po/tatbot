"""The palette lease (tatbot_session.lease): one run holds the palette, the node reads who, and the kernel drops a
crashed holder's lock."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from tatbot_session.lease import PaletteLease, held_zone

ZONE = {"arm": "left", "run": "20260930T130000Z-x-0001", "world_from_zone": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                                                               [0, 0, 0, 1]],
        "radius_m": 0.2, "height_m": 0.14}


def test_a_held_lease_is_read_and_a_second_run_is_refused(tmp_path):
    path = tmp_path / "palette.lease"
    assert held_zone(path) is None                         # no file: nobody holds it
    with PaletteLease(ZONE, path):
        assert held_zone(path) == ZONE
        with pytest.raises(RuntimeError, match="held by the left arm's run 20260930T130000Z-x-0001"):
            PaletteLease({**ZONE, "arm": "right"}, path)
    assert held_zone(path) is None                         # released: the file stays, the lock does not


def test_a_crashed_holder_leaves_no_lease(tmp_path):
    path = tmp_path / "palette.lease"
    code = ("import os, sys; from tatbot_session.lease import PaletteLease; "
            f"PaletteLease({json.dumps(ZONE)}, {str(path)!r}); os._exit(0)")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert path.exists() and held_zone(path) is None
