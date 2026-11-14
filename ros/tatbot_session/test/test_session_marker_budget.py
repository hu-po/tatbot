"""The draw's marker budget (drawrun.marker_loss): the compile's check again, against the print installed as the page."""
import sys

import numpy as np
import pytest
import yaml
from tatbot_description import repo_root

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from tatbot_session import drawrun  # noqa: E402

REPO = repo_root(None)


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """(where prints are installed, the forearm18 print's pattern id): one generated print, installed by its id."""
    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    import stencil_tile

    references = tmp_path_factory.mktemp("references")
    made = stencil_tile.artwork("forearm18", 0, references / "new")
    (references / "new").rename(references / made["pattern_id"])
    return references, made["pattern_id"]


def program(lines, **more):
    return {"resources": [{"id": "black", "rgb": [60, 60, 63], "tool": {"id": "tool", "line_width_m": None}}],
            "ops": [{"op": "stroke", "id": f"s{n:04d}", "resource_id": "black", "closed": False,
                     "generation_width_m": 0.0009, "points_m": line} for n, line in enumerate(lines)], **more}


HATCH = [[[-0.018, y], [0.018, y]] for y in np.arange(-0.028, 0.029, 0.002)]   # the whole tile
DOT = [[[0.0, 0.0], [0.004, 0.0]]]


def stack(pattern_id, **page):
    markers = yaml.safe_load((REPO / "ros/tatbot_bringup/config/stack.yaml").read_text())["page"]["markers"]
    return {"hardware": "real",
            "page": {"source": "stencil", "pattern_id": pattern_id, "static_run": "", "markers": markers, **page}}


def test_a_program_covering_the_installed_prints_markers_is_named(installed):
    references, pattern_id = installed
    lost = drawrun.marker_loss(program(HATCH), stack(pattern_id), references)
    assert lost.startswith("the program covers the markers its print is tracked by: after black (stroke s0028) the "
                           "overhead would read 0 of the print's 6 markers")
    assert drawrun.marker_loss(program(DOT), stack(pattern_id), references) is None


def test_what_is_drawn_without_tracking_is_not_asked(installed):
    references, pattern_id = installed
    covered = program(HATCH)
    assert drawrun.marker_loss(covered, stack(pattern_id, static_run="20261110T175827Z-node-ba54"), references) is None
    assert drawrun.marker_loss(covered, stack(pattern_id, source="fixed"), references) is None
    assert drawrun.marker_loss(covered, {**stack(pattern_id), "hardware": "mock"}, references) is None
    assert drawrun.marker_loss(covered, stack("stencil-not-installed"), references) is None
    assert drawrun.marker_loss({**covered, "preparation": {"allow_marker_loss": True}}, stack(pattern_id), references) is None
    assert drawrun.marker_loss({**covered, "rehearsal": {"of": "x"}}, stack(pattern_id), references) is None
    without = stack(pattern_id)
    del without["page"]["markers"]   # a stack.yaml from before the budget
    assert drawrun.marker_loss(covered, without, references) is None


def test_the_draw_refuses_only_when_the_budget_is_enforced(installed, monkeypatch):
    references, pattern_id = installed
    monkeypatch.setattr("tatbot_contracts.ros_program.validate_for_execution", lambda program: None)
    monkeypatch.setattr(drawrun.geometry, "off_print", lambda program, page: None)
    loss = drawrun.marker_loss
    monkeypatch.setattr(drawrun, "marker_loss", lambda program, stack: loss(program, stack, references))
    covered, on = program(HATCH, arm="xarm"), stack(pattern_id)
    assert "the overhead would read 0" in drawrun.check_program(covered, on, ["xarm"])   # said, not refused
    on["page"]["markers"]["enforce"] = True
    with pytest.raises(ValueError, match="covers the markers its print is tracked by"):
        drawrun.check_program(covered, on, ["xarm"])
    assert drawrun.check_program(program(DOT, arm="xarm"), on, ["xarm"]) is None
