"""drawn.svg renders planned and measured strokes in page millimetres."""
import xml.etree.ElementTree as ET

import pytest
from tatbot_session import drawn


def test_drawn_svg(tmp_path):
    program = {"ops": [{"op": "stroke", "id": "s0000", "closed": True,
                        "points_m": [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]]},
                       {"op": "pause", "id": "p0001"}]}
    path = drawn.write(tmp_path / "drawn.svg", program, {"s0000": [[[-0.01, -0.01], [0.0, -0.0101]], []]})
    root = ET.parse(path).getroot()
    paths = {p.get("id"): p.get("d") for p in root.iter("{http://www.w3.org/2000/svg}path")}
    assert paths["plan-s0000"].startswith("M 40.000 85.000") and paths["plan-s0000"].endswith("Z")
    assert paths["drawn-s0000-0"] == "M 40.000 85.000 L 50.000 85.100"
    assert root.get("viewBox") == "0 0 100.000 150.000"


@pytest.mark.parametrize('width', [.0004, None])
def test_each_resource_width_and_unknown_evidence_are_rendered(width):
    program = {'resources': [{'id': 'fine', 'tool': {'line_width_m': width}},
                             {'id': 'broad', 'tool': {'line_width_m': .001}}],
               'ops': [{'op': 'stroke', 'id': name, 'resource_id': name, 'closed': False,
                        'points_m': [[0, 0], [.001, 0]], 'generation_width_m': .0002} for name in ('fine', 'broad')]}
    root = ET.fromstring(drawn.render(program, {}))
    paths = {p.get('id'): p.get('stroke-width') for p in root.iter('{http://www.w3.org/2000/svg}path')}
    assert paths == {'plan-fine': '0.400' if width else '0.200', 'plan-broad': '1.000'}
    assert ('Physical width unknown' in ET.tostring(root).decode()) is (width is None)


def test_every_goals_measured_lines_come_back_from_the_ledger():
    rows = [{'arm': 'right', 'event': 'aborted', 'op': 's1', 'line': [[0, 0], [.001, 0]]},
            {'arm': 'right', 'event': 'sent', 'op': 's1'},
            {'arm': 'right', 'event': 'done', 'op': 's1', 'line': [[.001, 0], [.002, 0]]},
            {'arm': 'left', 'event': 'done', 'op': 's1', 'line': [[0, 0], [1, 1]]}]
    assert drawn.from_rows(rows, 'right') == {'s1': [rows[0]['line'], rows[2]['line']]}
