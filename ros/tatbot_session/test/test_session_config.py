"""The deployed revision in run meta, the page trim re-read at every goal, the trim's sign, and the page
taken from the installed print."""
import json

import numpy as np
from tatbot_description import repo_root
from tatbot_session import config, geometry


def test_revision_reads_deploys_file(tmp_path):
    assert config.revision(tmp_path) is None
    (tmp_path / "REVISION").write_text("0123abcd\ndirty=1\n")
    assert config.revision(tmp_path) == {"sha": "0123abcd", "dirty": True}
    (tmp_path / "REVISION").write_text("0123abcd\ndirty=0\n")
    assert config.revision(tmp_path) == {"sha": "0123abcd", "dirty": False}


def test_page_trim_is_read_from_the_launched_stack_yaml(tmp_path):
    path = tmp_path / "stack.yaml"
    assert config.page_trim(None) is None and config.page_trim(str(path)) is None
    path.write_text("page:\n  trim:\n    right: [0.0012, -0.0005]\n")
    assert config.page_trim(str(path)) == {"right": [0.0012, -0.0005]}
    path.write_text("page: {}\n")
    assert config.page_trim(str(path)) == {}


def test_trim_sign_moves_the_drawn_centre_by_the_trim():
    # base_from_page_used = touched . T(trim): the program's centre lands at +trim in the touched page
    # frame. A cross drawn with trim t0 whose centre sits at d from the printed centre (page axes: x to
    # the print's right, y toward its top) is corrected by trim = t0 - d.
    touched = np.eye(4)
    drawn_at = geometry.page_used(touched, np.eye(4), [0.0015, -0.002]) @ np.array([0, 0, 0, 1.0])
    assert np.allclose(drawn_at[:2], [0.0015, -0.002])
    error = np.array([0.0015, -0.002])  # where trim 0 drew the centre
    fixed = geometry.page_used(touched, np.eye(4), -error)[:2, 3] + error
    assert np.allclose(fixed, 0.0)


def test_shared_stack_reader_preserves_required_installed_session_configuration(tmp_path, monkeypatch):
    import sys

    import pytest
    from tatbot_bridge import stack
    from tatbot_calib import cli

    path = tmp_path/'ros/tatbot_bringup/config/stack.yaml'
    path.parent.mkdir(parents=True)
    path.write_text('registration: {right: fixture-registration}\n')
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', None)
    assert cli._stack(tmp_path) == stack.load(repo=tmp_path)
    assert config.load_stack(str(path)) == {'registration': {'right': 'fixture-registration'}}
    with pytest.raises(ImportError):
        config.load_stack()


def test_all_three_stack_readers_prefer_the_installed_share(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    from tatbot_bridge import stack
    from tatbot_calib import cli

    path = tmp_path/'installed/config/stack.yaml'
    path.parent.mkdir(parents=True)
    path.write_text('registration: {right: installed-registration}\n')
    package = ModuleType('ament_index_python.packages')
    package.get_package_share_directory = lambda name: str(path.parent.parent)
    monkeypatch.setitem(sys.modules, 'ament_index_python', ModuleType('ament_index_python'))
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', package)
    assert config.load_stack() == cli._stack(tmp_path) == stack.load(repo=tmp_path)


def test_the_page_comes_from_the_installed_print(tmp_path):
    """`ros up` takes the page's size, clear centre and inner border edges from the print page.pattern_id names
    (an 83 x 127 mm half-skin print here); without one installed, stack.yaml's own page stands."""
    page = {"source": "stencil", "pattern_id": "stencil-half", "size_m": [0.100, 0.150], "clear_m": [0.062, 0.112],
            "inner_edges_m": {"left": -0.03, "right": 0.03, "bottom": -0.05, "top": 0.05}, "max_lost_s": 5.0}
    assert config.print_page(repo_root(None), page, tmp_path) == {**page, "geometry": "stack.yaml"}
    installed = tmp_path / "stencils" / "references" / "stencil-half"
    installed.mkdir(parents=True)
    (installed / "tracking.json").write_text(json.dumps({
        "pattern_id": "stencil-half", "page_mm": [83.0, 127.0], "generator": {"svg_sha256": "abc"},
        "clear_center_uv": [19 / 83, 19 / 127, 64 / 83, 108 / 127]}))
    found = config.print_page(repo_root(None), page, tmp_path)
    assert found == {"source": "stencil", "pattern_id": "stencil-half", "size_m": [0.083, 0.127],
                     "clear_m": [0.045, 0.089], "max_lost_s": 5.0, "geometry": "print stencil-half"}
    (installed / "settings.json").write_text(json.dumps({"artwork_svg_sha256": "abc",
                                                         "border_inner_mm": [19.897, 19.643, 63.077, 107.357]}))
    edges = config.print_page(repo_root(None), page, tmp_path)["inner_edges_m"]
    assert edges == {"left": -0.021603, "right": 0.021577, "bottom": -0.043857, "top": 0.043857}
