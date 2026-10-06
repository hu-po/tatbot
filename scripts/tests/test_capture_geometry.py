"""A capture's declared camera views are the views it holds, on one arm."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import wrist_cameras
from capture_geometry import declared_roles


@pytest.fixture(autouse=True)
def configured_two_view_capture(tmp_path, monkeypatch):
    """Two configured views on the right arm, one on the left."""
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / 'config').mkdir()
    shutil.copyfile(repo / 'config/arms.json', tmp_path / 'config/arms.json')
    vision = tmp_path / 'rust/visiond/config/vision.toml'
    vision.parent.mkdir(parents=True)
    vision.write_text(''.join(f'''
[[cameras.realsense]]
name = "{name}"
serial = "{name}"
role = "wrist_{name}"
group = "d405"
arm = "{arm}"
owner_role = "test-owner"
''' for name, arm in (('upper', 'right'), ('lower', 'right'), ('left', 'left'))))
    actual = wrist_cameras.capture_arm
    monkeypatch.setattr(wrist_cameras, 'capture_arm', lambda roles, **_: actual(roles, repo=tmp_path))


def capture():
    data = {'camera_roles': np.array('["wrist_upper", "wrist_lower"]')}
    for role in ('wrist_upper', 'wrist_lower'):
        data[f'owner_frames_{role}'] = np.array('[]')
    return data


def test_declared_roles_are_the_views_the_capture_holds():
    assert declared_roles(capture()) == ['wrist_upper', 'wrist_lower']
    data = capture()
    del data['camera_roles']
    with pytest.raises(ValueError, match='lacks declared camera roles'):
        declared_roles(data)


@pytest.mark.parametrize('roles', [[], ['wrist_left'], ['wrist_probe'], ['wrist_upper', 'wrist_left'],
                                   ['wrist_upper', 'wrist_upper'], ['wrist_upper']])
def test_camera_declaration_cannot_hide_or_invent_views(roles):
    data = capture()
    data['camera_roles'] = np.array(json.dumps(roles))
    with pytest.raises(ValueError, match='camera roles'):
        declared_roles(data)
