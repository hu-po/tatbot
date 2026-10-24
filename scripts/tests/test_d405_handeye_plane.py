"""The plane hand-eye places link 6 where the drawing stack's arm model does."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import d405_handeye_plane as handeye
import kinematic_calibration
import numpy as np
import pytest
from urdf_kinematics import UrdfChain

# 2026-10-04's adopted pink offsets: the test's own, whatever config/workspace.yaml holds today.
OFFSETS = [0., .0098, .0003, -.0195, -.0023, 0., 0.]
JOINTS = np.array([.1, .9, .6, -.4, .2, .3])


def calibrated_chain(tmp_path, offsets):
    """urdf/tatbot.urdf as the drawing stack carries it: joint origins turned by the encoder-zero offsets."""
    tree = ET.parse(handeye.URDF)
    kinematic_calibration.apply_origins(tree.getroot().findall('joint'), 'right', offsets)
    tree.write(tmp_path / 'calibrated.urdf')
    return UrdfChain(tmp_path / 'calibrated.urdf')


def workspace(tmp_path, monkeypatch, offsets):
    path = tmp_path / 'workspace.yaml'
    path.write_text(f'right:\n  joint_offsets_rad: {json.dumps(offsets)}\n  paper_plane_z: 0.003\n')
    monkeypatch.setattr(handeye, 'WORKSPACE', path)


def captures(tmp_path):
    """One capture of a level plane 15 cm under each depth camera, stamped with JOINTS."""
    rng = np.random.default_rng(0)
    intrinsics = np.array([385., 385., 320., 240., 640., 480.])
    arrays = {'joints': JOINTS, 'carriage_m': 0.}
    for role in handeye.OPTICAL_FRAME:
        root_from_cam = handeye._look_at_plane(.15, 0., 0., 0., 0.)
        arrays[f'depth_{role}'] = handeye._synth_depth(root_from_cam, 0., intrinsics, 1e-4, 0., rng)
        arrays[f'intrinsics_{role}'] = intrinsics
        arrays[f'units_m_{role}'] = 1e-4
    directory = tmp_path / 'captures'
    directory.mkdir()
    np.savez_compressed(directory / 'capture-1.npz', **arrays)
    return directory


def test_link6_is_the_calibrated_model_at_the_measured_joints(tmp_path, monkeypatch):
    """FK(q + offsets) on the nominal URDF is apply_origins' FK(q), millimetres from nominal FK(q);
    the report names the offsets it used."""
    workspace(tmp_path, monkeypatch, OFFSETS)
    directory, nominal = captures(tmp_path), UrdfChain(handeye.URDF)
    per_role, _, _ = handeye.load_observations([directory], nominal, .5, np.asarray(OFFSETS))
    expected = calibrated_chain(tmp_path, OFFSETS).link_pose(handeye.LINK6, handeye.joint_map(JOINTS))
    drift = expected[:3, 3] - nominal.link_pose(handeye.LINK6, handeye.joint_map(JOINTS))[:3, 3]
    assert np.linalg.norm(drift) > 1e-3
    assert per_role and all(len(observations) == 1 for observations in per_role.values())
    for observations in per_role.values():
        np.testing.assert_allclose(observations[0][0], expected, atol=1e-12)
    handeye.run([directory], tmp_path / 'out', .5, None, handeye.URDF)
    report = json.loads((tmp_path / 'out/d405_handeye.json').read_text())
    assert report['provenance'][kinematic_calibration.KEY] == OFFSETS
    assert report['provenance']['paper_plane_z'] == .003


def test_a_yaw_joint_offset_is_refused(tmp_path, monkeypatch):
    workspace(tmp_path, monkeypatch, [.01, *OFFSETS[1:]])
    with pytest.raises(ValueError, match='joints 1-4'):
        handeye.run([captures(tmp_path)], tmp_path / 'out', .5, None, handeye.URDF)
    assert not (tmp_path / 'out').exists()
