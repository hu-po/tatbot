import json

import numpy as np
import pytest
from tatbot_sim.fiducial_benchmark import POSE_JOINT_NAMES, REPO, _load_pose_bank, _pose_path
from tatbot_sim.urdf import rig_from_follower_base


@pytest.mark.field_calibration
def test_checked_in_pose_banks_match_the_new_carriers_and_exclude_witnesses():
    poses, metadata, digest = _load_pose_bank(REPO / "config/fiducial_benchmark_poses.json")

    assert poses.shape == (13, len(POSE_JOINT_NAMES))
    assert metadata["source_session"] == "sweep-arm-pink-20260919_124146"
    assert len(metadata['hold_ids']) == 13
    assert all(not hold.startswith('witness-') for hold in metadata['hold_ids'])
    assert len(digest) == 64
    left, left_meta, _ = _load_pose_bank(
        REPO / 'config/fiducial_benchmark_poses_left.json', arm='left')
    assert left.shape == (13, 7)
    assert left_meta['source_session'] == 'sweep-arm-blue-20260919_124943'
    assert all(not hold.startswith('witness-') for hold in left_meta['hold_ids'])


@pytest.mark.field_calibration
def test_both_observed_layouts_match_the_inventory_and_have_rigid_tag_poses():
    from ee_fiducial import WristLayout

    for target, filename, ids in (
        ('wrist', 'wrist_tags_measured.json', {2, 3, 4}),
        ('wrist_left', 'wrist_tags_measured_left.json', {1, 5, 30}),
    ):
        layout = WristLayout.load(REPO / 'config' / filename, target=target)
        assert set(layout.ee_from_tag) == ids
        assert layout.parent_frame.startswith('right/' if target == 'wrist' else 'left/')


def test_pose_bank_rejects_wrong_joint_contract(tmp_path):
    path = tmp_path / "poses.json"
    path.write_text(
        json.dumps({"schema_version": 1, "joint_names": ["wrong"], "poses": [[0] * 7] * 3})
    )

    with pytest.raises(ValueError, match="pose-bank joints"):
        _load_pose_bank(path)


def test_pose_path_is_seeded_smooth_and_stays_between_observed_poses():
    poses = np.arange(28, dtype=float).reshape(4, 7)
    order, sample = _pose_path(poses, np.random.default_rng(17))

    assert sorted(order.tolist()) == [0, 1, 2, 3]
    np.testing.assert_allclose(sample(0.0), poses[order[0]])
    np.testing.assert_allclose(sample(0.125), (poses[order[0]] + poses[order[1]]) / 2)
    np.testing.assert_allclose(sample(1.0), poses[order[0]])


def test_follower_mount_is_derived_from_canonical_dual_arm_urdf():
    expected = np.eye(4)
    expected[1, 3] = -0.2675

    np.testing.assert_allclose(rig_from_follower_base(), expected)
