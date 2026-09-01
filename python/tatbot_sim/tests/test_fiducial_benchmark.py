import json

import numpy as np
import pytest
from tatbot_sim.fiducial_benchmark import POSE_JOINT_NAMES, REPO, _load_pose_bank, _pose_path
from tatbot_sim.urdf import rig_from_follower_base


def test_checked_in_pose_bank_is_finite_and_field_sourced():
    poses, metadata, digest = _load_pose_bank(REPO / "config/fiducial_benchmark_poses.json")

    assert poses.shape == (11, len(POSE_JOINT_NAMES))
    assert metadata["source_session"] == "sweep-20260825_192206"
    assert len(digest) == 64


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
