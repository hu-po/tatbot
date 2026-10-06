"""stencild samples (tatbot.target-pose/1) as the bridge reads them, and the pose helpers."""
import json
import math

import numpy as np
from tatbot_bridge import page

PATTERN = "stencil-" + "a" * 64


def envelope(source="measured", pose=None, pattern=PATTERN, **extra):
    payload = {"target_id": pattern, "target_frame": "world", "source": source,
               "world_from_target": (np.eye(4).tolist() if pose is None else pose) if source == "measured" else None,
               "calibration_id": "bundle", "translation_sigma_m": 0.001 if source == "measured" else None,
               "rotation_sigma_rad": 0.01 if source == "measured" else None, "tag_ids": [],
               "support": {"physical_instance_id": "print-7", "physical_instance_identity_verified": True,
                           "motion_authority": False}}
    payload.update(extra)
    return json.dumps({"schema": page.SCHEMA, "producer": {"role": "track"}, "seq": 5,
                       "stamp": {"mono_ns": 0, "wall_ns": 1_790_000_000_123_456_789, "basis": "x"},
                       "payload": payload}).encode()


def test_measured_sample():
    s = page.parse(envelope(), PATTERN)
    # Guaranteed non-None because envelope() constructs a valid tatbot.target-pose/1 payload for PATTERN
    assert s is not None
    assert s["source"] == "measured" and np.allclose(s["world_from_target"], np.eye(4))
    assert s["stamp_ns"] == 1_790_000_000_123_456_789 and s["seq"] == 5
    assert (s["print_id"], s["identity_verified"], s["calibration_id"]) == ("print-7", True, "bundle")
    assert s["translation_sigma_m"] == 0.001


def test_lost_sample_has_no_pose():
    s = page.parse(envelope("lost"), PATTERN)
    # Guaranteed non-None because envelope("lost") produces a valid lost sample for PATTERN
    assert s is not None
    assert s["source"] == "lost" and s["world_from_target"] is None and math.isnan(s["translation_sigma_m"])


def test_other_prints_and_malformed_samples_are_ignored():
    assert page.parse(envelope(pattern="stencil-" + "b" * 64), PATTERN) is None
    assert page.parse(b"not json", PATTERN) is None
    assert page.parse(json.dumps({"schema": "tatbot.other/1"}), PATTERN) is None
    assert page.parse(envelope(pose=(2 * np.eye(4)).tolist()), PATTERN) is None   # not rigid
    assert page.parse(envelope(source="guessed"), PATTERN) is None
    s = page.parse(envelope(), None)
    # Guaranteed non-None because passing pattern_id=None matches any target_id in a valid envelope
    assert s is not None
    assert s["pattern_id"] == PATTERN


def test_covariance_and_quaternion():
    cov = page.covariance(0.002, 0.01)
    assert np.allclose(np.diag(np.reshape(cov, (6, 6))), [4e-6] * 3 + [1e-4] * 3)
    assert page.covariance(math.nan, 0.0)[0] == -1.0
    rng = np.random.default_rng(3)
    randoms = []
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        randoms.append(q * np.sign(np.linalg.det(q)))
    # 180 degree turns (w = 0) about axes with one, two and three non-zero components
    flips = [2 * np.outer(a, a) - np.eye(3) for a in
             (np.array(v, float) / np.linalg.norm(v) for v in ([1, 0, 0], [0, 0, 1], [1, -1, 0], [0, 1, 1], [1, -2, 3]))]
    for q in randoms + flips + [np.diag([1.0, -1.0, -1.0]), np.eye(3)]:
        x, y, z, w = page.quaternion(q)
        back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(back, q, atol=1e-9)


def test_fixed_pose_from_xyz_rpy():
    m = page.from_xyz_rpy([0.371, -0.231, 0.0334], [0.0, 0.0, math.pi / 2])
    assert np.allclose(m[:3, 3], [0.371, -0.231, 0.0334])
    assert np.allclose(m[:3, 0], [0, 1, 0]) and np.allclose(m[:3, 2], [0, 0, 1])
    assert np.allclose(page.rpy_deg(page.from_xyz_rpy([0, 0, 0], np.radians([10, -20, 30]))[:3, :3]), [10, -20, 30])
