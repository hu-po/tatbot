"""Per-arm identities, the arm owner's telemetry, and byte verification of a retained per-arm capture."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import arm_calibration as recipe
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_identities_resolve_each_label_to_its_arm_controller_and_tag_triplet():
    pink = recipe.selected_arm(REPO, "pink")
    blue = recipe.selected_arm(REPO, "blue")
    assert (pink.arm_id, pink.controller_role, pink.wrist_target, pink.tag_parent) == ("right", "follower", "wrist", "right/gripper_left")
    assert (blue.arm_id, blue.controller_role, blue.wrist_target, blue.tag_parent) == ("left", "leader", "wrist_left", "left/carriage_left")
    assert recipe.fiducial_target(REPO, "wrist")["ids"] == [2, 3, 4] and recipe.fiducial_target(REPO, "wrist_left")["ids"] == [5, 30, 1]
    assert pink.other_arm_id == "left" and blue.other_arm_id == "right"
    with pytest.raises(recipe.RecipeError, match="unknown physical arm"):
        recipe.selected_arm(REPO, "right")


def test_root_and_arm_base_frames_are_distinct_and_composed_once():
    from urdf_kinematics import UrdfChain
    chain = UrdfChain(REPO / "urdf/tatbot.urdf")
    for arm_id, y in (("right", -0.2675), ("left", 0.2675)):
        root_from_base = chain.link_pose(f"{arm_id}/base_link", {})
        assert root_from_base[1, 3] == pytest.approx(y)
        z = np.eye(4)
        z[:3, :3] = rot([0, 0, 1], 0.7)
        z[:3, 3] = [0.4, -0.15, 0.09]
        world_from_base = z @ root_from_base
        # Applying the mount twice, or not at all, moves the base by the mount offset.
        assert np.linalg.norm(world_from_base[:3, 3] - z[:3, 3]) == pytest.approx(0.2675)
        assert not np.allclose(world_from_base, z @ root_from_base @ root_from_base)
        import robot_world
        assert np.allclose(robot_world.root_from_world({"world_from_base": z.tolist()}), np.linalg.inv(z))


def test_telemetry_reader_refuses_nonfinite_and_reversed_records(tmp_path):
    path = tmp_path / "telemetry.bin"
    header = recipe.TELEMETRY_MAGIC + struct.pack("<Q", recipe.TELEMETRY_RECORD.size)
    good = recipe.TELEMETRY_RECORD.pack(1, 10, 100, *([0.0] * 21), 3, 0, 1)
    later = recipe.TELEMETRY_RECORD.pack(2, 20, 200, *([0.0] * 21), 3, 0, 1)
    path.write_bytes(header + good + later)
    rows = recipe.read_telemetry(path)
    assert [row["tick"] for row in rows] == [1, 2]
    path.write_bytes(header + later + good)
    with pytest.raises(recipe.RecipeError, match="reversed"):
        recipe.read_telemetry(path)
    bad = recipe.TELEMETRY_RECORD.pack(3, 30, 300, float("nan"), *([0.0] * 20), 3, 0, 1)
    path.write_bytes(header + good + bad)
    with pytest.raises(recipe.RecipeError, match="nonfinite"):
        recipe.read_telemetry(path)
    assert recipe.unchanged_feedback_ns(rows) == 100, "identical tuples count as unchanged time"





def test_owner_capture_binding_keeps_per_frame_joints_and_reports_motion(tmp_path):
    import json

    capture = tmp_path / 'capture.npz'
    telemetry = tmp_path / 'telemetry.bin'
    header = recipe.TELEMETRY_MAGIC + struct.pack('<Q', recipe.TELEMETRY_RECORD.size)
    telemetry.write_bytes(header+b''.join(recipe.TELEMETRY_RECORD.pack(i+1, t, t,
        *([i*.01]*7 + [.2]*7 + [0.]*7), 1, 0, 1) for i,t in enumerate([100_000_000,110_000_000,120_000_000,130_000_000])))
    def metadata(stamp, seq):
        return {'sequence': seq, 'timestamps': {'normalized_unix_ns': stamp}}
    np.savez(capture, camera_roles=np.array(json.dumps(['wrist_left'])),
             owner_frames_wrist_left=np.array(json.dumps([{'metadata': metadata(112_000_000, 8)}])),
             owner_color_metadata_wrist_left=np.array(json.dumps(metadata(121_000_000, 9))))
    result = recipe.bind_wrist_capture(capture, telemetry)
    assert not result['motion_authority'] and result['physical_accuracy_bound_m'] is None
    assert [pair['stream'] for pair in result['pairs']] == ['depth', 'color']
    assert result['pairs'][0]['joints_rad'] == [.01]*6
    assert result['pairs'][1]['joints_rad'] == [.02]*6
    assert result['maximum_skew_ms'] == 2
    assert result['capture_joint_span_rad'] == pytest.approx([.02]*6)
    assert result['capture_max_measured_velocity_rad_s'] == .2
    np.savez(capture, camera_roles=np.array(json.dumps(['wrist_left'])),
             owner_frames_wrist_left=np.array(json.dumps([{'metadata': metadata(140_000_000, 10)}])),
             owner_color_metadata_wrist_left=np.array(json.dumps(metadata(121_000_000, 9))))
    with pytest.raises(recipe.RecipeError, match='bracketed'):
        recipe.bind_wrist_capture(capture, telemetry)


def test_original_owner_packet_binds_each_exposure_and_refuses_unbracketed_or_reversed_time(tmp_path):
    import hashlib
    import json

    capture, telemetry = tmp_path/'owner.bin', tmp_path/'telemetry.bin'
    header = recipe.TELEMETRY_MAGIC + struct.pack('<Q', recipe.TELEMETRY_RECORD.size)
    def flight(walls):
        telemetry.write_bytes(header+b''.join(recipe.TELEMETRY_RECORD.pack(i+1, wall, (i+1)*10_000_000,
            *([i*.01]*7+[.2]*7+[0.]*7), 1, 0, 1) for i, wall in enumerate(walls)))
    def packet(stamps):
        frames = [{'metadata': {'sensor_name': 'synthetic-'+kind, 'profile': {'stream': kind},
                   'sequence': i+1, 'timestamps': {'normalized_unix_ns': stamp}},
                   'payload': {kind: {'bytes': 1}}} for i, (kind, stamp) in enumerate(stamps)]
        h = json.dumps({'magic': 'tatbot-vision-frame-set', 'version': 1,
                        'envelope': {'producer': {'node': 'synthetic-owner'}}, 'frames': frames}).encode()
        capture.write_bytes(len(h).to_bytes(4, 'big')+h+b'x'*len(frames))
    flight([100_000_000, 110_000_000, 120_000_000, 130_000_000])
    packet([('color', 112_000_000), ('depth', 121_000_000)])
    result = recipe.bind_owner_packet(capture, telemetry)
    assert result['source_capture_sha256'] == hashlib.sha256(capture.read_bytes()).hexdigest()
    assert [p['joints_rad'] for p in result['pairs']] == [[.01]*6, [.02]*6]
    assert result['capture_joint_span_rad'] == pytest.approx([.02]*6)
    assert result['physical_accuracy_bound_m'] is None and not result['motion_authority']
    assert result['world_registration'] == 'unmeasured'
    packet([('color', 140_000_000)])
    with pytest.raises(recipe.RecipeError, match='bracketed'):
        recipe.bind_owner_packet(capture, telemetry)
    packet([('color', True)])
    with pytest.raises(recipe.RecipeError, match='normalized exposure'):
        recipe.bind_owner_packet(capture, telemetry)
    packet([('color', 112_000_000)])
    flight([100_000_000, 120_000_000, 110_000_000, 130_000_000])
    with pytest.raises(recipe.RecipeError, match='wall clock'):
        recipe.bind_owner_packet(capture, telemetry)
