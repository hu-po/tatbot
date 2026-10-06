"""Recorded-scene contracts: calibration, target masks, portable private assets, and starts at the stops."""

from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tatbot_travel.camera import Intrinsics
from tatbot_travel.episode import EpisodeConfig, EpisodeRunner
from tatbot_travel.kinematics import LIMIT_MARGIN
from tatbot_travel.profile import load_profile
from tatbot_travel.scene_bank import build_bank, deproject, surface_mesh
from tatbot_travel.selfview import Compositor, SelfViewLook, load_layers


def test_aligned_native_depth_recovers_the_colour_frame_z():
    intr = Intrinsics(2, 2, 4, 4, 0, 0)
    rot = Rotation.from_euler("y", 0.1).as_matrix()
    offset = np.array([0.002, 0, 0.003])
    rays = np.array([[[0, 0, 1], [0.25, 0, 1]], [[0, 0.25, 1], [0.25, 0.25, 1]]])
    colour_points = rays * 0.4
    native_z = ((colour_points - offset) @ rot)[..., 2]
    alignment = {"aligned_depth_value_axis": "native_depth_z", "rotation": rot.ravel(order="F").tolist(),
                 "translation": offset.tolist()}
    result = deproject(native_z, intr, {"depth_units_m": "1", "alignment_calibration": json.dumps(alignment)})
    assert np.allclose(result, colour_points, atol=1e-12)


def test_room_mesh_does_not_bake_masked_phantom_or_bridge_depth_edges():
    y, x = np.mgrid[:17, :17]
    xyz = np.stack([x / 100, y / 100, np.ones_like(x)], axis=-1)
    keep = np.ones((17, 17), bool)
    # The masked pixel is between mesh vertices: testing vertices alone would leave the target baked in.
    keep[4, 4] = False
    text = surface_mesh(xyz, keep, stride=8)
    faces = [line for line in text.splitlines() if line.startswith("f ")]
    assert len(faces) == 6
    assert not any("1/1 " in line for line in faces)
    xyz[8:, :, 2] += 1
    faces = [line for line in surface_mesh(xyz, np.ones_like(keep), stride=8).splitlines() if line.startswith("f ")]
    assert len(faces) == 4  # only the lower two cells: the depth boundary stays open


def test_profile_paths_move_with_the_bank_and_unknown_fields_fail(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "room").mkdir()
    spec = {"version": 1, "episode": {"world": {"scene_view": False, "real_assets": ".", "room_assets": "room"},
                                     "motion": {"placement": {"yaw_deg": [80, 100]}},
                                     "start_trace_prob": 0.2, "start_approach_prob": 0.2}}
    source = bank / "profile.json"
    source.write_text(json.dumps(spec))
    cfg = load_profile(source, 3)
    assert cfg.duration_s == 3 and not cfg.world.scene_view
    assert cfg.world.room_assets == str(bank / "room")
    assert cfg.motion.placement.yaw_deg == (80, 100)
    assert len(cfg.profile_id) == 64
    spec["episode"]["world"]["misspelled"] = 1
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unknown WorldConfig"):
        load_profile(source, 3)


def test_rest_observations_use_physical_stops_without_changing_ik_margin():
    runner = EpisodeRunner.__new__(EpisodeRunner)
    runner.cfg = EpisodeConfig(start_pose_rad=(0, 0, 0, 0, 0, np.pi / 2), park_noise_rad=0)
    runner.rng = np.random.default_rng(0)
    runner.kin = SimpleNamespace(lower=np.full(6, LIMIT_MARGIN), upper=np.full(6, 3 - LIMIT_MARGIN))
    runner.q_park = np.full(6, 0.1)
    assert np.array_equal(runner._start_pose(), runner.cfg.start_pose_rad)
    assert np.all(runner.kin.lower == LIMIT_MARGIN)


def test_private_cradle_alpha_also_controls_expert_visibility(tmp_path):
    bgra = np.full((480, 640, 4), 100, np.uint8)
    bgra[..., 3] = 0
    bgra[200:, :300, 3] = 255
    cv2.imwrite(str(tmp_path / "cradle.png"), bgra)
    layers = load_layers(str(tmp_path))
    mask = layers[0].alpha[..., 0] > 0.5
    result = Compositor(SelfViewLook(0, 1, (1, 1, 1), str(tmp_path)))(np.zeros((480, 640, 3), np.uint8))
    assert np.all(result[mask] == 100)
    assert np.all(result[~mask] == 0)


def test_scene_bank_refuses_to_write_lab_images_into_a_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(ValueError, match="outside Git checkouts"):
        build_bank(tmp_path / "captures", tmp_path / "layout.json", tmp_path / "bank")
