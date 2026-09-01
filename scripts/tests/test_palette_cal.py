"""Palette calibration: the artifact chooser, the writer, and the solve geometry.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_palette_cal.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "vision"))

import ink_spec  # noqa: E402


def _utc(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_choose_prefers_fresh_tip_then_vision_then_urdf():
    urdf = (0.126, 0.268, 0.085)
    tip = {"root_xyz_m": [0.13, 0.28, 0.06], "residual_mm": 1.2, "utc": _utc(1)}
    vis = {"root_xyz_m": [0.14, 0.27, 0.05], "residual_mm": 7.0, "utc": _utc(1)}
    # fresh tip wins over fresh vision
    c = ink_spec.choose_palette_root({"tip": tip, "vision": vis}, urdf)
    assert c["source"] == "tip" and c["root"] == (0.13, 0.28, 0.06) and c["residual_mm"] == 1.2
    # stale tip -> vision
    old_tip = {**tip, "utc": _utc(1000)}
    c = ink_spec.choose_palette_root({"tip": old_tip, "vision": vis}, urdf, max_age_h=168)
    assert c["source"] == "vision"
    # both stale -> urdf, with a note to re-measure
    c = ink_spec.choose_palette_root({"tip": old_tip, "vision": {**vis, "utc": _utc(1000)}}, urdf, max_age_h=168)
    assert c["source"] == "urdf" and c["root"] == urdf and "stale" in (c["note"] or "")
    # nothing at all -> urdf, no note
    c = ink_spec.choose_palette_root({}, urdf)
    assert c["source"] == "urdf" and c["note"] is None
    # undated is treated as usable (age unknown), not silently dropped
    c = ink_spec.choose_palette_root({"tip": {"root_xyz_m": [0.1, 0.2, 0.05], "residual_mm": 2.0}}, urdf)
    assert c["source"] == "tip"


def test_write_one_source_leaves_the_other(tmp_path):
    repo = tmp_path
    (repo / "config").mkdir()
    (repo / "config" / "palette_calibration.yaml").write_text("schema_version: 1\n")
    ink_spec.write_palette_cal("vision", {"root_xyz_m": [0.14, 0.27, 0.05], "residual_mm": 7.0, "utc": _utc(1)}, repo)
    ink_spec.write_palette_cal("tip", {"root_xyz_m": [0.13, 0.28, 0.06], "residual_mm": 1.2, "utc": _utc(1)}, repo)
    cal = ink_spec.load_palette_cal(repo)
    assert set(cal) == {"tip", "vision"}
    assert cal["tip"]["root_xyz_m"] == [0.13, 0.28, 0.06] and cal["vision"]["residual_mm"] == 7.0
    # re-writing tip does not drop vision
    ink_spec.write_palette_cal("tip", {"root_xyz_m": [0.131, 0.281, 0.061], "residual_mm": 1.1, "utc": _utc(0)}, repo)
    assert "vision" in ink_spec.load_palette_cal(repo)


def test_root_is_tag_centre_minus_the_urdf_offset():
    import palette_cal
    off = np.array(ink_spec.tag8_in_palette_root(REPO))
    center = np.array([0.30, -0.02, 0.16])
    assert np.allclose(palette_cal._root_from_tag_center(center), center - off)


def test_tip_pivot_recovers_a_known_tag_centre():
    """The authoritative solve: rolling the planted tip about one point recovers
    that point in the base frame, whatever the tip offset — the FK guarantee the
    tip source rests on."""
    import il_touchoff as touchoff
    rng = np.random.default_rng(0)
    tag = np.array([0.30, -0.02, 0.16])       # the planted point (base)
    tip = np.array([0.0597, -0.0032, -0.0005])  # tip in the EE frame
    poses = []
    for _ in range(12):
        ax = rng.normal(size=3)
        ax = ax / np.linalg.norm(ax)
        rmat = _rot(ax, rng.uniform(-0.5, 0.5))
        poses.append(_homog(rmat, tag - rmat @ tip))
    fit = touchoff.solve_pivot_trimmed(poses)
    assert np.linalg.norm(fit["pivot"] - tag) < 1e-6
    assert np.linalg.norm(fit["p"] - tip) < 1e-6


def _rot(axis, angle):
    axis = np.array(axis, float)
    x, y, z = axis / np.linalg.norm(axis)
    c, sn = np.cos(angle), np.sin(angle)
    k = 1 - c
    return np.array([
        [c + x * x * k, x * y * k - z * sn, x * z * k + y * sn],
        [y * x * k + z * sn, c + y * y * k, y * z * k - x * sn],
        [z * x * k - y * sn, z * y * k + x * sn, c + z * z * k]])


def test_base_from_root_is_the_inverse_of_the_urdf_mount():
    """base_from_root maps the arm's base origin (in root) back to the origin —
    the transform cal_tip needs because touchoff solves in the root frame."""
    bfr = ink_spec.base_from_root_matrix(REPO, "right")
    root = np.array(ink_spec.palette_root_in_base(REPO))       # palette_root in base
    # round trip: base -> root -> base is identity
    rfb = np.linalg.inv(bfr)
    back = (bfr @ rfb @ np.append(root, 1))[:3]
    assert np.allclose(back, root)
    # a pure rotation+translation (rigid): top-left 3x3 orthonormal
    assert np.allclose(bfr[:3, :3] @ bfr[:3, :3].T, np.eye(3), atol=1e-9)


def test_tip_holds_read_from_touches_then_wxtl(tmp_path):
    import palette_cal
    sess = tmp_path
    (sess / "touches.json").write_text(json.dumps({"tip_holds": [{"joints": [0.1] * 6}] * 5}))
    seqs, prov = palette_cal._tip_hold_joints(sess)
    assert len(seqs) == 5 and prov == "touches.json"
    (sess / "touches.json").unlink()
    seqs, prov = palette_cal._tip_hold_joints(sess)
    assert seqs == [] and "no touches" in prov


import json  # noqa: E402


@pytest.mark.skipif(__import__("importlib.util", fromlist=["find_spec"]).find_spec("cv2") is None,
                    reason="vision solve needs opencv")
def test_vision_geometry_round_trips(tmp_path):
    """Place tag8 at a known world pose, synthesize two calibrated cameras that
    look at it, and confirm cal_vision recovers palette_root through PnP and both
    bundle transforms — no hardware. (Verified against a hand-checked synthetic.)"""
    import json

    import palette_cal
    from fiducials import load_inventory
    half = load_inventory().target("palette").edge_m / 2
    off = np.array(ink_spec.tag8_in_palette_root(REPO))

    c_world = np.array([0.10, -0.05, 0.20])
    corners_tag = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    corners_world = np.array([c_world + _rot([0, 0, 1], 0.3) @ c for c in corners_tag])
    # the bundle's "world_from_base" is really world_from_ROOT (see the frame
    # note in palette_cal); the truth is palette_root in the RIGHT BASE frame.
    world_from_root = _homog(_rot([0.1, 0.2, 0.97], 1.1), [0.17, 0.26, 0.55])
    base_from_root = ink_spec.base_from_root_matrix(REPO, "right")
    tag_root = (np.linalg.inv(world_from_root) @ np.append(c_world, 1))[:3]
    tag_base = (base_from_root @ np.append(tag_root, 1))[:3]
    root_true = tag_base - off

    cam_k = np.array([[1600.0, 0, 1480], [0, 1600, 1080], [0, 0, 1]])
    cams, reports = {}, []
    for name, campos in {"camera1": [0.10, -0.05, 0.75], "camera2": [0.30, 0.10, 0.70]}.items():
        fwd = c_world - np.array(campos)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1.0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        r_wc = np.column_stack([right, -up, fwd])       # optical: x right, y down, z forward
        cam_from_world = np.linalg.inv(_homog(r_wc, campos))
        cc = np.array([(cam_from_world @ np.append(p, 1))[:3] for p in corners_world])
        assert (cc[:, 2] > 0).all()
        px = (cam_k @ cc.T).T
        px = px[:, :2] / px[:, 2:3]
        cams[name] = {"intrinsics": {"fx": 1600.0, "fy": 1600.0, "cx": 1480, "cy": 1080, "width": 2960, "height": 1668},
                      "distortion": {"coefficients": [0, 0, 0, 0, 0]},
                      "world_from_camera": {"rotation": r_wc.tolist(), "translation_m": list(campos)}}
        dets = [{"id": 8, "corners_px": px.tolist()}]
        if name == "camera1":
            dets.append({"id": 8, "corners_px": (px + 300).tolist()})  # a false-positive id-8, elsewhere in frame
        reports.append({"camera": name, "detections": dets})
    (tmp_path / "calib.json").write_text(json.dumps({"cameras": cams}))
    (tmp_path / "rw.json").write_text(json.dumps({"world_from_base": world_from_root.tolist(), "residual_mm_median": 6.9}))
    (tmp_path / "scan.json").write_text(json.dumps({"reports": reports}))

    written = {}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ink_spec, "write_palette_cal", lambda src, rec, repo=None: written.update({src: rec}) or Path("x"))

    class A:
        from_scan = str(tmp_path / "scan.json")
        calibration = str(tmp_path / "calib.json")
        robot_world = str(tmp_path / "rw.json")
        write = True
    rc = palette_cal.cal_vision(A())
    monkey.undo()
    assert rc == 0 and "vision" in written
    root = np.array(written["vision"]["root_xyz_m"])
    assert np.linalg.norm(root - root_true) < 1e-3, (root, root_true)


def _homog(rmat, t):
    m = np.eye(4)
    m[:3, :3] = rmat
    m[:3, 3] = t
    return m
