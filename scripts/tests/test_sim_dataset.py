"""Tests for dataset auditing (sim_dataset_audit.py) and frame sampling (sim_dataset_samples.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# Ensure pandas and tyro can be imported even in lightweight test environments.
if "pandas" not in sys.modules:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:

        class FakeSeries:
            def __init__(self, data):
                self._data = list(data)

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def to_numpy(self):
                return self._data

        class FakeRow:
            def __init__(self, d):
                self._d = d

            def __getattr__(self, k):
                if k in self._d:
                    return self._d[k]
                raise AttributeError(k)

            def __getitem__(self, k):
                return self._d[k]

        class FakeILoc:
            def __init__(self, df):
                self._df = df

            def __getitem__(self, idx):
                return FakeRow({k: v[idx] for k, v in self._df._data.items()})

        class FakeDataFrame:
            def __init__(self, data=None):
                self._data = data or {}

            def __len__(self):
                first = next(iter(self._data.values()), [])
                return len(first)

            def __getattr__(self, k):
                if k in self._data:
                    return FakeSeries(self._data[k])
                raise AttributeError(k)

            def __getitem__(self, k):
                return FakeSeries(self._data[k])

            @property
            def empty(self):
                return len(self) == 0

            def sort_values(self, col):
                return self

            def reset_index(self, drop=True):
                return self

            @property
            def iloc(self):
                return FakeILoc(self)

        class FakePandas:
            DataFrame = FakeDataFrame

            @staticmethod
            def concat(objs, ignore_index=True):
                if not objs:
                    return FakeDataFrame()
                combined = {}
                for k in objs[0]._data:
                    combined[k] = []
                    for o in objs:
                        combined[k].extend(o._data.get(k, []))
                return FakeDataFrame(combined)

            @staticmethod
            def read_parquet(f, columns=None, filters=None):
                path = Path(f)
                if path.exists():
                    try:
                        d = json.loads(path.read_text())
                        return FakeDataFrame(d)
                    except Exception:
                        pass
                return FakeDataFrame()

        sys.modules["pandas"] = FakePandas()

# Always monkeypatch read_parquet if real pandas was imported so parquet mocks work in any environment.
def _mock_read_parquet(f, columns=None, filters=None):
    path = Path(f)
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if hasattr(sys.modules["pandas"], "DataFrame"):
                return sys.modules["pandas"].DataFrame(d)
        except Exception:
            pass
    if hasattr(sys.modules["pandas"], "DataFrame"):
        return sys.modules["pandas"].DataFrame()
    return None

if "pandas" in sys.modules and not hasattr(sys.modules["pandas"], "FakePandas"):
    sys.modules["pandas"].read_parquet = _mock_read_parquet

if "tyro" not in sys.modules:
    try:
        import tyro  # noqa: F401
    except ImportError:
        sys.modules["tyro"] = types.ModuleType("tyro")

REPO = Path(__file__).resolve().parents[2]

spec_audit = importlib.util.spec_from_file_location(
    "sim_dataset_audit", REPO / "scripts" / "sim_dataset_audit.py"
)
sim_dataset_audit = importlib.util.module_from_spec(spec_audit)
sys.modules["sim_dataset_audit"] = sim_dataset_audit
spec_audit.loader.exec_module(sim_dataset_audit)

spec_samples = importlib.util.spec_from_file_location(
    "sim_dataset_samples", REPO / "scripts" / "sim_dataset_samples.py"
)
sim_dataset_samples = importlib.util.module_from_spec(spec_samples)
sys.modules["sim_dataset_samples"] = sim_dataset_samples
spec_samples.loader.exec_module(sim_dataset_samples)


def make_dataset_shard(
    root: Path,
    *,
    episodes: int = 2,
    frames_per_ep: int = 50,
    distribution: str | None = "paper-draw",
    tool_id: str = "lutin-ballpoint-dot",
    field_snapshots: int | None = None,
    skipped_batches: list | None = None,
    engaged: bool = True,
    corrupt_ep_table: bool = False,
    start_lead_rad: float = 0.01,
) -> Path:
    """Helper to build a valid or invalid dataset shard tree."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": episodes * frames_per_ep})
    )
    run_meta = {
        "config": {
            "distribution": distribution,
            "draw_clearance": 0.0,
            "tool_calibration_jitter": False,
            "tool_calibration_scale": 1.0,
        },
        # a fresh shard carries the follower that made it (fixed-mount-v2 since
        # 2026-08-30); one without this is a gripper-held-v1 shard to the audit
        "tool": {
            "tool_id": tool_id,
            "substrate": "paper",
            "embodiment": "fixed-mount-v2",
            "contact": True,
            "tool_geometry_version": "resolved-tool-v1",
            "geometry_status": "contact-qualified",
            "contact_geometry_status": "pivot-calibrated",
            "body_pose_status": "axis-inferred",
            "interaction_model": "rigid-contact-v1",
            "body_tip_offset_m": [0.0, 0.0, 0.06],
            "calibrated_tip_offset_m": [0.0, 0.0, 0.06],
            "calibration_delta_m": [0.0, 0.0, 0.0],
            "tip_offset_m": [0.0, 0.0, 0.06],
            "tcp_offset_m": [0.0, 0.0, 0.06],
        },
        "episodes": [{
            "kind": "language",
            "engaged": engaged,
            "interaction": {
                "frames": 10,
                "distance_min_m": -0.0001,
                "distance_mean_m": 0.0,
                "distance_max_m": 0.0002,
            },
        } for _ in range(episodes)],
    }
    if skipped_batches is not None:
        run_meta["skipped_batches"] = skipped_batches
    (meta / "run_meta.json").write_text(json.dumps(run_meta))

    ep_dir = meta / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    if field_snapshots is not None:
        fields_dir = meta / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)
        for i in range(field_snapshots):
            (fields_dir / f"episode_{i:06d}.png").write_bytes(b"png_data")

    if corrupt_ep_table:
        (ep_dir / "ep.parquet").write_text("not json parquet")
    else:
        # Our mock read_parquet parses json from parquet file
        ep_data = {
            "episode_index": list(range(episodes)),
            "tasks": [["mix_task"]] * episodes,
            "length": [frames_per_ep] * episodes,
            "videos/observation.images.wrist_upper/chunk_index": [0] * episodes,
            "videos/observation.images.wrist_upper/file_index": [0] * episodes,
            "videos/observation.images.wrist_upper/from_timestamp": [0.0] * episodes,
            "videos/observation.images.wrist_upper/to_timestamp": [5.0] * episodes,
            "videos/observation.images.wrist_lower/chunk_index": [0] * episodes,
            "videos/observation.images.wrist_lower/file_index": [0] * episodes,
            "videos/observation.images.wrist_lower/from_timestamp": [0.0] * episodes,
            "videos/observation.images.wrist_lower/to_timestamp": [5.0] * episodes,
        }
        (ep_dir / "ep.parquet").write_text(json.dumps(ep_data))

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "action": [[start_lead_rad, 0.0] for _ in range(episodes)],
        "observation.state": [[0.0, 0.0] for _ in range(episodes)],
        "frame_index": [0] * episodes,
    }
    (data_dir / "data.parquet").write_text(json.dumps(data))

    return root


# --- Audit Tests ---


def test_audit_shards_discovery(tmp_path: Path) -> None:
    # Single dataset root
    make_dataset_shard(tmp_path / "ds1")
    shards = sim_dataset_audit._shards(tmp_path / "ds1")
    assert shards == [tmp_path / "ds1"]

    # Parent directory holding shards
    root = tmp_path / "overnight"
    make_dataset_shard(root / "shard-a")
    make_dataset_shard(root / "shard-b")
    (root / ".hidden").mkdir()
    shards = sim_dataset_audit._shards(root)
    assert sorted(p.name for p in shards) == ["shard-a", "shard-b"]


def test_audit_valid_dataset_passes(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "valid-paper")
    args = sim_dataset_audit.Args(path=ds, verbose=True)
    assert sim_dataset_audit.main(args) == 0
    out = capsys.readouterr().out
    assert "all checks passed" in out
    assert "paper-draw" in out


def test_audit_requires_clean_stable_source_for_current_run_metadata(
        tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "dirty-source")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["schema_version"] = 2
    run_meta["software"] = {
        "repository": "example/tatbot",
        "revision_start": "a" * 40,
        "revision_end": "b" * 40,
        "dirty_start": False,
        "dirty_end": True,
    }
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1
    out = capsys.readouterr().out
    assert "source revision changed during generation" in out
    assert "source checkout was dirty or unknown" in out


def test_audit_accepts_bounded_self_consistent_calibration_jitter(
        tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "jittered-paper")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["config"]["tool_calibration_jitter"] = True
    run_meta["tool"]["contact_uncertainty_m"] = 0.004637
    run_meta["tool"]["calibration_delta_m"] = [0.001, -0.002, 0.003]
    run_meta["tool"]["tip_offset_m"] = [0.001, -0.002, 0.063]
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 0
    assert "all checks passed" in capsys.readouterr().out


def test_audit_rejects_unbounded_or_inconsistent_calibration_jitter(
        tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "bad-jitter")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["config"]["tool_calibration_jitter"] = True
    run_meta["tool"]["contact_uncertainty_m"] = 0.004
    run_meta["tool"]["calibration_delta_m"] = [0.005, 0.0, 0.0]
    # Deliberately leave tip_offset_m at the central calibration too.
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1
    out = capsys.readouterr().out
    assert "outside its 4.000 mm bound" in out
    assert "actual minus calibrated tip disagrees" in out


def test_audit_rejects_air_gap_geometry_unless_explicitly_historical(
        tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "air-gap")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["tool"].pop("tool_geometry_version")
    run_meta["tool"].pop("interaction_model")
    run_meta["config"]["draw_clearance"] = 0.004
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1
    assert "air-gap-v0" in capsys.readouterr().out
    assert sim_dataset_audit.main(
        sim_dataset_audit.Args(path=ds, allow_air_gap=True)) == 0


def test_audit_rejects_provisional_geometry_for_production(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "provisional")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["tool"]["geometry_status"] = "provisional"
    run_meta["tool"]["contact_geometry_status"] = "unqualified"
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1
    assert "quality-gated pivot TCP" in capsys.readouterr().out
    assert sim_dataset_audit.main(
        sim_dataset_audit.Args(path=ds, allow_provisional=True)) == 0


def test_audit_rejects_marks_outside_the_contact_band(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "floating-mark")
    run_meta_path = ds / "meta" / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text())
    run_meta["episodes"][0]["interaction"]["distance_max_m"] = 0.001
    run_meta_path.write_text(json.dumps(run_meta))

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1
    assert "above the surface" in capsys.readouterr().out


def test_audit_detects_tool_import_binding_mismatch(tmp_path: Path, capsys) -> None:
    # Distribution skin-erase expects picosecond-laser-pen, but tool recorded is lutin-ballpoint-dot
    ds = make_dataset_shard(tmp_path / "mismatched", distribution="skin-erase", tool_id="lutin-ballpoint-dot")
    args = sim_dataset_audit.Args(path=ds)
    assert sim_dataset_audit.main(args) == 1
    out = capsys.readouterr().out
    assert "built with the wrong geometry" in out
    assert "claims distribution 'skin-erase'" in out


def test_audit_detects_missing_distribution(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "no-dist", distribution=None)
    args = sim_dataset_audit.Args(path=ds)
    assert sim_dataset_audit.main(args) == 1
    out = capsys.readouterr().out
    assert "no distribution recorded" in out


def test_audit_detects_field_snapshot_count_mismatch(tmp_path: Path, capsys) -> None:
    # 2 episodes in info.json, but 1 field snapshot
    ds = make_dataset_shard(tmp_path / "field-mismatch", episodes=2, field_snapshots=1)
    args = sim_dataset_audit.Args(path=ds)
    assert sim_dataset_audit.main(args) == 1
    out = capsys.readouterr().out
    assert "1 pigment fields for 2 episodes" in out


def test_audit_detects_episode_table_mismatch_and_corruption(tmp_path: Path, capsys) -> None:
    # Corrupt table
    ds = make_dataset_shard(tmp_path / "corrupt", corrupt_ep_table=True)
    args = sim_dataset_audit.Args(path=ds)
    assert sim_dataset_audit.main(args) == 1
    out = capsys.readouterr().out
    assert "episode table unreadable" in out


def test_audit_rejects_large_episode_start_action_state_lead(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "bad-start", start_lead_rad=0.2)

    assert sim_dataset_audit.main(sim_dataset_audit.Args(path=ds)) == 1

    out = capsys.readouterr().out
    assert "episode-start action/state lead reaches 0.200 rad" in out
    assert "do not train on this shard" in out


def test_audit_shards_tool_disagreement(tmp_path: Path, capsys) -> None:
    root = tmp_path / "family"
    make_dataset_shard(root / "s1", distribution="paper-draw", tool_id="lutin-ballpoint-dot")
    make_dataset_shard(root / "s2", distribution="paper-draw", tool_id="other-tool")
    args = sim_dataset_audit.Args(path=root)
    assert sim_dataset_audit.main(args) == 1
    out = capsys.readouterr().out
    assert "shards disagree on the fitted tool" in out


def test_audit_reports_skipped_batches_and_idle_episodes(tmp_path: Path, capsys) -> None:
    ds = make_dataset_shard(
        tmp_path / "skipped-idle", skipped_batches=[{"batch": 1}], engaged=False
    )
    args = sim_dataset_audit.Args(path=ds, verbose=True)
    assert sim_dataset_audit.main(args) == 0
    out = capsys.readouterr().out
    assert "SKIPPED 1 batch(es)" in out
    assert "IDLE 2 episode(s)" in out


def test_audit_fails_on_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    args = sim_dataset_audit.Args(path=empty)
    with pytest.raises(SystemExit, match="no finished datasets"):
        sim_dataset_audit.main(args)


# --- Samples Tests ---


def test_samples_spread_spacing() -> None:
    assert sim_dataset_samples._spread(5, 10) == [0, 2, 4, 7, 9]
    assert sim_dataset_samples._spread(3, 3) == [0, 1, 2]
    assert sim_dataset_samples._spread(10, 2) == [0, 1]


def test_samples_extraction_single_shard(tmp_path: Path, monkeypatch, capsys) -> None:
    ds = make_dataset_shard(tmp_path / "ds-single", episodes=3, field_snapshots=3)
    out_dir = tmp_path / "samples_out"

    # Mock ffmpeg execution to return True without calling binary
    monkeypatch.setattr(sim_dataset_samples, "_ffmpeg", lambda args: True)

    args = sim_dataset_samples.Args(path=ds, out=out_dir, samples=2)
    sim_dataset_samples.main(args)

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    manifest = json.loads(manifest_file.read_text())
    assert manifest["shards"] == 1
    assert len(manifest["samples"]) == 2

    # Check field snapshot copy behavior
    field_copy = out_dir / "ep0000_field.png"
    assert field_copy.exists()
    assert field_copy.read_bytes() == b"png_data"


def test_samples_extraction_multi_shard(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "multi-ds"
    make_dataset_shard(root / "shard-01", episodes=2)
    make_dataset_shard(root / "shard-02", episodes=2)
    out_dir = tmp_path / "samples_out_multi"

    monkeypatch.setattr(sim_dataset_samples, "_ffmpeg", lambda args: True)

    args = sim_dataset_samples.Args(path=root, out=out_dir, samples=2)
    sim_dataset_samples.main(args)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["shards"] == 2
    assert len(manifest["samples"]) == 2
    shards_sampled = {s["shard"] for s in manifest["samples"]}
    assert "shard-01" in shards_sampled and "shard-02" in shards_sampled
