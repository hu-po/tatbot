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
            def read_parquet(f, columns=None):
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
def _mock_read_parquet(f, columns=None):
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
) -> Path:
    """Helper to build a valid or invalid dataset shard tree."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": episodes, "total_frames": episodes * frames_per_ep})
    )
    run_meta = {
        "config": {"distribution": distribution},
        # a fresh shard carries the follower that made it (fixed-mount-v2 since
        # 2026-08-30); one without this is a gripper-held-v1 shard to the audit
        "tool": {"tool_id": tool_id, "substrate": "paper", "embodiment": "fixed-mount-v2"},
        "episodes": [{"engaged": engaged} for _ in range(episodes)],
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
