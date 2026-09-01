"""tatbot_paths: one precedence order for every root. Stdlib only.

CLI > env > user config > repo config > XDG default, and a missing required
root raises PathConfigError naming the fix — never a silent CWD or /tmp write
(plan Phase 1 exit gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import tatbot_paths as tp  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("TATBOT_REPO", "TATBOT_CONFIG_DIR", "TATBOT_LOG_ROOT",
                "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)


def test_repo_root_is_file_relative_not_home():
    assert tp.repo_root() == REPO  # works from any clone location


def test_repo_root_env_wins_and_validates(monkeypatch, tmp_path):
    monkeypatch.setenv("TATBOT_REPO", str(tmp_path))
    with pytest.raises(tp.PathConfigError):
        tp.repo_root()  # not a checkout: no config/
    (tmp_path / "config").mkdir()
    assert tp.repo_root() == tmp_path


def test_config_dir_env_wins(monkeypatch, tmp_path):
    assert tp.config_dir() == REPO / "config"
    monkeypatch.setenv("TATBOT_CONFIG_DIR", str(tmp_path))
    assert tp.config_dir() == tmp_path
    monkeypatch.setenv("TATBOT_CONFIG_DIR", str(tmp_path / "gone"))
    with pytest.raises(tp.PathConfigError):
        tp.config_dir()


def test_log_root_precedence(monkeypatch, tmp_path):
    # env beats configured beats XDG default
    monkeypatch.setenv("TATBOT_LOG_ROOT", str(tmp_path / "env"))
    assert tp.log_root("~/rig-logs") == tmp_path / "env"
    monkeypatch.delenv("TATBOT_LOG_ROOT")
    assert tp.log_root(str(tmp_path / "cfg")) == tmp_path / "cfg"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert tp.log_root(None) == tmp_path / "xdg" / "tatbot" / "logs"


def test_log_root_default_never_tatbot_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert "tatbot-logs" not in str(tp.log_root(None))


def test_output_dir_explicit_beats_env_beats_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TATBOT_OUT", str(tmp_path / "env"))
    assert tp.output_dir(tmp_path / "cli", purpose="x", env="TATBOT_OUT") == tmp_path / "cli"
    assert tp.output_dir(None, purpose="x", env="TATBOT_OUT") == tmp_path / "env"
    monkeypatch.delenv("TATBOT_OUT")
    assert tp.output_dir(None, purpose="x", env="TATBOT_OUT",
                         default=tmp_path / "d") == tmp_path / "d"


def test_output_dir_missing_raises_with_hint():
    with pytest.raises(tp.PathConfigError, match="TATBOT_OUT"):
        tp.output_dir(None, purpose="renders", env="TATBOT_OUT")
