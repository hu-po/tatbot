"""Pin the EE microphone to K6 physical input 1 without plughw downmixing."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts" / "il_audio_record.sh"
ALSA_CONFIG = REPO / "config" / "audio" / "ee-input1.asoundrc"


def _fake_arecord(tmp_path: Path) -> Path:
    binary = tmp_path / "arecord"
    binary.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -l ]; then\n"
        "  echo 'card 2: K6 [Komplete Audio 6], device 0: USB Audio [USB Audio]'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"${ALSA_CONFIG_PATH-}\"\n"
    )
    binary.chmod(0o755)
    return binary


def _live_audio_module():
    path = REPO / "scripts" / "audio" / "live_audio.py"
    spec = importlib.util.spec_from_file_location("live_audio_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_route_selects_only_native_input_1():
    text = ALSA_CONFIG.read_text()
    assert 'pcm "hw:CARD=K6,DEV=0"' in text
    assert "channels 6" in text
    assert "ttable.0.0 1" in text
    assert "ttable.1" not in text


def test_helper_autodetects_named_route_not_plughw(tmp_path):
    _fake_arecord(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env.pop("TATBOT_AUDIO_DEVICE", None)
    result = subprocess.run(
        ["bash", str(HELPER), "devices"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "tatbot_ee"


def test_helper_does_not_fall_back_to_an_unrouted_usb_card(tmp_path):
    binary = _fake_arecord(tmp_path)
    binary.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = -l ]; then\n"
        "  echo 'card 3: UMC202HD [USB Audio], device 0: USB Audio [USB Audio]'\n"
        "  exit 0\n"
        "fi\n"
    )
    binary.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env.pop("TATBOT_AUDIO_DEVICE", None)
    result = subprocess.run(
        ["bash", str(HELPER), "devices"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert result.stdout == ""


def test_helper_applies_route_config_only_to_named_pcm(tmp_path):
    _fake_arecord(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    command = (
        f"source {HELPER!s}; "
        "audio::arecord tatbot_ee; "
        "audio::arecord custom_mono"
    )
    result = subprocess.run(
        ["bash", "-c", command], env=env, capture_output=True, text=True, check=True
    )
    routed, custom = result.stdout.splitlines()
    assert routed == str(ALSA_CONFIG)
    assert custom == ""


def test_live_producer_applies_same_route_config(monkeypatch):
    live_audio = _live_audio_module()
    monkeypatch.delenv("ALSA_CONFIG_PATH", raising=False)
    assert live_audio.capture_environment("tatbot_ee")["ALSA_CONFIG_PATH"] == str(ALSA_CONFIG)
    assert "ALSA_CONFIG_PATH" not in live_audio.capture_environment("custom_mono")
