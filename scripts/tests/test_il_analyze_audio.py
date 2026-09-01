"""Pin the audio contact analyzer's promises: detection, alignment, honesty.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_il_analyze_audio.py

The promise that matters most is the refusal: a recording with no believable
acoustic split (all-hover, dead mic) must report threshold.valid=false and
invent NO contact segments — Otsu always returns a cut, so the separation
gate is the only thing standing between "quiet" and "hallucinated drawing".
Second is alignment: segment times must land at audio_start.unix_seconds +
offset, because everything downstream joins on the flight CSV's t_wall.
No URDF/FK here — the geometric side of cross_check is fed synthetic masks.
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import il_analyze_audio as aa  # noqa: E402

FS = 48000
HOVER_NOISE = 10 ** (-50 / 20)   # broadband floor while airborne
CONTACT_NOISE = 10 ** (-20 / 20)  # tip-on-paper friction (white ⇒ fills 2-8 kHz)


def synth_wav(path: Path, duration_s: float, contact_spans, seed=0, channels=1):
    """EE-mic signal: 120 Hz motor tone + floor noise; contact adds friction."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    x = 0.05 * np.sin(2 * np.pi * 120.0 * t)
    x += HOVER_NOISE * rng.standard_normal(n)
    for a, b in contact_spans:
        sel = (t >= a) & (t < b)
        x[sel] += CONTACT_NOISE * rng.standard_normal(int(sel.sum()))
    if channels != 1:
        x = np.repeat(x[:, None], channels, axis=1)
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return path


def test_known_draw_spans_are_recovered(tmp_path):
    spans = [(2.0, 5.0), (7.0, 9.0)]
    synth_wav(tmp_path / "audio.wav", 12.0, spans)
    a = aa.analyze(tmp_path / "audio.wav", settle=0.0)
    assert a["audio"]["channels"] == 1
    assert a["threshold"]["valid"]
    assert a["metrics"]["n_contact_segments"] == len(spans)
    for (want_a, want_b), got in zip(spans, a["segments"], strict=True):
        assert got["t_start_s"] == pytest.approx(want_a, abs=0.15)
        assert got["t_end_s"] == pytest.approx(want_b, abs=0.15)
    # 5 s of contact in a 12 s window
    assert a["metrics"]["contact_pct_audio"] == pytest.approx(100 * 5.0 / 12.0, abs=5.0)


def test_segment_times_are_absolute_when_start_stamp_exists(tmp_path):
    t0 = 1_766_000_000.25
    synth_wav(tmp_path / "audio.wav", 8.0, [(3.0, 6.0)])
    (tmp_path / "audio_start.json").write_text(json.dumps({"unix_seconds": t0}))
    a = aa.analyze(tmp_path / "audio.wav", settle=0.0)
    seg = a["segments"][0]
    assert a["audio"]["start_unix"] == t0
    assert seg["t_start_unix"] == pytest.approx(t0 + seg["t_start_s"], abs=1e-3)
    assert seg["t_end_unix"] == pytest.approx(t0 + seg["t_end_s"], abs=1e-3)


def test_all_hover_recording_refuses_to_invent_contact(tmp_path):
    synth_wav(tmp_path / "audio.wav", 10.0, [])
    a = aa.analyze(tmp_path / "audio.wav", settle=0.0)
    assert not a["threshold"]["valid"]
    assert a["metrics"]["n_contact_segments"] == 0
    assert a["metrics"]["contact_pct_audio"] == 0.0
    assert any(c["name"] == "separation" and c["status"] == "warn" for c in a["checks"])


def test_non_mono_recording_is_rejected(tmp_path):
    synth_wav(tmp_path / "audio.wav", 1.0, [], channels=2)
    with pytest.raises(SystemExit, match="expected mono EE audio, got 2 channels"):
        aa.analyze(tmp_path / "audio.wav", settle=0.0)


def test_label_mode_scores_the_bench_recording(tmp_path):
    synth_wav(tmp_path / "bench.wav", 20.0, [(10.0, 20.0)])
    a = aa.analyze(tmp_path / "bench.wav", label_spec="hover:0-10,draw:10-20")
    lb = a["labels"]
    assert lb["accuracy_pct"] >= 90.0
    assert lb["d_prime"] is None or lb["d_prime"] > 2.0
    assert not lb["labels"]["hover"]["contact"]
    assert lb["labels"]["draw"]["contact"]


def test_cross_check_attributes_disagreement_to_a_side():
    t = np.arange(0.0, 10.0, 0.01)
    audio = (t >= 2.0) & (t < 6.0)
    geo = (t >= 2.0) & (t < 8.0)          # FK claims 2 s the mics never heard
    xc = aa.cross_check(audio, geo, t)
    assert xc["available"]
    assert xc["agreement_pct"] == pytest.approx(80.0, abs=1.0)
    assert xc["geo_only_pct"] == pytest.approx(20.0, abs=1.0)
    assert xc["audio_only_pct"] == pytest.approx(0.0, abs=0.5)
    who = {w["who"] for w in xc["disagreement_windows"]}
    assert who == {"geo_only"}


def test_micro_dropouts_merge_into_one_stroke():
    # 40 ms gaps inside a stroke are the pen skipping over grid ruling, not two
    # strokes; 50 ms blips are transients, not drawing.
    t = np.arange(0.0, 4.0, 0.005)
    mask = ((t >= 1.0) & (t < 1.5)) | ((t >= 1.54) & (t < 2.0)) | ((t >= 3.0) & (t < 3.05))
    segs = aa.find_segments(mask, t)
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(1.0, abs=0.02)
    assert segs[0][1] == pytest.approx(2.0, abs=0.02)
