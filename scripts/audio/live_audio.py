#!/usr/bin/env python3
"""Stream the EE-mounted piezo contact microphone into a live Rerun recording.

Runs on the node with the audio interface, started by
scripts/live/cockpit.sh. It pipes mono `arecord -t raw` through numpy and logs:

  audio/ee/levels/rms_dbfs       full-band level, one row per 1024-sample hop
  audio/ee/levels/friction_dbfs  2-8 kHz band energy — the same feature
                                 scripts/il_analyze_audio.py scores contact on
  audio/ee/spectrogram           rolling dB spectrogram image, rate-limited
  audio/status                   device, format, and any capture problem

Everything is on the `capture_time` timeline in wall-clock ns, so it lines
up with visiond frames and stream-teleop ticks from other nodes. The wall
clock is stamped immediately before arecord starts (the same anchor
il_audio_record.sh writes to audio_start.json); ALSA start-up latency is
therefore a fixed offset of a few tens of ms, not drift.

Memory contract: scalars go out as
send_columns batches, never one log call per hop; the spectrogram is logged
at most --spectrogram-fps times a second and only ever the latest window.

Audio is evidence, never a gate: no capture device means a status line and
exit 0, so the cockpit keeps running without it.

  uv run --no-project --with 'rerun-sdk==0.36.0' --with numpy \
    python scripts/audio/live_audio.py --connect rerun+http://HOST:9876/proxy \
    --recording-id <id>
  # or, to check the producer without a viewer:
  ... live_audio.py --output /tmp/audio.rrd --duration 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
EE_ALSA_CONFIG = REPO / "config" / "audio" / "ee-input1.asoundrc"
EE_ALSA_PCM = "tatbot_ee"
sys.path.insert(0, str(REPO / "scripts" / "vision"))
from rerun_metadata import log_producer_metadata  # noqa: E402

# Same analysis window as il_analyze_audio.py, but hop == window: the live
# view wants a bounded row rate (48000/1024 ≈ 47 rows/s per series), not
# the offline 5 ms frame pitch.
WIN = 1024
CHANNELS = 1
FRICTION_BAND_HZ = (2000.0, 8000.0)
SPECTRO_BINS = 256          # bins 0..255 of a 1024-point rfft = 0-12 kHz @ 48 kHz
SPECTRO_DB_RANGE = (-100.0, -20.0)  # dBFS per bin mapped onto u8 0..255
BATCH_S = 0.5               # send_columns batch length


def find_device() -> str | None:
    """The ALSA device il_audio_record.sh would record from, or None."""
    if os.environ.get("TATBOT_AUDIO_DEVICE"):
        return os.environ["TATBOT_AUDIO_DEVICE"]
    helper = REPO / "scripts" / "il_audio_record.sh"
    try:
        out = subprocess.run(["bash", str(helper), "devices"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    dev = out.stdout.strip()
    return dev if out.returncode == 0 and dev else None


def capture_environment(device: str) -> dict[str, str]:
    """Return arecord's environment, enabling the explicit K6 input-1 route."""
    env = os.environ.copy()
    if device == EE_ALSA_PCM:
        env["ALSA_CONFIG_PATH"] = str(EE_ALSA_CONFIG)
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sink = ap.add_mutually_exclusive_group(required=True)
    sink.add_argument("--connect", help="rerun+http://HOST:9876/proxy of the cockpit viewer")
    sink.add_argument("--output", help="write an .rrd instead of streaming (self-test)")
    ap.add_argument("--recording-id", help="join the cockpit's shared recording")
    ap.add_argument("--device", help="ALSA capture device (default: il_audio_record.sh devices)")
    ap.add_argument("--rate", type=int, default=int(os.environ.get("TATBOT_AUDIO_RATE", 48000)))
    ap.add_argument("--spectrogram-fps", type=float, default=2.0, help="0 disables the images")
    ap.add_argument("--spectrogram-seconds", type=float, default=10.0, help="rolling window")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until stopped")
    args = ap.parse_args()

    import rerun as rr  # noqa: PLC0415  (after argparse so --help needs no SDK)

    rr.init("tatbot_vision_v2", recording_id=args.recording_id)
    if args.output:
        rr.save(args.output)
    else:
        rr.connect_grpc(args.connect)
    log_producer_metadata(rr, "live_audio", args.recording_id)

    def status(text: str) -> None:
        print(f"audio: {text}", flush=True)
        rr.log("audio/status", rr.TextLog(text))

    device = args.device or find_device()
    if not device or not shutil.which("arecord"):
        status("no capture device found (arecord -l sees nothing usable; set TATBOT_AUDIO_DEVICE) — audio pane idle")
        return 0

    fs = args.rate
    frame_bytes = WIN * CHANNELS * 2
    cmd = [
        "arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(fs),
        "-c", str(CHANNELS), "-t", "raw",
    ]
    if args.duration > 0:
        cmd += ["-d", str(int(np.ceil(args.duration)))]
    start_ns = time.time_ns()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=capture_environment(device),
        )
    except OSError as error:
        status(f"arecord failed to start on {device}: {error} — audio pane idle")
        return 0
    status(json.dumps({
        "device": device,
        "rate": fs,
        "channels": CHANNELS,
        "microphone": "ee",
        "input_route": "K6 physical input 1" if device == EE_ALSA_PCM else "explicit device override",
        "alsa_config": str(EE_ALSA_CONFIG) if device == EE_ALSA_PCM else None,
        "window": WIN,
        "started_unix_ns": start_ns,
        "friction_band_hz": FRICTION_BAND_HZ,
    }))

    window = np.hanning(WIN)
    freqs = np.fft.rfftfreq(WIN, 1.0 / fs)
    band = (freqs >= FRICTION_BAND_HZ[0]) & (freqs <= FRICTION_BAND_HZ[1])
    hops_per_batch = max(1, int(round(BATCH_S * fs / WIN)))
    spectro_cols = max(1, int(round(args.spectrogram_seconds * fs / WIN)))
    spectro = np.zeros((SPECTRO_BINS, spectro_cols), dtype=np.uint8)
    spectro_interval = 1.0 / args.spectrogram_fps if args.spectrogram_fps > 0 else None
    last_spectro = 0.0
    lo, hi = SPECTRO_DB_RANGE

    hop_index = 0
    batch_t: list[int] = []
    batch_rms: list[float] = []
    batch_fric: list[float] = []

    def flush_batch() -> None:
        if not batch_t:
            return
        times = rr.TimeColumn("capture_time", timestamp=np.array(batch_t, dtype="datetime64[ns]"))
        rr.send_columns(
            "audio/ee/levels/rms_dbfs",
            indexes=[times],
            columns=rr.Scalars.columns(scalars=np.asarray(batch_rms)),
        )
        rr.send_columns(
            "audio/ee/levels/friction_dbfs",
            indexes=[times],
            columns=rr.Scalars.columns(scalars=np.asarray(batch_fric)),
        )
        batch_t.clear()
        batch_rms.clear()
        batch_fric.clear()

    assert proc.stdout is not None
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            t_ns = start_ns + int(round((hop_index * WIN + WIN / 2) * 1e9 / fs))
            hop_index += 1
            spec = np.abs(np.fft.rfft(x * window)) ** 2
            rms_db = 20.0 * np.log10(np.sqrt((x**2).mean()) + 1e-12)
            fric_db = 10.0 * np.log10(spec[band].sum() + 1e-12)
            batch_t.append(t_ns)
            batch_rms.append(rms_db)
            batch_fric.append(fric_db)
            if len(batch_t) >= hops_per_batch:
                flush_batch()

            if spectro_interval is not None:
                col_db = 10.0 * np.log10(spec[:SPECTRO_BINS] / WIN + 1e-12)
                col_u8 = np.clip((col_db - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
                spectro[:, :-1] = spectro[:, 1:]
                spectro[:, -1] = col_u8
                now = time.monotonic()
                if now - last_spectro >= spectro_interval:
                    last_spectro = now
                    rr.set_time("capture_time", timestamp=np.datetime64(t_ns, "ns"))
                    # Low frequencies at the bottom, like every spectrogram.
                    rr.log("audio/ee/spectrogram", rr.Image(spectro[::-1, :], color_model="L"))
    except KeyboardInterrupt:
        pass
    finally:
        flush_batch()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        if err:
            status(f"arecord: {err}")
        status(f"stopped after {hop_index * WIN / fs:.1f}s ({hop_index} hops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
