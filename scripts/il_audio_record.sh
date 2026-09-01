#!/usr/bin/env bash
# EE-mounted piezo contact-mic capture for rollouts and bench checks.
#
# The pen is audibly different hovering vs drawing (motor load + tip-on-paper
# friction), and the microphone hears that through the end-effector structure
# with none of the FK/plane-calibration drift that makes geometric "contact" a
# proxy. The hardware contract is mono: the EE microphone feeds input 1 only.
#
# SOURCE this file for the audio:: functions (il_rollout_async.sh does), or
# run it standalone on a bench:
#   scripts/il_audio_record.sh devices              # what would be captured
#   scripts/il_audio_record.sh probe [seconds]      # EE-mic wiring/gain check: RMS/peak/clip
#   scripts/il_audio_record.sh record <seconds> [out.wav]
#
# env: TATBOT_AUDIO_DEVICE   explicitly routed mono ALSA capture PCM (default:
#                            tatbot_ee when the Komplete Audio 6 is present)
#      TATBOT_AUDIO_ALSA_CONFIG
#                            ALSA config defining tatbot_ee (default: the
#                            checked-in input-1 route)
#      TATBOT_AUDIO_RATE     default 48000
#
# Every audio:: function is non-fatal (no device => warn and return 0): audio
# is evidence, never a gate — a rollout must run identically with the
# interface unplugged. Offline scoring: scripts/il_analyze_audio.py.

TATBOT_AUDIO_RATE="${TATBOT_AUDIO_RATE:-48000}"
TATBOT_AUDIO_ALSA_CONFIG="${TATBOT_AUDIO_ALSA_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config/audio/ee-input1.asoundrc}"
TATBOT_EE_AUDIO_PCM="tatbot_ee"

# Echo the ALSA device to capture from, or return 1 (callers guard).
audio::device() {
  if [ -n "${TATBOT_AUDIO_DEVICE:-}" ]; then
    echo "$TATBOT_AUDIO_DEVICE"
    return 0
  fi
  command -v arecord >/dev/null 2>&1 || return 1
  # `arecord -l` lines look like: card 1: K6 [Komplete Audio 6], device 0: ...
  # The checked-in route names that card and selects native capture channel 0
  # (physical input 1). Do not fall back to plughw: its mono conversion averages
  # all six K6 capture channels and attenuates the microphone by 15.6 dB.
  local cards
  cards=$(arecord -l 2>/dev/null | sed -n 's/^card [0-9]*: \([^ ]*\) \[.*/\1/p' | sort -u) || true
  echo "$cards" | grep -qx 'K6' || return 1
  echo "$TATBOT_EE_AUDIO_PCM"
}

# audio::arecord <device> <arecord args...> — apply the checked-in ALSA route
# only to the automatic Tatbot PCM. An explicit TATBOT_AUDIO_DEVICE is an expert
# override and must itself name a mono PCM that already selects the intended input.
audio::arecord() {
  local dev="$1"
  shift
  if [ "$dev" = "$TATBOT_EE_AUDIO_PCM" ]; then
    env ALSA_CONFIG_PATH="$TATBOT_AUDIO_ALSA_CONFIG" arecord -D "$dev" "$@"
  else
    arecord -D "$dev" "$@"
  fi
}

# audio::start <out_dir> <max_seconds> — background arecord into
# <out_dir>/audio.wav, stamping <out_dir>/audio_start.json with the wall clock
# immediately before the spawn (same-host time.time() is the alignment anchor
# against the flight CSV's t_wall; precedent: calib_sweep.sh narration).
# Exports AUDIO_PID and AUDIO_DEV. The -d cap is the backstop that bounds the
# recorder even if the calling shell is SIGKILLed — there is deliberately no
# trap here (the rollout script's signal path belongs to the arm landing).
audio::start() {
  local out_dir="$1" max_s="$2" dev
  dev=$(audio::device) || {
    echo "audio: no capture device found (set TATBOT_AUDIO_DEVICE) — recording skipped" >&2
    return 0
  }
  python3 -c "import json,time; json.dump({'unix_seconds': time.time()}, open('$out_dir/audio_start.json','w'))"
  audio::arecord "$dev" -q -f S16_LE -r "$TATBOT_AUDIO_RATE" -c 1 \
    -d "$max_s" "$out_dir/audio.wav" &
  AUDIO_PID=$!
  AUDIO_DEV="$dev"
  AUDIO_WAV="$out_dir/audio.wav"
  export AUDIO_PID AUDIO_DEV AUDIO_WAV
  echo "audio: recording $dev -> $out_dir/audio.wav (max ${max_s}s)"
}

# audio::stop — end the recording; bounded wait so a teardown never hangs.
audio::stop() {
  local pid="${AUDIO_PID:-}" i=0
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do
    sleep 0.1; i=$((i + 1))
  done
  unset AUDIO_PID
  # arecord killed before its -d cap leaves the WAV header claiming the full
  # cap (a 136 s session read as 7200 s, 2026-08-31); rewrite the RIFF and
  # data chunk sizes from the true file size. Non-fatal like everything here.
  if [ -n "${AUDIO_WAV:-}" ] && [ -f "$AUDIO_WAV" ]; then
    python3 - "$AUDIO_WAV" <<'PY' || true
import os, struct, sys
path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "r+b") as f:
    riff = f.read(12)
    if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
        sys.exit(0)
    pos = 12
    while True:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            sys.exit(0)
        chunk_id, chunk_len = header[:4], struct.unpack("<I", header[4:])[0]
        if chunk_id == b"data":
            actual = size - (pos + 8)
            if actual >= 0 and chunk_len != actual:
                f.seek(pos + 4)
                f.write(struct.pack("<I", actual))
                f.seek(4)
                f.write(struct.pack("<I", size - 8))
            break
        pos += 8 + chunk_len + (chunk_len & 1)
PY
  fi
  unset AUDIO_WAV
  return 0
}

# audio::levels <wav> — EE-mic RMS/peak dBFS and clip %, stdlib only
# (wave + array; audioop is gone in python 3.13).
audio::levels() {
  python3 - "$1" <<'PY'
import array, math, sys, wave
with wave.open(sys.argv[1], "rb") as w:
    ch, fs, n = w.getnchannels(), w.getframerate(), w.getnframes()
    if ch != 1:
        sys.exit(f"{sys.argv[1]}: expected mono EE audio, got {ch} channels")
    if w.getsampwidth() != 2:
        sys.exit(f"{sys.argv[1]}: expected 16-bit PCM, got {8*w.getsampwidth()}-bit")
    samples = array.array("h", w.readframes(n))
print(f"{sys.argv[1]}: {ch} ch, {fs} Hz, {n/fs:.1f}s")
x = samples
if not x:
    sys.exit(f"{sys.argv[1]}: empty")
peak = max(abs(v) for v in x) / 32768.0
rms = math.sqrt(sum(v * v for v in x) / len(x)) / 32768.0
clip = 100.0 * sum(1 for v in x if abs(v) >= 32767) / len(x)
def db(v):
    return 20 * math.log10(v) if v > 0 else float("-inf")
print(f"  ee mic: rms {db(rms):6.1f} dBFS   peak {db(peak):6.1f} dBFS   clipped {clip:.2f}%")
PY
}

# --- CLI (only when executed, not sourced) ---------------------------------
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  case "${1:-}" in
    devices)
      if dev=$(audio::device); then
        echo "$dev"
      else
        echo "no capture device found (arecord -l sees nothing usable)" >&2
        exit 1
      fi
      ;;
    probe)
      secs="${2:-5}"
      tmp=$(mktemp --suffix=.wav)
      trap 'rm -f "$tmp"' EXIT
      dev=$(audio::device) || { echo "no capture device found (arecord -l)" >&2; exit 1; }
      echo "probing $dev for ${secs}s — tap the EE microphone (want a clear response, 0% clip)"
      audio::arecord "$dev" -q -f S16_LE -r "$TATBOT_AUDIO_RATE" -c 1 \
        -d "$secs" "$tmp"
      audio::levels "$tmp"
      ;;
    record)
      [ -n "${2:-}" ] || { echo "usage: $0 record <seconds> [out.wav]" >&2; exit 2; }
      out="${3:-bench.wav}"
      dir=$(cd "$(dirname "$out")" && pwd)
      base=$(basename "$out")
      dev=$(audio::device) || { echo "no capture device" >&2; exit 1; }
      python3 -c "import json,time; json.dump({'unix_seconds': time.time()}, open('$dir/${base%.wav}_start.json','w'))"
      echo "recording $dev -> $dir/$base for ${2}s"
      audio::arecord "$dev" -q -f S16_LE -r "$TATBOT_AUDIO_RATE" -c 1 \
        -d "$2" "$dir/$base"
      audio::levels "$dir/$base"
      ;;
    *)
      sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
      exit 2
      ;;
  esac
fi
