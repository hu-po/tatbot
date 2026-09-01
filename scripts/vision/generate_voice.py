#!/usr/bin/env python3
"""Pre-render the session guide's voice clips with fal.ai kokoro TTS.

    FAL_KEY=... python3 generate_voice.py          # or --env <file with FAL_KEY>

Enumerates every phrase session_guide.py can speak (all_phrases — one source,
no drift), synthesizes the missing ones through `fal-ai/kokoro/american-english`,
runs a light ffmpeg phaser/echo pass for the robotic character, and drops
16 kHz mono WAVs into the guide's cache under the same content-hash names
Audio.say looks up. Idempotent: existing clips are never re-billed. Run once
per phrase-set change, ideally before going on the road — offline
sessions then still have the good voice, and machines without clips fall back
to espeak or tone patterns.

The whole worklist is ~3k characters (~80 short clips): fractions of a cent
per clip on kokoro. --max-phrases is a hard cap so an accidental phrase
explosion can never turn into a real bill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_guide import all_phrases  # noqa: E402

ENDPOINT = "https://fal.run/fal-ai/kokoro/american-english"
# A deep, even voice takes the robot treatment well.
DEFAULT_VOICE = "am_onyx"
# Deep and CLEAR (operator, 2026-08-26): pitch down 3 semitones with the
# tempo preserved, one light phaser for machine character, and nothing else.
#
# THE 16000 IS LOAD-BEARING, AND WAS WRONG UNTIL 2026-08-26. `asetrate`
# reinterprets the stream at a new rate, so the pitch ratio it applies is
# (new rate / SOURCE rate) — and kokoro returns 24 kHz, not 16 kHz. Hardcoding
# 16000*0.84 against a 24 kHz source pitched down ~10 semitones instead of 3
# and stretched duration 1.786x, of which atempo undid only 1.19x. Every clip
# rendered between 2026-08-22 and 2026-08-26 was 50% slow and 3x too deep —
# "inaudible", in the operator's words. The leading aresample=16000 is what
# makes the ratio mean what it says, whatever the TTS hands back.
#
# The echo went with it: aecho smears consonants, and these clips are heard
# once, over a robot arm, by someone whose hands are busy. loudnorm targets
# -14 LUFS rather than -18 for the same reason.
PITCH_RATIO = 0.84                      # 0.84 ~= -3 semitones
ROBOT_FX = (f"aresample=16000,asetrate=16000*{PITCH_RATIO},aresample=16000,"
            f"atempo={1 / PITCH_RATIO:.4f},"
            "aphaser=in_gain=0.7:out_gain=0.8:delay=2:decay=0.3:speed=0.5,"
            "silenceremove=start_periods=1:start_threshold=-45dB,"
            "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse")
LOUDNESS = "loudnorm=I=-14:TP=-1.5"


def synthesize(text, voice, key):
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"prompt": text, "voice": voice,
                         "speed": 1.3}).encode(),
        headers={"Authorization": f"Key {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    url = payload["audio"]["url"]
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def process(raw_wav, out_path, robotify):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(raw_wav)
        tmp.flush()
        filters = (ROBOT_FX + ",") if robotify else ""
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
             "-af", filters + LOUDNESS,
             "-ar", "16000", "-ac", "1", str(out_path)],
            capture_output=True, text=True)
    if result.returncode != 0:
        # ffmpeg missing a filter is not worth losing the clip over.
        out_path.write_bytes(raw_wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.environ.get("TATBOT_VOICE_ENV", ""),
                    help="env file holding FAL_KEY (skipped if FAL_KEY is set)")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--cache", default="~/.cache/tatbot-voice")
    ap.add_argument("--no-fx", action="store_true")
    ap.add_argument("--max-phrases", type=int, default=200,
                    help="hard budget cap on clips per run")
    ap.add_argument("--force", action="store_true",
                    help="re-render every clip from the cached raw TTS; only "
                         "phrases with no raw are re-billed")
    ap.add_argument("--raw-cache", default="~/.cache/tatbot-voice-raw",
                    help="untreated TTS, kept so the effect chain can be "
                         "retuned without paying for the same words twice")
    args = ap.parse_args()

    key = os.environ.get("FAL_KEY")
    if not key:
        env_path = Path(args.env).expanduser()
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                if line.startswith("FAL_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("no FAL_KEY in the environment or the env file")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    raws = Path(args.raw_cache).expanduser()
    raws.mkdir(parents=True, exist_ok=True)
    phrases = all_phrases()
    if len(phrases) > args.max_phrases:
        sys.exit(f"{len(phrases)} phrases exceeds --max-phrases "
                 f"{args.max_phrases} — is a profile or tour runaway?")

    rendered = skipped = failed = 0
    for text in phrases:
        clip = cache / f"say_{hashlib.sha1(text.encode()).hexdigest()[:16]}.wav"
        if clip.is_file() and not args.force:
            skipped += 1
            continue
        try:
            # The billed step is the TTS, not the effect chain. Keeping the
            # raw means a --force re-render after an FX change costs nothing —
            # which matters, because the FX is the part that gets tuned by ear
            # (the 2026-08-22 chain came out inaudible and stayed that way
            # until every clip had to be re-rendered).
            raw_path = raws / f"raw_{hashlib.sha1(text.encode()).hexdigest()[:16]}.wav"
            if raw_path.is_file():
                raw = raw_path.read_bytes()
                billed = ""
            else:
                raw = synthesize(text, args.voice, key)
                raw_path.write_bytes(raw)
                billed = "  (billed)"
            process(raw, clip, robotify=not args.no_fx)
            rendered += 1
            print(f"  {clip.name}  {text}{billed}")
        except Exception as error:  # noqa: BLE001 — keep rendering the rest
            failed += 1
            print(f"  FAILED: {text} ({error})")
    print(f"\n{rendered} rendered, {skipped} cached, {failed} failed -> {cache}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
