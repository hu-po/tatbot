#!/usr/bin/env python3
"""Transcribe a session's narration into notes (plus one control word).

Runs after the session, using faster-whisper locally — no network and no
remote services, because this has to work on the road.

The operator's voice is the session's lab notebook — setup facts only they
know (pad position, filled inks, lighting). Since the guide went fully paced
(2026-08-22), speech steers nothing: the only consumed control is 'scratch
that' (discard = drop the last sample); the 'touch' words survive solely for
labeling legacy pre-tip-phase sessions.

  ~/.venvs/tatbot-vision/bin/python transcribe_session.py <session_dir> \
      [--model small] [--device auto]

Reads `audio.wav` + `audio_start.json`, writes `transcript.jsonl` (segments
with absolute unix times) and `events.jsonl` (notes, with `kinds` marking the
discard control).
"""

import argparse
import json
import re
from pathlib import Path

# Matched on word boundaries against lowercased text, so "discarding"
# matches but "cardigan" does not. 'discard' is the one live control;
# 'touch' only labels legacy sessions.
CONTROL_WORDS = [
    ("discard", r"\b(discard|scratch that|ignore that|bad one|throw that out|skip that)\b"),
    ("touch", r"\b(touch|touching|plate|contact|pad|paper)\b"),
]


def parse_events(segments):
    events = []
    for segment in segments:
        text = segment["text"].strip()
        lowered = text.lower()
        kinds = [kind for kind, pattern in CONTROL_WORDS if re.search(pattern, lowered)]
        events.append({
            "start_unix": segment["start_unix"],
            "end_unix": segment["end_unix"],
            "kinds": kinds,
            "text": text,
        })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    session = Path(args.session_dir).expanduser()
    audio = session / "audio.wav"
    if not audio.is_file():
        raise SystemExit(f"no audio.wav in {session}")
    start_file = session / "audio_start.json"
    audio_start = json.loads(start_file.read_text())["unix_seconds"] if start_file.is_file() else 0.0
    if not audio_start:
        print("WARNING: no audio_start.json — times will be relative to the recording")

    from faster_whisper import WhisperModel
    device = args.device
    compute = "float16" if device == "cuda" else "int8"
    if device == "auto":
        # CPU int8 is quick enough for a few minutes of speech and avoids
        # fighting the GPU for memory while a viewer is running.
        device, compute = "cpu", "int8"
    print(f"transcribing {audio.name} with faster-whisper '{args.model}' on {device}...")
    model = WhisperModel(args.model, device=device, compute_type=compute)
    segments, info = model.transcribe(str(audio), language=args.language,
                                      vad_filter=True, beam_size=5)

    rows = []
    for segment in segments:
        rows.append({
            "start_unix": audio_start + segment.start,
            "end_unix": audio_start + segment.end,
            "start_offset_s": round(segment.start, 2),
            "end_offset_s": round(segment.end, 2),
            "text": segment.text.strip(),
        })
        print(f"  [{segment.start:6.1f}-{segment.end:6.1f}s] {segment.text.strip()}")

    with open(session / "transcript.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    events = parse_events(rows)
    with open(session / "events.jsonl", "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    tally = {}
    for event in events:
        for kind in event["kinds"]:
            tally[kind] = tally.get(kind, 0) + 1
    print(f"\n{len(rows)} segments over {info.duration:.1f}s of audio")
    print(f"control words: {tally or 'none recognised'}")
    print(f"wrote {session / 'transcript.jsonl'} and {session / 'events.jsonl'}")


if __name__ == "__main__":
    main()
