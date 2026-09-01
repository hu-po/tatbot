#!/usr/bin/env python3
"""Voice-guided calibration session: the machine paces, the operator drives.

    ssh <camera-node> 'sweep_session.py ...' | session_guide.py --session DIR \
        --phases board,wrist,tip [--profile debug|full] [--no-audio]

One mechanic for every phase (operator feedback 2026-08-22): a minimal
callout — "board 1 of 5" / "wrist 4 of 6" / "tip 2 of 9" — a gap to
reposition, then EXACTLY three beeps rising low -> mid -> high. The high beep
means BE STILL NOW; hold until the next callout. Capture is automatic while
still (the sweep's still-repeat keeps and the flight log's still intervals),
so the guide never waits on detections and never retries — the report card
judges yield afterwards.

The guide speaks three kinds of line and nothing else (operator, 2026-08-26):
"calibrating board, wrist, tip" once at the top, the numbered callouts, and
"calibration complete". Scene contracts and per-phase cues used to be read
aloud too; they were noise to anyone who runs this routinely, and they buried
the callouts that actually pace the work. Scene changes keep their silent
pause — see docs/vision.md for what to cover and uncover.

Phases:
  board  hold the calibration board somewhere new, tilted, each callout
  wrist  hold a new wrist orientation each callout
  tip    press the pen tip on ONE spot of the paper pad, a DIFFERENT wrist
         orientation each callout. Every hold is a pen-tip observation and
         the solve averages them; the pad is the surface the pen draws on, so
         the solved point doubles as the paper plane. Replaced the separate
         pose tour, and (2026-08-26) the pad-hover + palette-hold split

Everything is stamped into `guide_timeline.json`; fuse_session.py joins it
with the flight log and detections. The operator's voice is for setup notes
only ("scratch that" still drops the last sample).

Speech clips are pre-rendered by generate_voice.py (deep robotic voice);
without clips the guide falls back to espeak-ng or tone patterns — it must
work on the road.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

PROGRESS = re.compile(r"^progress cams=(\d+) wrist_cams=(\d+) kept=(\d+)")
KEPT = re.compile(r"^kept\s+(\d+)")

PROFILES = {
    # Distinct orientations are the information; the sweep's still-repeat
    # timer supplies per-pose sighting volume. still_s ~= 3 s guarantees a
    # still interval plus at least one timer keep per hold.
    # Counts raised 2026-08-22 ("so we really get a good read"): board and
    # wrist accuracy scale with distinct orientations, and extra palette
    # holds directly widen the tip solve's rotation spread. ~9.3 s per hold:
    # debug ~3.7 min, full ~5.7 min.
    # Operator-set counts 2026-08-22: 9 board / 9 wrist / 6 pad / 3 palette
    # (~4.2 min at the paced ~9.3 s per hold).
    # Operator-set 2026-08-26, replacing the pad-hover + palette-hold split
    # with 9 planted holds on ONE point of the paper pad.
    #
    # Two runs that day showed why. The laser pen's contact is a 15.4 mm blunt
    # lens face, not a point, so tilting migrates the contact across the face:
    # rms got WORSE as spread got better (3.32 mm at 26.7 deg, then 4.05 mm at
    # 71.9 deg). That residual is the tool's geometry, not the operator's aim,
    # and 3 samples cannot average it out. Nine holds can — the solve is an
    # average, and the operator's judgement is that a good-enough tip from nine
    # poses beats a perfect one that never passes.
    #
    # The pad, not the palette tag, because that is the surface the pen draws
    # on: planting there makes paper_plane_z the actual paper (n_pad > 0), so
    # tool_spec.derive_z_floor_m stops refusing on "no pad touches".
    "debug": {"board_holds": 9, "wrist_holds": 9, "tip_holds": 9,
              "move_s": 4.0, "still_s": 3.0, "scene_change_s": 12.0},
    "full": {"board_holds": 14, "wrist_holds": 12, "tip_holds": 9,
             "move_s": 4.0, "still_s": 3.0, "scene_change_s": 12.0},
}

DEFAULT_TOUR = {
    "paper_pad_over": "above the paper pad",
    "palette_center": "on the palette tag",
}


# --- every line the guide can speak ----------------------------------------
# generate_voice.py enumerates these to pre-render clips, and Audio.say
# hashes the exact text to find them — one source, no drift.

def phrase_target(name, index, count):
    # no comma: TTS renders punctuation as a pause, and pace matters here
    return f"{name} {index} of {count}"


# Canonical phases, and the pre-2026-08-22 names that all collapse onto tip.
CANONICAL_PHASES = ("board", "wrist", "tip")
PHASE_ALIASES = {"touch": "tip", "pivot": "tip", "poses": "tip"}


def resolve_phases(spec):
    """Canonical, de-duplicated phase order from a --phases string.

    Aliases collapse, so "touch,poses,pivot" is one tip run rather than three.
    """
    resolved = []
    for raw in spec.split(","):
        name = raw.strip()
        name = PHASE_ALIASES.get(name, name)
        if name in CANONICAL_PHASES and name not in resolved:
            resolved.append(name)
    return resolved


def phrase_opening(phases):
    # Commas here on purpose, unlike phrase_target: this line is read once,
    # and the TTS pause between phase names makes the list easier to parse.
    return "calibrating " + ", ".join(phases)


PHRASE_DONE = "calibration complete"


def all_phrases(*_ignored):
    """Every text the guide can emit, across all profiles.

    Cut back hard on 2026-08-26 (operator): the scene contracts and per-phase
    cues were "pointless and irritating" to someone who runs this routinely,
    and reading them aloud every session buried the two things that actually
    pace the work — which hold you are on, and when it is over. What survives
    is one opening line naming the phases, the numbered callouts, and the
    close. The scene contract still gets its silent pause; it is written down
    in docs/vision.md rather than recited.

    Index runs to `total`, not to the largest count across profiles: the guide
    can only ever say "board 5 of 9" for a profile with 9 board holds, and
    rendering "board 14 of 9" was paying for clips nothing can speak.
    """
    phrases = {PHRASE_DONE}
    for size in range(1, len(CANONICAL_PHASES) + 1):
        for combo in itertools.permutations(CANONICAL_PHASES, size):
            phrases.add(phrase_opening(list(combo)))
    for profile in PROFILES.values():
        for name, key in (("board", "board_holds"), ("wrist", "wrist_holds"),
                          ("tip", "tip_holds")):
            total = int(profile[key])
            for index in range(1, total + 1):
                phrases.add(phrase_target(name, index, total))
    return sorted(phrases)


def load_tour(tour_file):
    """`tour:` from config/poses.yaml — slug: "spoken label" pairs, in order.
    The guide no longer walks a tour (the tip phase carries the waypoints),
    but the pipeline still filters published poses by this list."""
    path = Path(tour_file).expanduser()
    if not path.is_file():
        return dict(DEFAULT_TOUR)
    tour = {}
    section = None
    for line in path.read_text().splitlines():
        raw = line.split("#")[0].rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent == 0:
            section = raw.split(":")[0].strip()
        elif section == "tour" and indent == 2:
            slug, _, label = raw.strip().partition(":")
            tour[slug.strip()] = label.strip().strip('"') or slug.strip()
    return tour or dict(DEFAULT_TOUR)


class Audio:
    """Tones synthesized locally; speech from pre-rendered clips, with
    espeak-ng or a tone pattern as fallback."""

    TONES = {"low": (660.0, 0.09), "mid": (990.0, 0.09),
             "high": (1320.0, 0.35), "skip": (440.0, 0.25)}

    def __init__(self, enabled, cache_dir):
        self.enabled = enabled
        self.cache = Path(cache_dir).expanduser()
        self.espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if enabled:
            self.cache.mkdir(parents=True, exist_ok=True)
            for name, (freq, seconds) in self.TONES.items():
                self._tone_file(name, freq, seconds)

    def _tone_file(self, name, freq, seconds, rate=16000):
        path = self.cache / f"tone_{name}.wav"
        if path.is_file():
            return path
        n = int(rate * seconds)
        with wave.open(str(path), "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(b"".join(
                struct.pack("<h", int(
                    9000 * math.sin(2 * math.pi * freq * i / rate)
                    * min(1.0, 8 * i / n) * min(1.0, 8 * (n - i) / n)))
                for i in range(n)))
        return path

    def _play(self, path, block=True):
        if not self.enabled:
            return
        try:
            proc = subprocess.Popen(["aplay", "-q", str(path)],
                                    stderr=subprocess.DEVNULL)
            if block:
                proc.wait()
        except FileNotFoundError:
            self.enabled = False

    def tone(self, name, block=False):
        if self.enabled:
            self._play(self.cache / f"tone_{name}.wav", block=block)

    def hold_tone(self, seconds):
        """The sustained high tone: still for as long as it sounds. Playback
        provides the timing; without audio, sleep stands in."""
        if not self.enabled:
            time.sleep(seconds)
            return
        path = self.cache / f"tone_hold_{seconds:.1f}.wav"
        if not path.is_file():
            rate = 16000
            n = int(rate * seconds)
            fade = int(rate * 0.08)
            with wave.open(str(path), "w") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(rate)
                f.writeframes(b"".join(
                    struct.pack("<h", int(
                        6000 * math.sin(2 * math.pi * 1320.0 * i / rate)
                        * min(1.0, i / fade, (n - i) / fade)))
                    for i in range(n)))
        self._play(path, block=True)

    def say(self, text):
        print(f"[guide] {text}", flush=True)
        if not self.enabled:
            return
        clip = self.cache / f"say_{hashlib.sha1(text.encode()).hexdigest()[:16]}.wav"
        if not clip.is_file() and self.espeak:
            subprocess.run([self.espeak, "-s", "150", "-w", str(clip), text],
                           check=False, stderr=subprocess.DEVNULL)
        if clip.is_file():
            self._play(clip, block=True)
            return
        for _ in range(2):
            self.tone("mid", block=True)


class Feed:
    """Echoes the sweep's stdout (swallowing machine-readable progress lines)
    so the operator's terminal stays readable. The paced guide does not react
    to detections, but the counters remain parsed for the audit trail."""

    def __init__(self):
        self.events = queue.Queue()
        self.kept = 0
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        for line in sys.stdin:
            line = line.rstrip("\n")
            match = PROGRESS.match(line)
            if match:
                self.kept = max(self.kept, int(match.group(3)))
                continue
            print(line, flush=True)
            match = KEPT.match(line)
            if match:
                self.kept = max(self.kept, int(match.group(1)))


class Timeline:
    def __init__(self, session, meta):
        self.path = Path(session).expanduser() / "guide_timeline.json"
        self.meta = meta
        self.entries = []

    def add(self, **entry):
        self.entries.append(entry)
        self.path.write_text(json.dumps(
            {**self.meta, "entries": self.entries}, indent=2))


class Guide:
    def __init__(self, args, profile):
        self.args = args
        self.profile = profile
        self.audio = Audio(not args.no_audio, args.audio_cache)
        self.feed = Feed()
        self.timeline = Timeline(args.session, {
            "profile": args.profile, "phases": args.phases.split(","),
            "targets": profile, "started_unix": time.time()})

    def paced_hold(self, phase, index, total, kind="hold", label=None):
        """Callout, reposition gap, then exactly three beeps low -> mid ->
        HIGH — and the high tone SUSTAINS for the whole stillness window:
        be still for as long as it sounds."""
        self.audio.say(phrase_target(phase, index, total))
        time.sleep(self.profile["move_s"])       # reposition window
        self.audio.tone("low", block=True)
        time.sleep(0.4)
        self.audio.tone("mid", block=True)
        time.sleep(0.4)
        start = time.time()
        self.audio.hold_tone(self.profile["still_s"])   # blocking = the hold
        entry = {"phase": phase, "kind": kind, "index": index,
                 "start_unix": start, "end_unix": time.time(),
                 "result": "paced"}
        if label is not None:
            entry["label"] = label
        self.timeline.add(**entry)

    def prepare_scene(self):
        """Silent gap for the operator to enforce the ID-reuse scene contract.

        Spoken until 2026-08-26; the contract lives in docs/vision.md now, and
        an operator who runs this weekly does not need it recited. The PAUSE is
        still load-bearing — board, wrist and palette reuse tag IDs, so the
        wrong thing left uncovered corrupts the phase, not just the hold."""
        time.sleep(self.profile["scene_change_s"])

    def phase_board(self):
        self.prepare_scene()
        count = int(self.profile["board_holds"])
        for index in range(1, count + 1):
            self.paced_hold("board", index, count)

    def phase_wrist(self):
        self.prepare_scene()
        count = int(self.profile["wrist_holds"])
        for index in range(1, count + 1):
            self.paced_hold("wrist", index, count)

    def phase_tip(self):
        """Holds with the tip planted on ONE point of the paper pad, a new
        wrist orientation each time: every hold is a pen-tip observation.

        Superseded the pad-hover + palette-hold split on 2026-08-26. The
        hovers only ever placed a waypoint, and the palette tag was a
        different surface from the one the pen draws on — so both halves are
        now the same nine planted holds on the pad. `label` stays on each
        entry because the fuser still reads the old labels out of archived
        sessions, where "pad" meant a hover rather than a touch."""
        self.prepare_scene()
        count = int(self.profile["tip_holds"])
        for index in range(1, count + 1):
            self.paced_hold("tip", index, count, kind="tip_hold",
                            label="pad_planted")

    def run(self):
        runners = {"board": self.phase_board, "wrist": self.phase_wrist,
                   "tip": self.phase_tip}
        phases = resolve_phases(self.args.phases)
        if not phases:
            return
        # One line, once: what this session is calibrating. Said BEFORE the
        # first scene pause, so the operator hears it while setting up rather
        # than after they have already committed to a scene.
        self.audio.say(phrase_opening(phases))
        for phase in phases:
            runners[phase]()
        self.audio.say(PHRASE_DONE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--phases", default="board,wrist,tip")
    ap.add_argument("--profile", default="debug", choices=sorted(PROFILES))
    ap.add_argument("--tour-file", default=str(
        Path(__file__).resolve().parents[2] / "config" / "poses.yaml"))
    ap.add_argument("--audio-cache", default="~/.cache/tatbot-voice")
    ap.add_argument("--no-audio", action="store_true")
    for key in PROFILES["debug"]:
        flag = "--" + key.replace("_", "-")
        ap.add_argument(flag, type=float, default=None)
    args = ap.parse_args()

    profile = dict(PROFILES[args.profile])
    for key in list(profile):
        override = getattr(args, key)
        if override is not None:
            profile[key] = int(override) if key.endswith("holds") else override
    guide = Guide(args, profile)
    try:
        guide.run()
    except KeyboardInterrupt:
        print("\n[guide] aborted — captured entries stand", flush=True)
        guide.timeline.add(phase="session", kind="abort",
                           start_unix=time.time(), end_unix=time.time(),
                           result="aborted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
