"""Pin the paced guide: minimal callouts, 3-beep countdown, timeline stamps.

    uvx --with pytest --with numpy pytest -q scripts/tests/test_session_guide.py

Since 2026-08-22 the guide never reacts to detections — it paces: callout,
reposition gap, exactly three beeps (low, mid, HIGH = be still), still hold,
stamped window. What must hold: the timeline windows start at the high beep
and cover the still period, every tip hold is a planted pad hold with
continuous numbering, legacy phase names collapse onto the tip phase exactly
once, and the sweep's stdout still flows through (progress lines swallowed).
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "vision"))

from session_guide import PROFILES, all_phrases, phrase_target  # noqa: E402


def run_guide(tmp_path, phases, stdin_script=(), extra=()):
    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "scripts" / "vision" / "session_guide.py"),
         "--session", str(session), "--phases", phases, "--profile", "debug",
         "--no-audio", "--move-s", "0.05", "--still-s", "0.05",
         "--scene-change-s", "0.01",
         "--board-holds", "2", "--wrist-holds", "2",
         "--tip-holds", "3", *extra],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for delay, line in stdin_script:
        time.sleep(delay)
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            break
    out, _ = proc.communicate(timeout=30)
    timeline = json.loads((session / "guide_timeline.json").read_text())
    return timeline, out


def test_paced_phases_stamp_still_windows(tmp_path):
    timeline, out = run_guide(tmp_path, "board,wrist,tip", [
        (0.1, "progress cams=5 wrist_cams=2 kept=3"),
        (0.1, "kept    3  cams 5  tags [0, 3]  wrist:[0, 3]  3.3s"),
    ])
    entries = timeline["entries"]
    board = [e for e in entries if e["phase"] == "board"]
    wrist = [e for e in entries if e["phase"] == "wrist"]
    tip = [e for e in entries if e["phase"] == "tip"]
    assert len(board) == 2 and len(wrist) == 2 and len(tip) == 3
    # window opens at the high beep and spans the still period
    assert all(e["end_unix"] > e["start_unix"] for e in entries)
    # Every hold is planted on the pad now. The label is NOT "pad": that
    # meant a hover in archived sessions, and the fuser still reads it that
    # way, so a planted hold must never borrow the name.
    assert [e.get("label") for e in tip] == ["pad_planted"] * 3
    assert [e["index"] for e in tip] == [1, 2, 3], "continuous numbering"
    assert all(e["kind"] == "tip_hold" for e in tip)
    # minimal callouts, spoken via the shared phrase functions
    assert "[guide] board 1 of 2" in out
    assert "[guide] tip 3 of 3" in out
    # One opening line naming what is being calibrated, then counts, then the
    # close. Nothing else is spoken: the scene contracts and per-phase cues
    # were dropped on 2026-08-26 for burying the callouts that pace the work.
    assert "[guide] calibrating board, wrist, tip" in out
    assert "[guide] calibration complete" in out
    spoken = [line for line in out.splitlines() if line.startswith("[guide] ")]
    assert spoken[0] == "[guide] calibrating board, wrist, tip"
    assert spoken[-1] == "[guide] calibration complete"
    for line in spoken[1:-1]:
        assert re.fullmatch(r"\[guide\] (board|wrist|tip) \d+ of \d+", line), line
    # sweep stdout: human lines pass through, progress lines are swallowed
    assert "kept    3  cams 5" in out
    assert "progress cams" not in out


def test_legacy_phase_names_collapse_to_one_tip_run(tmp_path):
    timeline, out = run_guide(tmp_path, "touch,poses,pivot")
    tip = [e for e in timeline["entries"] if e["phase"] == "tip"]
    assert len(tip) == 3, "aliases must not repeat the phase"


def test_profile_phrases_are_prerendered():
    phrases = all_phrases()
    for profile in PROFILES.values():
        total = int(profile["tip_holds"])
        assert phrase_target("tip", 1, total) in phrases
        assert phrase_target("board", 1, int(profile["board_holds"])) in phrases
        assert phrase_target("wrist", int(profile["wrist_holds"]),
                             int(profile["wrist_holds"])) in phrases
