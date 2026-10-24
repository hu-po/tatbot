"""What the ROS stack's Rerun bridge (ros/tatbot_rerun) says of a drawing session, as plain data: one item per thing the
operator checks, each a short text and a state.

  states   ok (green: fresh, fine), warn (yellow: stale, a warning, or unknown while it is expected), fault (red: STOP,
           latched), off (grey: not applicable, disabled, or the run has ended)
  board    the Demo page's status board (draw/ros/status_board/<item>): a fixed 860 x 150 canvas (x right, y down), each
           item a bar in its state's colour with its text under it, wrapped to the bar
  card     the same items as markdown (draw/ros/status), for the other tabs

Times are the node clock's seconds, so a bag replayed on its own clock reads as it did live. Board texts carry an age
only where it says something (a stale sample, an ended run), so an unchanged state logs nothing new. Display only:
nothing here gates motion. numpy-free.
"""
from __future__ import annotations

import re

OK, WARN, FAULT, OFF = "ok", "warn", "fault", "off"
RGB = {OK: (0x2E, 0xCC, 0x71), WARN: (0xF1, 0xC4, 0x0F), FAULT: (0xE7, 0x4C, 0x3C), OFF: (0x7F, 0x8C, 0x8D)}
FRESH_S = 2.0          # a Draw feedback or SafetyState sample older than this is stale
WRIST_FRESH_S = 3.0    # the bridge takes the wrist camera at 1 Hz
MACHINE_FRESH_S = 5.0  # the MCS1 reply's age
ACTIVE_S = 60.0        # a running run with no ledger row and no feedback for this long reads idle
CANVAS = (860.0, 150.0)
INK_CELL_CHARS = 38   # two lines of the ink cell (~20 characters each); a third touches the wrist bar under it
ROWS = ((("run", 190), ("op", 150), ("phase", 150), ("trim", 150), ("ink", 220)),
        (("estop", 190), ("probe", 150), ("machine", 150), ("latch", 150), ("wrist", 220)))


def age(seconds) -> str:
    """A short age: ms under a second, whole seconds under two minutes, then minutes, then hours."""
    if seconds is None:
        return ""
    seconds = max(float(seconds), 0.0)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min" if seconds < 5400 else \
        f"{seconds / 3600:.1f} h"


def _fresh(sample: dict | None, now_s: float, limit_s: float = FRESH_S) -> bool:
    return sample is not None and now_s - sample["t"] <= limit_s


def _ended(run: dict | None, now_s: float) -> bool:
    return run is not None and run.get("ended_s") is not None and run["ended_s"] <= now_s


def short_id(run_id: str) -> str:
    """A run id's start, HH:MMZ (run ids lead with their UTC stamp); any other id as it is."""
    match = re.match(r"\d{8}T(\d\d)(\d\d)\d\dZ", run_id or "")
    return f"{match[1]}:{match[2]}Z" if match else str(run_id)


def run_item(run, feedback, now_s):
    if run is None:
        return "run: none", OFF
    name = f"run {short_id(run['id'])}"
    if _ended(run, now_s):
        how = "ended" if run.get("status") in (None, "ok") else str(run["status"])
        return f"{name} {how} {age(now_s - run['ended_s'])} ago", OFF
    quiet = None if run.get("t") is None else now_s - run["t"]
    if _fresh(feedback, now_s) or (quiet is not None and quiet <= ACTIVE_S):
        return name + (f" · {age(now_s - run['started_s'])}" if run.get("started_s") else ""), OK
    return f"{name} · idle" + (f" {age(quiet)}" if quiet is not None else ""), WARN


def op_item(run, feedback, now_s):
    if _fresh(feedback, now_s):
        return f"op {feedback['index'] + 1}/{feedback['total']}", OK
    if run is None:
        return "op -", OFF
    if run.get("index") is None or _ended(run, now_s):
        return f"done {run.get('done', 0)}/{run['total']}", OK if run.get("index") is None else OFF
    if feedback is not None:
        return f"op {feedback['index'] + 1}/{feedback['total']} · {age(now_s - feedback['t'])} ago", WARN
    return f"op {run['index'] + 1}/{run['total']} (ledger)", WARN


def phase_item(run, feedback, now_s):
    if _fresh(feedback, now_s):
        phase = feedback["phase"]
        arc = f" · arc {feedback['arc_m'] * 1000:.1f} mm" if phase == "DRAW" else ""
        return f"phase {phase}{arc}", FAULT if phase == "LATCHED" else WARN if phase == "PAUSED" else OK
    if run is None or _ended(run, now_s) or run.get("index") is None:
        return "phase -", OFF
    if feedback is not None:
        return f"phase {feedback['phase']} · {age(now_s - feedback['t'])} ago", WARN
    return "phase: between goals", WARN


def trim_item(run, trim, now_s):
    """The operator's pen trim of the cartridge on the arm (Fn+W up, Fn+S deeper) and its limit, yellow at it; yellow
    too with the trim the stroke takes when that holds the keys back (a needle cartridge's, by the skin read)."""
    if run is None or _ended(run, now_s) or trim is None:
        return "trim -", OFF
    mm, limit = trim["m"] * 1000, trim.get("limit_m")
    if trim.get("used_m", trim["m"]) * 1000 > mm + 0.005:
        return f"trim {mm:+.1f} · held {trim['used_m'] * 1000:+.2f} mm", WARN
    if limit is None:
        return f"trim {mm:+.1f} mm", OK
    if abs(mm) > limit * 1000 - 0.05:
        return f"trim {mm:+.1f} mm: its limit", WARN
    return f"trim {mm:+.1f} of ±{limit * 1000:.1f} mm", OK


def ink_item(run, ink, palette_s, now_s, caps=None):
    if run is None or _ended(run, now_s) or not (ink and ink.get("ink")):
        return caps_item(caps)
    slot = f" @ {str(ink['slot']).removeprefix('inkcap_')}" if ink.get("slot") else ""   # a fitted pen has none
    fix = f" · palette fix {age(now_s - palette_s)}" if palette_s is not None else " · no palette fix"
    return f"ink {ink['ink']}{slot}{fix}", OK if palette_s is not None else WARN


def caps_item(caps):
    """Outside an ink op: the caps loaded and available to dip (ros_scene.caps' state, brief, alias and fill), or that
    none is, in the most words that fit two lines of the ink cell (INK_CELL_CHARS); the Scene's labels say the rest."""
    if caps is None:
        return "ink -", OFF
    loaded = [cap for cap in caps if cap["state"] == "loaded"]
    if not loaded:
        return "no ink cap loaded", OFF
    texts = ["loaded " + " · ".join(cap["brief"] for cap in loaded),
             "loaded " + " · ".join(f"{cap['alias']} ≥{cap['fill']:.0%}" for cap in loaded),
             "loaded " + " ".join(cap["alias"] for cap in loaded)]
    return next((text for text in texts if len(text) <= INK_CELL_CHARS), texts[-1]), OK


def estop_item(safety, now_s):
    if safety is None:
        return "e-stop: no state", WARN
    if safety["estop_source"] == 0:
        return "e-stop none", OFF
    if not safety["estop_ok"]:
        return "e-stop STOP", FAULT
    if not _fresh(safety, now_s):
        return f"e-stop OK · {age(now_s - safety['t'])} old", WARN
    return ("e-stop OK", OK) if safety["estop_age_s"] >= 0 else ("e-stop OK · no frame", WARN)


def probe_item(safety, enabled, now_s):
    # probe_triggered is false both at rest and with no fresh PRB1 frame; SafetyState carries no probe frame age
    if not enabled:
        return "probe disabled", OFF
    if safety is None:
        return "probe: no state", WARN
    if not _fresh(safety, now_s):
        return f"probe · {age(now_s - safety['t'])} old", WARN
    return ("probe triggered", WARN) if safety["probe_triggered"] else ("probe clear", OK)


def machine_item(safety, relay, now_s):
    if not relay:
        return "machine none", OFF
    reply = None if safety is None else safety.get("machine_reply_age_s")
    if reply is None:
        return "machine: no state", WARN
    if reply < 0:
        return "machine: no reply", WARN
    # the reply's age at the SafetyState stamp, plus that sample's own age: a silent session ages it too
    total, word = reply + max(0.0, now_s - safety["t"]), "on" if safety["machine_powered"] else "off"
    if total > MACHINE_FRESH_S:
        return f"machine {word} · {age(total)} old", WARN
    return f"machine {word}", OK if safety["machine_powered"] else OFF


def latch_item(safety, now_s):
    if safety is None:
        return "latch: no state", WARN
    if safety["latched"]:
        return f"LATCHED: {safety['latch']}", FAULT
    if not _fresh(safety, now_s):
        return f"latch clear · {age(now_s - safety['t'])} old", WARN
    return "latch clear", OK


def wrist_item(wrist, feedback, now_s):
    """The wrist D405 opens only inside a Draw goal: an old frame is a warning only while one runs."""
    if wrist is None:
        return "wrist: no frame yet (opens in a draw)", OFF
    if _fresh(wrist, now_s, WRIST_FRESH_S):
        return ("wrist live (latest tf)", WARN) if wrist.get("latest") else ("wrist live", OK)
    return f"wrist {age(now_s - wrist['t'])} ago", WARN if _fresh(feedback, now_s) else OFF


def items(now_s: float, *, run: dict | None = None, feedback: dict | None = None, safety: dict | None = None,
          ink: dict | None = None, trim: dict | None = None, wrist: dict | None = None, palette_s: float | None = None,
          caps: list | None = None, probe_enabled: bool = False,
          machine_relay: bool = False) -> dict[str, tuple[str, str]]:
    """{item: (text, state)} in board order. run {id, t (newest ledger row), index (next op; None when finished), total,
    done, started_s, ended_s, status}; feedback {t, index, total, phase (a name), arc_m}; safety {t, estop_source,
    estop_ok, estop_age_s, probe_triggered, latched, latch (a name), machine_powered, machine_reply_age_s}; ink {ink,
    slot}; trim {m, limit_m (None unknown), used_m (the trim the stroke takes, m by default)}; wrist {t (the frame's
    stamp), latest (placed by the newest tf, none at its stamp)}; palette_s: the station fix's measured time; caps:
    ros_scene.caps' (state, brief), which the ink item lists outside an ink op."""
    return {"run": run_item(run, feedback, now_s), "op": op_item(run, feedback, now_s),
            "phase": phase_item(run, feedback, now_s), "trim": trim_item(run, trim, now_s),
            "ink": ink_item(run, ink, palette_s, now_s, caps), "estop": estop_item(safety, now_s),
            "probe": probe_item(safety, probe_enabled, now_s), "machine": machine_item(safety, machine_relay, now_s),
            "latch": latch_item(safety, now_s), "wrist": wrist_item(wrist, feedback, now_s)}


def card(status: dict[str, tuple[str, str]], run_id: str | None = None) -> str:
    """The items as markdown, the board's two rows as two lines, the run's full id first; a non-ok item says its state."""
    def line(row):
        return " · ".join(status[key][0] + ("" if status[key][1] == OK else f" [{status[key][1]}]") for key, _ in row)
    return f"**run** {run_id or 'none'}\n\n" + "\n\n".join(line(row) for row in ROWS) + "\n"


def box(key: str, item: tuple[str, str]) -> dict:
    """Boxes2D arguments for one board item: a bar across its cell in the state's colour, the text labelled under it
    (Rerun draws a box's label below it, in the box's colour, wrapped to its width)."""
    for row, cells in enumerate(ROWS):
        x = 0.0
        for name, width in cells:
            if name == key:
                return {"centers": [[x + width / 2, 12.0 + row * CANVAS[1] / 2]], "half_sizes": [[width / 2 - 8, 3.0]],
                        "radii": [4.0], "colors": [RGB[item[1]]], "labels": [item[0]], "show_labels": True}
            x += width
    raise KeyError(key)
