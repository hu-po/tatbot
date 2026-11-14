"""The lab's records (docs/lab.md): a study, its trials, and the cells a plan draws them in.

Stdlib only. A study (`tatbot.lab-study/1`) fixes a question, the print its cells sit on, the constants held, the
cartridges and inks it may use, and the bounds of every setting it may vary. A trial (`tatbot.lab-trial/1`) is one
technique drawn with one settings vector in one cell; its record grows as the trial moves through proposed,
planned, drawn and read. A cells manifest (`tatbot.lab-cells/1`) is where a plan put each trial on the print, page
millimetres about the tile's centre, x right, y toward the print's top, with its strokes and where the program
dips among them: what the readers measure.
"""
from __future__ import annotations

import math
import re
from typing import Any

STUDY = "tatbot.lab-study/1"
TRIAL = "tatbot.lab-trial/1"
CELLS = "tatbot.lab-cells/1"

# Every setting a trial may carry: name -> type. What each means and where it is applied is tatbot_lab.space's.
SETTINGS: dict[str, type] = {
    "trim_mm": float,       # depth trim of the run, mm (negative is deeper)
    "speed_mm_s": float,    # pen-down speed of the program, mm/s
    "lines": int,           # strokes in the cell
    "length_mm": float,     # length of each stroke
    "pitch_mm": float,      # spacing between strokes
    "passes": int,          # times each stroke is drawn
    "angle_deg": float,     # stroke direction in the page, 0 along +x
    "deposition": float,    # whip: the lightest end's deposition, 0..1
    "mm_per_dip": float,    # line one dip draws before the next, over the cartridge datasheet's
    "dip_dwell_s": float,   # time the tube holds in the cap with the machine running, over the datasheet's
    "dip_above_ink_mm": float,   # how far over the declared ink level the tube's end holds, over the datasheet's
    "dip_drain_s": float,   # time the tube holds over the cap after the dwell, the machine running; 0 is no drain
    "dip_drain_above_rim_mm": float,   # how far over the cap's rim the tube's end drains, over the datasheet's
}
STATUSES = ("proposed", "planned", "drawn", "read", "failed", "dropped")
CONSTANTS = {"battery_v", "machine_hz", "stroke_mm"}
_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
_PATTERN = re.compile(r"stencil-[0-9a-f]{64}")
_SHA = re.compile(r"[0-9a-f]{64}")


def _keys(value: Any, required: set[str], optional: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    if missing := required - value.keys():
        raise ValueError(f"{where}: missing {sorted(missing)}")
    if extra := value.keys() - required - optional:
        raise ValueError(f"{where}: unknown {sorted(extra)}")
    return value


def _id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{where}: expected a lowercase id (letters, digits, _ . -)")
    return value


def _ids(value: Any, where: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{where}: expected a list of ids")
    for item in value:
        _id(item, where)
    if len(set(value)) != len(value):
        raise ValueError(f"{where}: repeated id")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number")
    return float(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(f"{where}: expected a bounded nonempty string")
    return value


def _setting(name: str, value: Any, where: str) -> float:
    kind = SETTINGS.get(name)
    if kind is None:
        raise ValueError(f"{where}: unknown setting {name!r}; known: {', '.join(SETTINGS)}")
    number = _number(value, f"{where}.{name}")
    if kind is int and (isinstance(value, float) or number < 1):
        raise ValueError(f"{where}.{name}: expected a whole number of at least 1")
    return number


def _numbers(value: Any, count: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{where}: expected {count} numbers")
    return [_number(v, where) for v in value]


def _stencil(stencil: Any, where: str) -> None:
    _keys(stencil, {"pattern_id", "layout", "tile_mm"}, set(), where)
    _pattern(stencil["pattern_id"], f"{where}.pattern_id")
    _id(stencil["layout"], f"{where}.layout")
    if min(_numbers(stencil["tile_mm"], 2, f"{where}.tile_mm")) <= 0:
        raise ValueError(f"{where}.tile_mm: expected [width, height] in mm")


def _pattern(value: Any, where: str) -> None:
    if not isinstance(value, str) or not _PATTERN.fullmatch(value):
        raise ValueError(f"{where}: expected a stencil-<sha256> print id")


def _palette(palette: Any, inks: list[str]) -> None:
    if not isinstance(palette, dict) or set(palette) != {*inks, "water"}:
        raise ValueError("study.palette: expected a cap for every ink and for water")
    for ink, slot in palette.items():
        _id(slot, f"study.palette.{ink}")
    if len(set(palette.values())) != len(palette):
        raise ValueError("study.palette: two liquids in one cap")


def _bounds(bounds: Any) -> None:
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("study.bounds: expected setting -> [low, high]")
    for name, span in bounds.items():
        if name not in SETTINGS:
            raise ValueError(f"study.bounds: unknown setting {name!r}; known: {', '.join(SETTINGS)}")
        low, high = _numbers(span, 2, f"study.bounds.{name}")
        if low > high:
            raise ValueError(f"study.bounds.{name}: low above high")


def study(record: Any) -> dict:
    """A study record, validated; returned unchanged."""
    _keys(record, {"schema", "id", "created", "question", "arm", "stencil", "cell_mm", "constants", "cartridges",
                   "inks", "palette", "techniques", "bounds"}, set(), "study")
    if record["schema"] != STUDY:
        raise ValueError(f"study: expected {STUDY}")
    for name in ("id", "arm"):
        _id(record[name], f"study.{name}")
    for name in ("created", "question"):
        _text(record[name], f"study.{name}")
    _stencil(record["stencil"], "study.stencil")
    if not 4.0 <= _number(record["cell_mm"], "study.cell_mm") <= 60.0:
        raise ValueError("study.cell_mm: expected 4 to 60 mm")
    constants = _keys(record["constants"], CONSTANTS, set(), "study.constants")
    if any(_number(value, f"study.constants.{name}") <= 0 for name, value in constants.items()):
        raise ValueError("study.constants: expected positive values")
    _ids(record["cartridges"], "study.cartridges")
    _ids(record["techniques"], "study.techniques")
    _palette(record["palette"], _ids(record["inks"], "study.inks"))
    _bounds(record["bounds"])
    return record


def _trial_optional(record: dict, where: str) -> None:
    for name in ("hypothesis", "note"):
        if record.get(name) is not None:
            _text(record[name], f"{where}: {name}")
    if record.get("repeat_of") is not None:
        _id(record["repeat_of"], f"{where}: repeat_of")
    if record.get("placement") is not None:
        _keys(record["placement"], {"plan", "run", "sheet", "cell"}, set(), f"{where}: placement")
    if record.get("run") is not None:   # what drew it: the program planned, then the ros-draw run that drew it
        _keys(record["run"], {"program_sha256"}, {"run_id"}, f"{where}: run")
        if not isinstance(record["run"]["program_sha256"], str) or not _SHA.fullmatch(record["run"]["program_sha256"]):
            raise ValueError(f"{where}: run.program_sha256 is a sha256")
        if record["run"].get("run_id") is not None:
            _text(record["run"]["run_id"], f"{where}: run.run_id")


def trial(record: Any, within: dict | None = None) -> dict:
    """A trial record, validated, and inside study `within`'s cartridges, inks, techniques and bounds when given."""
    _keys(record, {"schema", "study", "id", "created", "status", "technique", "cartridge", "inks", "settings"},
          {"hypothesis", "repeat_of", "placement", "run", "reads", "metrics", "note"}, "trial")
    if record["schema"] != TRIAL:
        raise ValueError(f"trial: expected {TRIAL}")
    where = f"trial {record.get('id')}"
    for name in ("study", "id", "technique"):
        _id(record[name], f"{where}: {name}")
    _text(record["created"], f"{where}: created")
    if record["status"] not in STATUSES:
        raise ValueError(f"{where}: status is one of {', '.join(STATUSES)}")
    cartridge = _keys(record["cartridge"], {"type"}, {"instance"}, f"{where}: cartridge")
    for name in ("type", "instance"):
        if name == "type" or cartridge.get(name) is not None:
            _id(cartridge[name], f"{where}: cartridge.{name}")
    _ids(record["inks"], f"{where}: inks")
    if not isinstance(record["settings"], dict):
        raise ValueError(f"{where}: settings: expected an object")
    for name, value in record["settings"].items():
        _setting(name, value, f"{where}: settings")
    _trial_optional(record, where)
    if within is not None:
        _within(record, within, where)
    return record


def _within(record: dict, s: dict, where: str) -> None:
    if record["study"] != s["id"]:
        raise ValueError(f"{where}: belongs to study {record['study']}, not {s['id']}")
    if record["technique"] not in s["techniques"]:
        raise ValueError(f"{where}: technique {record['technique']} is not one of the study's {s['techniques']}")
    if record["cartridge"]["type"] not in s["cartridges"]:
        raise ValueError(f"{where}: cartridge {record['cartridge']['type']} is not one of the study's")
    if stray := set(record["inks"]) - set(s["inks"]):
        raise ValueError(f"{where}: inks {sorted(stray)} are not the study's")
    for name, value in record["settings"].items():
        span = s["bounds"].get(name)
        if span is None:
            raise ValueError(f"{where}: the study holds {name} fixed; it gives it no bounds")
        if not span[0] <= value <= span[1]:
            raise ValueError(f"{where}: {name} {value} is outside the study's bounds {span}")


def _strokes(strokes: Any, where: str) -> None:
    if not isinstance(strokes, list):
        raise ValueError(f"{where}: strokes_mm is a list of polylines")
    for stroke in strokes:
        if not isinstance(stroke, list) or len(stroke) < 2:
            raise ValueError(f"{where}: a stroke is two or more [x, y] points")
        for point in stroke:
            _numbers(point, 2, f"{where}: a point")


def _cell(cell: Any) -> None:
    _keys(cell, {"id", "trial", "box_mm", "inks", "markers", "dips_mm", "strokes_mm"}, set(), "cells.cells[]")
    where = f"cell {_id(cell['id'], 'cell.id')}"
    _id(cell["trial"], f"{where}: trial")
    x0, y0, x1, y1 = _numbers(cell["box_mm"], 4, f"{where}: box_mm")
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"{where}: box_mm is [x0, y0, x1, y1] with x0 < x1, y0 < y1")
    _ids(cell["inks"], f"{where}: inks")
    if not isinstance(cell["markers"], list) or any(isinstance(m, bool) or not isinstance(m, int)
                                                    for m in cell["markers"]):
        raise ValueError(f"{where}: markers is a list of marker ids")
    if not isinstance(cell["dips_mm"], list) or any(_number(v, f"{where}: dips_mm") < 0 for v in cell["dips_mm"]):
        raise ValueError(f"{where}: dips_mm is the mm of the cell drawn before each dip")
    _strokes(cell["strokes_mm"], where)


def cells(record: Any) -> dict:
    """A cells manifest, validated; returned unchanged."""
    _keys(record, {"schema", "study", "plan", "run", "sheet", "pattern_id", "cells"}, set(), "cells")
    if record["schema"] != CELLS:
        raise ValueError(f"cells: expected {CELLS}")
    for name in ("study", "plan", "run", "sheet"):
        _id(record[name], f"cells.{name}")
    _pattern(record["pattern_id"], "cells.pattern_id")
    if not isinstance(record["cells"], list) or not record["cells"]:
        raise ValueError("cells.cells: expected a nonempty list")
    for cell in record["cells"]:
        _cell(cell)
    if len({cell["id"] for cell in record["cells"]}) != len(record["cells"]):
        raise ValueError("cells.cells: a cell is listed twice")
    return record
