"""The motion constants shared by the C++ samples planner and the Python planner.

`config/motion_constants.json` is the source; this module is its Python reader
and `scripts/gen_motion_constants.py` renders the C++ header from the same file.
`SHA` is written into every samples file (`constants_sha`) and the executor
refuses a file whose value differs from the one it was compiled with, so a
planner and an executor built from different numbers cannot meet on the arm.

Stdlib only, so the system python3 can read it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "config" / "motion_constants.json"
SCHEMA = "tatbot.draw-constants/1"


def _canonical(data: dict) -> str:
    return json.dumps({k: v for k, v in data.items() if not k.startswith("_")},
                      sort_keys=True, separators=(",", ":"))


def sha_of(data: dict) -> str:
    return hashlib.sha256(_canonical(data).encode()).hexdigest()[:12]


def load(path: Path = PATH) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r} is not {SCHEMA}")
    return data


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items() if not k.startswith("_")})
    if isinstance(value, list):
        return tuple(value)
    return value


_DATA = load()
C = _ns(_DATA)
SHA = sha_of(_DATA)


def flat(data: dict = _DATA, prefix: str = "") -> list[tuple[str, object]]:
    """(dotted.key, value) pairs in file order, skipping schema and _doc."""
    out: list[tuple[str, object]] = []
    for k, v in data.items():
        if k.startswith("_") or k == "schema":
            continue
        if isinstance(v, dict):
            out.extend(flat(v, f"{prefix}{k}."))
        else:
            out.append((f"{prefix}{k}", v))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--sha":
        print(SHA)
    elif len(sys.argv) == 2:
        node = C
        for part in sys.argv[1].split("."):
            node = getattr(node, part)
        print(node if not isinstance(node, tuple) else ",".join(map(str, node)))
    else:
        for key, value in flat():
            print(f"{key} = {value}")
