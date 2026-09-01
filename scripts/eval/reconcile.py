#!/usr/bin/env python3
"""Reconcile robot run COUNT against the index — a PRIMITIVE.

The 2026-08-24 lesson: never judge whether a launch was uncommanded from
async notification timing. Snapshot the index before your launches, snapshot
after, and assert the delta equals what you deliberately launched. Use it
however fits — around one launch, or a whole improvised battery.

    reconcile.py snapshot            # prints current rollout-run count
    reconcile.py check <N> <before>  # exits non-zero unless delta == N
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import tatbot_runlog  # noqa: E402

IDX = tatbot_runlog.log_root() / "index.jsonl"

def count():
    n = 0
    if IDX.is_file():
        for line in IDX.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("workflow") == "rollout_async" and d.get("status") == "running":
                n += 1
    return n

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "snapshot":
        print(count())
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        want, before = int(sys.argv[2]), int(sys.argv[3])
        got = count() - before
        ok = got == want
        print(f"reconcile: launched {want}, index gained {got} — "
              f"{'OK' if ok else 'MISMATCH — STOP, investigate before more motion'}")
        sys.exit(0 if ok else 1)
    else:
        sys.exit(__doc__)
