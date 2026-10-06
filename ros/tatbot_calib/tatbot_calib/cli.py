"""`station fix` measures the palette in an arm's frame using the shared roof-tag producer.
Retain capture/fix evidence, optionally compare an earlier fix, and report measurement/transport failures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tatbot_bridge import capture, stack
from tatbot_bridge.station import DARK_GRAY, UnmeasuredError, _stretched, _TagView, observe  # noqa: F401

WORKFLOW = "ros-station"


def _stack(repo: Path) -> dict:
    return stack.load(repo=repo)


def _earlier(value: str, runs: Path):
    """An earlier fix: a station.json path, or a ros-station run id."""
    from tatbot_calib.station import StationFix

    path = Path(value).expanduser()
    if not path.is_file():
        path = runs / value / "station.json"
    return StationFix.from_dict(json.loads(path.read_text()))


def measure(repo: Path, arm: str, shots: int, run_dir: Path):
    return observe(repo, arm, shots, run_dir, (_stack(repo).get('registration') or {}).get(arm))


def fix(args) -> int:
    from tatbot_description import repo_root

    from tatbot_calib import station

    repo = repo_root(None)
    capture._lib(repo)
    import tatbot_runlog

    run = tatbot_runlog.init(WORKFLOW, meta={"arm": args.arm, "source": "overhead_tag"}, attach_logging=False,
                             argv=["tatbot_calib", "station", "fix", "--arm", args.arm])
    out, code = {"ok": False, "run_dir": str(run.dir)}, 0
    try:
        found = measure(repo, args.arm, args.shots, run.dir)
        out.update(ok=True, station=found.as_dict())
        if args.against:
            moved = station.station_moved(_earlier(args.against, run.dir.parent), found)
            out.update(against=moved, ok=not moved["moved"])
            code = 1 if moved["moved"] else 0
    except (UnmeasuredError, ValueError) as error:
        out["message"], code = str(error), 3
    except RuntimeError as error:   # the D555's owner did not answer (register.Camera)
        out["message"], code = str(error), 5
    run.update(outcome="ok" if code == 0 else "fail")
    run.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps(out))
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib station")
    sub = parser.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("fix", help="the station in an arm's frame, measured now by the overhead D555")
    p.add_argument("--arm", required=True)
    p.add_argument("--shots", type=int, default=8, help="D555 captures averaged into the fix")
    p.add_argument("--against", default="", help="an earlier station.json or ros-station run id to compare with")
    args = parser.parse_args(argv)
    return fix(args)


if __name__ == "__main__":
    sys.exit(main())
