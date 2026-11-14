"""`tatbot ros cartridge status|fit` and `tatbot ros calib apply`'s tip, on the ROS owner: the arm's fitted cartridge
and each cartridge's measured tip, in calib/cartridge-<arm>.json beside the deployed repo (cartridge_fit), which deploy
never touches. Moves nothing; the CLI restarts a running stack afterwards so it plans with them.

    status --arm xarm
    fit lutin-5rl-mast --arm xarm --instance "box 2"
    adopt --arm xarm --run 20261110T161721Z-node-4625 --across-only --derive-gauges

Prints one JSON line, `ok` false with the exit code and why when it refuses.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cartridge_fit
import gauge_derive
import probe_calibration_adopt
import tool_spec
import tool_urdf

CALIB_RUNS = Path("~/tatbot-logs/ros-calib")   # where `calib run` and `calib sweep` leave candidate.json


class RefusedError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record(repo: Path, arm: str) -> dict:
    """The arm's record; its first is seeded from the deployed section."""
    workspace = tool_spec.read_workspace(repo)
    if arm not in workspace:
        raise RefusedError(2, f"config/workspace.yaml has no {arm} section")
    return cartridge_fit.load(repo, arm) or cartridge_fit.seeded(workspace[arm], arm)


def install(repo: Path, arm: str, new: dict) -> None:
    """Write the record once the robot description can carry it: tool_urdf refuses a tool with no mount on the arm
    and a measured tip far from its datasheet's or leaning off the bore, which would stop the stack from starting."""
    workspace = tool_spec.read_workspace(repo)
    try:
        tool_urdf.tip(repo, arm, {**workspace, arm: cartridge_fit.fitted_section(workspace[arm], new)})
    except (ValueError, tool_spec.ToolMountError) as error:
        raise RefusedError(3, str(error)) from None
    cartridge_fit.write(repo, new)


def _measured_with(calib: Path, arm: str, fitted: dict) -> dict:
    """What the station touch and each wrist gauge were measured with, and whether they predate the cartridge."""
    since = fitted.get("fitted_utc") or ""

    def read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    touch = read(calib / f"station-touch-{arm}.json")
    gauges = {}
    for path in sorted(calib.glob(f"wrist-gauge-{arm}*.json")):
        gauge = read(path)
        role = path.stem.removeprefix(f"wrist-gauge-{arm}").removeprefix("-")
        gauges[role or "session"] = {
            "measured_with": gauge.get("measured_with"), "utc": gauge.get("utc"),
            "derived": (gauge.get("derived") or [{}])[-1].get("run"),
            "current": gauge.get("measured_with") == fitted["tool_id"] and str(gauge.get("utc") or "") >= since}
    return {"station_touch": {"tool_id": touch.get("tool_id"), "measured_utc": touch.get("measured_utc"),
                              "height_source": touch.get("height_source") or None,
                              "before_fit": bool(touch) and str(touch.get("measured_utc") or "") < since},
            "gauges": gauges}


def status(repo: Path, arm: str) -> dict:
    """The fitted cartridge, its tip and how it stands, what was measured with it, and what to measure next."""
    rec = record(repo, arm)
    tip = rec["tips"][rec["tool_id"]]
    state = cartridge_fit.tip_state(rec)
    section = tool_spec.read_workspace(repo)[arm]
    measured = _measured_with(cartridge_fit.calib_dir(repo), arm, rec)
    todo = []
    if state != "measured":
        todo.append(f"tatbot ros calib sweep --arm {arm}, then tatbot ros calib apply --arm {arm} --run <its run> "
                    "(--across-only for a fresh cartridge of the type: its needles bottom where the last one's did)")
    if state != "measured" or measured["station_touch"]["before_fit"]:
        todo.append(f"tatbot ros calib run --arm {arm} --station-only, then tatbot ros calib height --arm {arm}")
    for role, gauge in measured["gauges"].items():
        if not gauge["current"]:
            camera = "" if role == "session" else f" --camera {role}"
            todo.append(f"tatbot ros calib wristcam --arm {arm}{camera} --adopt, then tatbot ros calib gauge "
                        f"--arm {arm}{camera}")
    return {"ok": True, "arm": arm, "tool_id": rec["tool_id"], "instance": rec.get("instance"),
            "fitted_utc": rec.get("fitted_utc"), "record": str(cartridge_fit.path(repo, arm))
            if cartridge_fit.load(repo, arm) else None,
            "tip": {"state": state, "utc": tip.get("utc"), "source": tip.get("source"),
                    "pen_tip_offset_m": [section[f"pen_tip_offset_{axis}"] for axis in "xyz"],
                    "touchoff": section.get("touchoff")},
            **measured, "next": todo}


def fit(repo: Path, arm: str, tool: str, instance: str | None) -> dict:
    """Record `tool` as the cartridge now in the arm's mount."""
    known = tool_spec.list_tools(repo)
    if tool not in known:
        raise RefusedError(2, f"{tool} is not a datasheet in config/tools/ (known: {', '.join(known)})")
    spec = tool_spec.load_tool(tool, repo)
    install(repo, arm, cartridge_fit.fit(record(repo, arm), tool, spec.touchoff_nominal_m, instance=instance or None,
                                         utc=_utc()))
    return status(repo, arm)


def follow_gauges(calib: Path, arm: str, tool_id: str, before: list[float], before_utc: str | None,
                  now: list[float], run: str, *, write: bool) -> tuple[dict, bool]:
    """Each wrist gauge of the arm against the tip `run` adopted (gauge_derive): how its tip moves in its camera and
    how far along its plane's normal, or why it cannot follow. `write` installs the derived gauges, each one it
    replaces kept beside it as .replaced-<utc>; without it they are only reported. Also whether any was written."""
    out, wrote = {}, False
    for path in sorted(calib.glob(f"wrist-gauge-{arm}*.json")):
        role = path.stem.removeprefix(f"wrist-gauge-{arm}").removeprefix("-") or "session"
        try:
            gauge = json.loads(path.read_text())
            then = gauge_derive.tip_before(gauge, tool_id, before, before_utc)
            derived = gauge_derive.derive(gauge, then, now, gauge_derive.hand_eye(gauge, CALIB_RUNS.expanduser()),
                                          tool_id=tool_id, run=run, utc=_utc())
        except (OSError, ValueError, gauge_derive.NotDerivedError) as error:
            out[role] = {"derived": False, "why": str(error)}
            continue
        step = derived["derived"][-1]
        out[role] = {"derived": write, "tip_moved_cam_mm": step["tip_moved_cam_mm"],
                     "along_normal_mm": gauge_derive.along_normal_mm(gauge, derived)}
        if write and any(step["tip_moved_mount_mm"]):
            wrote = True
            path.rename(path.with_name(path.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ", time.gmtime())))
            path.write_text(json.dumps(derived, indent=1) + "\n")
    return out, wrote


def adopt(repo: Path, arm: str, run: str, across_only: bool, derive_gauges: bool = False) -> dict:
    """A calib run's or sweep's tip as the fitted cartridge's measured one. Each wrist gauge's tip follows it when
    `derive_gauges`; without it the gauges stay as they are and what they would become is reported. A run whose tip
    is the installed one already writes nothing: `changed` says whether the record or a gauge was."""
    try:
        candidate = json.loads((CALIB_RUNS.expanduser() / run / "candidate.json").read_text())
    except (OSError, ValueError) as error:
        raise RefusedError(2, f"run {run}: no candidate ({error})") from None
    if candidate.get("arm") != arm:
        raise RefusedError(2, f"run {run} calibrated the {candidate.get('arm')} arm, not {arm}")
    rec, section = record(repo, arm), tool_spec.read_workspace(repo)[arm]
    calibration = tool_spec.load_tool(rec["tool_id"], repo).raw.get("calibration") or {}
    tip, why = probe_calibration_adopt.apply(candidate, section, calibration, across_only=across_only)
    if why:
        raise RefusedError(3, "not adopted: " + "; ".join(why))
    installed = rec["tips"][rec["tool_id"]]
    wrote = installed["source"] != candidate["run_id"] or installed["section"] != tip
    if wrote:
        install(repo, arm, cartridge_fit.adopt(rec, tip, run_id=candidate["run_id"], utc=candidate["utc"]))
    before, now = ([float(s[f"pen_tip_offset_{axis}"]) for axis in "xyz"] for s in (section, tip))
    followed, gauges_wrote = follow_gauges(cartridge_fit.calib_dir(repo), arm, rec["tool_id"], before,
                                           installed.get("utc"), now, candidate["run_id"], write=derive_gauges)
    moved = [round((b - a) * 1000, 3) for a, b in zip(before, now, strict=True)]
    return {**status(repo, arm), "adopted": run, "moved_mm": moved, "gauges_followed": followed,
            "changed": wrote or gauges_wrote}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("status", "fit", "adopt"))
    parser.add_argument("tool", nargs="?", default="")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--instance", default="")
    parser.add_argument("--run", default="")
    parser.add_argument("--across-only", dest="across_only", action="store_true")
    parser.add_argument("--derive-gauges", dest="derive_gauges", action="store_true")
    ns = parser.parse_args(argv)
    repo = Path(os.environ["TATBOT_REPO"])
    try:
        if ns.action == "fit" and not ns.tool:
            raise RefusedError(2, "fit needs the cartridge's datasheet id")
        if ns.action == "adopt" and not ns.run:
            raise RefusedError(2, "adopt needs --run <ros-calib run id>")
        result = {"status": lambda: status(repo, ns.arm), "fit": lambda: fit(repo, ns.arm, ns.tool, ns.instance),
                  "adopt": lambda: adopt(repo, ns.arm, ns.run, ns.across_only, ns.derive_gauges)}[ns.action]()
    except RefusedError as error:
        print(json.dumps({"ok": False, "code": error.code, "message": str(error)}))
        return 0
    except ValueError as error:   # a record or datasheet that does not read: said, never fallen back from
        print(json.dumps({"ok": False, "code": 1, "message": str(error)}))
        return 0
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
