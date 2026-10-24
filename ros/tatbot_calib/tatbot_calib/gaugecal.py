"""The wrist gauge's tip along the pen, on a surface at a known height (ros/README.md section 7).

The wrist gauge reads the skin's height under the tip from the wrist D405: the plane it sees, against where the tip
stands in that camera's view (calib/wrist-gauge-<arm>.json `tip_cam_m`). Its height is that point's distance from the
plane along the plane's normal, so the component of `tip_cam_m` along the normal is what sets every page's height.
The fit that made it touched paper with another cartridge (2026-10-07); the fitted one's touches trip in the air.

This measures that component on the probe body's white top, which stands BODY_TOP_BELOW_BALL_M under the ball's
centre on the probe's own axis, so the palette's tilt hardly moves it there. The ball's height is the palette
camera's (`calib height`), read through the tcp the arm commands, so the top's height is in the frame every dip and
stroke is commanded in, and an error in the tcp's length cancels. The arm holds the tool upright OFFSET_M off the
probe's axis at HEIGHTS_M over the top, toward the first side the way checks let it; at each hold the gauge reads
FRAMES depth frames (gauge.measure: the top square to the pen; none from a side where the pen hides it behind the wall). Each hold's known height less its reading is the
correction along the normal; their standard error says how well it is known. --adopt moves the installed tip by their mean
along the normal and records the fitted tool as measured_with, when that mean is known within MAX_SE_M.

Every hold is a probe-guarded move through Sweep.move: the tool's envelope off the station's parts and the ball,
the arm's bodies off the station. The run writes holds.jsonl, frames/, candidate.json and report.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tatbot_calib import halo, program, station
from tatbot_calib.tool import ToolRefusedError, fitted_tool

HEIGHTS_M = (0.011, 0.013, 0.015, 0.017)   # the tcp over the probe body's top: the cartridge's body keeps the 10 mm the
                                           # arm's bodies keep from the station (program.palette_check)
OFFSET_M = 0.014           # the tcp off the probe's axis: over the body's white top (17.5 mm in radius), the cartridge
                           # 10 mm off the probe's collar, as the arm's bodies keep from the station
NEAR_M = 0.006             # the gauge reads the top within this of the pen's axis: 8-20 mm off the probe's
YAWS_TRIED = 2             # the tool's yaws tried at each side: past the best two the stereo depth by the pen scattered
                           # 2-4 mm on the small top (2026-10-13)
SIDES_DEG = (225, 315, 45, 0, 180, 135, 270, 90)
# A hold is tried only where the wrist camera stands this far round the probe's axis from the hold's side, looking at
# the top under the tip across the probe's middle. Nearer the hold's side it looks past the pen, which hides the top
# under it, and the D405 reads the far, LED-lit part of the metal disc 2-7 mm deep however tight its frames: over six
# runs (2026-10-22) holds 18-63 deg round read that or nothing, 103-107 deg round (grazing the pen) 0-2.2 mm low, and
# 154-156 deg round -0.63 +- 0.13 mm in every rig run.
VIEW_ROUND_DEG = 120.0
FRAMES = 15
SETTLE_S = 1.0
FLOOR_OVER_TOP_M = 0.002   # the tool's lowest point stays this far over the probe body's top
MIN_HOLDS = 3
HOLDS_WANTED = 8
MAX_SE_M = 0.00025         # the corrections' mean is known to this (standard error) to adopt: one hold's reading
                           # scatters +-0.5 mm (2026-10-13, 15 frames each), so the mean of six is known to ~0.17
# A correction past this says the known heights are wrong, not the gauge: under the wrist tags' registration the
# station's height reads ~9 mm off, and a setup kept on it adopted +8.75 +- 0.09 mm (2026-10-22, restored at once)
MAX_CORRECTION_M = 0.003
FRAME_SPREAD_M = 0.0012    # a hold whose frames' middle half spreads past this is not counted: the top square to the
                           # pen reads its frames 0.5-1.2 mm apart, those holds' medians within 0.15 mm of each other


def known_height(tcp: np.ndarray, base_from_palette: np.ndarray, ball: np.ndarray) -> float:
    """The tcp's height over the probe body's top, along the palette's up."""
    up = base_from_palette[:3, 2]
    top = np.asarray(ball, float) - station.BODY_TOP_BELOW_BALL_M * up
    return float((np.asarray(tcp, float) - top) @ up)


def hold_poses(ball: np.ndarray, base_from_palette: np.ndarray, side_deg: float, yaw: float) -> list[np.ndarray]:
    up = base_from_palette[:3, 2]
    top = np.asarray(ball, float) - station.BODY_TOP_BELOW_BALL_M * up
    side = math.radians(side_deg)
    out = []
    for h in HEIGHTS_M:
        point = top + h * up + OFFSET_M * (math.cos(side) * base_from_palette[:3, 0]
                                           + math.sin(side) * base_from_palette[:3, 1])
        out.append(halo.tool_pose(point, yaw))
    return out


def view_round_deg(pose: np.ndarray, tcp_from_camera: np.ndarray, top: np.ndarray, up: np.ndarray) -> float:
    """How far round the probe's axis (through `top`, along `up`) the wrist camera stands from the hold's side."""
    camera = (pose @ tcp_from_camera)[:3, 3]
    a, b = camera - top, pose[:3, 3] - top
    a, b = a - (a @ up) * up, b - (b @ up) * up
    return math.degrees(math.acos(float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def correction(rows: list[dict]) -> tuple[float, float, np.ndarray]:
    """(the mean correction along the normal, its standard error, the mean normal in the camera)."""
    deltas = np.array([row["known_m"] - row["read_m"] for row in rows])
    normal = np.mean([row["normal_cam"] for row in rows], axis=0)
    return (float(deltas.mean()), float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
            normal / np.linalg.norm(normal))


def corrected(cal: dict, delta: float, normal: np.ndarray, tool: str, run: str, se: float, holds: int) -> dict:
    """The installed gauge with its tip moved `delta` along the plane's normal (toward the camera when positive)."""
    tip = np.asarray(cal["tip_cam_m"], float) + delta * normal
    out = dict(cal)
    out.update(tip_cam_m=tip.tolist(), offset_m=0.0, measured_with=tool, tool=tool, run=run, holds=holds,
               correction_m=delta, correction_se_m=se, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               source="known heights over the probe body's top (tatbot ros calib gauge); the pen's axis and cone "
                      f"end kept from {cal.get('run') or cal.get('runs') or 'the installed fit'}")
    out.pop("note", None)
    if "cone_end_cam_m" in cal and "axis_cam" in cal:
        out["tip_length_m"] = float((tip - np.asarray(cal["cone_end_cam_m"], float)) @ np.asarray(cal["axis_cam"]))
    return out


class DiscGauge:
    def __init__(self, cal: program.Calibration, sweep, touched: station.StationFix, camera, gauge_cal: dict):
        self.cal, self.sweep, self.touched, self.camera, self.gauge_cal = cal, sweep, touched, camera, gauge_cal
        self.rows: list[dict] = []

    def read(self, label: str, q) -> dict | None:
        """FRAMES depth frames through gauge.measure: the median height and normal, or None."""
        import cv2
        from tatbot_session import gauge

        colors, _ = self.camera.grab_both(FRAMES)
        out = self.cal.run.dir / "frames"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"{label}.png"), colors[0])
        np.savez_compressed(out / f"{label}.npz", depth_raw=self.camera.raw_depth[0], q=np.asarray(q, float),
                            meta=json.dumps(self.camera.depth_metadata))
        frames = [gauge.measure(d, c, self.camera.depth_metadata, self.gauge_cal, near_m=NEAR_M)
                  for d, c in zip(self.camera.raw_depth, colors, strict=True)]
        frames = [f for f in frames if f is not None]
        if len(frames) < FRAMES // 2:
            return None
        heights = np.sort([gauge.height(f, self.gauge_cal) for f in frames])
        cut = len(heights) // 4
        middle = heights[cut:len(heights) - cut]
        return {"read_m": float(np.median(heights)), "frames": len(frames),
                "frame_spread_m": float(middle.max() - middle.min()),
                "normal_cam": np.median([f.n for f in frames], axis=0).tolist()}

    def run(self) -> list[dict]:
        from tatbot_session import inspect as ins

        program._preflight(self.cal)
        ball, palette = self.cal.ball, self.touched.base_from_palette
        # the holds stand under the calibration's floor (10 mm under the ball's top): this run's floor is the top's
        self.cal.floor_z = float((ball - station.BODY_TOP_BELOW_BALL_M * palette[:3, 2])[2]) + FLOOR_OVER_TOP_M
        yaws = self.sweep.yaws()[:YAWS_TRIED]
        q = self.cal.rig.joints()
        tcp_from_camera = np.linalg.inv(self.cal.kin.fk(q)) @ self.cal.kin.frame(q, ins.camera_frame(self.cal.arm))
        up = palette[:3, 2]
        top = ball - station.BODY_TOP_BELOW_BALL_M * up
        for side, yaw in ((side, yaw) for side in SIDES_DEG for yaw in yaws):
            poses = hold_poses(ball, palette, side, math.radians(yaw))
            round_deg = view_round_deg(poses[0], tcp_from_camera, top, up)
            if round_deg < VIEW_ROUND_DEG:
                self.cal.say(f"side{side:03d}-yaw{yaw:+.0f}: skipped (the camera would stand {round_deg:.0f} deg round the "
                             "probe from the hold, looking past the pen)")
                continue
            for h, pose in reversed(list(zip(HEIGHTS_M, poses, strict=True))):   # down from the highest
                label = f"side{side:03d}-yaw{yaw:+.0f}-h{h * 1000:.0f}"
                try:
                    self.cal.approach(pose[:3, 3])
                    self.sweep.move(label, pose)
                except RuntimeError as error:
                    if program._latched(self.cal):
                        raise
                    self.cal.say(f"{label}: skipped ({error})")
                    break
                self.cal.rig.spin(SETTLE_S)
                q = self.cal.rig.joints()
                tcp = self.cal.kin.fk(q)[:3, 3]
                read = self.read(label, q)
                if read is None:
                    self.cal.say(f"{label}: the gauge did not read the probe's top")
                    continue
                row = {"label": label, "q": np.asarray(q).tolist(), "known_m": known_height(tcp, palette, ball), **read}
                self.cal.log("holds.jsonl", row)
                self.cal.say(f"{label}: {row['known_m'] * 1000:.2f} mm over the top by the arm, "
                             f"{read['read_m'] * 1000:.2f} by the gauge ({read['frames']} frames, middle half "
                             f"{read['frame_spread_m'] * 1000:.2f} mm)")
                if read["frame_spread_m"] <= FRAME_SPREAD_M:
                    self.rows.append(row)
            if len(self.rows) >= HOLDS_WANTED:
                break
        return self.rows


def run(args) -> int:
    from tatbot_description import repo_root, robot_description
    from tatbot_motion import Kinematics, load_motion
    from tatbot_session import inspect as ins

    from tatbot_calib import sweep

    repo = repo_root(None)
    try:
        tool, halo_tip = fitted_tool(repo, args.arm, args.tool)
    except ToolRefusedError as error:
        print(json.dumps({"ok": False, "message": str(error)}))
        return 3
    sys.path.insert(0, str(repo / "scripts" / "lib"))
    import tatbot_runlog

    fix = station.StationFix.from_dict(json.loads(Path(args.station).expanduser().read_text()))
    station.require_fresh(fix, args.run_started, max_age_s=args.max_age_s)
    kin = Kinematics(robot_description(None, arms=(args.arm,)), args.arm)
    run_log = tatbot_runlog.init(program.WORKFLOW, meta={"arm": args.arm, "tool": tool.tool_id, "method": "gauge",
                                                         "station": args.station},
                                 attach_logging=False, argv=["tatbot_calib", "gaugecal", "--arm", args.arm])
    cal = program.Calibration(None, run_log, repo, args.arm, halo_tip, 0.0, kin)
    sweep.equip(cal, fix, load_motion())
    touched = program.touched_fix(cal, fix, tool.tool_id)
    gauge_path = Path(f"~/tatbot-ros/calib/wrist-gauge-{args.arm}.json").expanduser()
    touch = station.load_touch(args.arm)
    if touched is fix or not touch.height_source or not gauge_path.is_file():
        cal.say("the gauge needs the station touched for this registration and tool, its ball's height by the "
                "palette camera (tatbot ros calib run --station-only, then tatbot ros calib height) and a wrist gauge "
                "file whose pen model it reads with")
        run_log.finalize(1, status="fail")
        print(json.dumps({"ok": False, "run_dir": str(run_log.dir)}))
        return 1
    program.place_station(cal, touched)
    gauge_cal = json.loads(gauge_path.read_text())
    camera = ins.WristCamera(serial=args.serial, depth=True)
    cal.rig, code = program.Rig(args.arm), 1
    try:
        disc = DiscGauge(cal, sweep.Sweep(cal, touched, tool.tool_id, args.station), touched, camera, gauge_cal)
        rows = disc.run()
        code = finish(cal, rows, gauge_cal, gauge_path, tool.tool_id, args.adopt)
    except RuntimeError as error:
        cal.say(f"stopped: {error}")
    finally:
        camera.close()
        if not program._latched(cal):
            program.retreat(cal.rig, kin)
        cal.rig.close()
        run_log.finalize(code, status="ok" if code == 0 else "fail")
    print(json.dumps({"ok": code == 0, "run_dir": str(run_log.dir)}), flush=True)
    return code


def finish(cal, rows: list[dict], gauge_cal: dict, gauge_path: Path, tool: str, adopt: bool) -> int:
    """The correction, its report and candidate; the gauge adopted when asked and the holds agree. 0, or 3."""
    if len(rows) < MIN_HOLDS:
        cal.say(f"only {len(rows)} holds read the probe's top, {MIN_HOLDS} needed: nothing measured")
        return 3
    delta, se, normal = correction(rows)
    run = cal.run.dir.name
    record = corrected(gauge_cal, delta, normal, tool, run, se, len(rows))
    lines = [f"- holds: {len(rows)}; the gauge read the probe's top {delta * 1000:+.2f} +- {se * 1000:.2f} mm off the "
             "arm's height (" + ", ".join(f"{(r['known_m'] - r['read_m']) * 1000:+.2f}" for r in rows) + ")",
             f"- the installed gauge ({gauge_cal.get('measured_with') or 'measured with an unrecorded tool'}, "
             f"{gauge_cal.get('note') or gauge_cal.get('source', '')}) moves {delta * 1000:+.2f} mm along the normal"]
    (cal.run.dir / "candidate.json").write_text(json.dumps({"rows": rows, "gauge": record}, indent=1) + "\n")
    (cal.run.dir / "report.md").write_text("# Wrist gauge on the probe's top\n\n" + "\n".join(lines) + "\n")
    for line in lines:
        cal.say(line)
    if se > MAX_SE_M:
        cal.say(f"not adopted: the correction is known to {se * 1000:.2f} mm, over {MAX_SE_M * 1000:.2f}")
        return 3
    if abs(delta) > MAX_CORRECTION_M:
        cal.say(f"not adopted: {delta * 1000:+.2f} mm is past {MAX_CORRECTION_M * 1000:.0f}; the station's height or "
                "the registration it was measured under is off, not the gauge")
        return 3
    if adopt:
        gauge_path.rename(gauge_path.with_name(gauge_path.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ",
                                                                               time.gmtime())))
        gauge_path.write_text(json.dumps(record, indent=1) + "\n")
        cal.say(f"adopted: {gauge_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib gaugecal", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--station", required=True)
    parser.add_argument("--run-started", type=float, required=True)
    parser.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    parser.add_argument("--serial", default="")
    parser.add_argument("--adopt", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
