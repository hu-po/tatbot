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
correction along the normal; their standard error says how well it is known. --adopt moves the installed tip so its
height changes by their mean, and records the fitted tool as measured_with, and that error with it: the best estimate
the holds give, however few a camera's side allows (2026-11-09: the xArm's D405-1 read the top from 3 holds,
+-0.27 mm). A mean past MAX_CORRECTION_M is not the gauge's, so nothing is adopted.

The tip moves along its own ray from the camera, keeping its pixel: the error the holds measure is between the tip and
the depth surface along that ray (a depth bias, or the hand-eye's range). Moved along the normal instead, the height came
out right but the tip stood correction x tan(view) sideways of the markers the holds carry onto the surface: 1.9 mm
toward the camera's view at the xArm's 47.5 deg, the ink's way on every lab cell (2026-11-10). `--from-run` re-derives an
adopted gauge from its own run's holds, with no arm move.

A gauge whose tip stands at the depth its own depth frames show the pen at (`depth_seen_m`, calib wristcam's since
2026-11-11) is not moved: the gauge reads the skin by that depth stream, and set to the arm's heights here instead
its tip stood 2.2 mm nearer the camera than the stream showed the pen, the needles about 1.5 mm deep. The run then
reports how far the arm's heights over the probe's top stand from the gauge's, and adopts nothing.

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
# A correction past this says the known heights are wrong, not the gauge: under the wrist tags' registration the
# station's height reads ~9 mm off, and a setup kept on it adopted +8.75 +- 0.09 mm (2026-10-22, restored at once)
MAX_CORRECTION_M = 0.003
FRAME_SPREAD_M = 0.0012    # a hold whose frames' middle half spreads past this is not counted: the top square to the
                           # pen reads its frames 0.5-1.2 mm apart, those holds' medians within 0.15 mm of each other
# The probe's white top must be lit: at dusk (2026-10-25, the frames' median V 13 of 255, the top V 45-79 and tinted by
# the LED ring) 0 of 8 holds read it, where the same holds read 4 of 4 on a morning (V 63); without the colour, the
# depth alone read 0.4-2.1 mm high there (the D405's stereo needs the light too)
DARK_V = 30


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


MIN_RAY_ON_NORMAL = 0.3   # the tip's ray this far onto the normal at least (72.5 deg off it): grazing, a small
                          # height error would move the tip far along the ray


def corrected(cal: dict, delta: float, normal: np.ndarray, tool: str, run: str, se: float, holds: int,
              tip_mount=None) -> dict:
    """The installed gauge with its tip moved along its ray from the camera, keeping its pixel, so its height along the
    plane's normal changes by `delta` (toward the camera when positive). `tip_mount`: the installed tip in the tool
    mount the holds were commanded with, recorded as the tip the gauge stands for (scripts/lib/gauge_derive.py)."""
    tip = np.asarray(cal["tip_cam_m"], float)
    ray = tip / np.linalg.norm(tip)
    on_normal = float(ray @ np.asarray(normal, float))
    if abs(on_normal) < MIN_RAY_ON_NORMAL:
        raise ValueError(f"the tip's ray lies {math.degrees(math.acos(min(1.0, abs(on_normal)))):.0f} deg off the "
                         "normal: too grazing to move it along")
    tip = tip + (delta / on_normal) * ray
    out = dict(cal)
    out.update(tip_cam_m=tip.tolist(), offset_m=0.0, measured_with=tool, tool=tool, run=run, holds=holds,
               correction_m=delta, correction_se_m=se, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               source="known heights over the probe body's top (tatbot ros calib gauge); the pen's axis and cone "
                      f"end kept from {cal.get('run') or cal.get('runs') or 'the installed fit'}")
    out.pop("note", None)
    out.pop("derived", None)   # measured now: a tip derived from the cartridge's (gauge_derive) is replaced by the holds'
    if tip_mount is not None:
        out["tip_mount_m"] = np.asarray(tip_mount, float).tolist()
    if "cone_end_cam_m" in cal and "axis_cam" in cal:
        out["tip_length_m"] = float((tip - np.asarray(cal["cone_end_cam_m"], float)) @ np.asarray(cal["axis_cam"]))
    return out


class DiscGauge:
    def __init__(self, cal: program.Calibration, sweep, touched: station.StationFix, camera, gauge_cal: dict,
                 frame: str = ""):
        self.cal, self.sweep, self.touched, self.camera, self.gauge_cal = cal, sweep, touched, camera, gauge_cal
        self.frame = frame   # the camera's colour optical frame; "" for the session camera's
        self.rows: list[dict] = []
        self.dark_v = 255.0   # the darkest unread hold's median brightness (V of 255)

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
            self.dark_v = min(self.dark_v, float(np.median(cv2.cvtColor(colors[0], cv2.COLOR_BGR2HSV)[..., 2])))
            return None
        heights = np.sort([gauge.height(f, self.gauge_cal) for f in frames])
        cut = len(heights) // 4
        middle = heights[cut:len(heights) - cut]
        return {"read_m": float(np.median(heights)), "frames": len(frames),
                "frame_spread_m": float(middle.max() - middle.min()),
                "normal_cam": np.median([f.n for f in frames], axis=0).tolist()}

    def run(self) -> list[dict]:
        from tatbot_session import inspect as ins

        frame = self.frame or ins.camera_frame(self.cal.arm)
        program._preflight(self.cal)
        ball, palette = self.cal.ball, self.touched.base_from_palette
        # the holds stand under the calibration's floor (10 mm under the ball's top): this run's floor is the top's
        self.cal.floor_z = float((ball - station.BODY_TOP_BELOW_BALL_M * palette[:3, 2])[2]) + FLOOR_OVER_TOP_M
        yaws = self.sweep.yaws()[:YAWS_TRIED]
        q = self.cal.rig.joints()
        tcp_from_camera = np.linalg.inv(self.cal.kin.fk(q)) @ self.cal.kin.frame(q, frame)
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
    serial, frame, gauge_path = program.wrist_camera(repo, args.arm, args.camera)
    touch = station.load_touch(args.arm)
    if touched is fix or not touch.height_source or not gauge_path.is_file():
        cal.say("the gauge needs the station touched for this registration and tool, its ball's height by the "
                "palette camera (tatbot ros calib run --station-only, then tatbot ros calib height) and a wrist gauge "
                f"file whose pen model it reads with ({gauge_path}: tatbot ros calib wristcam --adopt)")
        run_log.finalize(1, status="fail")
        print(json.dumps({"ok": False, "run_dir": str(run_log.dir)}))
        return 1
    program.place_station(cal, touched)
    gauge_cal = json.loads(gauge_path.read_text())
    camera = ins.WristCamera(serial=serial, depth=True)
    cal.rig, code = program.Rig(args.arm), 1
    try:
        disc = DiscGauge(cal, sweep.Sweep(cal, touched, tool.tool_id, args.station), touched, camera, gauge_cal,
                         frame)
        rows = disc.run()
        code = finish(cal, rows, gauge_cal, gauge_path, tool.tool_id, args.adopt, disc.dark_v)
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


def finish(cal, rows: list[dict], gauge_cal: dict, gauge_path: Path, tool: str, adopt: bool, dark_v=255.0) -> int:
    """The correction, its report and candidate; the gauge adopted when asked and the holds agree. 0, or 3."""
    if len(rows) < MIN_HOLDS:   # too few read in frames darker than DARK_V (dark_v, the darkest's median V): a light
        dark = f"; its frames are dark (median V {dark_v:.0f} of 255): light the probe's top" if dark_v < DARK_V else ""
        cal.say(f"only {len(rows)} holds read the probe's top, {MIN_HOLDS} needed: nothing measured{dark}")
        return 3
    delta, se, normal = correction(rows)
    run = cal.run.dir.name
    record = corrected(gauge_cal, delta, normal, tool, run, se, len(rows), cal.tip0)
    lines = [f"- holds: {len(rows)}; the gauge read the probe's top {delta * 1000:+.2f} +- {se * 1000:.2f} mm off the "
             "arm's height (" + ", ".join(f"{(r['known_m'] - r['read_m']) * 1000:+.2f}" for r in rows) + ")",
             f"- the installed gauge ({gauge_cal.get('measured_with') or 'measured with an unrecorded tool'}, "
             f"{gauge_cal.get('note') or gauge_cal.get('source', '')}) moves {delta * 1000:+.2f} mm along the normal, "
             "along its ray from the camera"]
    (cal.run.dir / "candidate.json").write_text(json.dumps({"rows": rows, "gauge": record}, indent=1) + "\n")
    (cal.run.dir / "report.md").write_text("# Wrist gauge on the probe's top\n\n" + "\n".join(lines) + "\n")
    for line in lines:
        cal.say(line)
    if abs(delta) > MAX_CORRECTION_M:
        cal.say(f"not adopted: {delta * 1000:+.2f} mm is past {MAX_CORRECTION_M * 1000:.0f}; the station's height or "
                "the registration it was measured under is off, not the gauge")
        return 3
    if adopt and gauge_cal.get("depth_seen_m"):
        cal.say(f"not adopted: the installed tip stands at the depth its camera's depth frames show the pen at "
                f"({gauge_cal['depth_seen_m'] * 1000:.1f} mm, calib wristcam), which the gauge reads the skin by; "
                f"the arm's heights over the probe's top stand {delta * 1000:+.2f} mm from it")
    elif adopt:
        gauge_path.rename(gauge_path.with_name(gauge_path.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ",
                                                                               time.gmtime())))
        gauge_path.write_text(json.dumps(record, indent=1) + "\n")
        cal.say(f"adopted: {gauge_path} (the correction known to {se * 1000:.2f} mm)")
    return 0


def rederive(installed: dict, run_dir: Path) -> dict:
    """An adopted gauge again from its run's holds (candidate.json), with no arm move: the tip the run started from,
    which the adoption moved along the normal before 2026-11-10, then the run's correction along its ray."""
    rows = json.loads((run_dir / "candidate.json").read_text())["rows"]
    if installed.get("run") != run_dir.name:
        raise ValueError(f"the installed gauge was adopted by {installed.get('run')}, not {run_dir.name}")
    delta, se, normal = correction(rows)
    before = dict(installed, tip_cam_m=(np.asarray(installed["tip_cam_m"], float) - delta * normal).tolist())
    record = corrected(before, delta, normal, installed["measured_with"], run_dir.name, se, len(rows))
    record.update(source=installed["source"], utc=installed["utc"],
                  rederived=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + ": the run's correction along the "
                                                                              "tip's ray, not the normal")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tatbot_calib gaugecal", description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tool", default=None)
    parser.add_argument("--station")
    parser.add_argument("--run-started", type=float)
    parser.add_argument("--max-age-s", type=float, default=station.MAX_AGE_S)
    parser.add_argument("--camera", default="", help="the wrist D405 by its vision.toml role (default: the session's)")
    parser.add_argument("--adopt", action="store_true")
    parser.add_argument("--from-run", type=Path, help="re-derive the gauge that run adopted from its holds: no arm move")
    args = parser.parse_args(argv)
    if args.from_run:
        from tatbot_description import repo_root

        gauge_path = program.wrist_camera(repo_root(None), args.arm, args.camera)[2]
        record = rederive(json.loads(gauge_path.read_text()), args.from_run.expanduser())
        moved = np.linalg.norm(np.subtract(record["tip_cam_m"], json.loads(gauge_path.read_text())["tip_cam_m"]))
        if args.adopt:
            gauge_path.rename(gauge_path.with_name(gauge_path.name + time.strftime(".replaced-%Y%m%dT%H%M%SZ",
                                                                                   time.gmtime())))
            gauge_path.write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps({"ok": True, "gauge": str(gauge_path), "adopted": args.adopt, "tip_moved_mm": round(moved * 1000, 3),
                          "tip_cam_m": record["tip_cam_m"]}))
        return 0
    if args.station is None or args.run_started is None:
        parser.error("--station and --run-started are required to measure")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
