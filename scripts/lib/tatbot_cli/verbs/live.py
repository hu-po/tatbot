"""live — every live sensor in one Rerun viewer (scripts/live/)."""

from __future__ import annotations

from tatbot_cli.registry import SENSOR, verb
from tatbot_cli.verbs._common import sh


def _cockpit_args(p):
    p.add_argument("--fps", help="PoE substream sets/s (default 2)")
    p.add_argument("--rs-fps", help="RealSense colour+depth sets/s (default 2)")
    p.add_argument("--rs-scale", help="RealSense colour+depth scale on the wire (default 0.5)")
    p.add_argument("--stream", choices=("sub", "main"), help="PoE stream (default sub)")
    p.add_argument("--duration", help="seconds before every producer stops (default 3600)")
    p.add_argument("--no-realsense", action="store_true", help="leave the wrist D405s alone")
    p.add_argument("--no-audio", action="store_true", help="skip the EE contact-mic producer")
    p.add_argument("--no-teleop", action="store_true", help="skip the URDF telemetry bridge")


@verb(noun="live", verb="cockpit", tier=SENSOR,
      summary="5 PoE + 2 D405 (RGB+depth) + URDF/teleop + mono EE audio in one capped Rerun viewer",
      role="viewer",
      wraps=("scripts/live/cockpit.sh", "scripts/audio/live_audio.py"),
      args=_cockpit_args, example=("--rs-fps", "2"), doc="docs/vision.md", tty=True,
      invariants=(
          "Read-only: no arm is opened; the robot animates only while `teleop start` runs on the arm node.",
          "The D405s are single-owner: refuses (exit 6) beside a LeRobot session on the arm node.",
          "Viewer memory and every producer's rate are capped; the launcher refuses an uncapped start.",
      ))
def cockpit(ctx, ns, rest):
    flags: list[str] = []
    for dest, flag in (("fps", "--fps"), ("rs_fps", "--rs-fps"), ("rs_scale", "--rs-scale"),
                       ("stream", "--stream"), ("duration", "--duration")):
        value = getattr(ns, dest)
        if value:
            flags += [flag, value]
    for dest, flag in (("no_realsense", "--no-realsense"), ("no_audio", "--no-audio"), ("no_teleop", "--no-teleop")):
        if getattr(ns, dest):
            flags.append(flag)
    return sh(ctx, "scripts/live/cockpit.sh", *flags, *rest)
