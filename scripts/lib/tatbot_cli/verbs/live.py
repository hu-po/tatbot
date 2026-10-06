"""live — every live sensor in one Rerun viewer (scripts/live/)."""

from __future__ import annotations

from tatbot_cli.registry import SENSOR, verb
from tatbot_cli.verbs._common import sh


def _cockpit_args(p):
    p.add_argument("--display-quality", choices=("normal", "showcase"), default="normal",
                   help="bounded preview profile: normal is cheap; showcase is explicit full-quality display")
    p.add_argument("--fps", help="bus preview sets/s (default 2; showcase 5)")
    p.add_argument("--duration", help="seconds before the subscriber stops (default 3600)")
    p.add_argument("--recording-id", help="join this existing session recording")
    p.add_argument("--robot-world", help="registration for tracking overlays (default: current local registration)")


@verb(effects=('read_files', 'write_files', 'sensor_read', 'network'), noun="live", verb="cockpit", tier=SENSOR,
      summary="every camera owner's bus previews and wrist tracking on the fleet Rerun viewer",
      role="operator",
      wraps=("scripts/live/cockpit_bus.sh",),
      args=_cockpit_args, example=("--fps", "2"), doc="docs/vision.md", tty=True,
      invariants=(
          "Opens no camera and no arm: it subscribes to the frames the camera owners already publish on the bus.",
          "Viewer memory and the subscriber's rate are capped; the launcher refuses an uncapped start.",
      ))
def cockpit(ctx, ns, rest):
    flags = ["--display-quality", ns.display_quality] if ns.display_quality != "normal" else []
    flags += [value for key in ("fps", "duration", "recording_id", "robot_world") if getattr(ns, key)
              for value in ("--" + key.replace("_", "-"), getattr(ns, key))]
    return sh(ctx, "scripts/live/cockpit_bus.sh", *flags, *rest)
