#!/usr/bin/env bash
# Manifested camera/bus services only; never starts an arm driver or viewer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$ROOT/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot deploy"
RELEASE="$(dirname "$ROOT")"
source "$ROOT/scripts/lib/runlog.sh"
runlog::init fleet-service --set service="${1:?service required}"
[[ -n "$RUN_DIR" ]] || { echo 'service requires a run log' >&2; exit 5; }
NODE="${TATBOT_NODE:?unit must state node}"
BUS="$(PYTHONPATH="$ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" python3 - "$ROOT" <<'PYTHON'
import sys
from pathlib import Path
from tatbot_cli.nodes import bus_endpoint, load
print(bus_endpoint(load(Path(sys.argv[1])), address="lan"))
PYTHON
)"
# Keep device assignments and node ownership from the same deployed source.
VISION_CONFIG="$ROOT/rust/visiond/config/vision.toml"
CALIBRATION="$HOME/tatbot-logs/vision/calibration-current.json"
d555_calibrated() {
  python3 - "$CALIBRATION" <<'PY'
import json, sys
try:
    cameras = json.load(open(sys.argv[1]))["cameras"]
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if {"overhead_depth_color", "overhead_depth_depth"} <= set(cameras) else 1)
PY
}
# The router on the rig address alone, which every client dials. With this node's fleet bus credential (an
# empty TATBOT_BUS_CREDENTIALS states none) it admits only its user: zenohd's usrpwd dictionary refuses anyone
# else at the handshake, and the access-control list denies all but that user (docs/fleet.md).
zenohd_args() {
  ZENOHD=(--config "$ROOT/config/zenoh/router.json5" --listen "$BUS")
  local creds="${TATBOT_BUS_CREDENTIALS-$HOME/.config/tatbot/bus-credentials}" user messages
  [[ -n "$creds" && -f "$creds" ]] || return 0
  [[ "$(stat -c %a "$creds")" == [4-7]00 ]] || { echo "zenohd: $creds is readable by others: chmod 600 it" >&2; exit 3; }
  user="$(grep -m 1 -v -E '^[[:space:]]*$' "$creds" | cut -d: -f1 | tr -d '[:space:]')"
  [[ "$user" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "zenohd: $creds has no user:password line" >&2; exit 3; }
  messages='"put","delete","declare_subscriber","query","reply","declare_queryable","liveliness_token","declare_liveliness_subscriber","liveliness_query"'
  ZENOHD+=(--cfg "transport/auth/usrpwd/dictionary_file:\"$creds\""
           --cfg "access_control:{enabled:true,default_permission:\"deny\",rules:[{id:\"fleet\",messages:[$messages],flows:[\"egress\",\"ingress\"],permission:\"allow\",key_exprs:[\"**\"]}],subjects:[{id:\"fleet\",usernames:[\"$user\"]}],policies:[{rules:[\"fleet\"],subjects:[\"fleet\"]}]}")
}
case "$1" in
  zenohd) zenohd_args; runlog::run "$RELEASE/bin/zenohd" "${ZENOHD[@]}" ;;
  zenoh-presence) runlog::run "$RELEASE/bin/fleetctl" --connect "$BUS" advertise --node "$NODE" --service zenohd --unit tatbot-zenohd.service ;;
  visiond-d405)
    # Select this host's devices before starting the capture executable. Keep
    # the d405 bus group; each producer already has its own node identity.
    D405_SENSORS="$(python3 - "$ROOT" "$NODE" "$VISION_CONFIG" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/lib'))
from wrist_cameras import owned_names
try:
    print('\n'.join(owned_names(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))))
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f'D405 ownership refused: {exc}')
PY
)"
    D405_ARGS=()
    while IFS= read -r sensor; do D405_ARGS+=(--sensor "$sensor"); done <<< "$D405_SENSORS"
    runlog::run "$RELEASE/bin/tatbot-visiond" --zenoh --bus-connect "$BUS" --bus-node "$NODE" capture-realsense-all "$VISION_CONFIG" --group d405 "${D405_ARGS[@]}" --duration-seconds 0 --no-record --socket /tmp/tatbot-d405-frames.sock ;;
  visiond-d555)
    D555_CALIBRATION=()
    d555_calibrated && D555_CALIBRATION=(--calibration "$CALIBRATION")
    # The D555 is found by DDS discovery, bound to this node's rig address:
    # discovery on every interface can answer the camera over the wrong route.
    DDS_ADDRESS="$(PYTHONPATH="$ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" python3 - "$ROOT" "$NODE" <<'PY'
import sys
from pathlib import Path
from tatbot_cli.nodes import load
address = load(Path(sys.argv[1])).get(sys.argv[2], {}).get("lan")
if not address:
    sys.exit(f"the D555 owner {sys.argv[2]} has no `lan` address in config/nodes.json")
print(address)
PY
)"
    runlog::run "$RELEASE/bin/tatbot-visiond" --zenoh --bus-connect "$BUS" --bus-node "$NODE" capture-realsense-all "$VISION_CONFIG" --group overhead-depth --dds-address "$DDS_ADDRESS" --duration-seconds 0 --no-record --socket /tmp/tatbot-d555-frames.sock "${D555_CALIBRATION[@]}" ;;
  trackd)
    # The follower wrist's tracker (`wrist`: its unsuffixed topic and lease),
    # on its calibrated tag layout, subscribed to the D555 owner's socket.
    d555_calibrated || { echo 'trackd: no calibrated fixed camera on this node' >&2; exit 3; }
    # A 47 mm wrist tag spans ~50 px in the D555's 1280x720 colour (~20 px at
    # 640x360, where it was searched at full scale: at a 0.3 reacquire scale it
    # was ~6 px and never acquired, 0 of 150 live frames). The full-frame search
    # at half scale keeps 640x360's pixels and its ~25 px tags; quad decimation
    # stays 1: on a 1280x720 frame of the parked arm, half scale with 2 found one
    # of its three tags, with 1 two, as 640x360 did. The ROI margin is 80 px doubled.
    runlog::run "$RELEASE/bin/trackd" --connect "$BUS" --node "$NODE" --socket /tmp/tatbot-d555-frames.sock --inventory "$ROOT/config/fiducials.json" --calibration "$CALIBRATION" --target wrist --wrist-layout "$ROOT/config/wrist_tags_measured.json" --temporal-initializers --reacquire-scale 0.5 --quad-decimate 1 --roi-margin-px 160 --output "$RUN_DIR/estimates.jsonl" ;;
  stencild)
    # One stencil observer for the fleet, beside the overhead RGB-D owner:
    # each turn its newest set, every installed print placed by its markers
    # and published as tatbot.target-pose/1 with its support. A subscriber to
    # the same owner socket as trackd; no device, no arm, no motion
    # authority. References are installed under the log root (one directory
    # per print, tracking.json beside its stencil.png and tile.json) and
    # rescanned by mtime, so installing one never restarts the service; none
    # installed leaves it idle. The observer's interpreter is the release's
    # own venv (scripts/vision/requirements-observer.txt).
    # shellcheck source=scripts/lib/paths.sh
    source "$ROOT/scripts/lib/paths.sh"
    STENCIL_REFERENCES="$(tatbot_paths::log_root)/stencils/references"
    mkdir -p "$STENCIL_REFERENCES"
    # A print is measured only in a calibrated world: without a D555 under a
    # bundle holding both of its entries there is nothing to observe (on the
    # demo stack: `tatbot ros register --adopt`).
    d555_calibrated || { echo 'stencild: no calibrated overhead camera on this node' >&2; exit 3; }
    # Captures are verified transient inputs. Keep them in the service's
    # systemd-managed tmpfs directory; run logs and estimates stay on disk.
    STENCIL_WORK="${RUNTIME_DIRECTORY:-$RUN_DIR}/captures"
    runlog::run "$RELEASE/bin/stencild" --connect "$BUS" --node "$NODE" --overhead-socket /tmp/tatbot-d555-frames.sock --calibration "$CALIBRATION" --references "$STENCIL_REFERENCES" --python "$RELEASE/observer-venv/bin/python" --observer "$ROOT/scripts/vision/stencil_observer.py" --work "$STENCIL_WORK" --output "$RUN_DIR/estimates.jsonl" ;;
  cockpit)
    # The demo cockpit: a display subscriber to what the camera owners
    # publish (the D555's colour and depth, the wrist tap, trackd and
    # stencild's overlays in the D555's camera-world frame), and the palette's
    # tag it reads in the D555's colour itself, streamed into
    # the standing recording. The hub binds every address
    # (scripts/viewer.sh serve), so beside it loopback reaches it whether or
    # not the rig address is up; elsewhere the hub's `lan`.
    HUB="$(PYTHONPATH="$ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" python3 - "$ROOT" "$NODE" <<'PY'
import sys
from pathlib import Path
from tatbot_cli.nodes import load, rerun_server
server = rerun_server(load(Path(sys.argv[1])))
if server is None:
    sys.exit("cockpit: no rerun-server node with a `lan` address in config/nodes.json")
print("127.0.0.1" if server[0] == sys.argv[2] else server[1])
PY
)"
    runlog::run "$RELEASE/bin/tatbot-visiond" --bus-connect "$BUS" subscribe --connect "rerun+http://$HUB:${RERUN_PORT:-9876}/proxy" --recording-id live-cameras --duration-seconds 0 --camera-world --fiducials "$ROOT/config/fiducials.json" ;;
  *) echo "unmanifested service: $1" >&2; exit 3 ;;
esac
