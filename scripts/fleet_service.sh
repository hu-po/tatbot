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
has_role() {
  PYTHONPATH="$ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" python3 - "$ROOT" "$NODE" "$1" <<'PY'
import sys
from pathlib import Path
from tatbot_cli.nodes import load, roles_of
raise SystemExit(0 if sys.argv[3] in roles_of(load(Path(sys.argv[1])), sys.argv[2]) else 1)
PY
}
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
case "$1" in
  zenohd) runlog::run "$RELEASE/bin/zenohd" --config "$ROOT/config/zenoh/router.json5" ;;
  zenoh-presence) runlog::run "$RELEASE/bin/fleetctl" --connect "$BUS" advertise --node "$NODE" --service zenohd --unit tatbot-zenohd.service ;;
  visiond-poe)
    SYNC_WAIT="${TATBOT_POE_SYNC_WAIT_MS:-60}"
    [[ "$SYNC_WAIT" =~ ^([1-9]|[1-5][0-9]|60)$ ]] || { echo 'PoE sync wait must be 1..60 ms' >&2; exit 3; }
    runlog::run "$RELEASE/bin/tatbot-visiond" --zenoh --bus-connect "$BUS" --bus-node "$NODE" capture-poe-all "$VISION_CONFIG" --decoded --stream main --duration-seconds 0 --no-record --fiducial-min-cameras 3 --fiducial-sync-tolerance-ms 40 --fiducial-max-sync-wait-ms "$SYNC_WAIT" --calibration "$CALIBRATION" --socket /tmp/tatbot-poe-frames.sock --socket-luma ;;
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
  trackd|trackd-left)
    # One tracker process per wrist: `wrist` (the follower, its unsuffixed
    # topic and lease) and `wrist_left` (the leader, `tatbot/tracking/ee/left`,
    # lease `trackd-left`). Each takes its own calibrated tag layout and lock;
    # both subscribe to the same owner sockets: the PoE cameras where the node
    # has them, the D555 alone on the demo stack.
    TRACKD_SOCKETS=()
    has_role poe-cameras && TRACKD_SOCKETS+=(--socket /tmp/tatbot-poe-frames.sock)
    d555_calibrated && TRACKD_SOCKETS+=(--socket /tmp/tatbot-d555-frames.sock)
    [ "${#TRACKD_SOCKETS[@]}" -gt 0 ] || { echo 'trackd: no calibrated fixed camera on this node' >&2; exit 3; }
    # A 47 mm wrist tag spans ~20 px in the D555's 640x360 colour. Searched at
    # the PoE default reacquire scale 0.3 it is ~6 px and never acquired (0 of
    # 150 live frames; 150 of 150 at 1.0), and the ROI's decimation halves it.
    TRACKD_DETECT=(--reacquire-scale 1 --quad-decimate 1)
    has_role poe-cameras && TRACKD_DETECT=(--quad-decimate 2)
    TRACKD_TARGET=(--target wrist --wrist-layout "$ROOT/config/wrist_tags_measured.json")
    [ "$1" = trackd-left ] && TRACKD_TARGET=(--target wrist_left --wrist-layout "$ROOT/config/wrist_tags_measured_left.json")
    runlog::run "$RELEASE/bin/trackd" --connect "$BUS" --node "$NODE" "${TRACKD_SOCKETS[@]}" --inventory "$ROOT/config/fiducials.json" --calibration "$CALIBRATION" "${TRACKD_TARGET[@]}" --temporal-initializers "${TRACKD_DETECT[@]}" --output "$RUN_DIR/estimates.jsonl" ;;
  stencild)
    # One stencil observer for the fleet: every fixed view each turn (the PoE
    # cameras where this node owns them, and the overhead RGB-D under the
    # shared bundle), every visible print published as tatbot.target-pose/1
    # with its support. A subscriber to the same owner sockets as trackd; no
    # device, no arm, no motion authority. References are installed under
    # the log root (one directory per print, tracking.json beside its
    # stencil.png) and rescanned by mtime, so installing one never restarts
    # the service; none installed leaves it idle. The observer's interpreter
    # is the release's own venv (scripts/vision/requirements-observer.txt).
    # shellcheck source=scripts/lib/paths.sh
    source "$ROOT/scripts/lib/paths.sh"
    STENCIL_REFERENCES="$(tatbot_paths::log_root)/stencils/references"
    mkdir -p "$STENCIL_REFERENCES"
    STENCILD_SOCKETS=()
    has_role poe-cameras && STENCILD_SOCKETS+=(--poe-socket /tmp/tatbot-poe-frames.sock)
    d555_calibrated && STENCILD_SOCKETS+=(--overhead-socket /tmp/tatbot-d555-frames.sock)
    # A print is measured only in a calibrated world: with neither the PoE
    # cameras nor a D555 under a bundle holding both of its entries there is
    # nothing to observe (on the demo stack: `tatbot ros register --adopt`).
    [ "${#STENCILD_SOCKETS[@]}" -gt 0 ] || { echo 'stencild: no calibrated fixed camera on this node' >&2; exit 3; }
    # Without the PoE cameras the D555 is the only fixed view, and it cannot
    # resolve a print (0.43 px/mm): let it place an unmeasured print by its
    # artwork on the table plane, identity unverified (docs/vision.md).
    STENCILD_ARTWORK=()
    has_role poe-cameras || STENCILD_ARTWORK=(--overhead-artwork-match)
    # Every fixed camera may anchor a print. The anchor is sticky (the one
    # that anchored last while it still sees the print), so the third camera
    # anchors only a print the others lose: its bundle entry disagreed by
    # 21-25 mm at the wrist, but on the page plane its print centre sits
    # 2-3.5 mm from the fourth's, which the drawing arm's wrist locate takes out.
    # Captures are verified transient inputs. Keep them in the service's
    # systemd-managed tmpfs directory; run logs and estimates stay on disk.
    STENCIL_WORK="${RUNTIME_DIRECTORY:-$RUN_DIR}/captures"
    # Each arm's wrist D405 (the registry's d405 cameras with an arm and an
    # owner role) is one more view, queried from its owner's capture
    # queryable at most once a second and posed through the arm's own
    # registration beside the bundle at the measured joints on the bus.
    runlog::run "$RELEASE/bin/stencild" --connect "$BUS" --node "$NODE" "${STENCILD_SOCKETS[@]}" "${STENCILD_ARTWORK[@]}" --calibration "$CALIBRATION" --vision-config "$VISION_CONFIG" --registrations "$(dirname "$CALIBRATION")" --urdf "$ROOT/urdf/tatbot.urdf" --references "$STENCIL_REFERENCES" --python "$RELEASE/observer-venv/bin/python" --observer "$ROOT/scripts/vision/stencil_observer.py" --work "$STENCIL_WORK" --output "$RUN_DIR/estimates.jsonl" ;;
  *) echo "unmanifested service: $1" >&2; exit 3 ;;
esac
