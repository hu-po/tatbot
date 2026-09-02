#!/usr/bin/env bash
# `tatbot draw run|scan`: one staged session — touch, orbit-scan with the wrist
# D405s, map, shadow, draw (docs/draw.md). This wrapper owns everything around
# the C++ executor: the arm gate, the draw dir and draw.json, the D405 capture
# server, the Rerun viewer, and the landing tail copied from teleop_spiral.sh.
# The executor (wxai_teleop --draw-dir) owns every motion decision.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot draw run"
# shellcheck source=scripts/lib/estop_guard.sh
source "$REPO/scripts/lib/estop_guard.sh"
# shellcheck source=scripts/lib/arm_gate.sh
source "$REPO/scripts/lib/arm_gate.sh"
# shellcheck source=scripts/lib/profile_env.sh
source "$REPO/scripts/lib/profile_env.sh"
profile_env::require || exit $?
# shellcheck source=scripts/lib/paths.sh
source "$REPO/scripts/lib/paths.sh"
export TATBOT_LOG_ROOT="${TATBOT_LOG_ROOT:-$(tatbot_paths::log_root)}"
# shellcheck source=scripts/lib/ee_tool.sh
source "$REPO/scripts/lib/ee_tool.sh"
# shellcheck source=scripts/vision/rerun_session.sh
source "$REPO/scripts/vision/rerun_session.sh"

estop_guard::reject_overrides "$@"
ee_tool::strip "$@"; set -- "${EE_TOOL_ARGS[@]}"
ee_tool::require || exit $?

# Defaults are the docs/draw.md draw.json example.
RADIUS_MM=15
TURNS=3
ROTATION_DEG=0
DURATION_S=120          # used only when DRAW_SPEED_MM_S is 0 (--duration-s given)
DRAW_SPEED_MM_S=3.5
EASE_S=2
STANDOFF_MM=80
TILT_DEG=15
ORBIT_MODE=camera
CAMERA_DISTANCE_MM=160
OFF_AXIS_DEG=35
POSES=5
ORBIT_SPEED_MM_S=20
CELL_MM=1.0
EXTENT_MM=60
HOLE_FILL_MM=0
LEAN_BUDGET_DEG=20
LEAN_DEADBAND_DEG=0
SCAN_ONLY=0
NO_RERUN=0
RERUN_CONNECT_ARG=""
VIEWER_MEMORY_LIMIT="${VIEWER_MEMORY_LIMIT:-3GB}"
REST=()
need_value() { [ "$#" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --radius-mm)       need_value "$@"; RADIUS_MM="$2"; shift 2 ;;
    --turns)           need_value "$@"; TURNS="$2"; shift 2 ;;
    --rotation-deg)    need_value "$@"; ROTATION_DEG="$2"; shift 2 ;;
    --duration-s)      need_value "$@"; DURATION_S="$2"; DRAW_SPEED_MM_S=0; shift 2 ;;
    --draw-speed-mm-s) need_value "$@"; DRAW_SPEED_MM_S="$2"; shift 2 ;;
    --ease-s)          need_value "$@"; EASE_S="$2"; shift 2 ;;
    --standoff-mm)     need_value "$@"; STANDOFF_MM="$2"; shift 2 ;;
    --orbit-mode)      need_value "$@"; ORBIT_MODE="$2"; shift 2 ;;
    --camera-distance-mm) need_value "$@"; CAMERA_DISTANCE_MM="$2"; shift 2 ;;
    --off-axis-deg)    need_value "$@"; OFF_AXIS_DEG="$2"; shift 2 ;;
    --tilt-deg)        need_value "$@"; TILT_DEG="$2"; shift 2 ;;
    --poses)           need_value "$@"; POSES="$2"; shift 2 ;;
    --orbit-speed-mm-s) need_value "$@"; ORBIT_SPEED_MM_S="$2"; shift 2 ;;
    --cell-mm)         need_value "$@"; CELL_MM="$2"; shift 2 ;;
    --extent-mm)       need_value "$@"; EXTENT_MM="$2"; shift 2 ;;
    --hole-fill-mm)    need_value "$@"; HOLE_FILL_MM="$2"; shift 2 ;;
    --lean-budget-deg) need_value "$@"; LEAN_BUDGET_DEG="$2"; shift 2 ;;
    --lean-deadband-deg) need_value "$@"; LEAN_DEADBAND_DEG="$2"; shift 2 ;;
    --scan-only)       SCAN_ONLY=1; shift ;;
    --no-rerun)        NO_RERUN=1; shift ;;
    --rerun-viewer)   need_value "$@"; RERUN_CONNECT_ARG="$2"; shift 2 ;;
    --)                shift; REST+=("$@"); break ;;
    *)                 REST+=("$1"); shift ;;
  esac
done

if [ "$EE_TOOL" != "lutin-ballpoint-dot" ]; then
  echo "tatbot draw is qualified only for --ee-tool lutin-ballpoint-dot (got '$EE_TOOL'):" >&2
  echo "  the carriage-IK envelope and the tip constant in the executor are the ballpoint's." >&2
  exit 2
fi

# The interpreter every draw-side Python stage runs in: the same candidate
# order as tatbot_cli.verbs._common.lerobot_py (system python3 has no numpy).
draw_python() {
  local home="${HOME:-~}" p
  for p in "$REPO/python/lerobot_robot_tatbot/.venv/bin/python" \
           "${TATBOT_SERVE_ROOT:-$home/il-serve}/.venv/bin/python" \
           "${TATBOT_TRAIN_ROOT:-$home/il-train}/.venv/bin/python"; do
    if [ -x "$p" ]; then echo "$p"; return 0; fi
  done
  return 1
}
DRAW_PY="$(draw_python)" || {
  echo "no LeRobot interpreter found (python/lerobot_robot_tatbot/.venv, ~/il-serve, ~/il-train):" >&2
  echo "  cd python/lerobot_robot_tatbot && uv sync" >&2
  exit 1
}

arm_gate::require || exit $?

# --- the draw dir --------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname -s 2>/dev/null || hostname)"
HEX="$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')"
DRAW_DIR="$TATBOT_LOG_ROOT/draw/$STAMP-$HOST-$HEX"
mkdir -p "$DRAW_DIR/capture"
echo "=== tatbot draw: $DRAW_DIR ==="

CAPTURE_PID=""
LAND_SENTINEL="/tmp/tatbot-probe-land-$$"
cleanup() {
  if [ -n "$CAPTURE_PID" ] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    touch "$DRAW_DIR/capture/server.stop"
    local attempt
    for ((attempt = 0; attempt < 30; attempt++)); do
      kill -0 "$CAPTURE_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill "$CAPTURE_PID" 2>/dev/null || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
  rerun_session::stop_viewer "${RERUN_VIEWER_PID:-}"
  rm -f "$LAND_SENTINEL"
}
trap cleanup EXIT

# --- Rerun viewer (capped; scripts/check rerun-caps insists on this helper) ---------
# Where the shadow goes, in order: --rerun-viewer; the launching node's viewer
# when this shell came in over ssh without a display (`tatbot draw viewer` on
# that node, found through SSH_CONNECTION); else a local capped viewer.
RERUN_CONNECT=""
if [ "$NO_RERUN" -eq 0 ]; then
  if [ -n "$RERUN_CONNECT_ARG" ]; then
    RERUN_CONNECT="$RERUN_CONNECT_ARG"
    echo "Rerun: streaming the shadow to $RERUN_CONNECT"
  elif [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ -n "${SSH_CONNECTION:-}" ]; then
    CLIENT_IP="${SSH_CONNECTION%% *}"
    if rerun_session::port_open "$CLIENT_IP" 9876; then
      RERUN_CONNECT="rerun+http://${CLIENT_IP}:9876/proxy"
      echo "Rerun: no display here; streaming the shadow to the launching node's viewer at $RERUN_CONNECT"
    else
      echo "Rerun: no display here and nothing listens on ${CLIENT_IP}:9876 — run 'tatbot draw viewer' on the" \
           "launching node first; the shadow is saved to $DRAW_DIR/shadow.rrd only" >&2
    fi
  elif rerun_session::start_viewer "$VIEWER_MEMORY_LIMIT" 512MB; then
    LAN_IP="$(rerun_session::lan_ip)"
    RERUN_CONNECT="rerun+http://${LAN_IP:-127.0.0.1}:9876/proxy"
    echo "Rerun viewer on $RERUN_CONNECT (memory cap $VIEWER_MEMORY_LIMIT)"
  else
    echo "Rerun viewer did not start; the shadow will be saved to $DRAW_DIR/shadow.rrd only" >&2
  fi
fi

# --- draw.json (wrapper -> stages) ---------------------------------------------------
TOOL="$EE_TOOL" RADIUS_MM="$RADIUS_MM" TURNS="$TURNS" ROTATION_DEG="$ROTATION_DEG" \
DURATION_S="$DURATION_S" DRAW_SPEED_MM_S="$DRAW_SPEED_MM_S" EASE_S="$EASE_S" SCAN_ONLY="$SCAN_ONLY" STANDOFF_MM="$STANDOFF_MM" \
TILT_DEG="$TILT_DEG" POSES="$POSES" ORBIT_SPEED_MM_S="$ORBIT_SPEED_MM_S" CELL_MM="$CELL_MM" \
ORBIT_MODE="$ORBIT_MODE" CAMERA_DISTANCE_MM="$CAMERA_DISTANCE_MM" OFF_AXIS_DEG="$OFF_AXIS_DEG" \
EXTENT_MM="$EXTENT_MM" HOLE_FILL_MM="$HOLE_FILL_MM" LEAN_BUDGET_DEG="$LEAN_BUDGET_DEG" \
LEAN_DEADBAND_DEG="$LEAN_DEADBAND_DEG" DRAW_PY="$DRAW_PY" \
RERUN_CONNECT="$RERUN_CONNECT" DRAW_DIR="$DRAW_DIR" python3 - <<'EOF'
import json, os
e = os.environ
num = lambda k: float(e[k])  # noqa: E731
cfg = {
    "schema": "tatbot.draw-config/1", "tool": e["TOOL"],
    "design": {"kind": "spiral", "radius_mm": num("RADIUS_MM"), "turns": num("TURNS"),
               "rotation_deg": num("ROTATION_DEG")},
    "duration_s": num("DURATION_S"), "draw_speed_mm_s": num("DRAW_SPEED_MM_S") or None,
    "ease_s": num("EASE_S"), "scan_only": e["SCAN_ONLY"] == "1",
    "orbit": {"mode": e["ORBIT_MODE"], "camera_distance_mm": num("CAMERA_DISTANCE_MM"), "off_axis_deg": num("OFF_AXIS_DEG"),
              "standoff_mm": num("STANDOFF_MM"), "tilt_deg": num("TILT_DEG"), "poses": int(float(e["POSES"])),
              "speed_mm_s": num("ORBIT_SPEED_MM_S")},
    "map": {"cell_mm": num("CELL_MM"), "extent_mm": num("EXTENT_MM"), "chart": "auto",
            "hole_fill_mm": num("HOLE_FILL_MM")},
    "lean_budget_deg": num("LEAN_BUDGET_DEG"), "lean_deadband_deg": num("LEAN_DEADBAND_DEG"),
    "python": e["DRAW_PY"], "rerun_connect": e["RERUN_CONNECT"],
}
with open(os.path.join(e["DRAW_DIR"], "draw.json"), "w") as fh:
    json.dump(cfg, fh, indent=1)
    fh.write("\n")
EOF
if [ "$DRAW_SPEED_MM_S" = "0" ]; then PACE="${DURATION_S}s"; else PACE="${DRAW_SPEED_MM_S}mm/s"; fi
echo "draw.json: spiral r=${RADIUS_MM}mm turns=$TURNS $PACE; orbit ${STANDOFF_MM}mm/${TILT_DEG}deg x$POSES;" \
     "map ${CELL_MM}mm/${EXTENT_MM}mm hole-fill ${HOLE_FILL_MM}mm; lean<=${LEAN_BUDGET_DEG}deg" \
     "deadband ${LEAN_DEADBAND_DEG}deg; scan_only=$SCAN_ONLY"

# --- capture server (both D405s) --------------------------------------------------
"$DRAW_PY" "$REPO/scripts/draw_capture.py" serve "$DRAW_DIR/capture" 2>&1 | sed -u 's/^/[capture] /' &
CAPTURE_PID=$!
for ((attempt = 0; attempt < 300; attempt++)); do
  [ -f "$DRAW_DIR/capture/server.ready" ] && break
  if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then break; fi
  sleep 0.1
done
if [ ! -f "$DRAW_DIR/capture/server.ready" ]; then
  echo "D405 capture server did not become ready within 30 s; refusing to start the session (exit 5)." >&2
  echo "  Is LeRobot or visiond holding the RealSenses? See [capture] lines above." >&2
  exit 5
fi
echo "D405 capture server ready."

if [ "$SCAN_ONLY" -eq 1 ]; then
  echo "SCAN ONLY: touch, orbit and map; no path is streamed."
fi
echo "Hand-guide to LIGHT CONTACT at the design centre. SPACE 1 = standoff orbit + map;"
echo "SPACE 2 = the compiled surface path (refused unless the preflight passed); Enter = land."

# --- the session ---------------------------------------------------------------------
export TATBOT_DRAW_ARMED=1
# The executor shells out to draw_stage.py with this interpreter (docs/draw.md).
export TATBOT_DRAW_PYTHON="$DRAW_PY"
export TATBOT_CARRIAGE_IK_ARMED=1
export TATBOT_PROBE_LAND_SENTINEL="$LAND_SENTINEL"
set +e
"$REPO/scripts/teleop_start.sh" \
  --ee-tool "$EE_TOOL" \
  --draw-dir "$DRAW_DIR" \
  --ff-gain 0 \
  "${REST[@]}"
PROBE_RC=$?
set -e

next_step() {
  echo "draw dir: $DRAW_DIR"
  echo "next: tatbot draw shadow $DRAW_DIR"
}

if [ "$PROBE_RC" -eq 130 ]; then
  echo "Draw session ended by emergency release; automatic landing is SKIPPED." >&2
  next_step
  exit "$PROBE_RC"
fi
if [ ! -f "$LAND_SENTINEL" ]; then
  echo "Draw session did not record an operator release; automatic landing is SKIPPED." >&2
  next_step
  exit "$PROBE_RC"
fi

echo "Draw session released: landing follower, then leader, to staged -> sleep -> idle."
echo "Keep both landing paths clear; E-stop operator stay ready."
LANDING_FAILED=0
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_FOLLOWER_IP" follower || LANDING_FAILED=1
"$REPO/scripts/il_recover_arm.sh" "$TATBOT_LEADER_IP" leader || LANDING_FAILED=1
next_step
if [ "$LANDING_FAILED" -ne 0 ]; then
  echo "Automatic landing did not verify for both arms; inspect the messages above." >&2
  exit 1
fi
exit "$PROBE_RC"
