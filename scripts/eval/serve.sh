#!/usr/bin/env bash
# Start one foreground async policy server with an explicit checkpoint contract.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/cli_hint.sh
source "$REPO/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot serve start"

usage() {
  cat <<'EOF'
usage: scripts/eval/serve.sh --policy PATH [options]

options:
  --policy-type TYPE  expected client policy type (default: inferred from config.json)
  --plausibility-contract PATH
                      reject unsafe chunks before returning them to a client
  --trace-dir PATH    atomically retain observation, normalized and decoded chunks
  --env-root PATH     serving uv environment (default: ~/il-serve)
  --host ADDRESS      listen address (default: 127.0.0.1)
  --port PORT         listen port (default: 8080)
  --fps FPS           action frequency (default: 30)
  --state-file PATH   live metadata (default: <env-root>/current-server.json)

The server runs in the foreground and removes its state file on clean exit.
Training-only nodes are always rejected.
EOF
}

POLICY=""
POLICY_TYPE=""
PLAUSIBILITY_CONTRACT=""
TRACE_DIR=""
ENV_ROOT="${TATBOT_SERVE_ROOT:-$HOME/il-serve}"
HOST="127.0.0.1"
PORT=8080
FPS=30
STATE_FILE=""
while (($#)); do
  case "$1" in
    --policy) POLICY="${2:?missing policy path}"; shift 2 ;;
    --policy-type) POLICY_TYPE="${2:?missing policy type}"; shift 2 ;;
    --plausibility-contract) PLAUSIBILITY_CONTRACT="${2:?missing contract path}"; shift 2 ;;
    --trace-dir) TRACE_DIR="${2:?missing trace directory}"; shift 2 ;;
    --env-root) ENV_ROOT="${2:?missing env root}"; shift 2 ;;
    --host) HOST="${2:?missing host}"; shift 2 ;;
    --port) PORT="${2:?missing port}"; shift 2 ;;
    --fps) FPS="${2:?missing fps}"; shift 2 ;;
    --state-file) STATE_FILE="${2:?missing state file}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$POLICY" ] || { echo "--policy is required" >&2; exit 2; }
[ -f "$POLICY/config.json" ] || { echo "checkpoint has no config.json: $POLICY" >&2; exit 2; }
PYTHON="$ENV_ROOT/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "serving interpreter missing: $PYTHON" >&2; exit 2; }

# A node whose roles include `train` but not `serve` must never serve a policy:
# training owns its GPU. Roles come from config/nodes.json, so the rule follows
# the deployment instead of a hostname list.
# shellcheck source=scripts/lib/nodes.sh
source "$REPO/scripts/lib/nodes.sh"
_this="$(tatbot_nodes::this_node)"  # maps a hostname alias to its node name
if python3 - "$REPO" "$_this" <<'PY'
import json, sys
from pathlib import Path
try:
    nodes = json.loads((Path(sys.argv[1]) / "config" / "nodes.json").read_text())
except (OSError, json.JSONDecodeError):
    sys.exit(1)
rec = nodes.get(sys.argv[2]) or {}
roles = rec.get("roles", [])
sys.exit(0 if ("train" in roles and "serve" not in roles) else 1)
PY
then
  echo "$_this is training-only; policy serving is forbidden" >&2
  exit 75
fi
if pgrep -f '[/]lerobot-train([[:space:]]|$)|lerobot[.]scripts[.]lerobot_train' >/dev/null; then
  echo "trainer is running on this host; refusing to share its GPU" >&2
  exit 75
fi
if pgrep -f '[l]erobot.*policy[_-]server|[p]olicy_server.*lerobot' >/dev/null; then
  echo "another policy server is already running" >&2
  pgrep -af '[l]erobot.*policy[_-]server|[p]olicy_server.*lerobot' >&2
  exit 75
fi

if [ -z "$POLICY_TYPE" ]; then
  POLICY_TYPE="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["type"])' "$POLICY/config.json")"
fi
if [ -n "$PLAUSIBILITY_CONTRACT" ]; then
  [ -f "$PLAUSIBILITY_CONTRACT" ] || {
    echo "plausibility contract is not a file: $PLAUSIBILITY_CONTRACT" >&2
    exit 2
  }
  PLAUSIBILITY_CONTRACT="$(cd "$(dirname "$PLAUSIBILITY_CONTRACT")" && pwd -P)/$(basename "$PLAUSIBILITY_CONTRACT")"
  "$PYTHON" - "$PLAUSIBILITY_CONTRACT" "$POLICY" "$FPS" <<'PY'
import hashlib, json, pathlib, sys
contract_path, policy_path = map(pathlib.Path, sys.argv[1:3])
serve_fps = float(sys.argv[3])
contract = json.loads(contract_path.read_text())
if contract.get("kind") != "demonstration-derived no-arm trajectory plausibility contract":
    raise SystemExit("unsupported plausibility contract kind")
if contract.get("schema_version") not in (1, 2):
    raise SystemExit("unsupported plausibility contract schema")
if contract.get("schema_version") == 2:
    if not isinstance(contract.get("quality_thresholds"), dict):
        raise SystemExit("schema-2 plausibility contract lacks quality thresholds")
    execution = contract.get("execution_model")
    if not isinstance(execution, dict):
        raise SystemExit("schema-2 plausibility contract lacks execution model")
config = json.loads((policy_path / "config.json").read_text())
expected_shape = (int(config["n_action_steps"]), int(config["output_features"]["action"]["shape"][0]))
actual_shape = (int(contract.get("horizon", -1)), int(contract.get("joints", -1)))
if actual_shape != expected_shape:
    raise SystemExit(f"plausibility contract shape {actual_shape} != checkpoint {expected_shape}")
if contract.get("schema_version") == 2:
    try:
        execution_fps = float(execution["fps"])
        target_velocity = float(execution["max_joint_velocity_rad_s"])
        controller_velocity = float(execution["controller_velocity_limit_rad_s"])
        actions_per_chunk = int(execution["actions_per_chunk"])
        aggregate = execution["aggregate_fn_name"]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid schema-2 execution model: {error}") from error
    if execution_fps != serve_fps:
        raise SystemExit(
            f"plausibility execution fps {execution_fps:g} != server fps {serve_fps:g}"
        )
    if not (0 < target_velocity <= controller_velocity):
        raise SystemExit("plausibility execution velocity limits are invalid")
    if not (0 < actions_per_chunk <= expected_shape[0]):
        raise SystemExit("plausibility execution actions_per_chunk is invalid")
    if aggregate != "weighted_average":
        raise SystemExit("plausibility execution aggregate_fn_name must be weighted_average")
postprocessor = policy_path / "policy_postprocessor.json"
digest = hashlib.sha256(postprocessor.read_bytes()).hexdigest()
if contract.get("postprocessor_sha256") != digest:
    raise SystemExit("plausibility contract postprocessor hash does not match checkpoint")
artifacts = contract.get("postprocessor_artifacts_sha256")
if not isinstance(artifacts, dict):
    raise SystemExit("plausibility contract lacks processor artifact hashes")
if artifacts is not None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise SystemExit("invalid plausibility processor artifact hash map")
    for name, expected in artifacts.items():
        artifact = pathlib.Path(name)
        if artifact.name != name or artifact.is_absolute():
            raise SystemExit(f"invalid plausibility processor artifact name: {name!r}")
        artifact = policy_path / name
        if not artifact.is_file():
            raise SystemExit(f"checkpoint lacks plausibility-bound processor artifact: {name}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"processor artifact hash does not match checkpoint: {name}")
PY
  export TATBOT_PLAUSIBILITY_CONTRACT="$PLAUSIBILITY_CONTRACT"
  export PYTHONPATH="$REPO/scripts/eval${PYTHONPATH:+:$PYTHONPATH}"
fi
if [ -n "$TRACE_DIR" ]; then
  mkdir -p "$TRACE_DIR"
  TRACE_DIR="$(cd "$TRACE_DIR" && pwd -P)"
  export TATBOT_INFERENCE_TRACE_DIR="$TRACE_DIR"
  export PYTHONPATH="$REPO/scripts/eval${PYTHONPATH:+:$PYTHONPATH}"
fi
STATE_FILE="${STATE_FILE:-$ENV_ROOT/current-server.json}"

"$PYTHON" "$REPO/scripts/il_patch_lerobot.py"
export TATBOT_EXPECTED_POLICY="$POLICY"
export TATBOT_EXPECTED_POLICY_TYPE="$POLICY_TYPE"

SERVER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [ -z "$SERVER_PID" ] || kill -TERM "$SERVER_PID" 2>/dev/null || true
  [ -z "$SERVER_PID" ] || wait "$SERVER_PID" 2>/dev/null || true
  if [ -f "$STATE_FILE" ] && grep -q "\"pid\": ${SERVER_PID:-null}" "$STATE_FILE"; then
    rm -f -- "$STATE_FILE"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

"$PYTHON" - "$STATE_FILE" "$POLICY" "$POLICY_TYPE" "$PLAUSIBILITY_CONTRACT" "$TRACE_DIR" "$HOST" "$PORT" "$FPS" <<'PY'
import json, os, socket, sys, tempfile, time
(
    path,
    policy,
    policy_type,
    plausibility_contract,
    trace_dir,
    host,
    port,
    fps,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "pid": os.getppid(),
    "host": socket.gethostname(),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "policy": policy,
    "policy_type": policy_type,
    "plausibility_contract": plausibility_contract or None,
    "trace_dir": trace_dir or None,
    "listen": f"{host}:{port}",
    "fps": int(fps),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix="current-server.")
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY

echo "server checkpoint: $POLICY"
echo "server policy type: $POLICY_TYPE"
[ -z "$PLAUSIBILITY_CONTRACT" ] || echo "server plausibility contract: $PLAUSIBILITY_CONTRACT"
[ -z "$TRACE_DIR" ] || echo "server inference trace: $TRACE_DIR"
echo "server listen: $HOST:$PORT fps=$FPS"
"$PYTHON" -m lerobot.async_inference.policy_server \
  --host="$HOST" --port="$PORT" --fps="$FPS" &
SERVER_PID=$!

# Refresh the provisional parent PID with the real child PID.
"$PYTHON" - "$STATE_FILE" "$SERVER_PID" <<'PY'
import json, os, sys, tempfile
path, pid = sys.argv[1], int(sys.argv[2])
payload = json.load(open(path))
payload["pid"] = pid
fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix="current-server.")
with os.fdopen(fd, "w") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
os.replace(temporary, path)
PY
wait "$SERVER_PID"
rm -f -- "$STATE_FILE"
SERVER_PID=""
