#!/usr/bin/env bash

# Verify Redis-backed state server end-to-end via MCP tools and Redis CLI.
#
# Checks:
# 1) Redis ping
# 2) MCP resource state://status on eek/hog/ook (redis connected)
# 3) Direct Redis set/get round-trip (no MCP state tools)
#
# Usage:
#   bash scripts/verify_state_server.sh
#   EEK_HOST=eek HOG_HOST=hog OOK_HOST=ook MCP_PORT=8000 REDIS_HOST=eek REDIS_PORT=6379 bash scripts/verify_state_server.sh

set -euo pipefail

EEK_HOST=${EEK_HOST:-eek}
HOG_HOST=${HOG_HOST:-hog}
OOK_HOST=${OOK_HOST:-ook}
MCP_PORT=${MCP_PORT:-8000}
REDIS_HOST=${REDIS_HOST:-eek}
REDIS_PORT=${REDIS_PORT:-6379}
VIZ_CHECK=${VIZ_CHECK:-false}
VIZ_NODE=${VIZ_NODE:-$OOK_HOST}
VIZ_SCENE=${VIZ_SCENE:-default}
VIZ_SERVER_NAME=${VIZ_SERVER_NAME:-stroke_viz}

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "✅ $1"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { echo "❌ $1"; FAIL_COUNT=$((FAIL_COUNT+1)); }

req() {
  local host="$1"; shift
  local payload="$1"; shift
  curl -sS "http://${host}:${MCP_PORT}/mcp/" \
    -H "Content-Type: application/json" \
    -d "$payload"
}

echo "--- Verifying Redis (${REDIS_HOST}:${REDIS_PORT}) ---"
if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping | grep -q PONG; then
    pass "Redis ping responded (PONG)"
  else
    fail "Redis ping failed"
  fi
else
  echo "ℹ️ redis-cli not found; skipping direct Redis ping"
fi

echo "--- Checking MCP state status resource on nodes ---"
for NODE in "$EEK_HOST" "$HOG_HOST" "$OOK_HOST"; do
  RESP=$(req "$NODE" '{"jsonrpc":"2.0","id":"sys","method":"resources/read","params":{"uri":"state://status"}}') || RESP=""
  if echo "$RESP" | grep -qi 'redis connected: true'; then
    pass "${NODE}: state://status -> redis_connected=true"
  else
    echo "$RESP" | sed 's/.\{0,0\}/  /'
    fail "${NODE}: state://status failed or redis_connected!=true"
  fi
done

echo "--- Skipping MCP set_state/get_state (removed); verifying Redis directly if available ---"
KEY="tatbot:test:$(date +%s)"
if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" set "$KEY" '{"hello":"world"}' EX 60 >/dev/null; then
    pass "redis-cli set succeeded"
    VALUE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" get "$KEY" || true)
    if echo "$VALUE" | grep -q 'hello' && echo "$VALUE" | grep -q 'world'; then
      pass "redis-cli get returned expected payload"
    else
      fail "redis-cli get unexpected payload: $VALUE"
    fi
  else
    fail "redis-cli set failed"
  fi
else
  echo "ℹ️ redis-cli not found; skipping direct set/get"
fi

echo "--- Read MCP resource state://status (via eek) ---"
RES_RESP=$(req "$EEK_HOST" '{"jsonrpc":"2.0","id":"r1","method":"resources/read","params":{"uri":"state://status"}}') || RES_RESP=""
if echo "$RES_RESP" | grep -qi 'redis connected'; then
  pass "resources/read state://status responded"
else
  echo "$RES_RESP" | sed 's/.\{0,0\}/  /'
  fail "resources/read state://status failed"
fi

if [ "$VIZ_CHECK" = "true" ]; then
  echo "--- Viz server start/status/stop on ${VIZ_NODE} ---"
  START_PAYLOAD=$(cat <<JSON
{"jsonrpc":"2.0","id":"vz1","method":"tools/call","params":{"name":"start_stroke_viz","arguments":{"scene":"${VIZ_SCENE}","enable_state_sync":true}}}
JSON
)
  START_RESP=$(req "$VIZ_NODE" "$START_PAYLOAD") || START_RESP=""
  if echo "$START_RESP" | grep -q '"success"\s*:\s*true' && echo "$START_RESP" | grep -q '"running"\s*:\s*true'; then
    pass "start_stroke_viz reported running"
  else
    echo "$START_RESP" | sed 's/.\{0,0\}/  /'
    fail "start_stroke_viz failed"
  fi

  STATUS_PAYLOAD='{"jsonrpc":"2.0","id":"vz2","method":"tools/call","params":{"name":"status_viz_server","arguments":{"server_name":"'"$VIZ_SERVER_NAME"'"}}}'
  STATUS_RESP=$(req "$VIZ_NODE" "$STATUS_PAYLOAD") || STATUS_RESP=""
  if echo "$STATUS_RESP" | grep -q '"running"\s*:\s*true'; then
    pass "status_viz_server confirms running"
  else
    echo "$STATUS_RESP" | sed 's/.\{0,0\}/  /'
    fail "status_viz_server did not confirm running"
  fi

  STOP_PAYLOAD='{"jsonrpc":"2.0","id":"vz3","method":"tools/call","params":{"name":"stop_viz_server","arguments":{"server_name":"'"$VIZ_SERVER_NAME"'"}}}'
  STOP_RESP=$(req "$VIZ_NODE" "$STOP_PAYLOAD") || STOP_RESP=""
  if echo "$STOP_RESP" | grep -q '"success"\s*:\s*true'; then
    pass "stop_viz_server succeeded"
  else
    echo "$STOP_RESP" | sed 's/.\{0,0\}/  /'
    fail "stop_viz_server failed"
  fi
fi

echo "--- Summary ---"
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
  exit 1
fi
