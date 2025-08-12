#!/usr/bin/env bash
set -euo pipefail

# Auto-switch rpi2 between two modes only: edge and home
# - edge: authoritative DNS + optional DHCP (as configured by mode-edge.conf)
# - home: DNS forwarder to home router
# Decision rule: if internet is reachable, prefer home; otherwise edge.

DNSMASQ_ACTIVE_LINK=/etc/dnsmasq.d/active.conf
CHECK_HOST=${CHECK_HOST:-8.8.8.8}
SLEEP_SECS=${SLEEP_SECS:-20}

get_current_mode() {
  local link
  link=$(readlink -f "${DNSMASQ_ACTIVE_LINK}" 2>/dev/null || true)
  if [[ "$link" == *mode-home.conf ]]; then echo home; return; fi
  if [[ "$link" == *mode-edge.conf ]]; then echo edge; return; fi
  echo unknown
}

set_mode() {
  local target=$1
  echo "[mode_auto] switching to ${target}"
  # Use uv runner within repo if available; fall back to system if needed
  if command -v uv >/dev/null 2>&1; then
    uv run -q -m tatbot.utils.mode_toggle --mode "${target}" || return 1
  else
    python3 -m tatbot.utils.mode_toggle --mode "${target}" || return 1
  fi
}

internet_ok() {
  ping -c1 -W2 "${CHECK_HOST}" >/dev/null 2>&1 && return 0
  return 1
}

echo "[mode_auto] starting (check=${CHECK_HOST}, interval=${SLEEP_SECS}s)"

while true; do
  current=$(get_current_mode)
  target=edge
  if internet_ok; then target=home; fi

  if [[ "$current" != "$target" ]]; then
    set_mode "$target" || true
  fi

  sleep "${SLEEP_SECS}"
done


