#!/usr/bin/env bash
set -euo pipefail

# Centralized mode toggle runner.
# Usage:
#   scripts/toggle_mode.sh status|home|edge
# - status: show rpi2 dnsmasq active profile and basic checks
# - home:   switch rpi2 to home mode (DNS forwarder), no DHCP
# - edge:   switch to edge mode (authoritative DNS + DHCP)

MODE=${1:-status}
RPI2_HOST=${RPI2_HOST:-rpi2}

echo "🔀 Toggle mode -> ${MODE} (DNS node: ${RPI2_HOST})"

run_on_rpi2() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${RPI2_HOST}" "$@"
}

case "${MODE}" in
  status)
    run_on_rpi2 "set -e; echo '--- dnsmasq active ---'; readlink -f /etc/dnsmasq.d/active.conf || echo no-symlink; echo '--- service ---'; systemctl is-active dnsmasq; echo '--- quick DNS ---'; dig +short ook.tatbot.lan @127.0.0.1 || true" || true
    ;;
  home)
    uv run -q -m tatbot.utils.mode_toggle --mode home || { echo "Failed to switch to home" >&2; exit 1; }
    run_on_rpi2 "sudo systemctl reload dnsmasq || sudo systemctl restart dnsmasq" || true
    ;;
  edge)
    uv run -q -m tatbot.utils.mode_toggle --mode edge || { echo "Failed to switch to edge" >&2; exit 1; }
    run_on_rpi2 "sudo systemctl reload dnsmasq || sudo systemctl restart dnsmasq" || true
    ;;
  *)
    echo "Unknown mode: ${MODE} (valid: status, home, edge)" >&2; exit 2
    ;;
esac

echo "✅ Done."


