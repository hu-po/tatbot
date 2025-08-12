#!/usr/bin/env bash
set -euo pipefail

# One-time per-node setup for edge-first operation:
# - Configure DNS to rpi2 (192.168.1.99) on all connections
# - Prefer Wi-Fi for internet when available (home mode)
# - Fall back to edge mode when Wi-Fi unavailable

RPI2_DNS_IP=${RPI2_DNS_IP:-192.168.1.99}

echo "🔧 Node bootstrap for edge-first operation (DNS -> ${RPI2_DNS_IP})"

if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli not found; please install NetworkManager and re-run" >&2
  exit 1
fi

# Discover active connections
ACTIVE=$(nmcli -t -f NAME,TYPE,DEVICE connection show --active | sed '/^lo:/d' || true)
WIFI_CONN=$(echo "$ACTIVE" | awk -F: '$2 ~ /wifi|802-11-wireless/ {print $1; exit}')
WIFI_DEV=$(echo  "$ACTIVE" | awk -F: '$2 ~ /wifi|802-11-wireless/ {print $3; exit}')
ETH_CONN=$(echo  "$ACTIVE" | awk -F: '$2 ~ /ethernet|802-3-ethernet/ {print $1; exit}')
ETH_DEV=$(echo   "$ACTIVE" | awk -F: '$2 ~ /ethernet|802-3-ethernet/ {print $3; exit}')

echo "Detected connections:"
echo "  Wi‑Fi:     ${WIFI_CONN:-<none>} (${WIFI_DEV:--})"
echo "  Ethernet:  ${ETH_CONN:-<none>} (${ETH_DEV:--})"

SUDO=sudo
if [[ $(id -u) -eq 0 ]]; then SUDO=""; fi

changed=false

if [[ -n "${WIFI_CONN}" ]]; then
  echo "Configuring Wi‑Fi connection '${WIFI_CONN}' (home mode when available)"
  $SUDO nmcli connection modify "${WIFI_CONN}" \
    ipv4.dns "${RPI2_DNS_IP}" ipv4.ignore-auto-dns yes \
    ipv4.never-default no ipv4.route-metric 600 || true
  changed=true
fi

if [[ -n "${ETH_CONN}" ]]; then
  echo "Configuring Ethernet connection '${ETH_CONN}' (edge mode fallback)"
  $SUDO nmcli connection modify "${ETH_CONN}" \
    ipv4.dns "${RPI2_DNS_IP}" ipv4.ignore-auto-dns yes \
    ipv4.never-default yes || true
  changed=true
fi

if [[ "$changed" == true ]]; then
  echo "Reapplying connections to pick up changes"
  [[ -n "${ETH_DEV}"  ]] && $SUDO nmcli device reapply "${ETH_DEV}"  || true
  [[ -n "${WIFI_CONN}" ]] && { $SUDO nmcli connection down "${WIFI_CONN}" || true; $SUDO nmcli connection up "${WIFI_CONN}"; } || true
fi

echo "Verifying DNS and routes"
echo "--- default route ---"; ip route show default || true
if command -v resolvectl >/dev/null 2>&1; then
  echo "--- resolvectl ---"; resolvectl status | sed -n '1,120p'
else
  echo "--- /etc/resolv.conf ---"; cat /etc/resolv.conf
fi

echo "--- tatbot DNS test ---"; (command -v dig >/dev/null 2>&1 && dig +short A eek.tatbot.lan @"${RPI2_DNS_IP}") || nslookup eek.tatbot.lan "${RPI2_DNS_IP}" || true
echo "Done."


