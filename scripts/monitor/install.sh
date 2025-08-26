#!/usr/bin/env bash
set -euo pipefail

# Install and enable Prometheus Node Exporter on the current host.
# Detects architecture, downloads pinned version from inventory, installs systemd unit from repo.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
INVENTORY="${ROOT_DIR}/config/monitoring/inventory.yml"

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Need curl or wget to download node_exporter." >&2
  exit 1
fi

VERSION="1.9.1"
if [[ -f "$INVENTORY" ]]; then
  ver=$(awk -F '"' '/node_exporter:/ {print $2; exit}' "$INVENTORY" || true)
  [[ -n "${ver:-}" ]] && VERSION="$ver"
fi

ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) TARBALL="node_exporter-${VERSION}.linux-amd64.tar.gz" ; DIR="node_exporter-${VERSION}.linux-amd64" ;;
  aarch64|arm64) TARBALL="node_exporter-${VERSION}.linux-arm64.tar.gz" ; DIR="node_exporter-${VERSION}.linux-arm64" ;;
  armv7l|armv7) TARBALL="node_exporter-${VERSION}.linux-armv7.tar.gz" ; DIR="node_exporter-${VERSION}.linux-armv7" ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

TMPDIR=$(mktemp -d)
cd "$TMPDIR"
URL="https://github.com/prometheus/node_exporter/releases/download/v${VERSION}/${TARBALL}"
echo "Downloading $URL ..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$URL" -o "$TARBALL"
else
  wget -q "$URL" -O "$TARBALL"
fi
tar -xzf "$TARBALL"

id -u nodeexp >/dev/null 2>&1 || useradd --no-create-home --shell /usr/sbin/nologin nodeexp
install -o nodeexp -g nodeexp -m 0755 "$DIR/node_exporter" /usr/local/bin/node_exporter

UNIT_SRC="${ROOT_DIR}/config/monitoring/exporters/$(hostname)/node_exporter.service"
if [[ ! -f "$UNIT_SRC" ]]; then
  echo "Missing unit file: $UNIT_SRC" >&2
  exit 1
fi
install -m 0644 "$UNIT_SRC" /etc/systemd/system/node_exporter.service
systemctl daemon-reload
systemctl enable --now node_exporter

echo "Node Exporter installed. Verifying..."
sleep 1
systemctl --no-pager --full status node_exporter | sed -n '1,15p'
curl -s --max-time 3 http://localhost:9100/metrics | grep -m1 '^node_cpu_seconds_total' || true
echo "Done."

