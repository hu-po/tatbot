#!/usr/bin/env bash
set -euo pipefail

# Unified installer helper.
# Usage:
#   sudo bash scripts/monitor/install.sh            # auto-detect role(s) by user/host
#   sudo bash scripts/monitor/install.sh node rpi   # explicit roles (optional)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
INVENTORY="${ROOT_DIR}/config/monitoring/inventory.yml"

require_root() {
  if [[ ${EUID:-0} -ne 0 ]]; then
    echo "Please run as root (sudo)." >&2; exit 1
  fi
}

node_exporter_install() {
  bash "${ROOT_DIR}/scripts/monitor/install_node_exporter.sh"
}

rpi_exporter_install() {
  local host unit_src arch
  host="$(hostname)"
  unit_src="${ROOT_DIR}/config/monitoring/exporters/${host}/rpi_exporter.service"
  if [[ ! -f "$unit_src" ]]; then echo "Missing unit file: $unit_src" >&2; return 1; fi
  arch=$(uname -m)
  case "$arch" in aarch64|arm64) ;; *) echo "rpi_exporter intended for ARM64 hosts." >&2; return 1 ;; esac
  cd /tmp
  curl -fsSL -o rpi_exporter.tar.gz "https://github.com/lukasmalkmus/rpi_exporter/releases/download/v0.4.0/rpi_exporter-0.4.0.linux-arm64.tar.gz"
  tar -xzf rpi_exporter.tar.gz
  install -m 0755 rpi_exporter-*/rpi_exporter /usr/local/bin/rpi_exporter
  install -m 0644 "$unit_src" /etc/systemd/system/rpi_exporter.service
  systemctl daemon-reload
  systemctl enable --now rpi_exporter
}

jetson_exporter_install() {
  local unit_src
  unit_src="${ROOT_DIR}/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service"
  if [[ ! -f "$unit_src" ]]; then echo "Missing unit file: $unit_src" >&2; return 1; fi
  pip3 install -U pip
  pip3 install "jetson-stats==$(awk -F '"' '/jetson_stats:/ {print $2}' "$INVENTORY" 2>/dev/null || echo 4.3.2)" \
                "jetson-stats-node-exporter==$(awk -F '"' '/jetson_stats_exporter:/ {print $2}' "$INVENTORY" 2>/dev/null || echo 0.1.2)"
  install -m 0644 "$unit_src" /etc/systemd/system/jetson-stats-node-exporter.service
  systemctl daemon-reload
  systemctl enable --now jetson-stats-node-exporter
}

dcgm_exporter_install() {
  # Requires docker + NVIDIA driver
  if ! command -v docker >/dev/null 2>&1; then echo "Docker not found; skipping DCGM exporter." >&2; return 0; fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo "nvidia-smi not found; skipping DCGM exporter." >&2; return 0; fi
  local image tag name listen
  image="nvidia/dcgm-exporter"
  tag="$(awk -F '"' '/dcgm_exporter:/ {print $2}' "$INVENTORY" 2>/dev/null || echo 4.4.0-4.5.0-ubi9)"
  name="dcgm-exporter"
  listen=":9400"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null || true
  else
    docker run -d --restart=always --gpus all --cap-add SYS_ADMIN --net host \
      --name "$name" -e DCGM_EXPORTER_LISTEN="$listen" "$image:$tag" || true
  fi
}

intel_gpu_exporter_install() {
  # Requires docker and /dev/dri
  if ! command -v docker >/dev/null 2>&1; then echo "Docker not found; skipping Intel GPU exporter." >&2; return 0; fi
  if [[ ! -d /dev/dri ]]; then echo "/dev/dri not present; skipping Intel GPU exporter." >&2; return 0; fi
  local image tag name
  image="$(awk -F '"' '/intel_gpu_exporter_image:/ {print $2}' "$INVENTORY" 2>/dev/null || echo restreamio/intel-prometheus)"
  tag="$(awk -F '"' '/intel_gpu_exporter_tag:/ {print $2}' "$INVENTORY" 2>/dev/null || echo latest)"
  name="intel-gpu-exporter"
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
    docker start "$name" >/dev/null || true
  else
    docker run -d --restart=always --net host --name "$name" --privileged \
      -v /sys:/sys:ro -v /dev/dri:/dev/dri "$image:$tag" || true
  fi
}

detect_actions() {
  local actor host
  actor="${SUDO_USER:-${USER:-unknown}}"
  host="$(hostname)"
  # Prefer $SUDO_USER if set, else hostname; both appear to match node names in this fleet.
  local key="$actor"
  [[ "$key" =~ ^(unknown|root)$ ]] && key="$host"
  case "$key" in
    rpi1|rpi2)
      echo "node rpi" ;;
    ojo)
      echo "jetson" ;;
    ook|oop)
      echo "node dcgm" ;;
    hog)
      echo "node intel" ;;
    eek)
      echo "node" ;;
    *)
      # Default to node exporter only
      echo "node" ;;
  esac
}

run_action() {
  case "$1" in
    node) node_exporter_install ;;
    rpi) rpi_exporter_install ;;
    jetson) jetson_exporter_install ;;
    dcgm) dcgm_exporter_install ;;
    intel) intel_gpu_exporter_install ;;
    *) echo "Unknown action: $1" >&2; return 1 ;;
  esac
}

require_root

if [[ $# -eq 0 ]]; then
  actions=( $(detect_actions) )
else
  actions=( "$@" )
fi

echo "Detected install actions: ${actions[*]}"
for a in "${actions[@]}"; do
  run_action "$a"
done

echo "Install complete."
