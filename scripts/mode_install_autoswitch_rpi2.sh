#!/usr/bin/env bash
set -euo pipefail

# Install a systemd service on rpi2 to auto-switch between Edge and Home.
# Default on boot: Edge (via symlink). The service flips to Home when internet is reachable.

RPI2_HOST=${RPI2_HOST:-rpi2}
SERVICE_NAME=tatbot-mode-auto.service

read -r -d '' UNIT <<'EOF'
[Unit]
Description=Tatbot Mode Auto Switcher (Edge/Home)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rpi2
WorkingDirectory=/home/rpi2/tatbot
ExecStart=/bin/bash -lc 'source /home/rpi2/tatbot/scripts/setup_env.sh && /home/rpi2/tatbot/scripts/mode_auto.sh'
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

echo "Installing ${SERVICE_NAME} on ${RPI2_HOST}..."

ssh -o BatchMode=yes -o ConnectTimeout=5 "$RPI2_HOST" "set -e; cat >/tmp/${SERVICE_NAME} <<'UNIT'
$UNIT
UNIT
sudo mv /tmp/${SERVICE_NAME} /etc/systemd/system/${SERVICE_NAME}
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}
systemctl status ${SERVICE_NAME} --no-pager -l | sed -n '1,80p'" || {
  echo "Failed to install service on ${RPI2_HOST}" >&2
  exit 1
}

echo "✅ Installed and started ${SERVICE_NAME} on ${RPI2_HOST}."


