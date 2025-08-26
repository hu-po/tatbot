---
summary: Prometheus + Grafana for edge nodes with pinned exporters
tags: [monitoring, prometheus, grafana]
updated: 2025-08-26
audience: [dev, agent]
---

# 📊 Monitoring

A single wallboard (Prometheus + Grafana) on **rpi1** (Raspberry Pi 5) showing live CPU/memory/disk/net and GPU metrics across all nodes with minimal overhead, using **only FOSS** and unified fleet overview dashboard.

---

## Executive Summary

We standardize on **Prometheus + Grafana** with per-node exporters:

- **Host metrics (all nodes):** Prometheus **Node Exporter** (tiny static binary).
- **NVIDIA dGPU (ook/oop):** **DCGM Exporter** (official NVIDIA).
- **Jetson (ojo / AGX Orin):** **jetson-stats (jtop)–based exporter** that exposes system + GPU.
- **Intel Arc/iGPU (hog):** Intel GPU exporter that parses `intel_gpu_top -J`.
- **Raspberry Pi SoC (rpi1, rpi2):** lightweight **rpi_exporter** for temps/voltages/clock.

Prometheus + Grafana run centrally on **eek** (System76 Meerkat). **rpi1** displays Grafana in **kiosk mode**.

---

## Fleet & Roles

| Host | Hardware | Role(s) | Exporters |
|---|---|---|---|
| **eek** | System76 Meerkat (i5‑1340P) | **Prometheus + Grafana server**; NFS | `node_exporter` |
| **ook** | Acer Nitro V 15 (i7‑13620H + **RTX 4050**) | Wi‑Fi NAT; GPU compute | `node_exporter`, **DCGM exporter** |
| **oop** | Desktop PC (RTX 3090) | GPU compute; development | `node_exporter`, **DCGM exporter** |
| **hog** | GEEKOM GT1 Mega (Core Ultra 9 + **Intel Arc**) | Robot control, RealSense | `node_exporter`, **Intel GPU exporter** |
| **ojo** | **Jetson AGX Orin** | Agent inference | **jetson‑stats node exporter** (includes CPU/mem/GPU) |
| **rpi1** | Raspberry Pi 5 (8GB) | **Wallboard**; app frontend | `node_exporter`, **rpi_exporter**; **Grafana kiosk** client |
| **rpi2** | Raspberry Pi 5 (8GB) | DNS/DHCP | `node_exporter`, **rpi_exporter** |

> **Jetson note:** the selected exporter exposes **both** host and GPU metrics on port **:9100**; do **not** run a separate node_exporter there to avoid port conflicts.

---

## High‑Level Architecture

```
                           [ rpi1 ]  ──> Chromium/Grafana Kiosk (read‑only)
                                        http://eek:3000/d/tatbot-compute/tatbot-compute?kiosk=tv&refresh=5s

                ┌──────────────────────── [ eek ] ──────────────────────────┐
                │  Prometheus (9090)  ← scrape 15s   Grafana OSS (3000)     │
                │  retention: 7d or 2GB (whichever first)                   │
                └───────────────────────────────────────────────────────────┘
                         ▲           ▲           ▲           ▲           ▲
                         │           │           │           │           │
                node_exporter   DCGM exporter  Intel GPU   rpi_exporter  jetson-stats
                    :9100          :9400         :8080       :9110         :9100
                 [eek/ook/hog]    [ook/oop]      [hog]     [rpi1/rpi2]      [ojo]
```

---

## Design Decisions

1. **Centralize Prometheus+Grafana on `eek`** to keep compute nodes light.
2. **Keep exporters tiny**:
   - Node Exporter is a **single static binary**; use systemd (no container overhead).
   - DCGM exporter runs as a small container only on the NVIDIA host.
   - Jetson uses a **jtop‑based** exporter (CPU/mem/temps/GPU) with one port.
   - Intel GPU exporters wrap `intel_gpu_top -J` and expose Prometheus metrics.
   - rpi_exporter reads VC hardware directly (no `vcgencmd`), very low overhead.
3. **Modest scrape interval**: `global.scrape_interval: 15s`.
4. **Bounded storage**: `--storage.tsdb.retention.time: 7d` **and** `--storage.tsdb.retention.size: 2GB`.
5. **Version pinning** everywhere (Docker tags, PyPI packages, binary releases).

---

## Inventory (Single Source of Truth)

Inventory lives at `~/tatbot/config/monitoring/inventory.yml` (versions, scrape interval, and nodes). IPs should match `src/conf/nodes.yaml`. Keep them in sync and regenerate Prometheus config (see "Generate Prometheus Config").

> **Why versions here?** This makes `inventory.yml` the **true** source of truth for **both topology and versions**. The agent can template Docker tags, download URLs, and PyPI requirements from these fields to produce deterministic configs and installers.

---

## Per‑Node Installation

> The CLI agent commits config/service files; a human runs the commands below (root ssh).
> Replace versions with those from **inventory.yml** if you change them.

Prereqs by node
- eek: Docker Engine + compose plugin (section 7.1).
- ook (RTX 4050): Docker Engine + NVIDIA Container Toolkit (section 6.2); NVIDIA driver working (`nvidia-smi`).
- hog (Intel GPU): Docker Engine (section 6.4).
- ojo (Jetson): Python3/pip; install jetson-stats + exporter (section 6.3).
- rpi1/rpi2: None beyond systemd and curl.

### Common host metrics (node_exporter) — eek, ook, hog, rpi1, rpi2
Quick installer (recommended):
```bash
cd ~/tatbot && git pull &&sudo bash scripts/monitor/install.sh
```

Manual install (x86_64) — On each host (repo at `~/tatbot`), download and install Node Exporter v1.9.1:

**eek, ook, hog** (Intel/AMD x86_64):
```bash
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-amd64.tar.gz
tar -xzf node_exporter-1.9.1.linux-amd64.tar.gz -C /tmp
sudo useradd --no-create-home --shell /usr/sbin/nologin nodeexp || true
sudo install -o nodeexp -g nodeexp -m 0755 /tmp/node_exporter-1.9.1.linux-amd64/node_exporter /usr/local/bin/node_exporter
sudo install -o root -g root -m 0644 ~/tatbot/config/monitoring/exporters/$(hostname)/node_exporter.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now node_exporter
curl -sS --no-progress-meter http://localhost:9100/metrics | head -n 20
```

**rpi1, rpi2** (Raspberry Pi 5 ARM64):
```bash
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-arm64.tar.gz
tar -xzf node_exporter-1.9.1.linux-arm64.tar.gz -C /tmp
sudo useradd --no-create-home --shell /usr/sbin/nologin nodeexp || true
sudo install -o nodeexp -g nodeexp -m 0755 /tmp/node_exporter-1.9.1.linux-arm64/node_exporter /usr/local/bin/node_exporter
sudo install -o root -g root -m 0644 ~/tatbot/config/monitoring/exporters/$(hostname)/node_exporter.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now node_exporter
curl -sS --no-progress-meter http://localhost:9100/metrics | head -n 20
```

> Do not install node_exporter on `ojo` (Jetson). It runs `jetson-stats-node-exporter` on :9100 (see "Jetson (ojo): jtop/jetson‑stats Exporter").

### NVIDIA dGPU (ook/oop): DCGM Exporter (container, **pinned tag**)
Prereqs (Docker engine + NVIDIA Container Toolkit on Ubuntu 24.04):
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker
sudo systemctl enable --now docker
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Test GPU access inside container (optional):
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Run DCGM exporter (pinned tag from inventory) on each NVIDIA host (ook, oop):
```bash
docker run -d --restart=always --gpus all --cap-add SYS_ADMIN --net host \
  --name dcgm-exporter -e DCGM_EXPORTER_LISTEN=":9400" \
  nvidia/dcgm-exporter:4.4.0-4.5.0-ubi9
```

Or use the systemd unit in the repo: `~/tatbot/config/monitoring/exporters/ook/dcgm-exporter.service` (for ook) or `~/tatbot/config/monitoring/exporters/oop/dcgm-exporter.service` (for oop), then:
`sudo systemctl daemon-reload && sudo systemctl enable --now dcgm-exporter`

Verify (ook): `curl -sS --no-progress-meter http://192.168.1.90:9400/metrics | head -n 20`

Verify (oop): `curl -sS --no-progress-meter http://192.168.1.51:9400/metrics | head -n 20`

### Jetson (ojo): jtop/jetson‑stats Exporter (**no jtop.service dependency required**)
```bash
sudo -H pip3 install "jetson-stats==4.3.2"
sudo -H pip3 install "jetson-stats-node-exporter==0.1.2"
sudo install -m 0644 ~/tatbot/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now jetson-stats-node-exporter
curl -sS --no-progress-meter http://192.168.1.96:9100/metrics | head -n 20
```

> We **removed** the `Requires=jtop.service` dependency. The exporter uses the **Python API** from jetson‑stats directly and does not require the `jtop` systemd service to be running. See "Systemd Unit Files (Exporters)" for the corrected unit file.

### Intel Arc/iGPU (hog): Intel GPU Exporter
**hog** needs Node Exporter (section 6.1) PLUS Intel GPU monitoring:

First install Docker if not present:
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group changes
```

Then run Intel GPU exporter (replace the tag with your pinned tag from inventory):
```bash
docker run -d --restart=always --net host --name intel-gpu-exporter --privileged \
  -v /sys:/sys:ro -v /dev/dri:/dev/dri \
  restreamio/intel-prometheus:latest  # replace 'latest' with a pinned tag
curl -sS --no-progress-meter http://192.168.1.88:8080/metrics | head -n 20
```

### Raspberry Pi SoC telemetry — rpi1, rpi2
Download and install rpi_exporter for ARM64:
```bash
cd /tmp
wget https://github.com/lukasmalkmus/rpi_exporter/releases/download/v0.4.0/rpi_exporter-0.4.0.linux-arm64.tar.gz
tar -xzf rpi_exporter-0.4.0.linux-arm64.tar.gz -C /tmp
sudo install -o root -g root -m 0755 /tmp/rpi_exporter /usr/local/bin/rpi_exporter
sudo install -m 0644 ~/tatbot/config/monitoring/exporters/$(hostname)/rpi_exporter.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now rpi_exporter
curl -sS --no-progress-meter http://$(hostname):9110/metrics | head -n 20
```

> rpi_exporter flag change: service files in this repo now use `--web.listen-address=:9110` (older docs sometimes show `-addr`). If you had a prior unit with `-addr`, update it to `--web.listen-address` or copy the unit from `config/monitoring/exporters/<host>/rpi_exporter.service`.

---

## Prometheus & Grafana on eek (Pinned Images)

### Docker Compose Setup (eek)
Install Docker with modern compose plugin:
```bash
# Remove legacy docker-compose if installed
sudo apt-get remove -y docker-compose || true

# Add Docker's official repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && newgrp docker

# Verify
docker compose version
```

Start Prometheus + Grafana:
```bash
cd ~/tatbot && make -C config/monitoring up
```

### Prometheus Config
File: `~/tatbot/config/monitoring/prometheus/prometheus.yml` (generated from inventory). Alert rules: `~/tatbot/config/monitoring/prometheus/rules/`.

### (Optional) Starter Alerts
See: `~/tatbot/config/monitoring/prometheus/rules/edge.rules.yml`.

---

## Grafana Provisioning & Dashboards (Tatbot Compute)

### Provision Prometheus Datasource
File: `~/tatbot/config/monitoring/grafana/provisioning/datasources/prometheus.yaml`.

### Provision Dashboards
File: `~/tatbot/config/monitoring/grafana/provisioning/dashboards/dashboards.yaml`.

### Included dashboards (JSON files under `grafana/dashboards/`)
- **Tatbot Compute** — `tatbot-compute.json` (**installed in repo**, uid=tatbot-compute)  

> The **Tatbot Compute** dashboard is the default kiosk target and summarizes CPU, Memory, Disk, Network, and GPU across all hosts. You can drill into the detailed dashboards when needed.

### Tatbot Compute Dashboard
Installed at `~/tatbot/config/monitoring/grafana/dashboards/tatbot-compute.json` (uid=tatbot-compute). Intel exporter metric names may vary (`igpu_*`).

<!-- Dashboard JSON lives in the repo; see path above. -->

---

## rpi1 Wallboard (Kiosk)

### Quick Start: Monitoring Kiosk Script
**Run on rpi1** (wallboard display node) to launch the monitoring dashboard in kiosk mode:
```bash
# Start monitoring kiosk (default: eek:3000, 5s refresh)
cd ~/tatbot && bash scripts/monitor/kiosk.sh

# Custom refresh interval  
cd ~/tatbot && bash scripts/monitor/kiosk.sh eek 10

# Specific IP address
cd ~/tatbot && bash scripts/monitor/kiosk.sh 192.168.1.97
```

**When to run:** Once per boot, or when you want to restart the wallboard display.

### Manual Kiosk URL
Point Chromium (or `grafana-kiosk`) at:
```
http://eek:3000/d/tatbot-compute/tatbot-compute?kiosk=tv&refresh=5s
```

### Optional: grafana-kiosk systemd Service
Unit file: `~/tatbot/config/monitoring/exporters/rpi1/grafana-kiosk.service`.
Enable: `sudo systemctl daemon-reload && sudo systemctl enable --now grafana-kiosk`

---

## Systemd Unit Files (Exporters)

- Node Exporter: `~/tatbot/config/monitoring/exporters/<host>/node_exporter.service`
- DCGM exporter (docker): `~/tatbot/config/monitoring/exporters/ook/dcgm-exporter.service`, `~/tatbot/config/monitoring/exporters/oop/dcgm-exporter.service`
- Jetson exporter: `~/tatbot/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service`
- Intel GPU exporter (docker): `~/tatbot/config/monitoring/exporters/hog/intel-gpu-exporter.service`
- rpi_exporter: `~/tatbot/config/monitoring/exporters/rpi{1,2}/rpi_exporter.service`
- Grafana kiosk (optional): `~/tatbot/config/monitoring/exporters/rpi1/grafana-kiosk.service`

---

## Repo Layout

```
config/monitoring/
├─ inventory.yml
├─ compose/
│  └─ docker-compose.yml
├─ prometheus/
│  ├─ prometheus.yml
│  └─ rules/
│     └─ edge.rules.yml
├─ grafana/
│  ├─ provisioning/
│  │  ├─ datasources/prometheus.yaml
│  │  └─ dashboards/dashboards.yaml
│  └─ dashboards/
│     └─ tatbot-compute.json
└─ exporters/
   ├─ eek/node_exporter.service
   ├─ ook/{node_exporter.service,dcgm-exporter.service}
   ├─ oop/{node_exporter.service,dcgm-exporter.service}
   ├─ hog/{node_exporter.service,intel-gpu-exporter.service}
   ├─ ojo/jetson-stats-node-exporter.service
   └─ rpi{1,2}/{node_exporter.service,rpi_exporter.service}
scripts/
├─ monitor/
│  ├─ server.sh           # Start/verify monitoring stack on eek
│  ├─ kiosk.sh            # Start kiosk display on rpi1
│  └─ clean.sh            # Clean monitoring volumes and cache
src/tatbot/utils/
└─ gen_prom_config.py      # Generate Prometheus config from inventory
```

## Generate Prometheus Config

Prometheus targets are generated from `inventory.yml`.

- From repo root: `python3 src/tatbot/utils/gen_prom_config.py`
- Or: `make -C config/monitoring gen-prom`

Output: `config/monitoring/prometheus/prometheus.yml`.
After changes, restart the stack on eek:
- `make -C config/monitoring restart`

## Monitoring Server Management

### Single Entry Point Script

**Run on eek** to start/verify the complete monitoring system:

```bash
# Start and verify monitoring system
cd ~/tatbot && ./scripts/monitor/server.sh

# Restart services and verify
cd ~/tatbot && ./scripts/monitor/server.sh --restart
```

### Cache Cleanup Script

**Run on any node** to clean all cached monitoring data:

```bash
# Clean cache (works on any node)
cd ~/tatbot && ./scripts/monitor/clean.sh

# For complete reset: clean + restart
cd ~/tatbot && ./scripts/monitor/clean.sh && ./scripts/monitor/server.sh
```

**server.sh** features:
- Verifies it's running on eek (monitoring server host)
- Optionally restarts Prometheus + Grafana containers  
- Performs comprehensive diagnostics on all nodes
- Tests connectivity, services, and HTTP endpoints
- Provides installation commands for missing exporters
- Shows detailed Prometheus target status

**clean.sh** features:
- Stops and removes Docker containers/volumes (on eek)
- Clears browser cache and temp profiles
- Removes log files and temporary data
- Works on any node (cleans local cache)
- Safe to run anytime for fresh start

### Manual Verification Checklist

If running manual verification:
- `curl http://eek:9090/-/ready` → `Prometheus Server is Ready.`
- `curl http://eek:9090/targets` shows all targets **UP**.
- `curl http://ook:9400/metrics` includes `DCGM_FI_DEV_GPU_UTIL`.
- `curl http://oop:9400/metrics` includes `DCGM_FI_DEV_GPU_UTIL`.
- `curl http://ojo:9100/metrics` includes Jetson system + GPU metrics.
- `curl http://192.168.1.88:8080/metrics` includes Intel `igpu_*` metrics.
- Grafana at `http://eek:3000/` shows dashboards, including **Tatbot Compute**.
- rpi1 displays the **Tatbot Compute** URL with `?kiosk=tv&refresh=5s`.

---

## Security & Operations

- LAN‑only exposure; firewall Prometheus (9090) and Grafana (3000) to your subnet.
- Grafana uses **anonymous Viewer**; remove when not needed.
- Prometheus retention capped by **time and size** to avoid disk exhaustion.
- If historical retention grows, consider remote_write to **VictoriaMetrics** later.
- Back up Grafana’s `/var/lib/grafana` (provisioned dashboards are already in Git).

---

## References (Selected)

- Prometheus downloads & retention flags  
  - https://prometheus.io/download/  
  - https://prometheus.io/docs/prometheus/latest/storage/  
  - https://prometheus.io/docs/prometheus/latest/migration/  

- Node Exporter release (v1.9.1)  
  - https://github.com/prometheus/node_exporter/releases

- NVIDIA DCGM exporter  
  - Docs: https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html  
  - Tags: https://hub.docker.com/r/nvidia/dcgm-exporter/tags

- Jetson Stats & exporter  
  - jetson-stats (PyPI): https://pypi.org/project/jetson-stats/  
  - jetson-stats-node-exporter (PyPI): https://pypi.org/project/jetson-stats-node-exporter/  
  - Project: https://github.com/laminair/jetson_stats_node_exporter

- Intel GPU exporters  
  - restreamio/intel-prometheus: https://hub.docker.com/r/restreamio/intel-prometheus  
  - go-intel-gpu-exporter: https://pkg.go.dev/gitlab.com/leandrosansilva/go-intel-gpu-exporter  
  - Example metrics (igpu_*): https://github.com/onedr0p/intel-gpu-exporter

- Grafana OSS (v12.x) & kiosk  
  - Releases: https://github.com/grafana/grafana/releases  
  - Docker: https://grafana.com/docs/grafana/latest/setup-grafana/installation/docker/  
  - grafana-kiosk: https://github.com/grafana/grafana-kiosk  

- Reference dashboards  
  - Node Exporter Full (1860): https://grafana.com/grafana/dashboards/1860  
  - NVIDIA DCGM (12239): https://grafana.com/grafana/dashboards/12239  
  - NVIDIA Jetson (14493 / 21727): https://grafana.com/grafana/dashboards/14493-nvidia-jetson/  
  - Intel GPU Metrics (23251): https://grafana.com/grafana/dashboards/23251-intel-gpu-metrics/
