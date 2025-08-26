# Edge Fleet Monitoring Design (Prometheus + Grafana)

**Audience:** humans deploying on device, and a CLI coding agent committing config into this repo.  
**Outcome:** a single wallboard on **rpi1** (Raspberry Pi 5) showing live CPU/memory/disk/net and GPU metrics across all nodes with minimal overhead, using **only FOSS** — now with **explicit version pinning** and a **unified Fleet Overview dashboard**.

_Last updated: 2025-08-26._

---

## 1) Executive summary

We standardize on **Prometheus + Grafana** with per-node exporters:

- **Host metrics (all nodes):** Prometheus **Node Exporter** (tiny static binary).
- **NVIDIA dGPU (ook / RTX 4050):** **DCGM Exporter** (official NVIDIA).
- **Jetson (ojo / AGX Orin):** **jetson-stats (jtop)–based exporter** that exposes system + GPU.
- **Intel Arc/iGPU (hog):** Intel GPU exporter that parses `intel_gpu_top -J`.
- **Raspberry Pi SoC (rpi1, rpi2):** lightweight **rpi_exporter** for temps/voltages/clock.

Prometheus + Grafana run centrally on **eek** (System76 Meerkat). **rpi1** displays Grafana in **kiosk mode**.

**Why this fits your constraints**

- **Unified dashboard** on a Pi display.
- **Minimal overhead**: exporters read `/proc` or vendor APIs; scrape every 15s.
- **FOSS** end‑to‑end; no cloud dependency.
- **Reproducible**: all images/binaries **pinned to specific versions** (and optionally digests).

---

## 2) Fleet & roles

| Host | Hardware | Role(s) | Exporters |
|---|---|---|---|
| **eek** | System76 Meerkat (i5‑1340P) | **Prometheus + Grafana server**; NFS | `node_exporter` |
| **ook** | Acer Nitro V 15 (i7‑13620H + **RTX 4050**) | Wi‑Fi NAT; GPU compute | `node_exporter`, **DCGM exporter** |
| **oop** | Desktop PC | GPU compute; development | `node_exporter` |
| **hog** | GEEKOM GT1 Mega (Core Ultra 9 + **Intel Arc**) | Robot control, RealSense | `node_exporter`, **Intel GPU exporter** |
| **ojo** | **Jetson AGX Orin** | Agent inference | **jetson‑stats node exporter** (includes CPU/mem/GPU) |
| **rpi1** | Raspberry Pi 5 (8GB) | **Wallboard**; app frontend | `node_exporter`, **rpi_exporter**; **Grafana kiosk** client |
| **rpi2** | Raspberry Pi 5 (8GB) | DNS/DHCP | `node_exporter`, **rpi_exporter** |

> **Jetson note:** the selected exporter exposes **both** host and GPU metrics on port **:9100**; do **not** run a separate node_exporter there to avoid port conflicts.

---

## 3) High‑level architecture

```
                           [ rpi1 ]  ──> Chromium/Grafana Kiosk (read‑only)
                                        http://eek:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=5s

                ┌──────────────────────── [ eek ] ──────────────────────────┐
                │  Prometheus (9090)  ← scrape 15s   Grafana OSS (3000)     │
                │  retention: 7d or 2GB (whichever first)                   │
                └───────────────────────────────────────────────────────────┘
                         ▲           ▲           ▲           ▲           ▲
                         │           │           │           │           │
                node_exporter   DCGM exporter  Intel GPU   rpi_exporter  jetson-stats
                    :9100          :9400         :8080       :9110         :9100
                 [eek/ook/hog]      [ook]        [hog]     [rpi1/rpi2]      [ojo]
```

---

## 4) Design decisions

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

## 5) Inventory (single source of truth)

Inventory lives at `~/tatbot/config/monitoring/inventory.yml` (versions, scrape interval, and nodes). IPs should match `src/conf/nodes.yaml`. Keep them in sync and regenerate Prometheus config (see §12).

> **Why versions here?** This makes `inventory.yml` the **true** source of truth for **both topology and versions**. The agent can template Docker tags, download URLs, and PyPI requirements from these fields to produce deterministic configs and installers.

---

## 6) Human‑performed installation (per node)

> The CLI agent commits config/service files; a human runs the commands below (root ssh).
> Replace versions with those from **inventory.yml** if you change them.

Prereqs by node
- eek: Docker Engine + compose plugin (section 7.1).
- ook (RTX 4050): Docker Engine + NVIDIA Container Toolkit (section 6.2); NVIDIA driver working (`nvidia-smi`).
- hog (Intel GPU): Docker Engine (section 6.4).
- ojo (Jetson): Python3/pip; install jetson-stats + exporter (section 6.3).
- rpi1/rpi2: None beyond systemd and curl.

### 6.1 Common host metrics (node_exporter) — eek, ook, hog, rpi1, rpi2
On each host (repo at `~/tatbot`), download and install Node Exporter v1.9.1:

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

> Do not install node_exporter on `ojo` (Jetson). It runs `jetson-stats-node-exporter` on :9100 (see §6.3).

### 6.2 NVIDIA dGPU (ook / RTX 4050): DCGM exporter (container, **pinned tag**)
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

Run DCGM exporter (pinned tag from inventory):
```bash
docker run -d --restart=always --gpus all --cap-add SYS_ADMIN --net host \
  --name dcgm-exporter -e DCGM_EXPORTER_LISTEN=":9400" \
  nvidia/dcgm-exporter:4.4.0-4.5.0-ubi9
```

Or use the systemd unit in the repo: `~/tatbot/config/monitoring/exporters/ook/dcgm-exporter.service`, then:
`sudo systemctl daemon-reload && sudo systemctl enable --now dcgm-exporter`

Verify: `curl -sS --no-progress-meter http://192.168.1.90:9400/metrics | head -n 20`

### 6.3 Jetson (ojo): jtop/jetson‑stats exporter (**no jtop.service dependency required**)
```bash
sudo -H pip3 install "jetson-stats==4.3.2"
sudo -H pip3 install "jetson-stats-node-exporter==0.1.2"
sudo install -m 0644 ~/tatbot/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now jetson-stats-node-exporter
curl -sS --no-progress-meter http://192.168.1.96:9100/metrics | head -n 20
```

> We **removed** the `Requires=jtop.service` dependency. The exporter uses the **Python API** from jetson‑stats directly and does not require the `jtop` systemd service to be running. See §10 for the corrected unit file.

### 6.4 Intel Arc/iGPU (hog): Intel GPU exporter
**hog** needs Node Exporter (section 6.1) PLUS Intel GPU monitoring:

First install Docker if not present:
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group changes
```

Then run Intel GPU exporter:
```bash
docker run -d --restart=always --net host --name intel-gpu-exporter --privileged -v /sys:/sys:ro -v /dev/dri:/dev/dri restreamio/intel-prometheus:latest
curl -sS --no-progress-meter http://192.168.1.88:8080/metrics | head -n 20
```

### 6.5 Raspberry Pi SoC telemetry — rpi1, rpi2
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

---

## 7) Prometheus & Grafana on **eek** (pinned images)

### 7.1 Docker Compose setup (eek)
Install Docker with modern compose plugin:
```bash
# Remove broken legacy docker-compose if installed
sudo apt-get remove -y docker-compose

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


### 7.2 Prometheus config
File: `~/tatbot/config/monitoring/prometheus/prometheus.yml` (generated from inventory). Alert rules: `~/tatbot/config/monitoring/prometheus/rules/`.

### 7.3 (Optional) starter alerts
See: `~/tatbot/config/monitoring/prometheus/rules/edge.rules.yml`.

---

## 8) Grafana provisioning & dashboards (with **Fleet Overview**)

### 8.1 Provision Prometheus datasource
File: `~/tatbot/config/monitoring/grafana/provisioning/datasources/prometheus.yaml`.

### 8.2 Provision dashboards
File: `~/tatbot/config/monitoring/grafana/provisioning/dashboards/dashboards.yaml`.

### 8.3 Included dashboards (JSON files under `grafana/dashboards/`)
- **Fleet Overview** — `fleet-overview.json` (**installed in repo**)  
- **Node Exporter Full** — `node-exporter-full-1860.json` (ID 1860)  
- **NVIDIA DCGM** — `nvidia-dcgm-12239.json` (ID 12239)  
- **NVIDIA Jetson** — `jetson-14493.json` (ID 14493, alt 21727)  
- **Intel GPU Metrics** — `intel-gpu-23251.json` (ID 23251)

> The **Fleet Overview** dashboard is the kiosk target and summarizes CPU, Memory, Disk, Network, and GPU across all hosts. You can drill into the detailed dashboards when needed.

Run this on eek to pull community dashboards into `~/tatbot/config/monitoring/grafana/dashboards/`:
`cd ~/tatbot && bash scripts/fetch_dashboards.sh` (requires `jq`).

### 8.4 Fleet Overview dashboard
Installed at `~/tatbot/config/monitoring/grafana/dashboards/fleet-overview.json` (uid=fleet-overview). Intel exporter metric names may vary (`igpu_*`).

<!-- Dashboard JSON lives in the repo; see path above. -->

---

## 9) rpi1 wallboard (kiosk)

### 9.1 Quick start with monitoring kiosk script
**Run on rpi1** (wallboard display node) to launch the monitoring dashboard in kiosk mode:
```bash
# Start monitoring kiosk (default: eek:3000, 5s refresh)
cd ~/tatbot && bash scripts/monitoring_kiosk.sh

# Custom refresh interval  
cd ~/tatbot && bash scripts/monitoring_kiosk.sh eek 10

# Specific IP address
cd ~/tatbot && bash scripts/monitoring_kiosk.sh 192.168.1.97
```

**When to run:** Once per boot, or when you want to restart the wallboard display.

### 9.2 Manual kiosk URL
Point Chromium (or `grafana-kiosk`) at:
```
http://eek:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=5s
```

### 9.3 Optional: grafana-kiosk systemd service
Unit file: `~/tatbot/config/monitoring/exporters/rpi1/grafana-kiosk.service`.
Enable: `sudo systemctl daemon-reload && sudo systemctl enable --now grafana-kiosk`

---

## 10) Systemd unit files (exporters) — repository paths

- Node Exporter: `~/tatbot/config/monitoring/exporters/<host>/node_exporter.service`
- DCGM exporter (docker): `~/tatbot/config/monitoring/exporters/ook/dcgm-exporter.service`
- Jetson exporter: `~/tatbot/config/monitoring/exporters/ojo/jetson-stats-node-exporter.service`
- Intel GPU exporter (docker): `~/tatbot/config/monitoring/exporters/hog/intel-gpu-exporter.service`
- rpi_exporter: `~/tatbot/config/monitoring/exporters/rpi{1,2}/rpi_exporter.service`
- Grafana kiosk (optional): `~/tatbot/config/monitoring/exporters/rpi1/grafana-kiosk.service`

---

## 11) Repo layout

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
│     └─ fleet-overview.json
└─ exporters/
   ├─ eek/node_exporter.service
   ├─ ook/{node_exporter.service,dcgm-exporter.service}
   ├─ hog/{node_exporter.service,intel-gpu-exporter.service}
   ├─ ojo/jetson-stats-node-exporter.service
   └─ rpi{1,2}/{node_exporter.service,rpi_exporter.service}
scripts/
├─ fetch_dashboards.sh
└─ gen_prom_config.py
```

---

## 12) Coding agent tasks (deterministic)

1. Edit `~/tatbot/config/monitoring/inventory.yml` (hosts, versions).
2. Generate Prom config: `make -C ~/tatbot/config/monitoring gen-prom`.
3. Install exporters using units under `~/tatbot/config/monitoring/exporters/<host>/...`.
4. Start Prometheus + Grafana: `make -C ~/tatbot/config/monitoring up`.
5. Optional: run `~/tatbot/scripts/fetch_dashboards.sh` to fetch reference dashboards.

---

## 13) Verification checklist

- `curl http://eek:9090/-/ready` → `Prometheus is Ready.`
- `curl http://eek:9090/targets` shows all targets **UP**.
- `curl http://ook:9400/metrics` includes `DCGM_FI_DEV_GPU_UTIL`.
- `curl http://ojo:9100/metrics` includes Jetson system + GPU metrics.
- `curl http://192.168.1.88:8080/metrics` includes Intel `igpu_*` metrics.
- Grafana at `http://eek:3000/` shows dashboards, including **Fleet Overview**.
- rpi1 displays the **Fleet Overview** URL with `?kiosk=tv&refresh=5s`.

---

## 14) Version pinning policy (why & how)

- **Never use `:latest`** for Docker images or unversioned binary URLs.
- Keep **all versions in `inventory.yml`**. The agent must template every reference from there.
- For Docker: prefer **semantic tags** (e.g., `prom/prometheus:v3.5.0`) and/or **digest pinning** (`name@sha256:...`).
- For PyPI: use **exact `==` pins** (e.g., `jetson-stats==4.3.2`).
- For GitHub releases: download by **tag** and verify checksums/signatures when available.
- Revisit pins quarterly; update, test, and commit as a single change.

---

## 15) Security & operations

- LAN‑only exposure; firewall Prometheus (9090) and Grafana (3000) to your subnet.
- Grafana uses **anonymous Viewer**; remove when not needed.
- Prometheus retention capped by **time and size** to avoid disk exhaustion.
- If historical retention grows, consider remote_write to **VictoriaMetrics** later.
- Back up Grafana’s `/var/lib/grafana` (provisioned dashboards are already in Git).

---

## 16) References (selected)

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
