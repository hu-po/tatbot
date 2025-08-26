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

The **coding agent** should generate all configs from this file.

```yaml
# monitoring/inventory.yml

versions:
  # Server-side
  prometheus:   "v3.5.0"          # prom/prometheus:<this>
  grafana_oss:  "12.1.1"          # grafana/grafana-oss:<this>

  # Exporters
  node_exporter: "1.9.0"          # github.com/prometheus/node_exporter
  dcgm_exporter: "4.4.0-4.5.0-ubi9" # nvidia/dcgm-exporter:<this> (UBI9 build)
  jetson_stats:  "4.3.2"          # PyPI: jetson-stats
  jetson_stats_exporter: "0.1.2"  # PyPI: jetson-stats-node-exporter
  intel_gpu_exporter_image: "restreamio/intel-prometheus" # see §6.4 for pinning
  intel_gpu_exporter_tag:   "latest"  # optionally replace with a private, pinned image
  rpi_exporter: "0.4.0"            # if packaging locally; otherwise use upstream binary

scrape_interval: "15s"

nodes:
  eek:
    addr: "eek:9100"
    roles: [node, server]
  ook:
    addr: "ook:9100"
    roles: [node]
    gpu:
      nvidia_dcgm: { addr: "ook:9400" }
  hog:
    addr: "192.168.1.88:9100"
    roles: [node]
    gpu:
      intel: { addr: "192.168.1.88:8080" }
  ojo:
    # jetson-stats exporter includes CPU/mem + GPU on 9100; no separate node_exporter
    jetson: { addr: "ojo:9100" }
  rpi1:
    addr: "rpi1:9100"
    roles: [node, kiosk]
    rpi: { addr: "rpi1:9110" }
  rpi2:
    addr: "rpi2:9100"
    roles: [node]
    rpi: { addr: "rpi2:9110" }
```

> **Why versions here?** This makes `inventory.yml` the **true** source of truth for **both topology and versions**. The agent can template Docker tags, download URLs, and PyPI requirements from these fields to produce deterministic configs and installers.

---

## 6) Human‑performed installation (per node)

> The CLI agent commits config/service files; a human runs the commands below (root ssh).
> Replace versions with those from **inventory.yml** if you change them.

### 6.1 Common host metrics (node_exporter) — eek, ook, hog, rpi1, rpi2
```bash
# Download Node Exporter v{versions.node_exporter} for your OS/arch:
# https://github.com/prometheus/node_exporter/releases/tag/v{versions.node_exporter}
useradd --no-create-home --shell /usr/sbin/nologin nodeexp || true
install -o nodeexp -g nodeexp -m 0755 node_exporter /usr/local/bin/node_exporter
install -o root -g root -m 0644 monitoring/exporters/<host>/node_exporter.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now node_exporter
# Verify: curl http://<host>:9100/metrics
```

### 6.2 NVIDIA dGPU (ook / RTX 4050): DCGM exporter (container, **pinned tag**)
```bash
# Install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Use a pinned image tag from inventory.yml (e.g. 4.4.0-4.5.0-ubi9)
docker run -d --restart=always --gpus all --cap-add SYS_ADMIN --net host   --name dcgm-exporter   -e DCGM_EXPORTER_LISTEN=":9400"   nvidia/dcgm-exporter:{versions.dcgm_exporter}
# Verify: curl http://ook:9400/metrics | head
```

### 6.3 Jetson (ojo): jtop/jetson‑stats exporter (**no jtop.service dependency required**)
```bash
sudo -H pip3 install "jetson-stats=={versions.jetson_stats}"
sudo -H pip3 install "jetson-stats-node-exporter=={versions.jetson_stats_exporter}"
install -m 0644 monitoring/exporters/ojo/jetson-stats-node-exporter.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now jetson-stats-node-exporter
# Verify: curl http://ojo:9100/metrics | head
```
> We **removed** the `Requires=jtop.service` dependency. The exporter uses the **Python API** from jetson‑stats directly and does not require the `jtop` systemd service to be running. See §10 for the corrected unit file.

### 6.4 Intel Arc/iGPU (hog): Intel GPU exporter
**Recommendation:** use the **container method (A)** for simplicity and to avoid managing Go/Python deps on the host. For strict reproducibility, build a **private image** from a pinned commit of a known exporter and reference that image + digest here.

**A. Container (simple):**
```bash
# restreamio/intel-prometheus (exposes :8080/metrics)
docker run -d --restart=always --net host --name intel-gpu-exporter   --privileged -v /sys:/sys:ro -v /dev/dri:/dev/dri   {versions.intel_gpu_exporter_image}:{versions.intel_gpu_exporter_tag}
# Verify: curl http://192.168.1.88:8080/metrics | head
```

**B. Go exporter (binary) — alternative:**
- https://gitlab.com/leandrosansilva/go-intel-gpu-exporter (wraps `intel_gpu_top -J`).
- Build from a known tag/commit; install as a systemd service similar to node_exporter.

### 6.5 Raspberry Pi SoC telemetry — rpi1, rpi2
```bash
# rpi_exporter for temps/voltages/clocks on :9110
# If using a packaged binary, pin to version {versions.rpi_exporter}
install rpi_exporter /usr/local/bin/rpi_exporter
install -m 0644 monitoring/exporters/<host>/rpi_exporter.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now rpi_exporter
# Verify: curl http://rpi1:9110/metrics | head
```

---

## 7) Prometheus & Grafana on **eek** (pinned images)

### 7.1 Docker Compose (compose/docker-compose.yml)
```yaml
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:{versions.prometheus}
    user: "65534:65534"
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prom
      - --storage.tsdb.retention.time=7d
      - --storage.tsdb.retention.size=2GB
    ports: ["9090:9090"]
    volumes:
      - ../prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prom_data:/prom
    restart: unless-stopped

  grafana:
    image: grafana/grafana-oss:{versions.grafana_oss}
    environment:
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: "Viewer"
    ports: ["3000:3000"]
    volumes:
      - grafana_data:/var/lib/grafana
      - ../grafana/provisioning:/etc/grafana/provisioning:ro
      - ../grafana/dashboards:/var/lib/grafana/dashboards:ro
    restart: unless-stopped

volumes:
  prom_data: {}
  grafana_data: {}
```

> **Optional: digest pinning.** For extra determinism, resolve each image to a **content digest** and use `image: name@sha256:...`. The coding agent can automate this step.

### 7.2 Prometheus config (generated) — `prometheus/prometheus.yml`
```yaml
global:
  scrape_interval: {scrape_interval}
  evaluation_interval: {scrape_interval}

scrape_configs:
  - job_name: 'nodes'
    static_configs:
      - targets: ['eek:9100','ook:9100','192.168.1.88:9100','rpi1:9100','rpi2:9100']

  - job_name: 'jetson'
    static_configs:
      - targets: ['ojo:9100']

  - job_name: 'nvidia_dgpu'
    static_configs:
      - targets: ['ook:9400']

  - job_name: 'intel_gpu'
    static_configs:
      - targets: ['192.168.1.88:8080']

  - job_name: 'rpi_soc'
    static_configs:
      - targets: ['rpi1:9110','rpi2:9110']
```

### 7.3 (Optional) starter alerts — `prometheus/rules/edge.rules.yml`
```yaml
groups:
- name: edge-basics
  rules:
  - alert: InstanceDown
    expr: up == 0
    for: 2m
    labels: { severity: critical }
    annotations:
      summary: "Target {{ $labels.instance }} down"

  - alert: HighCPU
    expr: 100 - (avg by (instance)(irate(node_cpu_seconds_total{{mode="idle"}}[5m])) * 100) > 90
    for: 10m
    labels: { severity: warning }
    annotations:
      summary: "High CPU on {{ $labels.instance }}"

  - alert: NvidiaGpuHighUtil
    expr: avg_over_time(DCGM_FI_DEV_GPU_UTIL[5m]) > 90
    for: 5m
    labels: { severity: warning }
    annotations:
      summary: "High NVIDIA GPU util on {{ $labels.instance }}"
```

---

## 8) Grafana provisioning & dashboards (with **Fleet Overview**)

### 8.1 Provision Prometheus datasource — `grafana/provisioning/datasources/prometheus.yaml`
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    isDefault: true
    url: http://prometheus:9090
```

### 8.2 Provision dashboards — `grafana/provisioning/dashboards/dashboards.yaml`
```yaml
apiVersion: 1
providers:
  - name: 'Edge Dashboards'
    type: file
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

### 8.3 Included dashboards (JSON files under `grafana/dashboards/`)
- **Fleet Overview** — `fleet-overview.json` (**new**, provided below)  
- **Node Exporter Full** — `node-exporter-full-1860.json` (ID 1860)  
- **NVIDIA DCGM** — `nvidia-dcgm-12239.json` (ID 12239)  
- **NVIDIA Jetson** — `jetson-14493.json` (ID 14493, alt 21727)  
- **Intel GPU Metrics** — `intel-gpu-23251.json` (ID 23251)

> The **Fleet Overview** dashboard is the kiosk target and summarizes CPU, Memory, Disk, and GPU across all hosts. You can drill into the detailed dashboards when needed.

#### `scripts/fetch_dashboards.sh` (agent convenience; pins latest revision at fetch-time)
```bash
#!/usr/bin/env bash
set -euo pipefail

fetch() { id="$1"; out="$2"; rev="$(curl -sL https://grafana.com/api/dashboards/${id}/revisions | jq '.[-1].revision')"
  curl -sL "https://grafana.com/api/dashboards/${id}/revisions/${rev}/download" -o "$out"; }

mkdir -p grafana/dashboards
fetch 1860 grafana/dashboards/node-exporter-full-1860.json
fetch 12239 grafana/dashboards/nvidia-dcgm-12239.json
fetch 14493 grafana/dashboards/jetson-14493.json || true
fetch 21727 grafana/dashboards/jetson-21727.json || true
fetch 23251 grafana/dashboards/intel-gpu-23251.json
```

### 8.4 **Fleet Overview** dashboard (minimal, ready-to-use)
Save as `grafana/dashboards/fleet-overview.json`. The **uid** is fixed so the kiosk URL is stable (`/d/fleet-overview/fleet-overview`).

> Notes for Intel panel: some exporters expose `igpu_*` metrics (e.g., `igpu_engines_render_3d_0_busy`). Adjust the query to match your exporter if needed.

```json
{
  "uid": "fleet-overview",
  "title": "Fleet Overview",
  "timezone": "browser",
  "schemaVersion": 39,
  "version": 1,
  "refresh": "5s",
  "templating": {
    "list": [
      {
        "type": "query",
        "name": "instance",
        "label": "Instance",
        "datasource": "Prometheus",
        "query": "label_values(up, instance)",
        "refresh": 2,
        "multi": true,
        "includeAll": true
      }
    ]
  },
  "panels": [
    {
      "type": "timeseries",
      "title": "CPU Utilization (%) by Instance",
      "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "100 - (avg by (instance)(irate(node_cpu_seconds_total{mode="idle",instance=~"$instance"}[5m])) * 100)",
          "legendFormat": "{{instance}}"
        }
      ]
    },
    {
      "type": "timeseries",
      "title": "Memory Utilization (%) by Instance",
      "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "(1 - (avg by (instance)(node_memory_MemAvailable_bytes{instance=~"$instance"}) / avg by (instance)(node_memory_MemTotal_bytes{instance=~"$instance"}))) * 100",
          "legendFormat": "{{instance}}"
        }
      ]
    },
    {
      "type": "timeseries",
      "title": "NVIDIA GPU Utilization (%) by Instance",
      "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "avg by (instance) (DCGM_FI_DEV_GPU_UTIL{instance=~"$instance"})",
          "legendFormat": "{{instance}}"
        }
      ]
    },
    {
      "type": "timeseries",
      "title": "Intel GPU Busy (%) by Instance",
      "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
      "targets": [
        {
          "datasource": "Prometheus",
          "expr": "avg by (instance) (igpu_engines_render_3d_0_busy{instance=~"$instance"})",
          "legendFormat": "{{instance}}"
        }
      ]
    }
  ]
}
```

---

## 9) rpi1 wallboard (kiosk)

Point Chromium (or `grafana-kiosk`) at:

```
http://eek:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=5s
```

**Optional: grafana-kiosk service** (ARM binary pinned separately if desired).

`exporters/rpi1/grafana-kiosk.service`
```ini
[Unit]
Description=Grafana Kiosk on Chromium (rpi1)
After=network-online.target

[Service]
Environment=KIOSK_URL=http://eek:3000/d/fleet-overview/fleet-overview?kiosk=tv&refresh=5s
ExecStart=/usr/local/bin/grafana-kiosk --kiosk-mode=tv --url ${KIOSK_URL}
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

---

## 10) Systemd unit files (exporters) — **updated**

**Node Exporter** (`exporters/<host>/node_exporter.service`)
```ini
[Unit]
Description=Prometheus Node Exporter
After=network-online.target

[Service]
User=nodeexp
Group=nodeexp
ExecStart=/usr/local/bin/node_exporter --web.listen-address=":9100"
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**DCGM exporter (docker)** (`exporters/ook/dcgm-exporter.service`) — **pinned image tag**
```ini
[Unit]
Description=NVIDIA DCGM Exporter
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/bin/docker run --rm --gpus all --cap-add SYS_ADMIN --net host   --name dcgm-exporter -e DCGM_EXPORTER_LISTEN=":9400"   nvidia/dcgm-exporter:{versions.dcgm_exporter}
ExecStop=/usr/bin/docker stop dcgm-exporter
Restart=always

[Install]
WantedBy=multi-user.target
```

**Jetson exporter** (`exporters/ojo/jetson-stats-node-exporter.service`) — **no `jtop.service` dependency**
```ini
[Unit]
Description=Jetson Stats Node Exporter
After=network-online.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -m jetson_stats_node_exporter
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Intel GPU exporter (docker)** (`exporters/hog/intel-gpu-exporter.service`) — **recommend container**
```ini
[Unit]
Description=Intel GPU Exporter (container)
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/bin/docker run --rm --privileged --net host --name intel-gpu-exporter   -v /sys:/sys:ro -v /dev/dri:/dev/dri   {versions.intel_gpu_exporter_image}:{versions.intel_gpu_exporter_tag}
ExecStop=/usr/bin/docker stop intel-gpu-exporter
Restart=always

[Install]
WantedBy=multi-user.target
```

**rpi_exporter** (`exporters/rpi1/rpi_exporter.service`)
```ini
[Unit]
Description=Raspberry Pi Hardware Exporter
After=network-online.target

[Service]
ExecStart=/usr/local/bin/rpi_exporter -addr=:9110
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

(Repeat for `rpi2`.)

---

## 11) Repo layout (agent will create)

```
monitoring/
├─ DESIGN.md                            # this file
├─ inventory.yml                        # hosts, roles, versions, ports
├─ compose/
│  └─ docker-compose.yml                # Prometheus + Grafana on eek (pinned images)
├─ prometheus/
│  ├─ prometheus.yml                    # generated from inventory.yml
│  └─ rules/
│     └─ edge.rules.yml                 # optional starter alerts
├─ grafana/
│  ├─ provisioning/
│  │  ├─ datasources/prometheus.yaml
│  │  └─ dashboards/dashboards.yaml
│  └─ dashboards/
│     ├─ fleet-overview.json            # unified wallboard (uid=fleet-overview)
│     ├─ node-exporter-full-1860.json
│     ├─ nvidia-dcgm-12239.json
│     ├─ jetson-14493.json
│     └─ intel-gpu-23251.json
├─ exporters/
│  ├─ eek/node_exporter.service
│  ├─ ook/node_exporter.service
│  ├─ ook/dcgm-exporter.service
│  ├─ hog/node_exporter.service
│  ├─ hog/intel-gpu-exporter.service
│  ├─ ojo/jetson-stats-node-exporter.service
│  ├─ rpi1/node_exporter.service
│  ├─ rpi1/rpi_exporter.service
│  ├─ rpi1/grafana-kiosk.service
│  ├─ rpi2/node_exporter.service
│  └─ rpi2/rpi_exporter.service
└─ scripts/
   ├─ fetch_dashboards.sh               # grabs JSON by ID
   └─ gen_prom_config.py                # renders prometheus.yml from inventory.yml
```

---

## 12) Coding agent tasks (deterministic)

1. **Create repo tree** (above).
2. **Write `inventory.yml`** with your hostnames/IPs and **versions** (§5).
3. **Render `prometheus/prometheus.yml`** from `inventory.yml`:
   - jobs: `nodes`, `jetson`, `nvidia_dgpu`, `intel_gpu`, `rpi_soc`.
4. **Drop systemd units** under `exporters/<host>/…` using the templates (§10).
5. **Render Compose** under `compose/docker-compose.yml` with **pinned tags** (§7.1).
6. **Add Grafana provisioning** files (§8.1–§8.2).
7. **Install Fleet Overview** dashboard JSON (§8.4).
8. **Run `scripts/fetch_dashboards.sh`** to fetch reference dashboards.
9. **Emit a `Makefile`** with shortcuts:
   - `make up` → `docker compose -f monitoring/compose/docker-compose.yml up -d`
   - `make down` → `docker compose -f monitoring/compose/docker-compose.yml down`
   - `make dashboards` → run `scripts/fetch_dashboards.sh`

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

- Node Exporter release (v1.9.0)  
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
