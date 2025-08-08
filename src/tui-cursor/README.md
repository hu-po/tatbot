## Tatbot Nodes TUI (Cursor)

A lightweight, dependency-free terminal UI for monitoring Tatbot nodes. It reads `src/conf/nodes.yaml`, checks node reachability, and shows CPU load and GPU memory per node via SSH.

### Features
- Shows whether each node is online (ICMP ping)
- Displays CPU load averages (1/5/15 minutes) from `/proc/loadavg`
- Displays GPU memory usage if `nvidia-smi` is available
- Auto-refresh every 2 seconds; press `r` to refresh now, `q` to quit

### Requirements
- `ssh` access to nodes listed in `src/conf/nodes.yaml` (public key auth recommended)
- Each node should have:
  - `/proc/loadavg` (standard on Linux)
  - `nvidia-smi` if it has an NVIDIA GPU
- Local system tools: `ping`, `ssh`

### Install & Run
This project uses `uv` for Python environments.

```bash
# From repo root
source scripts/setup_env.sh

# Run the TUI (direct execution)
python src/tui-cursor/tui.py

# Alternatively, if PYTHONPATH is configured:
PYTHONPATH=. python -m src.tui-cursor.tui
```

### Configuration
Configure nodes in `src/conf/nodes.yaml`:

```yaml
nodes:
  - name: ook
    emoji: 🦧
    ip: 192.168.1.90
    user: ook
  - name: oop
    emoji: 🦊
    ip: 192.168.1.51
    user: oop
```

Optionally provide hardware metadata in `src/tui-cursor/nodes_meta.yaml` (this repo folder) to enrich the UI (totals shown even if remote queries fail). This file is used automatically; you do not need to modify `src/conf`:

```yaml
nodes_meta:
  ook:
    cpu_cores: 16
    gpu_total_mb: 24576
    gpu_count: 1
  rpi1:
    cpu_cores: 4
    gpu_total_mb: 0
    gpu_count: 0
```

Edit `src/tui-cursor/nodes_meta.yaml` directly to match your hardware.

### Notes
- Nodes without GPUs or without `nvidia-smi` installed show GPU as `n/a`.
- Offline nodes show `-` for metrics and are colored red.
- SSH failures degrade gracefully; only the failed metric shows `n/a`.

### Troubleshooting
- Ensure passwordless SSH works:
  ```bash
  ssh <user>@<ip> true
  ```
- Allow ICMP echo replies on nodes if online status always shows off.
- For NVIDIA GPUs, install `nvidia-smi` (part of NVIDIA drivers).
