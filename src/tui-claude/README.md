# Tatbot Node Monitor TUI

A clean, fast text-based user interface for monitoring Tatbot nodes with real-time system statistics.

## Features

- **Real-time Monitoring**: Updates every 5 seconds (configurable) with parallel SSH
- **Clean Interface**: Simplified panels and summary table
- **CPU Monitoring**: Load percentage and core count (physical cores preferred)
- **Memory Monitoring**: Usage in GB and percentage with color coding
- **GPU Monitoring**: VRAM usage, temperature, and utilization for NVIDIA GPUs
- **Metadata Fallback**: Shows hardware specs when nodes are offline
- **Source Indicators**: [R]=Remote data, [M]=Metadata fallback
- **Color Coding**: Green (healthy) → Yellow (warning) → Red (critical/offline)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or if using the main tatbot environment:
```bash
pip install rich
```
(paramiko and pyyaml are already in the main requirements)

## Usage

### Basic Usage

Run the TUI with default settings (5-second update interval):
```bash
python tui.py
```

### Custom Update Interval

Specify a custom update interval in seconds:
```bash
python tui.py --interval 10  # Update every 10 seconds
python tui.py -i 2           # Update every 2 seconds
```

### Exit

Press `Ctrl+C` to exit the application.

## Requirements

- Python 3.11+
- SSH key-based authentication configured for all nodes
- Network connectivity to node IPs listed in `src/conf/nodes.yaml`

## Node Configuration

The TUI reads node configuration from `/home/oop/tatbot/src/conf/nodes.yaml`. Each node entry should have:
- `name`: Node identifier
- `emoji`: Visual emoji for the node
- `ip`: IP address of the node
- `user`: SSH username for the node

Example:
```yaml
nodes:
  - name: oop
    emoji: 🦊
    ip: 192.168.1.51
    user: oop
```

## SSH Authentication

The TUI uses SSH to connect to nodes and gather system information. Ensure:
1. SSH key-based authentication is set up for all nodes
2. The SSH keys are loaded in your SSH agent or available in `~/.ssh/`
3. The user specified in the config has permissions to run system commands

### SSH Configuration

You can customize SSH connection settings using environment variables:

```bash
export TATBOT_TUI_SSH_KEY=~/.ssh/your_custom_key
export TATBOT_TUI_SSH_PORT=2222  # if using non-standard port
python tui.py
```

## Monitored Nodes

The following nodes are configured:
- 🦧 **ook** (192.168.1.90) - May have GPU
- 🦊 **oop** (192.168.1.51) - May have GPU
- 🦎 **ojo** (192.168.1.96) - May have GPU
- 🍓 **rpi1** (192.168.1.98) - Raspberry Pi (CPU only)
- 🍇 **rpi2** (192.168.1.99) - Raspberry Pi (CPU only)
- 🦾 **trossen-ai** (192.168.1.97) - Robot arm controller

## Display Layout

The TUI displays:
1. **Header**: Application title and last update timestamp
2. **Node Panels**: Individual panels for each node showing:
   - Connection status
   - CPU load with progress bar
   - Memory usage with progress bar
   - GPU information (if available) with progress bars
3. **Summary Table**: Quick overview of all nodes

## Troubleshooting

### Node Shows as Offline
- Check network connectivity: `ping <node_ip>`
- Verify SSH access: `ssh user@node_ip`
- Ensure the node is powered on

### No GPU Information
- GPU monitoring requires `nvidia-smi` to be installed on the node
- Only NVIDIA GPUs are currently supported
- CPU-only nodes (like Raspberry Pis) will show "No GPU detected"

### SSH Connection Errors
- Verify SSH key authentication is set up
- Check if SSH agent is running: `ssh-add -l`
- Add SSH key if needed: `ssh-add ~/.ssh/id_rsa`

## Node Metadata

The TUI supports an optional metadata file (`nodes_meta.yaml`) that provides hardware specifications for fallback when nodes are offline:

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

This allows the TUI to show hardware specs even when nodes are unreachable, marked with `[M]` indicators.

## Architecture

**Simplified Design:**
- `NodeStats`: Single data class containing all metrics
- `NodeInfo`: Node config + current stats
- `NodeMonitor`: Parallel SSH updates with metadata fallback
- `TatbotTUI`: Clean Rich-based interface

## Performance

- Parallel node updates via ThreadPoolExecutor
- SSH connection reuse with health checks
- Physical CPU core detection (lscpu → nproc fallback)
- Metadata fallback for offline nodes
- 3-second SSH timeouts for responsiveness