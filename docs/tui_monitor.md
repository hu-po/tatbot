---
summary: Terminal UI monitor for nodes and stroke progress
tags: [tui, monitor, state]
updated: 2025-08-21
audience: [dev, operator]
---

# TUI System Monitor

A real-time terminal-based dashboard for monitoring the distributed tatbot system state via Redis parameter server.

## 🔍 Overview

The TUI monitor provides live visualization of:

- **System Status**: Redis connectivity, active sessions, node summary
- **Stroke Progress**: Real-time progress bars for active tattoo sessions
- **Node Health**: Status and connectivity of all tatbot nodes
- **Event Stream**: Recent system events and errors

## 🛠️ Installation

Install TUI dependencies:

```bash
uv pip install -e .[tui]
```

## ⚡ Usage

### Command Line

```bash
# Start monitor on rpi1 (default)
tatbot-monitor

# Custom node ID and Redis host
tatbot-monitor --node-id rpi2 --redis-host 192.168.1.97

# Python module
python -m tatbot.tui.monitor
```

### Running

The TUI monitor is designed to run directly on rpi1 using the terminal command. Simply SSH to rpi1 and start it:

```bash
# SSH to rpi1 and start the monitor
ssh rpi1@192.168.1.98
source scripts/setup_env.sh
uv run tatbot-monitor
```

## 🖥️ Display Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🤖 TATBOT SYSTEM MONITOR                          │
│          Redis: 🟢 CONNECTED  Nodes: 5/7  Updated: 14:32:15        │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────┬─────────────────────────────────────────────────┐
│   📊 System Status  │              🖥️  Node Health                    │
│ ┌─────────────────┐ │ ┌─────────────────────────────────────────────┐ │
│ │Redis Server  🟢 │ │ │eek        🟢 UP                            │ │
│ │Stroke Sessions🟡│ │ │hog        🟢 UP                            │ │  
│ │Nodes Online  🟡 │ │ │ojo        🟢 UP                            │ │
│ └─────────────────┘ │ │ook        🟢 UP                            │ │
├─────────────────────┤ │oop        🔴 DOWN                          │ │
│   🎨 Stroke Progress│ │rpi1       🟢 UP                            │ │
│ Session: logo@hog   │ │rpi2       ⚪ UNKNOWN                       │ │
│ ████████████░░░░░░░░│ │ └─────────────────────────────────────────────┘ │
│ 67/100 (67.0%)     │ ├─────────────────────────────────────────────────┤
│ Pose: 23/50 (46%)  │ │              📡 Recent Events                   │
│ Status: EXECUTING  │ │ [14:32:10] hog: Progress Update (67/100)        │ │
└─────────────────────┘ │ [14:32:08] hog: Progress Update (66/100)        │ │
                        │ [14:31:45] eek: Session Start (logo)            │ │
                        │ [14:30:12] rpi1: System Start                   │ │
                        └─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│                    Controls: Ctrl+C - Exit                          │
└─────────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Data

- **Auto-refresh**: Configurable refresh rate (0.5-10.0 seconds)  
- **Live events**: Subscribes to Redis pub/sub channels for immediate updates
- **Progress tracking**: Visual progress bars for stroke execution
- **Health monitoring**: Node connectivity and status updates

### Display

- **Rich formatting**: Colors, icons, and progress bars
- **Responsive layout**: Adapts to terminal size
- **Clean interface**: Organized panels with clear information hierarchy

### Background  

- **Detached processes**: Can run monitors in background
- **Process management**: List, start, and stop multiple monitor instances
- **Resource efficient**: Low CPU/memory footprint

## Configuration

### Refresh Rate

- **Fast**: 0.5-1.0s for active monitoring during operations
- **Normal**: 1.0-3.0s for general monitoring  
- **Slow**: 3.0-10.0s for background monitoring

### Node Assignment

Run monitors on different nodes based on role:

- **rpi1/rpi2**: Primary monitoring nodes with displays
- **oop/ook**: Secondary monitoring on workstation nodes
- **eek/hog**: Avoid running monitors on robot/camera nodes

## Troubleshooting

### Connections

```bash
# Test Redis connectivity
redis-cli -h eek -p 6379 ping

# Check network connectivity  
ping eek

# Verify MCP server is running
curl -sS "http://rpi1:8000/mcp/health"
```

### Display Problems

- **Terminal size**: Ensure terminal is at least 80x24 characters
- **Dependencies**: Install with `uv pip install -e .[tui]`
- **Colors**: Some terminals may not support all colors/formatting

### Performance

- **High CPU**: Increase refresh rate or reduce concurrent monitors
- **Memory usage**: Monitor processes typically use 10-20MB RAM
- **Network load**: Each monitor creates persistent Redis connections

## Development

### Custom Panels

```python
def create_custom_panel(self) -> Panel:
    """Create a custom monitoring panel."""
    content = Text("Custom data here")
    return Panel(content, title="📈 Custom Panel", box=box.ROUNDED)

def setup_layout(self) -> None:
    """Update layout to include custom panel."""
    # Add custom panel to layout structure
    pass
```

### Events

```python
async def handle_custom_event(self, event_data: Dict[str, Any]) -> None:
    """Handle custom system events."""
    if event_data.get("type") == "custom_event":
        # Process custom event
        self.custom_data.append(event_data)
```

The TUI monitor provides essential visibility into the distributed tatbot system, enabling operators to track system health and stroke execution progress in real-time.
