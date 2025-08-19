# TUI System Monitor

A real-time terminal-based dashboard for monitoring the distributed tatbot system state via Redis parameter server.

## Overview

The TUI monitor provides live visualization of:

- **System Status**: Redis connectivity, active sessions, node summary
- **Stroke Progress**: Real-time progress bars for active tattoo sessions
- **Node Health**: Status and connectivity of all tatbot nodes
- **Event Stream**: Recent system events and errors

## Installation

Install TUI dependencies:

```bash
uv pip install -e .[tui]
```

## Usage

### Direct Command Line

```bash
# Start monitor on rpi1 (default)
tatbot-monitor

# Custom node ID and Redis host
tatbot-monitor --node-id rpi2 --redis-host 192.168.1.97

# Python module
python -m tatbot.tui.monitor
```

### MCP Tools

The monitor can be controlled via MCP tools on monitoring nodes:

```bash
# Start monitor in foreground (blocks until Ctrl+C)
curl -sS "http://rpi1:8000/mcp/" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"start","method":"tools/call","params":{"name":"start_tui_monitor","arguments":{"background":false}}}'

# Start monitor in background
curl -sS "http://rpi1:8000/mcp/" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"start","method":"tools/call","params":{"name":"start_tui_monitor","arguments":{"background":true}}}'

# List running monitors
curl -sS "http://rpi1:8000/mcp/" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"list","method":"tools/call","params":{"name":"list_tui_monitors","arguments":{}}}'

# Stop all monitors
curl -sS "http://rpi1:8000/mcp/" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"stop","method":"tools/call","params":{"name":"stop_tui_monitor","arguments":{}}}'
```

## Display Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🤖 TATBOT SYSTEM MONITOR                          │
│          Redis: 🟢 CONNECTED  Nodes: 5/7  Updated: 14:32:15        │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────┬─────────────────────────────────────────────────┐
│   📊 System Status  │              🖥️  Node Health                    │
│ ┌─────────────────┐ │ ┌─────────────────────────────────────────────┐ │
│ │Redis Server  🟢 │ │ │eek     🟢 UP    Redis+Cams     Active      │ │
│ │Stroke Sessions🟡│ │ │hog     🟢 UP    Robot          Active      │ │  
│ │Nodes Online  🟡 │ │ │ook     🟢 UP    GPU+Monitor    Active      │ │
│ └─────────────────┘ │ │oop     🔴 DOWN  GPU+Monitor    2m ago      │ │
├─────────────────────┤ │ojo     🟢 UP    Vision         Active      │ │
│   🎨 Stroke Progress│ │rpi1    🟢 UP    Monitor        Active      │ │
│ Session: logo@hog   │ │rpi2    ⚪ UNKNOWN Monitor       Never       │ │
│ ████████████░░░░░░░░│ │ └─────────────────────────────────────────────┘ │
│ 67/100 (67.0%)     │ ├─────────────────────────────────────────────────┤
│ Pose: 23/50 (46%)  │ │              📡 Recent Events                   │
│ Status: EXECUTING  │ │ [14:32:10] hog: Progress Update (67/100)        │ │
└─────────────────────┘ │ [14:32:08] hog: Progress Update (66/100)        │ │
                        │ [14:31:45] eek: Session Start (logo)            │ │
                        │ [14:30:12] rpi1: System Start                   │ │
                        └─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│     Controls: Ctrl+C - Exit  R - Refresh Rate  Refresh: 2.0s       │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### Real-time Data

- **Auto-refresh**: Configurable refresh rate (0.5-10.0 seconds)  
- **Live events**: Subscribes to Redis pub/sub channels for immediate updates
- **Progress tracking**: Visual progress bars for stroke execution
- **Health monitoring**: Node connectivity and status updates

### Interactive Display

- **Rich formatting**: Colors, icons, and progress bars
- **Responsive layout**: Adapts to terminal size
- **Clean interface**: Organized panels with clear information hierarchy

### Background Operation  

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

### Connection Issues

```bash
# Test Redis connectivity
redis-cli -h eek -p 6379 ping

# Check network connectivity  
ping eek

# Verify MCP server is running
curl -sS "http://rpi1:8000/mcp/health"
```

### Display Issues

- **Terminal size**: Ensure terminal is at least 80x24 characters
- **Dependencies**: Install with `uv pip install -e .[tui]`
- **Colors**: Some terminals may not support all colors/formatting

### Performance

- **High CPU**: Increase refresh rate or reduce concurrent monitors
- **Memory usage**: Monitor processes typically use 10-20MB RAM
- **Network load**: Each monitor creates persistent Redis connections

## Development

### Adding Display Panels

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

### Event Handling

```python
async def handle_custom_event(self, event_data: Dict[str, Any]) -> None:
    """Handle custom system events."""
    if event_data.get("type") == "custom_event":
        # Process custom event
        self.custom_data.append(event_data)
```

The TUI monitor provides essential visibility into the distributed tatbot system, enabling operators to track system health and stroke execution progress in real-time.