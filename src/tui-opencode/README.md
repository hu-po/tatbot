# TUI-Opencode

A Terminal User Interface (TUI) for monitoring Tatbot nodes, designed specifically for the opencode environment.

## Features

- Real-time monitoring of node status (online/offline)
- CPU usage and core count display
- Memory usage monitoring
- GPU information including memory usage and temperature
- Rich visual interface with color-coded status indicators
- Support for both remote data and metadata fallback
- Configurable update intervals

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the TUI with default settings:

```bash
python tui.py
```

Or specify a custom update interval:

```bash
python tui.py --interval 10
```

### Command Line Options

- `--interval`, `-i`: Update interval in seconds (default: 5)

## Configuration

The TUI reads node configuration from `/home/oop/tatbot/src/conf/nodes.yaml` and metadata from `nodes_meta.yaml` in the same directory.

### Environment Variables

- `TATBOT_TUI_SSH_KEY`: Path to SSH private key file
- `TATBOT_TUI_SSH_PORT`: SSH port (default: 22)

## Interface

The TUI displays:

1. **Header**: Shows current time and application name
2. **Node Panels**: Individual panels for each node showing:
   - Online/offline status
   - CPU usage and core count
   - Memory usage
   - GPU information (if available)
   - Temperature and utilization
3. **Summary Table**: Compact overview of all nodes with source indicators:
   - `[R]`: Data from remote node
   - `[M]`: Data from metadata file

## Controls

- `q`: Quit the application
- `r`: Force refresh all nodes
- `i`: Decrease update interval by 1 second (minimum 1s)
- `I`: Increase update interval by 1 second (maximum 60s)
- `Ctrl+C`: Emergency exit

## Design

This TUI is based on the existing `tui-claude` implementation but customized for the opencode environment with:

- Enhanced visual styling using magenta theme colors
- Improved panel layouts with rounded borders
- Better GPU information display
- Cleaner status indicators

The implementation uses the Rich library for terminal rendering and Paramiko for SSH connections to remote nodes.