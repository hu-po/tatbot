---
summary: Real-time visualization tools and usage
tags: [viz, tools]
updated: 2025-08-21
audience: [dev, operator]
---

# Visualization

Tatbot provides interactive 3D visualization tools using [`viser`](https://github.com/nerfstudio-project/viser) for debugging, teleoperation, and stroke execution monitoring.

## 🖥️ Architecture

The visualization system has two modes of operation:

### Standalone Scripts
Located in `src/tatbot/viz/`:
- `base.py`: Base class for all visualizations
- `stroke.py`: Visualizes a full `StrokeBatch` execution
- `teleop.py`: Provides interactive teleoperation via inverse kinematics
- `map.py`: Tool for debugging 2D-to-3D surface mapping

Run standalone with:
```bash
uv run python -m tatbot.viz.stroke --scene=tatbotlogo
uv run python -m tatbot.viz.teleop --enable-robot --enable-depth
uv run python -m tatbot.viz.map --scene=default
```

### MCP Tools
Located in `src/tatbot/tools/viz/`:
- `stroke_viz.py`: Start stroke visualization server via MCP
- `teleop_viz.py`: Start teleoperation server via MCP
- `map_viz.py`: Start surface mapping visualization via MCP
- `control.py`: Stop servers and list running servers

## ⚡ Usage

### Start Stroke Viz
```json
{
  "tool": "start_stroke_viz",
  "input": {
    "scene": "tatbotlogo",
    "align": false,
    "enable_robot": false,
    "enable_depth": false,
    "speed": 1.0
  }
}
```

### Start Teleop Viz
```json
{
  "tool": "start_teleop_viz",
  "input": {
    "scene": "default",
    "enable_robot": true,
    "enable_depth": true,
    "transform_control_scale": 0.2
  }
}
```

### Start Surface Mapping
```json
{
  "tool": "start_map_viz",
  "input": {
    "scene": "default",
    "stroke_point_size": 0.0005,
    "skin_ply_point_size": 0.0005
  }
}
```

### Control
```json
// List all running servers
{
  "tool": "list_viz_servers",
  "input": {}
}

// Stop a specific server
{
  "tool": "stop_viz_server",
  "input": {
    "server_name": "stroke_viz"
  }
}
```

## ✨ Features

### Stroke Visualization
- Real-time robot arm movement during stroke execution
- Point cloud visualization of stroke paths
- Joint position monitoring
- Optional depth camera integration
- Playback speed control

### Teleoperation
- Interactive end-effector control via transform controls
- Real-time inverse kinematics solving
- Pose saving and loading
- Direct robot control (when enabled)
- Calibrator positioning
- EE offset calculation and saving

### Surface Mapping
- 2D stroke designs in 3D space
- PLY point cloud data from depth cameras
- Interactive mesh building from point clouds
- Stroke mapping to 3D surface mesh
- Design pose adjustment controls

## 🛠️ Configuration

Visualization tools accept standard configuration parameters:
- `scene`: Scene configuration name
- `enable_robot`: Connect to real robot hardware
- `enable_depth`: Enable depth camera visualization
- `speed`: Playback speed multiplier
- `fps`: Frame rate for visualization loop (default: 30.0)
- `bind_host`: Host interface to bind to (default: "0.0.0.0" for all interfaces)
- `env_map_hdri`: Environment map for lighting
- `view_camera_position`: Initial camera position
- `view_camera_look_at`: Initial camera target

## Notes

- Visualization servers run in background threads with proper lifecycle management
- Only one server of each type can run at a time
- Servers can be cleanly stopped and will release all resources
- Frame rate is limited to prevent CPU burn (configurable via `fps` parameter)
- Access servers via the URL returned when starting (uses node IP for cross-node access)
- Servers bind to all network interfaces by default for remote accessibility
- Startup includes health checks to ensure server is ready before returning success
