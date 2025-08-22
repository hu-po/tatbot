---
summary: tatbot documentation index and quick links
tags: [index]
updated: 2025-08-21
audience: [all]
---

# tatbot

```{image} logos/dark.svg
:alt: tatbot logo  
:class: only-dark
:align: center
:width: 200px
```

<br>

tattoo robot system composed of 2 robot arms, 7 cameras, and 6 computers controlling 2 tattoo machines. Built in public, all open source, [reserve your tattoo](https://forms.gle/Zys6f5iLEtYCG8VW7), [follow the progress](docs/progress.md)

---

## Quick Start

```{admonition} Essential Commands
:class: tip

- `./scripts/mcp_run.sh` - Start MCP server
- `./scripts/setup_env.sh` - Install dependencies
```



## Documentation Index

### 🤖 Hardware & Control
- [**Robot System**](robot.md) - Trossen arms, URDF models, inverse kinematics
- [**Vision System**](vision.md) - Cameras, AprilTag tracking, 2D→3D mapping
- [**Tattoo Gear**](gear.md) - Physical tattoo equipment and setup

### 🌐 Software Architecture  
- [**MCP Protocol**](mcp.md) - Model Context Protocol for distributed control
- [**Tools System**](tools.md) - Unified operation framework and tool registry
- [**Network Architecture**](network_architecture.md) - Automatic dual-mode networking
- [**Agent Interface**](agent.md) - LLM-based control system

### 🎨 Art & Visualization
- [**Artwork Pipeline**](artwork.md) - From images to tattoo designs
- [**3D Visualization**](viz.md) - Real-time robot visualization with Viser
- [**VGGT System**](vggt.md) - 3D reconstruction and surface mapping

### 🧠 AI & Models
- [**Model Plans**](plans/vla_plan/index.md) - Vision-language-action model development
- [**Training Data**](plans/models/data.md) - Dataset collection and management
- [**Gr00t Integration**](plans/models/gr00t.md) - Foundation model interface
- [**SmolVLA**](plans/models/smolvla.md) - Compact vision-language-action model

### 🔧 Operations & Monitoring
- [**State Server**](state_server.md) - Redis-based state management
- [**TUI Monitor**](tui_monitor.md) - Real-time system dashboard
- [**Development Workflows**](development.md) - Coding, testing, and deployment

### 📚 Reference
- [**Style Guide**](style_guide.md) - Code and documentation standards
- [**Progress Tracking**](progress.md) - Development milestones and status
- [**Research Ideas**](ideas.md) - Future development directions
- [**Academic Paper**](paper/index.md) - Research publication and figures

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

development
nodes
robot
vision
mcp
tools
viz
agent
artwork
gear
progress
state_server
ideas
network_architecture
tui_monitor
vggt
logos/index
paper/index
style_guide
```