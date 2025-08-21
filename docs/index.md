---
summary: tatbot documentation index and quick links
tags: [index]
updated: 2025-08-21
audience: [all]
---

# 🤖 tatbot

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: development
:link-type: doc
:class-header: bg-light

🚀 Development Guide
^^^
Quick setup and development guide
+++
Install dependencies, configure nodes, and understand the source code architecture.
:::

:::{grid-item-card}
:link: nodes
:link-type: doc
:class-header: bg-light

🌐 Distributed Compute
^^^
Network topology & compute nodes
+++
Understand the distributed system architecture and node capabilities.
:::

::::

```{admonition} Quick Reference
:class: tip

**Essential Commands:**
- `./scripts/mcp_run.sh <node>` - Start MCP server
- `uv pip install .[bot,viz,cam]` - Install dependencies
- `uv run python -m tatbot.viz.teleop --enable-robot` - Launch teleop interface
```

## 🛠️ Core Systems

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
**🤖 Robot Arms and Vision**
- [Robot System](robot.md) - Trossen arms, URDF models, and inverse kinematics
- [Vision System](vision.md) - Cameras, AprilTag tracking, and 2D to 3D mapping
:::

:::{grid-item}
**🔗 Distributed Architecture** 
- [Nodes](nodes.md) - Network topology & compute nodes
- [Network Architecture](network_architecture.md) - Automatic dual-mode networking
- [MCP Protocol](mcp.md) - Model Context Protocol for distributed control
- [Tools System](tools.md) - Unified operation framework
- [Agent Interface](agent.md) - LLM-based control system
:::

:::{grid-item}
**🧠 Models & Training**
- [Datasets](plans/models/data.md) - Training data collection
- [Gr00t](plans/models/gr00t.md) - Foundation model
- [SmolVLA](plans/models/smolvla.md) - Vision-language-action model
- [VLA Plans](plans/vla_plan/index.md) - Model planning approaches
:::

:::{grid-item}
**🖼️ Artwork Generation**
- [Artwork Pipeline](artwork.md) - From images to tattoo designs
- [Tattoo Gear](gear.md) - Physical tattoo equipment
- [3D Visualization](viz.md) - Real-time robot visualization with Viser
:::

::::

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
plans/models/index
plans/vla_plan/index
plans/vggt/claude_plan
plans/vggt/codex_plan
plans/vggt/_prompts
plans/padding/claude_plan
plans/padding/codex_plan
plans/padding/cursor_plan
plans/padding/_prompts
```
