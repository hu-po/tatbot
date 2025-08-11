# 🎨 Tatbot Documentation

Welcome to **tatbot** - an autonomous tattoo robot system with dual Trossen arms and distributed compute.

::::{grid} 1 1 2 3
:class-container: text-center
:gutter: 3

:::{grid-item-card}
:link: setup
:link-type: doc
:class-header: bg-light

⚡ Quick Setup
^^^
Get tatbot running in minutes
+++
Install dependencies, configure nodes, and start your first session.
:::

:::{grid-item-card}
:link: nodes
:link-type: doc
:class-header: bg-light

🖥️ Hardware Overview
^^^
Network topology & compute nodes
+++
Understand the distributed system architecture and node capabilities.
:::

:::{grid-item-card}
:link: dev
:link-type: doc
:class-header: bg-light

🛠️ Development Guide
^^^
Contributing to tatbot
+++
Development workflow, coding standards, and how to contribute.
:::

::::

```{admonition} Quick Reference
:class: tip

**Essential Commands:**
- `./scripts/run_mcp.sh <node>` - Start MCP server
- `uv pip install .[bot,viz,cam]` - Install dependencies
- `uv run python -m tatbot.viz.teleop --enable-robot` - Launch teleop interface

**Key Nodes:** {{ook}} • {{oop}} • {{ojo}} • {{eek}} • {{hog}} • {{rpi1}} • {{rpi2}}
```

## 🔧 Core Systems

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
**🦾 Robot Control**
- [Trossen Arms](trossen.md) - WidowXAI arm configuration and control
- [URDF Models](urdf.md) - Robot kinematic models  
- [Inverse Kinematics](kinematics.md) - Motion planning algorithms
:::

:::{grid-item}
**👁️ Vision & Sensing**
- [Cameras](cameras.md) - RealSense depth and Amcrest IP cameras
- [AprilTags](apriltags.md) - Fiducial marker tracking for calibration
- [2D to 3D Mapping](mapping.md) - Surface projection algorithms
:::

:::{grid-item}
**🌐 Distributed Architecture** 
- [MCP Protocol](mcp.md) - Model Context Protocol for distributed control
- [Tools System](tools.md) - Unified operation framework
- [Source Code](source.md) - Module documentation and API reference
:::

:::{grid-item}
**📊 Visualization & UI**
- [3D Visualization](viz.md) - Real-time robot visualization with Viser
- [Agent Interface](agent.md) - LLM-based control system
:::

::::

## 🎨 Specialized Topics

::::{tab-set}

:::{tab-item} Art Generation
:sync: art

**🖼️ From Concept to Ink**
- [Artwork Pipeline](artwork.md) - From images to tattoo designs
- [Tattoo Gear](gear.md) - Physical tattoo equipment

```{note}
The art generation pipeline transforms digital images into precise tattoo strokes through G-code parsing, surface mapping, and trajectory optimization.
```
:::

:::{tab-item} AI Models & Training  
:sync: models

**🤖 Machine Learning Pipeline**
- [Datasets](models/data.md) - Training data collection
- [Gr00t](models/gr00t.md) - Foundation model
- [SmolVLA](models/smolvla.md) - Vision-language-action model
- [VLA Plans](models/index.md) - Model planning approaches

```{note}
Tatbot uses vision-language-action models for autonomous control, trained on demonstration data from human operators.
```
:::

:::{tab-item} Development & Research
:sync: dev

**📚 Development Resources**
- [Progress Tracking](progress.md) - Development milestones
- [Ideas & Future Work](ideas.md) - Research directions
- [Network Refactor](network_refactor/index.md) - Architecture improvements
- [Academic Paper](paper/index.md) - Publication and research

```{tip}
Check the progress page for the latest updates and roadmap status.
```
:::

::::

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

setup
nodes
dev
trossen
urdf
kinematics
cameras
apriltags
mapping
mcp
tools
source
viz
agent
artwork
gear
progress
ideas

models/index
network_refactor/index
paper/index
```


