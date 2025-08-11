# Tatbot Documentation

Autonomous tattoo robot system with dual Trossen arms and distributed compute.

## Quick Start
- [Setup & Installation](setup.md) - Get tatbot running
- [Hardware Overview](nodes.md) - Network topology and compute nodes
- [Development Guide](dev.md) - Contributing to tatbot

## Core Systems

### Robot Control
- [Trossen Arms](trossen.md) - WidowXAI arm configuration and control
- [URDF Models](urdf.md) - Robot kinematic models
- [Inverse Kinematics](kinematics.md) - Motion planning algorithms

### Vision & Sensing  
- [Cameras](cameras.md) - RealSense depth and Amcrest IP cameras
- [AprilTags](apriltags.md) - Fiducial marker tracking for calibration
- [2D to 3D Mapping](mapping.md) - Surface projection algorithms

### Distributed Architecture
- [MCP Protocol](mcp.md) - Model Context Protocol for distributed control
- [Tools System](tools.md) - Unified operation framework
- [Source Code](source.md) - Module documentation and API reference

### Visualization & UI
- [3D Visualization](viz.md) - Real-time robot visualization with Viser
- [Agent Interface](agent.md) - LLM-based control system

## Specialized Topics

### Art Generation
- [Artwork Pipeline](artwork.md) - From images to tattoo designs
- [Tattoo Gear](gear.md) - Physical tattoo equipment

### Models & Training
- [Datasets](models/data.md) - Training data collection
- [Gr00t](models/gr00t.md) - Foundation model
- [SmolVLA](models/smolvla.md) - Vision-language-action model
- [VLA Plans](models/index.md) - Model planning approaches

### Development Resources
- [Progress Tracking](progress.md) - Development milestones
- [Ideas & Future Work](ideas.md) - Research directions
- [Network Refactor](network_refactor/index.md) - Architecture improvements
- [Academic Paper](paper/index.md) - Publication and research

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
models/data
models/gr00t
models/smolvla

network_refactor/index
paper/index
```


