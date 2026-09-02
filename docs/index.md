---
summary: Public Tatbot developer and artist documentation
tags: [index, public]
updated: 2026-09-01
audience: [dev, artist, contributor]
---

# Tatbot

Tatbot is an open robotics project for exploring robot-assisted tattoo design
and placement. This site documents public technical interfaces: simulation,
teleoperation, vision concepts, Inkmap, contribution workflow, and safety scope.

```{admonition} Scope
:class: warning

The public project is a research and software collaboration. It is not a
tattooing service, medical device, or authorization to operate a robot around a
person. Use simulation or an instrumented non-human fixture while developing.
```

## Start here

- [Overview and installation](development.md)
- [Run the simulation](simulation.md)
- [For artists: design and placement data](design-format.md)
- [For developers: architecture](architecture.md)
- [Contributing](contributing.md)

## Public systems

- [Robot model and kinematics](robot.md)
- [Teleoperation](teleop_tuning.md)
- [Vision and fiducials](vision.md)
- [Imitation-learning interfaces](imitation_learning.md)
- [Inkmap preview](inkmap.md)
- [Tools and configuration](tools.md)
- [Run-log format](run_logs.md)

## Safety and reference

- [Safety scope](safety.md)
- [Hardware e-stop contract](estop.md)
- [Command-line reference](cli.md)
- [Configuration](configuration.md)
- [Documentation style](style_guide.md)
- [Release notes](release-notes.md)

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

development
simulation
design-format
architecture
contributing
robot
teleop_tuning
draw
vision
fiducials
ee_fiducial_tracking
imitation_learning
inkmap
ink
tools
run_logs
safety
estop
cli
configuration
style_guide
release-notes
logos/index
```
