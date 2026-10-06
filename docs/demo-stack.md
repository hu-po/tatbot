---
summary: The demo stack, the portable two-arm subset of the rig - its hardware, network, e-stops and what runs where
tags: [demo, hardware, network, fleet]
updated: 2026-09-28
audience: [operator, dev]
---

# Demo stack

The demo stack is the smallest part of the rig that still runs both
demonstrations at once: the right arm drawing with a ballpoint through the
ROS 2 stack, and the left arm running a learned ink-tracing policy with a
non-emitting laser-pen prop. It has no dedicated viewer computer, no camera
computer and no PoE scene cameras; one networked depth camera does the
overhead work. Deployment-specific addresses and the device inventory are
private.

## Hardware

| Item | Role |
|---|---|
| Two Trossen WidowX AI arms, each with its controller and power supply | the right arm draws, the left arm runs the policy |
| Two NVIDIA Jetson Thor computers | the **ros node** drives the right arm; the **arm node** drives the left arm and runs the policy on its GPU |
| One unmanaged gigabit switch | the rig's wire: both computers, both controllers, an optional uplink |
| Two RealSense D405 wrist cameras | one per arm, on the USB of the computer that drives that arm |
| One RealSense D555 (PoE) | the only overhead camera |
| A Raspberry Pi 5 on the palette | relays the one hardware e-stop button to both arms over the network; also carries the palette's touch probe and camera |
| A PoE source | powers the D555 and the palette Pi |
| A portable power station | powers the computers, the switch and the arm supplies |

The ballpoint is a cartridge, so the drawing never dips; the palette's ink
caps are not used by either demo.

## Network

Everything shares one flat wire. The rig subnet (`__rig__` in
`config/nodes.json`) holds static addresses only: the arm controllers and the
D555 keep theirs in device configuration, and each computer carries its rig
address as a static secondary address on its wired profile, beside whatever
upstream address it gets. There is no rig gateway and no DHCP server on the
rig subnet, so the demos run with the uplink unplugged. An uplink adds
internet and the tailnet, which deploys and remote operation need.

The ros node also hosts the fleet bus router and is the fleet's Rerun viewer
endpoint (roles `bus-router` and `rerun-server`); neither demo needs either to
move an arm.

## E-stop

One button stops both arms. The palette Pi reads its normally-closed contact
and sends the same heartbeat to the drawing driver on the ros node
(`tatbot ros up --estop udp`) and to the policy runner's monitor on the arm node
(the profile's `driver.estop_device` names the relay). Each reader accepts the
relay's address only and stops its own arm on a press, a malformed stream, or
silence; the policy runner refuses to take its arm without a healthy
heartbeat. Contract: [E-stop](estop.md), `ros/README.md`.

## What runs where

| Computer | Runs | Drives |
|---|---|---|
| ros node | the ROS 2 drawing stack (`tatbot ros`); the D555's owner and the stencil observer, which place the page by its artwork, refined by the wrist camera and three guarded touches | the right arm |
| arm node | the ink-tracing policy runner, with the wrist camera | the left arm |
| palette Pi | the e-stop relay to both computers (and the probe relay, used only for tool calibration) | nothing |

The two stacks never share an arm: the ROS stack connects only to the right
controller, the policy runner only to the left. Each checks only its own arm's
links, so the arms are placed with their working areas apart.

## The overhead camera

The D555 replaces the PoE scene cameras. The functions they served move to it
one at a time:

| Function | Status |
|---|---|
| Scene view for the ink-tracing policy | in progress: needs DDS support on the arm node and a policy trained for the D555's lens and pose; wrist-only checkpoints run without it |
| Camera-to-arm registration | the right arm: `tatbot ros register` (its wrist tags at 27 holds the stack drives) |
| Page pose for stencil drawing | the right arm's page by its artwork on the table plane (`stencild`, identity unverified), then the wrist camera and touches |
| Palette and station tag measurement | `tatbot ros station --arm <arm>`: the palette's roof tag from the D555, through the arm's registration |
| End-effector fiducial tracking | not yet; not needed by either demo |

## Bring-up

```bash
tatbot status --fleet                  # every node, both arms reachable, e-stop state
tatbot ros deploy --units              # the drawing stack on the ros node
tatbot ros up --hardware real --page fixed --estop udp --arms right
tatbot ros status                      # units, arm safety state, what the stack runs with
```

Before the first draw on a new table, measure where the sheet lies: one slow
guarded touch from a rough position (`tatbot ros touch --start X Y Z`) and the
measured position in `page.fixed` in `ros/tatbot_bringup/config/stack.yaml`.
A starting estimate far below the paper lets the touch's fast approach reach
it unguarded.

The policy runner starts on the arm node once the relay's heartbeat reaches it: a
shadow run first (the policy plans, the arm stays still), then a holding run,
then a moving one.
