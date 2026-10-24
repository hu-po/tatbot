---
summary: The demo stack, the portable subset of the rig - its hardware, network, e-stops, what runs where and the demo viewer
tags: [demo, hardware, network, fleet, viewer]
updated: 2026-10-22
audience: [operator, dev]
---

# Demo stack

The demo stack is the smallest part of the rig that runs the demonstration:
the right arm drawing with its fitted tool through the ROS 2 stack, and the
station beside it carrying the agent people talk to and the demo viewer. The
left arm is the teleop leader alone and is out of the demo. The stack has no dedicated viewer computer, no
camera computer and no PoE scene cameras; one networked depth camera does the
overhead work. Deployment-specific addresses and the device inventory are
private.

## Hardware

| Item | Role |
|---|---|
| Two Trossen WidowX AI arms, each with its controller and power supply | the right arm draws; the left arm is not in the demo |
| Two NVIDIA Jetson Thor computers | the **ros node** drives the right arm; the **arm node** holds the left arm for teleop |
| One unmanaged gigabit switch | the rig's wire: both computers, both controllers, an optional uplink |
| Two RealSense D405 wrist cameras | both on the right wrist, one above it and one under it, on the ros node's USB |
| One RealSense D555 (PoE) | the only overhead camera |
| A Raspberry Pi 5 on the palette | relays the one hardware e-stop button over the network (to the arm node too when its profile names the relay); also carries the palette's touch probe and camera |
| A PoE source | powers the D555 and the palette Pi |
| A portable power station | powers the computers, the switch and the arm supplies |
| A DGX Spark-class computer with a portrait touchscreen and a keyboard, the **station** | the agent people at a demo talk to, and the demo viewer's window |
| A direct 10 GbE cable from the station to the ros node | the demo viewer's stream, off the rig wire |

The fitted tool is the one `config/workspace.yaml` records. A needle cartridge
dips in the palette's ink caps; a ballpoint never does.

## Network

Everything shares one flat wire behind a travel router, the rig gateway
(`__rig__` in `config/nodes.json`). It is the only DHCP server on the rig
subnet: the arm controllers and the D555 keep static addresses in device
configuration, the computers take DHCP reservations and also carry their rig
address as a static secondary, so the arms, cameras and e-stop keep talking if
the router is down. The router's uplink is Wi-Fi in repeater mode, a venue
network or a phone hotspot; with no uplink the rig is a closed local network
and the demo still runs. An uplink adds internet and the tailnet, which
deploys and remote operation need. Keep every other network off the rig wire.

The arm controllers' network stack is small. A router that ARP-probes its
clients every few seconds made one controller drop most of its packets, and
a dropped connection idles the arm mid-motion. On the GL.iNet router the rig
uses, its client tracker and traffic inspection are therefore removed at every
boot.

The ros node also hosts the fleet bus router and is the fleet's Rerun viewer
endpoint (roles `bus-router` and `rerun-server`); the demo needs neither to
move the arm.

The ros node's one rig port carries both the right arm's controller and
anything the viewer hub sends over the rig wire. The station's viewer window
therefore dials the hub over the direct cable while the station's kernel
routes the hub's cable address off the rig interface and the hub answers there
within a second; otherwise it dials the rig address, so a dead cable never
costs the window. `tatbot viewer status` names the path. A window keeps the
path it opened on.

## E-stop

One button stops the drawing arm. The palette Pi reads its normally-closed
contact and sends a heartbeat to the drawing driver on the ros node
(`tatbot ros up --estop udp`). The reader accepts the relay's address only and
stops the arm on a press, a malformed stream, or silence. Contract:
[E-stop](estop.md), `ros/README.md`.

## What runs where

| Computer | Runs | Drives |
|---|---|---|
| ros node | the ROS 2 drawing stack (`tatbot ros`), with its Rerun bridge and the wrist camera's display tap; the D555's owner, the stencil observer, which places the page by the stencil's markers, refined by the wrist camera and three guarded touches, and the wrist-cube tracker; the viewer hub, the demo cockpit and a load monitor | the right arm |
| palette Pi | the e-stop relay (and the probe relay, used only for tool calibration) | nothing |
| station | the demo agent; the demo viewer's window and a load monitor | nothing |

The ROS stack connects only to the right controller and models only that arm;
the left arm is not in its description, so it must stay clear of the working area.

## The overhead camera

The D555 replaces the PoE scene cameras:

| Function | How |
|---|---|
| Camera-to-arm registration | `tatbot ros setup`: a coarse `tatbot ros register` (the wrist tags at the holds the stack drives), then `tatbot ros calib overhead --adopt` through the wrist camera, the registration kept |
| Page pose for stencil drawing | the stencil observer (`stencild`) places the fiducial tile by its markers; the wrist camera and touches refine it |
| Palette and station tag measurement | `tatbot ros station --arm <arm>`: the palette's tag from the D555, through the arm's registration |
| End-effector fiducial tracking | `trackd` on the ros node: the wrist cube's tags in the D555 |

## The demo viewer

The station's portrait screen shows a drawing session on the fleet Rerun
viewer's **Demo** tab, the first of its tabs. The station opens it at boot,
fullscreen behind the agent terminal, and the operator switches between the
two. Top to bottom:

| Row | Shows |
|---|---|
| Status board | run, op, phase (with the arc along a stroke being drawn), the pen trim and ink over e-stop, probe, machine, latch and wrist: one bar each, green when fine, yellow when stale or warning, red on STOP or a latch, grey when not applicable or the run has ended. Outside an ink op the ink bar lists the caps loaded and available to dip, with their recorded fill while two lines hold them, else their names. The trim bar is the depth keys' trim of the cartridge on the arm and its limit (`trim -1.5 of ±4.0 mm`), yellow at the limit, and yellow with the depth a needle stroke holds it at while keys pressed past that are held back (`trim -4.0 · held -2.65 mm`: deeper than 2.5 mm reaches only skin the wrist read sunk) |
| Scene | in the arm base's frame, z up, so the table reads level however the D555 is tilted, from a fixed eye: the D555's RGB depth cloud out to 0.85 m (faintly tinted blue), the robot, the page or its fitted surface, the strokes planned, done and being drawn, the tip trail, the palette in the arm's black with its tag and its caps by state, each marked over its rim (loaded and available to dip in green, empty or no cap grey, unknown yellow, the next dip cyan) and labelled in a column beside it: a loaded cap's ink, recorded level and fill and the run's dips, the others grouped by state, the wrist tags on the end effector's cube, and the wrist D405's RGB cloud (faintly tinted amber) |
| Design, Wrist | the design map in the print's millimetres, centred on the design, each drawn stroke on the print where the design has it (less the correction stencil tracking drew it with), over that print's markers in faint violet; beside it the wrist D405's colour image upright in a 9:16 pane, with its age, the print's markers each reading decoded in that frame (green used, amber in a rejected fit, red left out) |
| D555, load | the D555's colour image with its age and the fiducials the trackers read in it outlined where they read them: the wrist cube's tags (blue those the pose was solved from, grey the others, with their ids) and the print's markers the stencil observer kept (amber), beside CPU % for the palette Pi, the station and the ros node, and TEMP for them and the robot (its hottest motor), each labelled with its host's platform and keyed by one legend, over the last ten minutes |

| Producer | Runs on | Sends |
|---|---|---|
| viewer hub (`tatbot-viewer@`) | ros node | buffers 512 MB of the standing recording and serves the windows and the browser viewer |
| demo cockpit (`tatbot-cockpit.service`) | ros node | the D555's colour (640x360 at 5 Hz) and depth (320x180 at 1 Hz), the wrist colour (2 Hz) with each wrist reading's markers drawn on it, the trackers' poses with the fiducials they read outlined on the D555's colour, every image chunk within 64 KiB, and the D555's depth again as the Scene's RGB cloud: about 0.4 MB/s |
| the drawing stack's Rerun bridge (`tatbot_rerun`) | ros node, in the stack | robot, strokes, design map, page, palette, wrist cloud and status board |
| the session's wrist tap | ros node, in the session | the wrist D405 frames the session already takes, and the print's markers each reading decodes in them, on the bus; only while a draw, or a page touch's set-up, has the camera open |
| load monitors (`tatbot-sysmon@`) | ros node and station | CPU, memory and temperature at 1 Hz; the station's adds its GPU and reads the palette Pi over ssh (nothing installed there); the ros node's adds the robot from the drawing stack's run files |
| the window (`tatbot-viewer-window`) | station | renders the page, capped at 4 GB |

All of it is display only: nothing in it gates, latches or stops motion. The
window, the hub, the cockpit and both load monitors are part `station` in
`tatbot rig check`, so a display never makes `pink drawing` NOT READY.

- Deploy or restart any of it between draws only. A fresh install goes hub
  first: `tatbot viewer install` (it restarts the hub empty, and the cockpit
  and load monitor beside it), the cockpit's
  `tatbot deploy <ros node> --service tatbot-cockpit.service`,
  `tatbot ros deploy` (it also installs the bridge's SDK) and `tatbot ros up`,
  then on the station `tatbot viewer station-install` and
  `tatbot viewer open --restart --memory-limit 4GB --fullscreen`.
- A window that connects receives the hub's whole buffer at once, oldest
  first. Over the rig port that burst can delay the controller's packets, so a
  window on the rig path, or a browser viewer on the rig LAN, is not opened or
  restarted during a draw.
- An ad hoc `tatbot live cockpit` or `tatbot viewer view` opens a recording of
  its own, and the station's page switches to it: not during a demo.

Details: [Vision](vision.md) (the viewer, the Demo tab and the cockpit) and
`ros/README.md` (the bridge and the wrist tap).

## Bring-up

```bash
tatbot status --fleet                  # every node, both arms reachable, e-stop state
tatbot ros deploy --units              # the drawing stack on the ros node
tatbot ros up --hardware real --estop udp --arms right
tatbot ros status                      # units, arm safety state, what the stack runs with
tatbot ros setup --arm right           # register, station probe, gauge, then the page
```

The demo viewer follows its install order above.
