# Rig sleep and wake

The rig's cameras stream and its trackers run around the clock unless
something switches them off. `tatbot rig sleep` does that for the night and
`tatbot rig wake` undoes it in the morning, both from any node with ssh to the
fleet. Neither moves an arm, cuts arm power, or touches the drawing stack, the
bus it holds a session to, or the e-stop relay. `tatbot rig off` is for the end
of a demo: it lands the arms and shuts the hosts down so the power strip can be
switched off without a hard power cut.

```
tatbot rig off --plan            # the landing and shutdown order; no ssh
tatbot rig off                   # land, then shut down every "off" node, this one last
tatbot rig sleep --plan          # what sleep stops, closes and leaves up per node; no ssh
tatbot rig sleep                 # cameras and trackers down, the demo window closed
tatbot rig wake --plan           # the start stages and the check each must pass; no ssh
tatbot rig wake                  # services started and verified on the bus, the window reopened
tatbot rig status                # the marker, and every rig node's live state
tatbot rig check                 # is every piece of demo hardware up and in place
tatbot rig lan-install --plan    # the generated ssh fallback and clock files; no ssh
tatbot rig lan-install           # install them on every demo-case host
```

## What sleep does

A *rig node* is a node whose `config/nodes.json` record carries a `power`
object. Sleep touches those nodes only, in the modes the record lists:

| mode | at sleep | at wake |
|---|---|---|
| `services` | stop the node's running manifested `services` units: displays (part `station`, such as the demo cockpit) first, best-effort, then the rest in reverse manifest order; never the bus router or its presence reporter | start the enabled ones in stages and verify them on the bus (below) |
| `window` | close the node's demo viewer window (`tatbot viewer stop`, which closes only the window anywhere but the hub's node, so the hub's node may not carry this mode) | open it, best-effort, with the flags the station's launcher uses at boot (`tatbot viewer open --memory-limit 4GB --fullscreen`), behind the window that holds the focus; a window that runs is left as it is |

```json
"power": {"sleep": ["services"]}
"power": {"sleep": ["window"]}
```

On the demo rig the ros node is a `services` node: sleep stops the demo
cockpit, then the tool tracker, the stencil tracker and the overhead D555's
owner, trackers first so they do not exit when their camera goes. Stopping a
RealSense owner is what idles the sensor: its ASIC and imagers run only while
a pipeline streams. The station is a `window` node: its demo viewer window is
the one display that renders all night.

What stays up, and why:

- **The ROS 2 drawing stack and its router.** `tatbot ros up|down` own them.
  The arm stays connected and as its last session left it; sleep reports each
  arm's safety state (latched, landed), never lands it, and refuses while it
  has not landed (below).
- **The fleet bus router and its presence reporter.** The drawing stack's
  bridge holds a session to the router, and a night-long outage under it has
  never been shown to heal; the two idle at a few hundredths of a core.
- **The viewer hub and both load monitors.** The drawing stack's Rerun bridge
  reaches a restarted hub only by being respawned, so a stopped hub would
  respawn it all night; they idle at under a hundredth of a core.
- **The station's agent, its model, the image generator and the keys relay.**
- **The e-stop relay.** Its silence stops and latches a running arm, so a
  `power` record on a node with role `estop-relay` is refused.
- **The arm controllers, switches and rig router**, which have no software off,
  and every node without a `power` record. Sleep suspends no host.

`--plan` prints all of this per node: what stops and in which order, what is
closed, what stays up, where the busy check asks and where the marker goes.

The station's launcher (`station/bin/station-launch`) opens the same window
at boot by the same rules: only once the hub answers on the direct cable, and
not while the station's marker says the rig sleeps.

## What wake does

Per node, the bus router's node first:

1. **The bus.** The units that carry the router's own liveliness key (the
   router and its presence reporter) are started, a no-op when they run, and
   the key must answer on the bus within 20 s (`fleetctl services --require`).
   If it does not, no node's other services are started: each dials the
   router at start and exits without it.
2. **The services.** The node's other enabled services start together without
   blocking. Each must be active within 150 s (the D555 owner's start timeout
   is 120 s, and a start that lost the race to the bus heals on its own 5 s
   restart), hold its liveliness key on the bus, and not restart over the next
   5 s. A unit that is not enabled was off before the night too, and is left
   as it is.
3. **The displays**, best-effort: the cockpit, and the node's display
   `hardware` units (the viewer hub and its load monitor) in case one died.
4. **The window**, best-effort, once every node's services are up, and only
   over the node's direct cable to the hub where it has one: a window
   streaming over the rig port (and a reconnect's replay of the hub's whole
   buffer) delays the arm controller's packets queued behind it. When the
   cable does not answer, wake leaves the window closed and says how to open
   it.

Then, as notes that never hold the gate: the drawing stack's state from
`tatbot ros status` (each arm latched or not and landed or not, the page on
the bus), waiting up to 60 s for a measured page while the stack follows the
stencil, and `tatbot rig check`, part by part.

## What off does

1. Refuses (exit 6) while any workflow but a service's or the drawing
   stack's runs on a demo host it would shut down, or a drawing or camera run
   is open on this node or an operator node.
2. Runs `tatbot ros down`, which lands every arm that has not landed and stops
   the drawing stack. If the landing fails, nothing is shut down and off exits
   with the landing's own exit code. A ros node that does not answer ssh is
   noted and skipped.
3. Shuts down every node whose `config/nodes.json` record says `"off": true`,
   one at a time, waiting until each stops answering ssh: the arm and camera
   hosts first, then the e-stop relay (it is the stop path while an arm host
   is up), then the node running the command, five seconds after it prints
   its report. `--no-self` leaves that node on.

The arm controllers, switches and router are on the strip with no software
off; once every screen and status light is dark, switch the strip off. The
hosts come back when power returns or their power buttons are pressed, with
their services enabled as before; off leaves no marker.

## What check does

`tatbot rig check` probes every demo-case host (the nodes marked `"off": true`)
once, in parallel, and reports three parts: **pink drawing**, **station** and
**blue arm**, each READY or NOT READY. It is read-only: it pings, lists devices
and reads unit state, and never starts, stops or opens anything. It exits 5
while any part has a failed check; warnings never fail it. `--all` lists the
passing checks too, `--json` the full report.

| check | fails | warns |
|---|---|---|
| host | does not answer ssh (often a Thor waiting for its power button) | clock not NTP-synced, more than 2 s off this node's, disk under 10 % free, 85 C or hotter, a failed systemd unit |
| rig link | the node's `lan` address is missing; the D555 owner's link is under MTU 9000 | negotiated under 1 Gb/s |
| direct links | | a static NetworkManager profile whose interface has no carrier (an SFP peer off or unplugged) |
| USB | a wrist D405 missing from its owner's USB, or a `hardware` USB id missing | a D405 below USB 3 |
| devices, units | a `hardware` device path missing; a manifested or `hardware` unit not active | |
| peers | the rig router, the arm controllers or the D555 do not answer ping from the node that uses them | |

Expectations come from the files that own them: roles and `services` in
`config/nodes.json`, the wrist cameras' owners in vision.toml, the controller
addresses in the hardware profile. A node's `hardware` list names what nothing
else records, each entry with a `name`, an optional `part`, and one of `usb`
(`vid:pid`), `dev` (a path), `unit` or `user_unit`; a template unit
`name@.service` is expected as the node's login-user instance. A manifested
service may name its own `part` too. The demo display (the viewer hub, the
cockpit, the station's window and both load monitors) is part `station`, so a
display never makes `pink drawing` NOT READY, and a failed unit that an entry
expects is reported once, under its own part. The station touchscreen's
USB hub is powered by the panel itself, so it is missing until the panel's
own power button is on. While the rig sleeps, check reports what sleep
stopped (the trackers, the D555's owner, the window) as not active.

## What lan-install does

The rig has to run with no uplink at all. Two things otherwise depend on one:
every `ssh` target in `config/nodes.json` is a tailnet address, and the tailnet
may not come up after a cold boot with no internet; and hosts without a clock
battery boot on their last shutdown time. `tatbot rig lan-install` installs a
fallback for each on every demo-case host (the nodes marked `"off": true`),
generated from the node map (`tatbot_cli.rig_lan`):

- **ssh.** `~/.ssh/tatbot-rig.conf`, included as the first line of
  `~/.ssh/config` (the old file is kept once as `config.bak-tatbot-rig`). For
  each node with an ssh target and a `lan` address on the rig subnet it holds a
  `Match` block: while this host has an address on the rig subnet, the node's
  tailnet address does not answer ssh within a second and its rig address
  does, ssh dials the rig address, checking the host key already trusted for
  the tailnet address (`HostKeyAlias`). The block also matches the node's name
  and hostname alias. Only a fallback: over the tailnet a node may be served by
  Tailscale SSH, which admits by tailnet identity, while its rig address admits
  only `authorized_keys`, so a direction that works over the tailnet needs its
  key in the target's `authorized_keys` to work offline, allowed from the rig
  subnet where the entry has a `from=`. The key offered is the one ssh picks
  for the name as typed, not one a `Host` block names for the rig address.
- **Clock.** The node with role `rig-clock` (one whose clock keeps time through
  a power-off) gets chrony, installed with apt if missing, with a drop-in that
  serves the rig subnet and, with no internet source reachable, its own clock
  at stratum 10. Every other demo-case host takes its time from that node
  first: a chrony host through a preferred `server` in
  `/etc/chrony/sources.d/`, a systemd-timesyncd host through
  `NTP=<rig clock> pool.ntp.org` in `/etc/systemd/timesyncd.conf.d/`. The verb
  waits up to 30 s for each client to report the rig clock as its source.

It is idempotent: an unchanged file is not rewritten and its service is not
restarted. It touches no arm, camera, network profile or drawing-stack unit.
The first run needs an uplink if chrony is missing on the rig clock. It writes
a `rig-lan-install` run log with a per-node `report.json`, and exits 5 when a
host did not answer ssh. `tatbot rig check` reports the rig clock's chrony as a
`station` unit, and warns on any host whose clock is not synced or is more
than 2 s off the checking node's.

## Refusals

Sleep checks these first, in this order, and changes nothing when one refuses:

- **Exit 3** while the drawing stack runs on real hardware and reports an arm
  that has not landed: sleep moves no arm, so it would hold its pose all
  night. `tatbot ros down` lands each arm that answers and stops the stack;
  an arm whose controller reports an error lands only after a restart, by
  `tatbot ros up` with the stack's flags (the refusal prints them). A stack
  that is down or does not answer holds no arm and refuses nothing.
- **Exit 6** while a drawing or camera run is open (`ros-*` but `ros-stack`,
  `vision*`, `calib*`, `surface-rgbd`) on a rig node, on the node running
  sleep or on any operator node: a draw, touch or calibration on the ros node,
  or the `ros-cli` run of a setup started from the station or a workstation,
  which lives on the node it ran from. Training and simulation runs do not
  refuse. Sleep reads each node's own `tatbot status` and never opens an arm
  connection; a node that does not answer ssh is noted and left as it is. The
  check runs right before the first stop, so a run opened after it is not
  seen.
- **Exit 5** when a node answers but cannot say what it runs, or does not say
  it in time.

## The marker

Sleep writes `~/tatbot-logs/rig/state.json` on every rig node it reached, on
the node it ran from and, where they answer, on every other operator node.
While it says `asleep`:

- `tatbot status` shows `rig_power` as failed with the reason.
- Every motion verb (the drawing stack's draw, setup, touch, jog and the
  rest), and every verb whose role sleep switched off (the overhead camera's
  owner, the trackers), refuses with exit 5 and the fix `tatbot rig wake`. A
  dry run prints the note and still plans. The gate reads the invocation: a
  spelling that mints no launch id and names no arm or sensor still runs.
- The gate is the invoking node's own marker. The station is a rig node and
  holds one, so its agent's motion verbs refuse too.
- `tatbot ros up|down|status`, the `viewer` verbs and read-only status remain
  available during sleep.

A node rebooted overnight starts its enabled services at boot but keeps the
marker, so the refusal holds until `rig wake` verifies them. Wake works from
the node map and each unit's enabled state, not from the marker, so it can run
at any time: on an awake rig its starts are no-ops, a window that runs is left
as it is, and a failed attempt sets no gate. The node the command runs on is
driven through its own shell, never ssh, so a rig node can sleep or wake
itself alongside the rest without holding its own key.

Wake clears the marker on every node that can hold the gate. Required: each
`services` node it reached and the node that ran `rig sleep` (the marker's
`by`, taken from this node's marker or, when the wake runs elsewhere, from the
markers the rig nodes hold). A wake that cannot reach the sleeper (or finds it
missing from the map) fails and keeps the gate everywhere, like any other
unfinished `services` node. Then every other rig node and every `operator`
node, best-effort: a `window` node the sleep reached that does not take the
marker now (it does not answer) keeps refusing its own motion verbs. Wake
clears the gate everywhere else, names that node under `gate_kept` and exits
1; run `rig wake` again once it answers.

A failed wake reports `state: wake_failed` and exits nonzero. The sleep gate
stays set until every `services` node's services pass their checks. The marker
keeps the sleep's own provenance (when the rig slept, from which node, which
run) and records the attempt under `wake_failed`, so a retry's note still
names the sleep. Failure to publish a required remote marker also fails the
command and keeps the local gate set. On a rig that was not asleep, a failed
wake writes no gate at all. Inspect the run's per-node errors, restore the
unavailable dependency and run `rig wake` again.

## Run logs

Sleep and wake each write a run log (`rig-sleep`, `rig-wake`) with a
per-node `report.json`; `tatbot logs last rig-wake` is where a service that did
not come back is named, and `tatbot logs last fleet-service` on that node is
where its own console is.

## Caveats

- The arm controllers are not touched, and the arm stays connected to the
  drawing stack all night. Sleep refuses an arm that has not landed; one that
  latched before the night needs `tatbot ros up` before the next session,
  whatever wake reports.
- A RealSense owner starts fresh at wake; one that restarts within the steady
  window is reported unhealthy and keeps the gate.
