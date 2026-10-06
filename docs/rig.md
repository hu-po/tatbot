# Rig sleep and wake

The rig's cameras stream, its bus runs and its hosts stay up around the
clock unless something switches them off. `tatbot rig sleep` does that for the
night and `tatbot rig wake` undoes it in the morning, both from any node with
ssh to the fleet. Nothing here moves an arm or cuts arm power.

```
tatbot rig sleep --plan          # what sleep would do to each node; no ssh
tatbot rig sleep                 # services down, hosts suspended
tatbot rig sleep --no-hosts      # services only
tatbot rig sleep --wake-at 07:00 # also arm every suspended host's clock alarm
tatbot rig wake                  # hosts up, services started and verified
tatbot rig status                # the marker, and every rig node's live state
```

## What sleep does

A *rig node* is a node whose `config/nodes.json` record carries a `power`
object. Sleep touches those nodes only, in the modes the record lists:

| mode | at sleep | at wake |
|---|---|---|
| `services` | stop the node's manifested `services` units in reverse order | start them in manifest order, then judge each active and not restarting |
| `suspend` | suspend the host to RAM | Wake-on-LAN (`wake: wol`, needs `mac`) from a node on the rig LAN, or the host's own clock (`wake: rtc`) |

Explicit `power.units` are stopped and restored in every mode.

```json
"power": {"sleep": ["services", "suspend"], "wake": "wol", "mac": "aa:bb:cc:dd:ee:ff"}
"power": {"sleep": ["services", "suspend"], "wake": "rtc", "units": ["tatbot-extra.service"]}
```

Stopping a RealSense owner is what idles the sensor: the ASIC and imagers run
only while a pipeline streams. The wrist camera that visiond owns stops with its
node's services. The other wrist camera is opened directly by the ROS 2 stack on
its own node, and sleep leaves that stack running. Host suspension requires a
separately qualified wake path.
A network camera behind an unmanaged PoE switch
keeps encoding regardless; stopping its consumer only idles the node that was
decoding it. Cutting that power is a smart plug or a managed switch, not a
software verb, and is deliberately not part of this command.

A node whose `wake` is `rtc` cannot be woken by anything but its own clock, so
sleep suspends it only when `--wake-at` names the time; otherwise it sleeps its
services and stays up. With `--wake-at`, every suspended host gets the same
alarm, which doubles as a safety net should a wake packet be lost.

Order: camera and arm nodes first, the bus router last at sleep so services can
deregister, and first at wake so they have a bus to register on. Every service
dials the bus at start and exits when it cannot, so if the router does not come
back nothing else is started and wake says so.

Give a host `suspend` only after proving, with `--wake-at` armed as the safety
net, that it answers ssh again after a magic packet. Sleep never changes the
kernel's sleep state (`/sys/power/mem_sleep`): a NIC that ignores the packet in
modern standby (s2idle) may not resume from deep S3 at all, and a host that
comes back neither way needs its power button. Such a host sleeps its
`services` only.

## Refusals

- **Exit 6** while any arm workflow is running on the arm node. Sleep reads
  the arm node's own `tatbot status` and never opens an arm connection; the
  arms stay landed and idle exactly as their last session left them. Land them
  with `tatbot arm recover` first if needed.
- **Exit 4** when run from a node that sleep would suspend: the command would
  cut itself off mid-way. Run it from an operator node, or pass `--no-hosts`.
- **Exit 5** when the arm node cannot be asked whether it is busy.

## The marker

Sleep writes `~/tatbot-logs/rig/state.json` on every node it touched and on
the node it ran from. While it says `asleep`:

- `tatbot status` shows `rig_power` as failed with the reason.
- Every motion verb, and every verb whose role sleep switched off (arm,
  e-stop, cameras, viewer), refuses with exit 5 and the fix
  `tatbot rig wake`. A dry run prints the note and still plans. The gate reads
  the invocation: a spelling that mints no launch id and names no arm or
  sensor still runs.
- Read-only `viewer status` remains available during sleep.

A host woken by its power button keeps the marker and its stopped services, so
the refusal holds until `rig wake` restores them. Wake works from the node map,
not from the marker, so it is safe to run at any time and idempotent. The node
the command runs on is driven through its own shell, never ssh, so a rig node
can sleep or wake itself alongside the rest without holding its own key.

Wake clears the marker on every node that can hold the gate, not only the rig
nodes: the node that ran `rig sleep` (the marker's `by`, taken from this node's
marker or, when the wake runs elsewhere, from the markers the rig nodes hold)
is required, and every other `operator` node in the map is best-effort. An
operator node that does not answer ssh is noted in the report, not a failure,
since a gate there could only come from a sleep it ran itself, which the
sleeper rule covers. A wake that cannot reach the sleeper (or finds it missing
from the map) fails and keeps the gate, like any other unfinished node.

A failed wake reports `state: wake_failed` and exits nonzero. The sleep gate
stays set until all planned hosts and services are ready;
waking a host alone does not clear it. The marker keeps the sleep's own
provenance (when the rig slept, from which node, which run) and records the
attempt under `wake_failed`, so a retry's note still names the sleep. Failure to publish a remote marker also
fails the command and keeps the local gate set. Inspect the run's per-node
errors, restore the unavailable dependency and run `rig wake` again.

## Run logs

Sleep and wake each write a run log (`rig-sleep`, `rig-wake`) with a
per-node `report.json`; `tatbot logs last rig-wake` is where a service that did
not come back is named, and `tatbot logs last fleet-service` on that node is
where its own console is.

## Caveats

- Suspending the arm node interrupts anything else it runs, including a
  training job under its `train` role. Sleep refuses only for arm workflows.
- The arm controllers are not touched. Idle motors hold no torque, so the arms
  draw little overnight; the e-stop's heartbeat silence during host suspend is
  already a stop, never a power cut.
- RealSense devices re-enumerate after a host resume; wake starts their owner
  fresh and reports it unhealthy if it restarts within the settle window.
