---
summary: Fleet deploys, staged releases, supervised services and the bus presence tool
tags: [fleet, deploy, services]
updated: 2026-09-26
audience: [dev, operator]
---

# Fleet: deploy, releases and services

The camera, tracker, bus and viewer services run as systemd units on the nodes
whose roles own them. `config/nodes.json` is the service manifest: each node
lists the units it runs, and every deploy target is resolved from a role (the
fleet viewer, the camera node, the arm node), never from a hostname. The ROS 2
drawing stack is not part of this manifest; it deploys itself with
`tatbot ros deploy` (`ros/README.md`, section 12).

## `tatbot deploy`

```bash
tatbot deploy all --build-only     # stage and compile everywhere, install nothing
tatbot deploy all                  # stage, compile, install and restart the services
tatbot deploy NODE --service UNIT  # one manifested unit on one node; repeatable
```

A deploy builds only the pushed `origin/main`: it refuses when `HEAD` differs.
The source is the exact `git archive` of that commit, with a generated
`source-manifest.json` naming every file's bytes, mode and symlink target.
Every selected node builds successfully before any node installs anything.
Tracked edits and untracked files in a node's checkout are preserved, and a
node without enough free space for the release and the build cache refuses
before compiling.

A deploy never starts, stops or replaces an arm process or a Rerun viewer.
Every installed node exposes the deployed launcher as `/usr/local/bin/tatbot`.

## Staged releases

Each node stages a release under `~/.local/share/tatbot/releases/<sha>/`:

| Path | Contents |
| --- | --- |
| `source/` | The archived source, verified against `source-manifest.json` before and after compiling |
| `bin/` | The Rust service binaries the node's units run, `fleetctl`, and `zenohd` where a unit needs it |
| `build.json` | The build receipt (`tatbot.receipt/1`, kind `build`), written only after every build succeeded |
| `build.incomplete` | Present while a build is running or after one failed |

The arm node's release also builds, from `cpp/teleop`, the offline samples
planner `path_plan_check`, `wxai_teleop` and `arm_recover`. The receipt names
the sha256 of every binary a checkout resolves through it (`fleetctl`, and on
the arm node `path_plan_check`) and of the staged service executables, so an
unchanged source can reuse a complete earlier build.

A full deploy fast-forwards the node's checkout (the one `config/nodes.json`
names, which the CLI hop runs in) to the built revision and copies the receipt
there as `.tatbot-build.json`. `scripts/lib/fleet_release.py verify` compares
the checkout's `HEAD`, tracked edits and each named binary's digest with that
receipt without building anything. A `--service` deploy writes no receipt and
neither merges nor re-stamps the checkout.

Three releases are kept per node, plus any release a unit or the checkout's
stamp still names; older ones are pruned at install.

## Services

Units run `scripts/fleet_service.sh <service>` from their release with a run
log under `~/tatbot-logs/fleet-service/`. The script opens only camera, tracker
and bus processes; it never starts an arm driver or a viewer.

| Service | Role | Does |
| --- | --- | --- |
| `zenohd` | bus router | The fleet's Zenoh router |
| `visiond-poe` | PoE cameras | Decoded PoE capture, fiducial observations and the local frame socket |
| `visiond-d555` | overhead depth | The fixed D555's aligned RGB-D and snapshot queries |
| `visiond-d405` | wrist cameras | The wrist D405 this node owns (`rust/visiond/config/vision.toml`) |
| `trackd`, `trackd-left` | track | One EE fiducial tracker per wrist, on the shared frame sockets |
| `stencild` | track | The stencil observer: every visible print as `tatbot.target-pose/1` |
| `zenoh-presence` | bus router | A `fleetctl advertise` token while `zenohd` is active |

The ROS 2 stack reads the stencil pose and the EE fiducials from the bus
(`tatbot_bridge`); it owns its arm's wrist D405 itself, so that camera has no
`visiond-d405` unit.

## `fleetctl`

`rust/fleetctl` is the bus presence tool. `fleetctl advertise --node N
--service S --unit U` holds a liveliness token while a supervised unit is
active. `fleetctl services`
lists the tokens as `tatbot.services/1`, and with `--expected-sha` or
`--require` checks them; a full deploy runs it on the bus-router node after
installing to confirm each selected service came back on the deployed source.
It opens no arm and no serial device.

## Fleet observations

`tatbot status` includes a 200 ms aggregate host CPU sample from `/proc/stat`,
available/used memory from `/proc/meminfo`, and free space for the checkout and
run-log filesystems. CPU percentage uses all host CPU capacity as its
denominator; it is not a percentage of one core or a renderer measurement.

System and user systemd scopes report loaded Tatbot service states and restart
counts. An unavailable user bus is unknown. An empty list means no loaded
matching services were reported; it does not establish that all expected
services exist. Collecting these observations starts, restarts or repairs
nothing.

Fleet collection requests local collectors with `--no-service-discovery`, then
attaches one shared `fleetctl services` observation collected at the caller.
At most four node collectors share one overall deadline.

## Material target tracking

`tatbot vision track-target` publishes a measured rigid target (a tag layout
attached to the material) from the existing PoE frame-owner socket. Add a
`rigid_target` entry to `config/fiducials.json` with at least two IDs unused by
every other target, the measured tag edge length and a material
`parent_frame`, and measure its layout in the existing layout-file schema
(`schema_version: 2`, `calibration_status: calibrated`, matching inventory
digest; for a target, `ee_from_tag` means material-target-from-tag in metres).
Pending layouts and tag IDs shared with the wrist, board or palette are
refused.

Run it on the tracking node with the target ID, the layout, the camera
calibration, the bus endpoint, the frame socket and an output path outside the
source tree. It opens no cameras or arm connections and runs beside wrist
tracking. It publishes `tatbot.target-pose/1` on
`tatbot/tracking/target/<target_id>` with capture time, producer identity,
calibration and layout digests, observed IDs, world pose and
translation/rotation uncertainty. Nothing it publishes authorizes motion.
