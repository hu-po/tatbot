# Rust workspace

One Cargo workspace (`rust/Cargo.toml`). Every crate builds and tests with
fixtures, without access to the Tatbot deployment.

| Crate | What it is |
|---|---|
| `visiond` | camera ingestion: the wrist D405s and the overhead D555 (feature `realsense`), timestamped recording, synchronization, the frame socket and bus publisher (`zenoh`), and Rerun replay (`rerun`) |
| `tatbot-bus` | the versioned bus envelope, its Zenoh transport and fleet roles resolved from `config/nodes.json` |
| `trackd` | subscribers to a camera owner's frame socket: the wrist-cube tracker (`trackd`) and the stencil observer (`stencild`) |
| `fleetctl` | service presence on the bus: `advertise` while a unit runs, `services` to list and check them |
| `tatbot-arm` | the arm control boundary: a fault-injectable mock, and the native backend behind feature `trossen` |
| `trossen-arm-sys` | the CXX bridge to the pinned Trossen vendor SDK (feature `sdk`) |

## Build and test

```bash
cd rust
cargo test
cargo build --release --features rerun
```

`scripts/check rust` builds the workspace once with every feature this node
has and runs every crate's tests; `rust/check-profiles.json` names the
feature sets the deployed binaries are built with.

Optional capture backends should be selected explicitly. A frame record must
carry device identity, requested and active profile, timestamp domain, sequence,
calibration revision, and health flags.

## Replay boundary

Replay and synchronization checks validate data handling, not geometry or safe
robot behavior. Keep live endpoints, credentials, calibration snapshots, and
deployment runbooks in the private engineering tree.
