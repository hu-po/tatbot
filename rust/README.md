# visiond

`rust/visiond` is the public Rust component for camera ingestion, timestamped
recording, synchronization, and replay. It can be built and tested with
fixtures without access to the Tatbot deployment.

## Build and test

```bash
cargo test
cargo build --release
```

Optional capture backends should be selected explicitly. A frame record must
carry device identity, requested and active profile, timestamp domain, sequence,
calibration revision, and health flags.

## Replay boundary

Replay and synchronization checks validate data handling, not geometry or safe
robot behavior. Keep live endpoints, credentials, calibration snapshots, and
deployment runbooks in the private engineering tree.
