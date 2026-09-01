---
summary: Public vision pipeline and timestamp contract
tags: [vision, cameras, replay]
updated: 2026-08-31
audience: [dev, contributor]
---

# Vision

`rust/visiond/` ingests camera streams, records synchronized frames, and
supports replay with the shared URDF. It is designed so downstream code can see
which profile and timestamp domain produced a frame.

## Frame contract

Each capture record should identify the device, requested and active profile,
timestamp domain, sequence number, calibration revision, and dropped-frame or
transport warnings. A consumer must not silently substitute a profile or
calibration.

## Developing the pipeline

```bash
cd rust/visiond
cargo test
cargo build --release
```

Use fixture streams or recorded data for tests. Keep live addresses, camera
identifiers, credentials, and calibration snapshots in private acceptance
records rather than in this page.

## Evidence limits

Timestamp alignment is not geometric calibration, and a synchronized recording
is not proof of safe robot behavior. State those limits in every published
result.
