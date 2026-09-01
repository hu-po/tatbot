# Teleoperation

This directory contains the low-latency C++ leader/follower teleoperation
component. It is a public build target; deployment addresses, CPU pinning, and
measured tuning baselines are kept in private engineering records.

## Build

```bash
cmake -B build -S .
cmake --build build
ctest --test-dir build --output-on-failure
```

The executable accepts explicit controller and safety-device configuration. Do
not hard-code a site address or bypass the e-stop in a contribution.

## Design invariants

- The loop must preserve the leader/follower mapping and units.
- E-stop transitions hold measured state and reject stale targets.
- Teleoperation and another exclusive arm driver cannot run concurrently.
- A run records the source revision and the effective configuration.

Use simulator/replay fixtures for new behavior. Physical acceptance belongs to
the private safety record and is never implied by a successful build.
