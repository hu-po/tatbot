# Arm control boundary

The default backend is the fault-injectable mock. Feature `trossen` adds the
native `TrossenArm` implementation through the pinned `trossen-arm-sys` CXX
bridge. It does **not** enable hardware session CLI commands.

```sh
CARGO_BUILD_JOBS=2 cargo test --manifest-path rust/Cargo.toml -p tatbot-arm --features trossen
```

The native constructor requires an explicit controller address, role, golden
configuration path, the shared monitor atomic and an `Arc<HardwareLease>`.
Only `HardwareLease::acquire()` can construct the production lease; it locks
`/tmp/tatbot-arm-driver.lock`, shared with C++ and LeRobot. The lease outlives
the vendor object. Construction performs no SDK connection. Use
`Worker::spawn_with` to create/use/drop the native object on the control thread;
there is no unsafe Send implementation. A constructor must not arm motion
before the worker begins its tick-level checks.

Native measurement separates six rotational joints from the linear carriage.
Known healthy SDK error strings are normalized; malformed feedback or mixed
modes cannot appear healthy. `CarriageMeasured.target_m` is optional: connection
and configuration invalidate it, and only a successful hold/command supplies
it. Unknown target means no valid contact assessment and no normal stream.
Measured re-seed establishes all axes before the six-joint stream starts.

Normal commands preserve that carriage target and check all seven positions
and average displacement rates against live controller limits. The stop atomic
is checked again after feedback reads, immediately before commanding. A failed
SDK command invalidates the target. The qualified follower-carriage trip retract remains
32 mm in 0.6 s; the leader gripper refuses this primitive. Its measured
verification, 20 N contact cap and 40-tick trip
window remain in the independent worker. A hold uses actual measured positions
and is allowed while STOP is latched; it never enables a mode or retracts.

Reconnect destroys the old SDK session and configures a fresh driver with the
legacy five-second timeout and error clear. Recover.Freeze owns the prior
freeze attempt: a wedged old session need not accept another hold to reconnect.
ClearFault loads the selected golden and reconnects, then checks the actual
firmware error. Configuration refreshes live limits and the selected standard
end-effector model. Measured rotary feedback may occupy the controller's
configured tolerance band; takeover clips its hold target to the nominal range.
The carriage remains exact because its boot-limit overtravel selects the
golden/re-read/clamp recovery path. Commanded targets retain nominal limits.
