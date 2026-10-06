# Trossen vendor boundary

Feature `sdk` builds a CXX bridge against the same unmodified v1.8.5 vendor
revision as `cpp/teleop` (`fdfd9f68f57b3bd05c4e85011fa5c11296525b2f`). Set
`TROSSEN_ARM_SDK_ROOT` to that SDK checkout, or use the CMake download under
`cpp/teleop/build/_deps/trossen_arm_sdk-src`. No network fetch occurs in the
build script. Linux x86_64 and aarch64 libraries are supported.

```sh
CARGO_BUILD_JOBS=2 cargo test --manifest-path rust/Cargo.toml -p trossen-arm-sys --features sdk
```

Native tests exercise pure argument validation and an unconfigured vendor
object. They never call vendor configuration, change modes or send positions;
all controls refuse before reaching the SDK. They verify seven-element/finite/
time checks, failed configuration guards and C++ exception translation. The
unconfigured-object test is also traced for connect/send syscalls.
`scripts/check rust` includes it when the SDK cache is present, otherwise
reports a named SKIP. Default builds contain no vendor interface.

This crate is the ABI used by the feature-gated `tatbot-arm::trossen` adapter. It exposes explicit
configuration, measurement, mode, nonblocking seven-joint command, immediate
measured hold, configuration load and cleanup. It never requests controller
reboot. Six joint positions are radians and carriage position is metres; the
same mixed units apply to velocity and acceleration. The caller must supply
the physical e-stop path, driver lease, measured re-seed, calibrated tool
geometry, limits, preflight and CLI gates. Configuration load reapplies the
selected standard leader/follower end-effector model, matching the existing
C++ configure path; old arm files cannot silently replace that model. A failed
configuration never enables subsequent bridge calls.

`limits()` reads all seven live controller limits, including position, velocity,
effort and their tolerances; malformed or non-finite limits refuse. `hold()`
returns the exact seven measured positions submitted to the SDK after it accepts
the command. This receipt lets the adapter retain the actual carriage
target rather than inventing it from a later measurement. It does not verify
that the controller executed the hold. No command target is inferred at connect.

This feature enables no hardware command line of its own. Passing these tests
does not qualify a tool or authorize human contact. Fixtures are
paper or silicone, never a person.
