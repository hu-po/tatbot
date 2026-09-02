# Release Notes

## 2026-09-01

### Teleoperation & Leader Friction Compensation
Leader arm joint friction coefficients and effort correction factors have been retuned to improve transparency and tracking fidelity. Teleoperation CLI flags `--damping` and `--assist` now incorporate runaway protection and base-joint velocity filtering to prevent motion instability.

### CLI & Hardware Profile Interface
Hardware profile selection is now exposed via dedicated CLI verbs and flags, allowing profile configuration to be explicitly forwarded across remote execution hops. Dataset hub publishing defaults to opt-in execution, and log outputs standardise hints on CLI verb names.

### Imitation Learning & Policy Evaluation Contracts
Policy training and evaluation pipelines now enforce effort-masking contracts, preventing unmasked external effort inputs from reaching models trained on masked modalities. Policy evaluation routines fix state validation for 14-DOF policy states and guard against invalid held-out claims on training fixtures.

### Simulation & 3D Ink Mapping
Simulation contact geometry has been re-aligned with measured tool center points and scene reconfiguration start states. 3D tattoo mapping routines now define contracts for posed scenarios and align tool geometry specifications.

Commits: fed7217 8952f43 7a7727a 52988bd 7163798 c70808a 9fe6c84 fba0eff 89105dc 5595c3d e68f63a 24964dc 44f18ff 6f88de8 715deec f2cac69 726f09f 81cb9a3 0b1d133 265da2b 11cf126 16b09fb 89519de 9db6f1b bed0952 5f2af1d 10b9380 723f62a 8e28e69 81bbaf5 64839f9 49b244c 85d6ce8 459c9ee 55facda c5f9802 913a37d 88a7e3a c1315ac 767af82
