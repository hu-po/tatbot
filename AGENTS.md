# Tatbot

Dual Trossen WXAI arms, imitation learning with LeRobot. Independent build
roots, no root Python package:

- `cpp/teleop/`: 400 Hz leader→follower teleop. CMake; the Trossen SDK is
  fetched at configure time and the hardware-independent libraries build and
  test without it.
- `rust/visiond/`: camera ingestion, sync, recording, replay. Cargo. Sensors
  are described in a registry file — see `rust/visiond/config/vision.example.toml`.
- `python/lerobot_robot_tatbot/`: LeRobot leader/follower plugins. uv,
  Python 3.12. `mock_driver.MockDriver` is the hardware-free backend.
- `python/tatbot_sim/`: ManiSkill data factory (x86_64 only).
- `web/inkmap/`, `web/inkgen/`: tattoo mapping and design generation.
- `firmware/estop_pico/`: hardware e-stop heartbeat firmware.
- `scripts/tatbot`: the CLI. `scripts/check` runs every check this tree can.

Hardware requires a complete profile in `config/profiles/` — see
`trossen-wxai.json` (fill in your arms' addresses) and `example.json`
(synthetic, cannot drive hardware). Example configuration lives under
`config/examples/`; copy and adapt, never guess measured values.

Safety: this code can move a robot arm holding a sharp tool. Keep the e-stop
required, keep the workspace clear, and validate on your own hardware before
any use near a person. Nothing here is a medical device.
