# tatbot

tattoo robot 🦾🖋️🎨 — [tatbot.ai](https://tatbot.ai)

Dual-arm robot that draws and tattoos: 400 Hz C++ teleoperation, a Rust
camera daemon, LeRobot imitation-learning plugins, a ManiSkill data factory,
and web tools for design and placement.

- **Researchers**: `python/tatbot_sim/` generates datasets;
  `python/lerobot_robot_tatbot/` is the robot plugin (with a mock driver).
- **Artists**: `web/inkmap/` places designs on a 3D body; `web/inkgen/`
  generates designs.
- **Robot builders**: `cpp/teleop/` + `firmware/estop_pico/` +
  `config/profiles/trossen-wxai.json` reproduce the rig on Trossen WXAI arms.

Start with `AGENTS.md` for the build roots; `scripts/check` runs every check
this tree can. Hardware needs a completed profile in `config/profiles/`.

This code can move a robot arm holding a sharp tool. Keep the e-stop
required, validate on your own hardware, and keep people out of the
workspace. MIT licensed; see `LICENSE`.
