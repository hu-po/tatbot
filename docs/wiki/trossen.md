# Trossen Robot Arms

- [`trossen_arm`](https://github.com/TrossenRobotics/trossen_arm)
- [Trossen WidowXAI arms](https://docs.trossenrobotics.com/trossen_arm/main/specifications.html)
- [Driver API Docs](https://docs.trossenrobotics.com/trossen_arm/main/api/library_root.html#)

see:

- trossen config in `tatbot/config/trossen/arm_{l|r}.yaml`
- `tatbot/tatbot/data/arms.py`
- `tatbot/config/arms/bimanual.yaml`

## Update Firmware

First update the firmware to the latest version:

https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html#software-upgrade

Download latest firmware:

https://docs.trossenrobotics.com/trossen_arm/main/downloads.html

```bash
cd ~/Downloads && wget <get link from above>
unzip firmware-wxai_v0-v1.8.4.zip
teensy_loader_cli --mcu=TEENSY41 -s firmware-wxai_v0-v1.8.4.hex
```

TODO: Set the velocity_tolerance to 0.2 times the velocity max
https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html#joint-limits
TODO: Edit end effector parameters for tattoo ee
https://docs.trossenrobotics.com/trossen_arm/main/api/structtrossen__arm_1_1EndEffector.html#struct-documentation
TODO: Create trossen_arm.StandardEndEffector.wxai_v0_tatbot_l and trossen_arm.StandardEndEffector.wxai_v0_tatbot_r