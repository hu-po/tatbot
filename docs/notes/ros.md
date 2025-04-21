we use ROS2 Humble

to calibrate the amcrest poe cameras, run:

```bash
./scripts/oop/launch.oop-camera-calibration.sh
```

to generate the moveit config, run:

```bash
./scripts/oop/launch.oop-moveit-setup.sh
```

TODO: test these workflows on `trossen-ai` so that we don't need to use `oop`