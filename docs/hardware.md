# Hardware

the main body of tatbot is made of 2020 T-slot aluminum
[aluminum](notes/aluminum.md)

a thin central black PVC board holds the ojo, rpis, and switches
`switch-main` has 2 keyhole mounting holes, is mounted upside down
`switch-poe` has 2 keyhole mounting holes, is mounted upside down
`rpi1` and `rpi2` have 2 keyhole mount points requiring M3 screws, mounted on top of the pvc board
`ojo` has 4 mounting holes on flanges, is mounted upside down
a wider black PVC board holds the arm-leader and arm-follower, and robot power supply
`arm-leader` has 4 mounting holes, is mounted upside down
`arm-follower` has 4 mounting holes, is mounted upside down
heat-set threaded inserts and M4 mounting screws are used to mount the compute nodes to the PVC board
[pvc](notes/pvc.md)

the robot is a Trossen Robotics WidowXAI V0 arm
the leader arm has a right handed human teacher handle with finger ring grips
the follower arm has a parallel jaw gripper
[trossen](notes/trossen.md)

to put the arms back in the sleep position run:

```bash
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-sleep.py
```

to configure the arms (config files are in `~/dev/tatbot-dev/cfg/trossen`):

```bash
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-config.py
python ~/dev/tatbot-dev/scripts/trossen-ai/arms-config.py --push
```

to reset the realsense cameras run:

```bash
sudo bash ~/dev/tatbot-dev/scripts/trossen-ai/reset-realsenses.sh
```
