# ROS

## Generic ROS2 tips

https://docs.ros.org/en/humble/index.html

to debug ROS2 system, you can run this script:

```bash
./scripts/oop/launch.oop-raw.sh
```

this will give you a shell into the oop container, where you can run `ros2` commands:

```bash
ros2 pkg list
ros2 topic list
ros2 node list
ros2 param list
```

for packages:

```bash
export PACKAGE=tf2
ros2 pkg xml $PACKAGE | grep version
```

for messages:

```bash
export MESSAGE=apriltag_msgs/msg/AprilTagDetectionArray
ros2 interface show $MESSAGE
```


for an individual topic:

```bash
export TOPIC=/apriltag_2/detections
ros2 topic type $TOPIC
ros2 topic info -v $TOPIC
ros2 topic hz $TOPIC
```

## AprilTag detection using docker containers and rpis, ojo

a description of the ros nodes running on each compute node:

-`rpi1`
    - inside docker container `tatbot-rpi` build and launch using `scripts/rpi/launch.rpi1.sh`
        - handles a subset of cameras defined in `cfg/cameras.yaml`
        - uses `gscam2` to convert RTSP streams into ROS2 image topics (e.g., /camera_002_image_raw)
        - uses `image_proc` to rectify those images (e.g., /apriltag_002/image_rect)
        - uses `apriltag_ros` to detect AprilTags, publishing poses as AprilTagDetectionArray messages on topics like /apriltag_002/detections
-`rpi2`
    - inside docker container `tatbot-rpi` build and launch using `scripts/rpi/launch.rpi2.sh`
        - handles a subset of cameras defined in `cfg/cameras.yaml`
        - uses `gscam2` to convert RTSP streams into ROS2 image topics (e.g., /camera_002_image_raw)
        - uses `image_proc` to rectify those images (e.g., /apriltag_002/image_rect)
        - uses `apriltag_ros` to detect AprilTags, publishing poses as AprilTagDetectionArray messages on topics like /apriltag_002/detections
- `oop`
    - inside docker container `tatbot-oop-tf2pose` build and launch using `scripts/oop/launch.oop-tf2pose.sh`
        - uses `tf2` to aggregate and publish poses for other nodes
    - inside docker container `tatbot-oop-rerun` build and launch using `scripts/oop/launch.oop-rerun.sh`
        - uses rerun to visualize the robot
    - inside docker container `tatbot-oop-rviz` build and launch using `scripts/oop/launch.oop-rviz.sh`
        - uses rviz2 to visualize the robot's state

## MoveIt using meerkat

Trossen (Interbotix) does not seem to have the widowx arms working with ROS yet
https://github.com/Interbotix/interbotix_ros_manipulators/tree/main
their manipulator repos are quite dated and only for the older arms

MoveIt seems like the best option for ROS, they have a setup guide for new robots
https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html

MoveIt seems quite heavy tbh, and alot of the complexity is for collisions and null space

> Collision checking is a very expensive operation often accounting for close to 90% of the computational expense during motion planning.

MoveIt does support docker containers

https://moveit.picknik.ai/main/doc/how_to_guides/how_to_setup_docker_containers_in_ubuntu.html

MoveIt creates `JointTrajectoryAction` commands, which the robot executes. Unfortunately that node is not implemented for the widowx arms.

https://moveit.picknik.ai/main/doc/concepts/move_group.html#the-move-group-node

IsaacSim has a cumotion, which seems to wrap over moveit

https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html#quickstart