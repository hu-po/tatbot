#!/bin/bash
CAMERA=${1:-1}
source "scripts/util/validate_backend.sh"
docker build \
-f $TATBOT_ROOT/docker/ros2/Dockerfile.camera_calibration.$BACKEND \
-t tatbot-ros2-camera_calibration-$BACKEND \
$TATBOT_ROOT
xhost +local:root
docker run $GPU_FLAG -it --rm \
--privileged \
--network=host \
-e CAMERA=$CAMERA \
-e DISPLAY=$DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority \
-v $TATBOT_ROOT/config/camera_calibration:/root/tatbot/config/camera_calibration \
tatbot-ros2-camera_calibration-$BACKEND