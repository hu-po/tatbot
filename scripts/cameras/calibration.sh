#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/ros2/Dockerfile.camera_calibration.$BACKEND -t tatbot-ros2-camera_calibration-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm \
-v $TATBOT_ROOT/config/camera_calibration:/root/tatbot/config/camera_calibration \
tatbot-ros2-camera_calibration-$BACKEND