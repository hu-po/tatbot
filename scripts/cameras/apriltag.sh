#!/bin/bash
CAMERA=${1:-1}
source "scripts/util/validate_backend.sh"

docker build \
-f $TATBOT_ROOT/docker/ros2/apriltag/Dockerfile.$BACKEND \
-t tatbot-ros2-apriltag-$BACKEND \
$TATBOT_ROOT

docker run $GPU_FLAG -it --rm \
--privileged \
--network=host \
-e CAMERA=$CAMERA \
tatbot-ros2-apriltag-$BACKEND