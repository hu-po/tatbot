#!/bin/bash
CAMERA=${1:-1}

# Source the .env file
ENV_FILE="$TATBOT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Sourcing environment variables from $ENV_FILE"
    set -a
    . "$ENV_FILE"
    set +a
else
    echo "Warning: $ENV_FILE not found. Camera passwords might be missing."
fi

# Get the specific camera password
PASSWORD_VAR_NAME="CAMERA${CAMERA}_PASSWORD"
CAM_PASSWORD="${!PASSWORD_VAR_NAME}"

if [ -z "$CAM_PASSWORD" ]; then
    echo "Error: Password for camera $CAMERA (${PASSWORD_VAR_NAME}) not found in environment."
    exit 1
fi

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
-e CAM_PASSWORD="$CAM_PASSWORD" \
-e DISPLAY=$DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority \
-v $TATBOT_ROOT/config/camera_calibration:/root/tatbot/config/camera_calibration \
tatbot-ros2-camera_calibration-$BACKEND