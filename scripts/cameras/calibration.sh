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

# Create output directory
HOST_OUTPUT_DIR="$TATBOT_ROOT/output/calibration_results/camera_${CAMERA}"
echo "Ensuring host output directory exists: $HOST_OUTPUT_DIR"
mkdir -p "$HOST_OUTPUT_DIR"

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
-v "$HOST_OUTPUT_DIR":/output \
tatbot-ros2-camera_calibration-$BACKEND
DOCKER_EXIT_CODE=$?

echo "Docker container finished with exit code $DOCKER_EXIT_CODE."

# Extract Calibration Results on Host
CALIBRATION_TAR_FILE="$HOST_OUTPUT_DIR/calibrationdata.tar.gz"
CAMERA_NAME="camera_${CAMERA}"

# Check if docker run succeeded AND the file was copied
if [ $DOCKER_EXIT_CODE -eq 0 ] && [ -f "$CALIBRATION_TAR_FILE" ]; then
    echo "Attempting to extract results on host from $CALIBRATION_TAR_FILE..."
    # Change directory to output dir to extract files there
    if cd "$HOST_OUTPUT_DIR"; then
        if tar xzf calibrationdata.tar.gz; then
            echo "Extracted successfully."
            # Rename the common output filename (often ost.yaml)
            if [ -f ost.yaml ]; then
                YAML_OUTPUT_FILE="${CAMERA_NAME}.yaml"
                echo "Renaming ost.yaml to ${YAML_OUTPUT_FILE}"
                mv ost.yaml "${YAML_OUTPUT_FILE}"
                echo "Calibration YAML saved to: $HOST_OUTPUT_DIR/${YAML_OUTPUT_FILE}"
            else
                echo "WARN: Expected ost.yaml not found after extraction in $HOST_OUTPUT_DIR."
            fi
        else
            echo "ERROR: Extraction of $CALIBRATION_TAR_FILE failed."
        fi
        # Go back to original directory
        cd - > /dev/null
    else
         echo "ERROR: Could not change directory to $HOST_OUTPUT_DIR"
    fi
else
    echo "WARN: Docker exited with code $DOCKER_EXIT_CODE or $CALIBRATION_TAR_FILE not found on host. Cannot extract."
fi

echo "Calibration script for camera $CAMERA finished."