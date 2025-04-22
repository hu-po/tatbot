#!/bin/bash
source "scripts/util/validate_backend.sh"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <usd_file_path>"
    echo "Example: $0 output/stencil.usdz"
    echo "Example: $0 assets/3d/real_leg/leg.usda"
    exit 1
fi

USD_FILE=$1

echo "Building TatBot viewer Docker image..."
docker build -f $TATBOT_ROOT/docker/viewer/Dockerfile -t tatbot-viewer $TATBOT_ROOT

echo "Running TatBot viewer container..."
docker run -p 8080:8080 \
  -v "$TATBOT_ROOT/output:/app/output" \
  -v "$TATBOT_ROOT/assets:/app/assets" \
  -e USD_FILE="$USD_FILE" \
  --rm \
  tatbot-viewer
