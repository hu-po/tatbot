#!/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: visualize.sh <path/to/scene.usd>"
  exit 1
fi

USD_FILE="$1"
GLB_FILE="$(basename "${USD_FILE%.usd}").glb"

# build the two images & start them detached (viewer will 404 until GLB exists)
docker compose -f docker/viewer/docker-compose.yml up --build -d viewer api

# run the conversion INSIDE the api container
docker compose -f docker/viewer/docker-compose.yml run --rm \
  -v "$(pwd)/tatbot/assets:/app/assets" \
  api \
  python /app/usd_to_gltf.py "/tatbot/${USD_FILE}" "/app/assets/scenes/${GLB_FILE}"

echo
echo "✅  Converted  ${USD_FILE}  ➜  tatbot/assets/scenes/${GLB_FILE}"
echo "🌐  Open: http://localhost:8080?file=scenes/${GLB_FILE}"