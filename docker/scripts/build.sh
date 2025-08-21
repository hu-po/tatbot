#!/usr/bin/env bash
set -euo pipefail

# Build Tatbot GPU containers
# Usage:
#   docker/scripts/build.sh jax   [tag]
#   docker/scripts/build.sh torch [tag]

TARGET=${1:-}
TAG=${2:-}

if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 <jax|torch> [tag]" >&2
  exit 1
fi

case "$TARGET" in
  jax)
    DOCKERFILE=docker/Dockerfile.jax
    DEFAULT_TAG=tatbot-jax:latest
    ;;
  torch)
    DOCKERFILE=docker/Dockerfile.torch
    DEFAULT_TAG=tatbot-torch:latest
    ;;
  *)
    echo "Unsupported target: $TARGET (use 'jax' or 'torch')" >&2
    exit 1
    ;;
esac

TAG=${TAG:-$DEFAULT_TAG}

echo "Building $TARGET image as $TAG ..."
DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" -t "$TAG" .
echo "Done: $TAG"

