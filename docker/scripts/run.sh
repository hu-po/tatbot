#!/usr/bin/env bash
set -euo pipefail

# Run Tatbot GPU containers with NVIDIA runtime
# Usage:
#   docker/scripts/run.sh jax   [extra docker args...] [-- <cmd>]
#   docker/scripts/run.sh torch [extra docker args...] [-- <cmd>]

TARGET=${1:-}
if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 <jax|torch> [docker args ...] [-- <cmd>]" >&2
  exit 1
fi
shift

case "$TARGET" in
  jax)
    IMAGE=tatbot-jax:latest
    ;;
  torch)
    IMAGE=tatbot-torch:latest
    ;;
  *)
    echo "Unsupported target: $TARGET (use 'jax' or 'torch')" >&2
    exit 1
    ;;
esac

# Split args at optional --
DOCKER_ARGS=()
CMD_ARGS=()
SEEN_DASH_DASH=false
for arg in "$@"; do
  if ! $SEEN_DASH_DASH && [[ "$arg" == "--" ]]; then
    SEEN_DASH_DASH=true
    continue
  fi
  if $SEEN_DASH_DASH; then
    CMD_ARGS+=("$arg")
  else
    DOCKER_ARGS+=("$arg")
  fi
done

WORKDIR_HOST=$(pwd)

# Common mounts
MOUNTS=(
  -v "$WORKDIR_HOST:/workspace/tatbot"
)

# Mount NFS logs if present
if [[ -d /nfs/tatbot ]]; then
  MOUNTS+=( -v /nfs/tatbot:/nfs/tatbot )
fi

# Cache for uv to speed up repeated installs
mkdir -p "$HOME/.cache/uv"
MOUNTS+=( -v "$HOME/.cache/uv:/root/.cache/uv" )

echo "Running $IMAGE ..."
exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HYDRA_FULL_ERROR=1 \
  -w /workspace/tatbot \
  "${MOUNTS[@]}" \
  "${DOCKER_ARGS[@]}" \
  "$IMAGE" \
  ${CMD_ARGS:+bash -lc "${CMD_ARGS[*]}"}

