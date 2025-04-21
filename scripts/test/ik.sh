#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/warp/Dockerfile.$BACKEND -t tatbot-warp-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm \
tatbot-warp-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv pip freeze && \
bash \${TATBOT_ROOT}/scripts/util/specs.sh && \
if [ ! -z \"\${NVIDIA_VISIBLE_DEVICES:-}\" ]; then
    uv run python \${TATBOT_ROOT}/tatbot/util/cuda_device_properties.py
fi && \
uv run python \${TATBOT_ROOT}/tatbot/ik/test.py"