#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/ik/Dockerfile.$BACKEND -t tatbot-ik-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm --user="root" \
tatbot-ik-$BACKEND bash -c "
source /root/tatbot/.venv/bin/activate && \
source /root/tatbot/.env && \
uv pip freeze && \
bash /root/tatbot/scripts/util/specs.sh && \
if [ ! -z \"\${NVIDIA_VISIBLE_DEVICES:-}\" ]; then
    uv run python /root/tatbot/tatbot/util/cuda_device_properties.py
fi && \
uv run python /root/tatbot/tatbot/ik/test.py"