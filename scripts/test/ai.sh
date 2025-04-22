#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/warp/Dockerfile.$BACKEND -t tatbot-warp-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm --user="root" \
tatbot-warp-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv run python \${TATBOT_ROOT}/tatbot/ai.py --test"