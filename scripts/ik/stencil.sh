#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/ik/Dockerfile.$BACKEND -t tatbot-ik-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm \
-v $TATBOT_ROOT/output:/root/tatbot/output \
-v $TATBOT_ROOT/assets:/root/tatbot/assets \
tatbot-ik-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv run python \${TATBOT_ROOT}/tatbot/ik/stencil.py"