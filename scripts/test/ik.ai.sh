#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/ik/Dockerfile.$BACKEND -t tatbot-ik-$BACKEND $TATBOT_ROOT
docker run $GPU_FLAG -it --rm --user="root" \
tatbot-ik-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv run python \${TATBOT_ROOT}/tatbot/ai.py --test"