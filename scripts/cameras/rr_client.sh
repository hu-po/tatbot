#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/cameras/Dockerfile.rr_client.$BACKEND -t tatbot-rr_client-$BACKEND $TATBOT_ROOT
docker run -it --rm \
--network host \
-v $TATBOT_ROOT/output:/root/tatbot/output \
-v $TATBOT_ROOT/assets:/root/tatbot/assets \
tatbot-rr_client-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv run python \${TATBOT_ROOT}/tatbot/cameras/rr_client.py"