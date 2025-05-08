#!/bin/bash
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/cameras/Dockerfile.rr_server.$BACKEND -t tatbot-rr_server-$BACKEND $TATBOT_ROOT
docker run -it --rm \
-p 9876:9876 \
--network host \
-v $TATBOT_ROOT/output:/root/tatbot/output \
-v $TATBOT_ROOT/assets:/root/tatbot/assets \
tatbot-rr_server-$BACKEND bash -c "
source \${TATBOT_ROOT}/.venv/bin/activate && \
source \${TATBOT_ROOT}/.env && \
uv run python \${TATBOT_ROOT}/tatbot/cameras/rr_server.py"