#!/bin/bash
docker build -f docker/Dockerfile.meerkat-moveit -t tatbot-meerkat-moveit .
docker run -it --rm --privileged \
--network=host \
tatbot-meerkat-moveit