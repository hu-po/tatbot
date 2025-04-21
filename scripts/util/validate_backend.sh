#!/bin/bash
echo "validating environment variables"
if [ -z "${BACKEND}" ]; then
    echo "Error: BACKEND environment variable is not set"
    exit 1
fi
valid_backends=("x86-3090" "x86-meerkat" "arm-rpi" "arm-agx" "arm-gh200")
if [[ ! " ${valid_backends[@]} " =~ " ${BACKEND} " ]]; then
    echo "Error: Invalid BACKEND. Must be one of: ${valid_backends[*]}"
    exit 1
fi 
echo "BACKEND: ${BACKEND}"
if [ -z "${TATBOT_ROOT}" ]; then
    echo "Error: TATBOT_ROOT environment variable is not set"
    exit 1
fi
if [ ! -d "${TATBOT_ROOT}" ]; then
    echo "Error: TATBOT_ROOT does not exist"
    exit 1
fi
echo "TATBOT_ROOT: ${TATBOT_ROOT}"
if [ ! -v GPU_FLAG ]; then
    echo "Error: GPU_FLAG environment variable is not set"
    exit 1
fi
echo "GPU_FLAG: ${GPU_FLAG}"