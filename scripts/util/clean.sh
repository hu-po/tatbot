#!/bin/bash
source "scripts/util/validate_backend.sh"
OUTPUT_DIR="$TATBOT_ROOT/output"
sudo rm -rf "$OUTPUT_DIR"/*
echo "🧹 🧼 Cleaned output directory at: $OUTPUT_DIR"