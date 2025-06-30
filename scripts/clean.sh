#!/usr/bin/env bash

# Clean the output directory with emoji feedback
OUTPUT_DIR="$(dirname "$0")/../output"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "🚫 Output directory not found: $OUTPUT_DIR"
  exit 1
fi

# Confirm action
echo "🧹 Cleaning output directory: $OUTPUT_DIR"

# Remove all files and subdirectories in output/
find "$OUTPUT_DIR" -mindepth 1 -exec rm -rf {} +

if [ $? -eq 0 ]; then
  echo "✅ Output directory cleaned!"
else
  echo "❌ Failed to clean output directory."
  exit 1
fi