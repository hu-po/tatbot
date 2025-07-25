#!/bin/bash
CODEBASE_NAME="tatbot"
TATBOT_ROOT="$HOME/tatbot"
OUTPUT_FILE="$TATBOT_ROOT/nfs/context.txt"
echo "Populating context for $CODEBASE_NAME"
echo "Output file: $OUTPUT_FILE"
if [ -f "$OUTPUT_FILE" ]; then
  echo "Removing existing context file"
  rm -f "$OUTPUT_FILE"
fi

declare -A IGNORE_FILES=(
  # ----------------------------- ADD FILES TO IGNORE HERE
  ["*.pyc"]=""
  ["*.pyo"]=""
  ["*.pyd"]=""
  ["*.so"]=""
  ["*.egg"]=""
  ["*.egg-info"]=""
  ["*.egg-info/*"]=""
  ["uv.lock"]=""
  [".DS_Store"]=""
  ["*.log"]=""
  ["*.rviz"]=""
  ["*.env"]=""
)

declare -A IGNORE_DIRS=(
  # ----------------------------- ADD DIRECTORIES TO IGNORE HERE
  ["$TATBOT_ROOT/.git"]=""
  ["$TATBOT_ROOT/.cursor"]=""
  ["$TATBOT_ROOT/.venv"]=""
  ["$TATBOT_ROOT/__pycache__"]=""
  ["$TATBOT_ROOT/config/dbv3"]=""
  ["$TATBOT_ROOT/config/polyscope"]=""
  ["$TATBOT_ROOT/build"]=""
  ["$TATBOT_ROOT/docs/paper"]=""
  ["$TATBOT_ROOT/docs/logos"]=""
  ["$TATBOT_ROOT/tatbot.egg-info"]=""
  ["$TATBOT_ROOT/.ruff_cache"]=""
  ["$TATBOT_ROOT/nfs"]=""
  ["$TATBOT_ROOT/urdf"]=""
)

declare -A DIRECTORIES=(
  # ----------------------------- ADD DIRECTORIES HERE
  ["$TATBOT_ROOT"]=""
)

declare -A FILES=(
  # ----------------------------- ADD FILES HERE
  ["$TATBOT_ROOT/README.md"]=""
  ["$TATBOT_ROOT/docs/tech.md"]=""
)

echo "Below is a list of files for the $CODEBASE_NAME codebase." >> "$OUTPUT_FILE"

# Function to check if a file should be ignored
should_ignore_file() {
  local file="$1"
  for pattern in "${!IGNORE_FILES[@]}"; do
    if [[ "$file" == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

# Function to check if a directory should be ignored
should_ignore_dir() {
  local dir="$1"
  for pattern in "${!IGNORE_DIRS[@]}"; do
    if [[ "$dir" == $pattern ]]; then
      return 0
    fi
  done
  return 1
}

process_file() {
  local file="$1"
  if should_ignore_file "$(basename "$file")"; then
    echo "Ignoring file: $file"
    return
  fi
  echo "Processing: $file"
  echo -e "\n\n--- BEGIN FILE: $file ---\n" >> "$OUTPUT_FILE"
  cat "$file" >> "$OUTPUT_FILE"
  echo -e "\n--- END FILE: $file ---\n" >> "$OUTPUT_FILE"
}

for specific_file in "${!FILES[@]}"; do
  if [ -f "$specific_file" ]; then
    process_file "$specific_file"
  else
    echo "File not found: $specific_file"
  fi
done

for dir in "${!DIRECTORIES[@]}"; do
  if [ -d "$dir" ]; then
    if should_ignore_dir "$dir"; then
      echo "Ignoring directory: $dir"
      continue
    fi
    # Build prune expression for ignored directories
    PRUNE_ARGS=()
    for ignore_dir in "${!IGNORE_DIRS[@]}"; do
      PRUNE_ARGS+=( -path "$ignore_dir" -prune -o )
    done
    # Add the main find expression
    PRUNE_ARGS+=( -type f -not -name "*.env" -print )

    find "$dir" "${PRUNE_ARGS[@]}" | while IFS= read -r file; do
      process_file "$file"
    done
  else
    echo "Directory not found: $dir"
  fi
done

echo -e "\n\n--- END OF CONTEXT ---\n" >> "$OUTPUT_FILE"
TOTAL_FILES=$(grep -c "^--- BEGIN FILE:" "$OUTPUT_FILE")
TOTAL_SIZE=$(du -h "$OUTPUT_FILE" | awk '{print $1}')
echo "Context file created at $OUTPUT_FILE"
echo "Total files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"