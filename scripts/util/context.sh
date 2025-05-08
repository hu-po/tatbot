#!/bin/bash
source "scripts/util/validate_backend.sh"
echo "Creating context file for $TATBOT_ROOT"
CODEBASE_NAME="tatbot"
OUTPUT_FILE="$TATBOT_ROOT/output/context.txt"
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
  ["__pycache__"]=""
  [".DS_Store"]=""
  ["*.log"]=""
  ["*.rviz"]=""
)

declare -A IGNORE_DIRS=(
  # ----------------------------- ADD DIRECTORIES TO IGNORE HERE
  [".venv"]=""
  [".vscode"]=""
  ["assets"]=""
  ["config/camera_calibration"]=""
  ["config/networking"]=""
  ["config/trossen-ui"]=""
  ["docs/logs"]=""
  ["docs/notes"]=""
  ["docs/paper"]=""
  ["output"]=""
  ["__pycache__"]=""
  [".git"]=""
  ["venv"]=""
  ["env"]=""
  ["node_modules"]=""
  ["dist"]=""
  ["build"]=""
)

declare -A DIRECTORIES=(
  # ----------------------------- ADD DIRECTORIES HERE
  ["docker"]=""
  # ["docs"]=""
  ["config"]=""
  ["scripts"]=""
  ["tatbot"]=""
  # # Viewer sub-project
  # ["viewer"]=""
  # ["tatbot/viewer"]=""
  # ["scripts/viewer"]=""
  # ["docker/viewer"]=""
)

declare -A FILES=(
  # ----------------------------- ADD FILES HERE
  ["README.md"]=""
  ["pyproject.toml"]=""
  # ["tatbot/ik/morphs/base.py"]=""
  # ["tatbot/ik/morphs/gpt-e409cb.py"]=""
  # ["tatbot/ik/morphs/gemini-71f9bf.py"]=""
  [".env.example"]=""
  # [".dockerignore"]=""
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
    eval find "$dir" -type f -not -name "*.env" ${DIRECTORIES[$dir]} | while IFS= read -r file; do
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