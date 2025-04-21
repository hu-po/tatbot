#!/bin/bash
source "scripts/util/validate_backend.sh"
echo "Creating context file for $TATBOT_ROOT"
OUTPUT_FILE="$TATBOT_ROOT/output/context.txt"
echo "Output file: $OUTPUT_FILE"
if [ -f "$OUTPUT_FILE" ]; then
  echo "Removing existing context file"
  rm -f "$OUTPUT_FILE"
fi
declare -A DIRECTORIES=(
  # ----------------------------- ADD DIRECTORIES HERE
  ["docker"]=""
  ["docs"]=""
  ["config"]=""
  ["scripts"]=""
  ["tatbot"]=""
)
CODEBASE_NAME="tatbot"
declare -A FILES=(
  # ----------------------------- ADD FILES HERE
  ["README.md"]=""
  # [".env.example"]=""
  # ["pyproject.toml"]=""
  # [".dockerignore"]=""
)
echo "Below is a list of files for the $CODEBASE_NAME codebase." >> "$OUTPUT_FILE"
process_file() {
  local file="$1"
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