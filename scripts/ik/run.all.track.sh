#!/bin/bash
TATBOT_ROOT="$(dirname "$(dirname "$0")")" # the warp-ik directory
SEED=${1:-999}
if [ -z "${BACKEND}" ]; then
    echo "Error: BACKEND environment variable is not set"
    exit 1
fi
if [ -z "${NUM_ENVS}" ]; then
    echo "Error: NUM_ENVS environment variable is not set"
    exit 1
fi

# Initialize arrays to track results
declare -a failed_morphs
declare -a successful_morphs

docker build -f docker/Dockerfile.$BACKEND -t warp-ik-$BACKEND .
GPU_FLAG=""
if [[ "$BACKEND" != "x86-meerkat" && "$BACKEND" != "arm-rpi" ]]; then
    GPU_FLAG="--gpus all"
fi

# Find all morph files, excluding template.py and hidden files
MORPH_FILES=$(find "$TATBOT_ROOT/warp_ik/morphs" -name "*.py" ! -name "base.py" ! -path "*/\.*")
total_morphs=$(echo "$MORPH_FILES" | wc -w)
current_morph=0

for MORPH_FILE in $MORPH_FILES; do
    MORPH=$(basename "$MORPH_FILE" .py)
    ((current_morph++))
    echo "[$current_morph/$total_morphs] Running morph: $MORPH"
    
    # Run the morph and capture its exit status
    if docker run $GPU_FLAG -it --rm --user="root" \
        -v $TATBOT_ROOT/output:/root/warp-ik/output \
        -v $TATBOT_ROOT/assets:/root/warp-ik/assets \
        -v $TATBOT_ROOT/warp_ik/morphs:/root/warp-ik/warp_ik/morphs \
        warp-ik-$BACKEND bash -c "
        source /root/warp-ik/.venv/bin/activate && \
        source /root/warp-ik/.env && \
        uv run python /root/warp-ik/warp_ik/src/morph.py --backend $BACKEND --morph $MORPH --track --headless --num_envs $NUM_ENVS --seed $SEED"; then
        successful_morphs+=("$MORPH")
        echo "✓ Successfully completed morph: $MORPH"
    else
        failed_morphs+=("$MORPH")
        echo "✗ Failed to process morph: $MORPH"
    fi
    echo "----------------------------------------"
done

# Print summary
echo
echo "=== Execution Summary ==="
echo "Total morphs processed: $total_morphs"
echo "Successful: ${#successful_morphs[@]}"
echo "Failed: ${#failed_morphs[@]}"

if [ ${#failed_morphs[@]} -gt 0 ]; then
    echo
    echo "Failed morphs:"
    printf '%s\n' "${failed_morphs[@]}" | sed 's/^/  - /'
fi

# Exit with error if any morphs failed
[ ${#failed_morphs[@]} -eq 0 ]