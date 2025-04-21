#!/bin/bash
SEED=${1:-999}
source "scripts/util/validate_backend.sh"
docker build -f $TATBOT_ROOT/docker/ik/Dockerfile.$BACKEND -t tatbot-ik-$BACKEND $TATBOT_ROOT
declare -a failed_morphs
declare -a successful_morphs
MORPH_FILES=$(find "$TATBOT_ROOT/warp_ik/morphs" -name "*.py")
total_morphs=$(echo "$MORPH_FILES" | wc -w)
current_morph=0
for MORPH_FILE in $MORPH_FILES; do
    MORPH=$(basename "$MORPH_FILE" .py)
    ((current_morph++))
    echo "[$current_morph/$total_morphs] running morph: $MORPH"
    if docker run $GPU_FLAG -it --rm \
        -v $TATBOT_ROOT/output:/root/tatbot/output \
        -v $TATBOT_ROOT/assets:/root/tatbot/assets \
        -v $TATBOT_ROOT/tatbot/ik/morphs:/root/tatbot/tatbot/ik/morphs \
        tatbot-ik-$BACKEND bash -c "
        source \${TATBOT_ROOT}/.venv/bin/activate && \
        source \${TATBOT_ROOT}/.env && \
        uv run python \${TATBOT_ROOT}/tatbot/ik/morph.py --morph $MORPH --track --headless --seed $SEED"; then
        successful_morphs+=("$MORPH")
        echo "✓ success $MORPH"
    else
        failed_morphs+=("$MORPH")
        echo "✗ failed $MORPH"
    fi
    echo "----------------------------------------"
done
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
[ ${#failed_morphs[@]} -eq 0 ]