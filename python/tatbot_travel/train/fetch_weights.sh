#!/usr/bin/env bash
# Download the FLUX 3 Action base trunk and its frozen encoders (about 25 GB)
# at the revision the SO-101 recipe pins, into $TRAVEL_HOME/weights.
set -euo pipefail

TRAVEL_HOME="${TRAVEL_HOME:-$HOME/travel}"
BASE_REV="${BASE_REV:-62878e2925e59b7a89ec14463ce89932624c490d}"
DEST="$TRAVEL_HOME/weights/flux-3-action-base"
HF="$TRAVEL_HOME/.venv/bin/hf"

"$HF" download black-forest-labs/flux-3-action-base --revision "$BASE_REV" \
  --include 'flux-3-action-base.safetensors' --include 'video_vae.safetensors' --include 'text_encoder/*' \
  --local-dir "$DEST"
echo "$DEST"
