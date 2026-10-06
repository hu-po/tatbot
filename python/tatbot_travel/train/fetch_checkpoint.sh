#!/usr/bin/env bash
# Bring one LoRA checkpoint (raw and EMA) and its dataset's metadata to this node, pointed at a local base.
#
#   python/tatbot_travel/train/fetch_checkpoint.sh <ssh target> <run> <step> <dataset dir on the source>
#   e.g. fetch_checkpoint.sh trainer ink-base 010000 travel/data/v2all/merged
#
# The adapter names the base it was trained on by the source node's path. Every
# base exported from flux-3-action-base with this recipe is the same trunk and
# the adapter carries the fresh heads, so it is repointed at this node's base
# ($TRAVEL_HOME/bench/base, from prep: export_base with supplied statistics),
# unless BASE= names another (e.g. the SO-101-trunk run's own base). Text encoder
# and VAE paths follow $TRAVEL_HOME/weights. Lands in $TRAVEL_HOME/ckpt/<run>-<step>/.
set -euo pipefail

SRC="${1:?ssh target of the training node}"
RUN="${2:?run name}"
STEP="${3:?checkpoint step, e.g. 010000}"
DATA="${4:?dataset directory on the source, relative to its home}"
TRAVEL_HOME="${TRAVEL_HOME:-$HOME/travel}"
BASE="${BASE:-$TRAVEL_HOME/bench/base}"
OUT="$TRAVEL_HOME/ckpt/$RUN-$STEP"
DATA_OUT="$TRAVEL_HOME/data/$RUN"

mkdir -p "$OUT" "$DATA_OUT"
rsync -a "$SRC:travel/runs/$RUN/train/checkpoints/$STEP/pretrained_model" \
  "$SRC:travel/runs/$RUN/train/checkpoints/$STEP/pretrained_model_ema" "$OUT/"
rsync -a "$SRC:$DATA/meta" "$DATA_OUT/"
for d in "$OUT/pretrained_model" "$OUT/pretrained_model_ema"; do
  python3 - "$d" "$BASE" "$TRAVEL_HOME/weights/flux-3-action-base" <<'PY'
import json, sys
from pathlib import Path
package, base, weights = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
adapter = json.loads((package / "adapter_config.json").read_text())
adapter["base_model_name_or_path"] = base
(package / "adapter_config.json").write_text(json.dumps(adapter, indent=2) + "\n")
config = json.loads((package / "config.json").read_text())
config["pretrained_path"] = base
for key in ("text_encoder_id", "video_vae_id", "trunk_weights"):
    value = config.get(key)
    if isinstance(value, str) and "/weights/flux-3-action-base" in value:
        config[key] = weights + value.split("/weights/flux-3-action-base", 1)[1]
(package / "config.json").write_text(json.dumps(config, indent=4) + "\n")
PY
done
echo "checkpoint $OUT (dataset metadata $DATA_OUT)"
