#!/usr/bin/env bash
# Fresh blue-arm heads on the FLUX 3 Action base, then the flux3 LoRA recipe.
#
#   python/tatbot_travel/train/train.sh <dataset_root> <run_name>
#   TRUNK=~/travel/weights/so101-trunk.safetensors train.sh <dataset_root> <run_name>
#
# TRUNK starts from another pretrained trunk (package_trunk.py makes one from a
# policy package). Training runs through tatbot_travel.flux3.train_main: fp32
# master weights for the fresh heads, and the NATTEN shim when F3_NATTEN_BACKEND
# is set (a Thor); NUM_WORKERS= trims the dataloader to fit; STEPS= trains
# longer (the cooldown stays at the end); CAMERAS=scene,wrist takes the SO-101
# package's two-view layout for datasets that carry a scene camera; AUGMENT=strong
# jitters brightness, contrast, saturation and hue (LeRobot's image transforms). floor_stats.py keeps every
# normalisation span above a floor, so a channel that barely varies in the
# data cannot turn model noise into motion.
#
# 1. export_base (LeRobot's examples/flux3): the base trunk with fresh 6-channel
#    action and conditioning heads, and processors whose normalisation comes
#    from the dataset's training episodes. No training step.
# 2. lerobot-train on that base with the recipe in flux3_blue.json: rank-32
#    LoRA on the trunk, heads in full, EMA 0.995, batch 2 x 4 accumulation.
# Runs detached; the log and everything else land in $TRAVEL_HOME/runs/<run_name>.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAVEL_HOME="${TRAVEL_HOME:-$HOME/travel}"
DATASET="$(realpath "${1:?dataset root (a LeRobot v3 dataset directory)}")"
RUN="$TRAVEL_HOME/runs/${2:?run name}"
WEIGHTS="$TRAVEL_HOME/weights/flux-3-action-base"
TRUNK="${TRUNK:-$WEIGHTS/flux-3-action-base.safetensors}"
PY="$TRAVEL_HOME/.venv/bin/python"

[ -f "$TRUNK" ] || { echo "no trunk $TRUNK: run fetch_weights.sh (or package_trunk.py)" >&2; exit 1; }
[ -e "$RUN" ] && { echo "$RUN exists; pick a new run name" >&2; exit 1; }
mkdir -p "$RUN"
sed -e "s#@DATASET@#$DATASET#g" -e "s#@WEIGHTS@#$WEIGHTS#g" -e "s#@OUTPUT@#$RUN/train#g" \
  "$HERE/flux3_blue.json" > "$RUN/config.json"
git -C "$HERE" rev-parse HEAD > "$RUN/tatbot_rev.txt" 2>/dev/null || true
echo "$TRUNK" > "$RUN/trunk.txt"
if [ -n "${NUM_WORKERS:-}" ]; then  # fewer dataloader workers where memory is short (a Thor)
  sed -i "s/\"num_workers\": [0-9]*/\"num_workers\": $NUM_WORKERS/" "$RUN/config.json"
fi
if [ -n "${STEPS:-}" ]; then  # before the camera edit below, which rewrites the file's layout
  sed -i "s/\"steps\": [0-9]*, \"save_freq\"/\"steps\": $STEPS, \"save_freq\"/" "$RUN/config.json"
fi
if [ "${CAMERAS:-wrist}" = "scene,wrist" ]; then  # the SO-101 package's two views, scene left of wrist
  "$PY" - "$RUN/config.json" <<'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
config["policy"].update(camera_keys=["observation.images.scene", "observation.images.wrist"],
                        camera_layout="side_by_side", canvas_hw=[256, 512])
json.dump(config, open(path, "w"), indent=1)
PY
fi
# No sharpness jitter: its convolution fails in oneDNN's aarch64 JIT on the training node ("label is too far");
# the simulator's camera model already varies focus blur and sharpening.
if [ "${AUGMENT:-}" = "strong" ]; then  # photometric jitter, one draw across a sample's history; no geometry
  "$PY" - "$RUN/config.json" <<'PY'
import json, sys
path = sys.argv[1]
config = json.load(open(path))
def jitter(kind, **kwargs):
    return {"weight": 1.0, "type": kind, "kwargs": kwargs}
config["dataset"]["image_transforms"] = {"enable": True, "max_num_transforms": 3, "random_order": True, "tfs": {
    "brightness": jitter("ColorJitter", brightness=[0.6, 1.4]),
    "contrast": jitter("ColorJitter", contrast=[0.6, 1.4]),
    "saturation": jitter("ColorJitter", saturation=[0.4, 1.6]),
    "hue": jitter("ColorJitter", hue=[-0.08, 0.08])}}
json.dump(config, open(path, "w"), indent=1)
PY
fi

"$PY" "$TRAVEL_HOME/lerobot/examples/flux3/export_base.py" --config "$RUN/config.json" \
  --dataset local/travel-ink --dataset-root "$DATASET" \
  --trunk-weights "$TRUNK" --output "$RUN/base" --device cpu \
  2>&1 | tee "$RUN/export_base.log"
"$PY" "$HERE/floor_stats.py" "$RUN/base" 2>&1 | tee -a "$RUN/export_base.log"

nohup "$PY" -c "import sys; from tatbot_travel.flux3 import train_main; sys.exit(train_main())" \
  --config_path="$RUN/base/so101_train.json" > "$RUN/train.log" 2>&1 &
echo "training pid $! -> $RUN/train.log"
