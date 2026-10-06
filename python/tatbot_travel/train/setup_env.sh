#!/usr/bin/env bash
# Build the travel demo's generate+train environment on the training node.
#
#   python/tatbot_travel/train/setup_env.sh          # -> $TRAVEL_HOME (default ~/travel)
#
# One venv holds LeRobot (pinned to the commit that added the flux3 policy),
# torch from the CUDA 13 index, the NATTEN wheel the FLUX 3 video VAE needs,
# and this package, so the generator can write LeRobot datasets directly.
# Re-running is safe: it re-pins and re-installs in place.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAVEL_HOME="${TRAVEL_HOME:-$HOME/travel}"
LEROBOT_REV="${LEROBOT_REV:-87db15c88afc98824ffb182aaa789a81d4262fc4}"  # main after flux3 (#4739)
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
NATTEN="${NATTEN:-natten==0.21.6+torch2110cu130}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

mkdir -p "$TRAVEL_HOME"
if [ ! -d "$TRAVEL_HOME/lerobot/.git" ]; then
  git clone --quiet https://github.com/huggingface/lerobot.git "$TRAVEL_HOME/lerobot"
fi
git -C "$TRAVEL_HOME/lerobot" fetch --quiet origin
git -C "$TRAVEL_HOME/lerobot" checkout --quiet "$LEROBOT_REV"

[ -x "$TRAVEL_HOME/.venv/bin/python" ] || "$UV" venv --quiet -p 3.12 "$TRAVEL_HOME/.venv"
PY="$TRAVEL_HOME/.venv/bin/python"
# torch first, from the CUDA index: PyPI's aarch64 torch wheels are CPU-only.
"$UV" pip install --quiet -p "$PY" "torch==2.11.*" "torchvision==0.26.*" --index-url "$TORCH_INDEX"
"$UV" pip install --quiet -p "$PY" -e "$TRAVEL_HOME/lerobot[training,flux3,peft,diffusion]" \
  --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
"$UV" pip install --quiet -p "$PY" "$NATTEN" -f https://whl.natten.org/
"$UV" pip install --quiet -p "$PY" -e "$HERE/.."

"$PY" - <<'PY'
import lerobot, mujoco, natten, torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} natten {natten.__version__} "
      f"mujoco {mujoco.__version__} lerobot {getattr(lerobot, '__version__', '?')}")
PY
