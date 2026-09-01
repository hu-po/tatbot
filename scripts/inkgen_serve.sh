#!/usr/bin/env bash
# Run the Inkmap design generator (web/inkgen/app.py) on this machine.
#   scripts/inkgen_serve.sh [--port 8600] [--host 0.0.0.0] [--model <hf id>] [--cpu]
# First run creates ~/.cache/tatbot/inkgen/venv with uv (torch + diffusers, a
# few GB) and downloads the model into the Hugging Face cache. Same code as
# the ZeroGPU Space; no Hub account involved.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/lib/cli_hint.sh
source "$repo/scripts/lib/cli_hint.sh"; cli_hint::note "tatbot inkgen serve"
port=8600; host=0.0.0.0; model=""; cpu=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --cpu) cpu=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "inkgen_serve: unknown argument $1" >&2; exit 2 ;;
  esac
done
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "inkgen_serve: uv is required (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2; exit 3; }
root="${INKGEN_HOME:-$HOME/.cache/tatbot/inkgen}"
mkdir -p "$root"
if [ ! -x "$root/venv/bin/python" ]; then
  echo "inkgen_serve: creating $root/venv (first run; torch is large)"
  uv venv -q -p 3.12 "$root/venv"
fi
"$root/venv/bin/python" -c "import diffusers, gradio, torch" 2>/dev/null || \
  uv pip install -q -p "$root/venv/bin/python" -r "$repo/web/inkgen/requirements.txt.local"
[ -n "$model" ] && export INKGEN_MODEL="$model"
[ "$cpu" = 1 ] && export CUDA_VISIBLE_DEVICES=""
export INKGEN_HOST="$host" INKGEN_PORT="$port"
echo "inkgen_serve: http://$host:$port/  (health: /api/health)  model=${INKGEN_MODEL:-Tongyi-MAI/Z-Image-Turbo}"
cd "$repo/web/inkgen" && exec "$root/venv/bin/python" app.py
