#!/bin/bash
echo "🚀  Go to project directory and update"
cd ~/tatbot
git pull || true

echo "🧹  Clean old environment (ignore errors)"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  deactivate || true
fi
rm -rf .venv || true
rm -f uv.lock || true

echo "🛠️  Setup new uv environment"
uv venv --prompt="tatbot"
source .venv/bin/activate

echo "📦  Installing base dependencies"
uv pip install .

echo "🔍  Detecting current node and installing dependencies"
# Get extras using the Python utility
EXTRAS=$(uv run -m tatbot.utils.node_config 2>/dev/null || echo "")

# Trim whitespace and check if EXTRAS is non-empty
EXTRAS_TRIMMED=$(echo "$EXTRAS" | tr -d '[:space:]')
if [ -n "$EXTRAS_TRIMMED" ] && [ "$EXTRAS_TRIMMED" != "" ]; then
    echo "📦  Installing extras: [$EXTRAS]"
    uv pip install ".[$EXTRAS]"
fi

echo "🔑  Source environment variables (keys, tokens, etc.)"
source /nfs/tatbot/.env || true
