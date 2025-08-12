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
uv pip install .

echo "🔑  Source environment variables (keys, tokens, etc.)"
source /nfs/tatbot/.env || true
