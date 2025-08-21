#!/bin/bash
echo "🧹  Linting and formatting docs"
source ~/tatbot/scripts/setup_env.sh
uv pip install .[docs]
uv run sphinx-build -W docs docs/_build
echo "✅ Done"
