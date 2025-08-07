#!/bin/bash
echo "🧹  Linting and formatting code"
source ~/tatbot/scripts/setup_env.sh
uv pip install .[dev]
uv run isort .
uv run ruff check --config pyproject.toml --fix
echo "✅ Done"