#!/bin/bash
echo "🧹 Linting and formatting code"
uv run isort .
uv run ruff check --config pyproject.toml --fix
echo "✅ Done"