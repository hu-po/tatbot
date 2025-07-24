#!/bin/bash
uv run isort .
uv run ruff check --config pyproject.toml --fix