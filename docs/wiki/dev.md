# Development

```bash
uv pip install .[dev]
uv run isort
uv run ruff format --config pyproject.toml
uv run ruff check --config pyproject.toml --fix
```