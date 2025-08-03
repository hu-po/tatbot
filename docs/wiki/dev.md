# Development

linting and formatting:

```bash
uv pip install .[dev]
uv run isort .
uv run ruff format --config pyproject.toml
uv run ruff check --config pyproject.toml --fix
```

updating forked repos when merge conflics arise:

```bash
cd ~/lerobot # example using hu-po/lerobot fork
git pull
git fetch upstream
git merge upstream/main
git push origin main
```

tips for models:
- use "uv pip install" and "uv run python" inside the ~/tatbot/.venv