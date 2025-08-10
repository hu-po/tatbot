# Development

## Linting and Formatting
This project uses `ruff` for both linting and formatting, plus `isort` for import sorting.

To run all code quality checks, use the lint script:
```bash
./scripts/lint.sh
```

## Helpful Commands

oneliner to get diff for browser-based models:
```bash
rm -rf diff.txt && git diff main...HEAD > /tmp/diff.txt && xclip -selection clipboard < /tmp/diff.txt
```

when merge conflicts arise in forked repos (e.g., `lerobot`), follow this process:
```bash
cd ~/lerobot # or other forked repo
git pull
git fetch upstream
git merge upstream/main
git push origin main
```

## Documentation
Generate project documentation:
```bash
uv pip install .[docs]
uv run sphinx-build docs docs/_build
```

## General Tips
- Always work within the `uv` virtual environment (`source .venv/bin/activate`)
- Use `uv pip install` and `uv run` for consistency