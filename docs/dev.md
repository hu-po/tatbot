# Development

## Linting and Formatting
This project uses `ruff` for both linting and formatting, plus `isort` for import sorting.

To run all code quality checks, use the lint script:
```bash
./scripts/lint.sh
```

Or run commands manually:
```bash
uv pip install .[dev]
uv run isort .
uv run ruff check --config pyproject.toml --fix
```

Note: The project currently uses `ruff check --fix` for both linting and formatting. Type checking is handled by IDE integration.

Helpful oneliner to get diff for browser-based models:
```bash
rm -rf diff.txt && git diff main...HEAD > /tmp/diff.txt && xclip -selection clipboard < /tmp/diff.txt
```

## Prompts

helpful prompts:

```
Give feedback on the tatbot project below - high level and down to the nitty gritty. How can we improve it? Are there any big architectural changes we should make? What are the biggest opportunities for improvement? Does the design and choice of data structures, abstractions, and implementation make sense?
```

## Forked Repositories
When merge conflicts arise in forked repos (e.g., `lerobot`), follow this process:
```bash
cd ~/lerobot # or other forked repo
git pull
git fetch upstream
git merge upstream/main
git push origin main
```

## General Tips
- Always work within the `uv` virtual environment (`source .venv/bin/activate`).
- Use `uv pip install` and `uv run python` for consistency.