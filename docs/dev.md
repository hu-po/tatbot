# Development

## Linting and Formatting
This project uses `ruff` for both linting and formatting.
```bash
uv pip install .[dev]
uv run isort .
uv run ruff format --config pyproject.toml
uv run ruff check --config pyproject.toml --fix
```

helpful oneliner to get diff for browser based models:

```bash
rm -rf diff.txt && git diff main...configrefactor > /tmp/diff.txt && xclip -selection clipboard < /tmp/diff.txt
```

To run all quality gates, including tests and type checking, use the lint script:
```bash
./scripts/lint.sh
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