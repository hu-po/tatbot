# Repository Guidelines

## Project Structure & Module Organization
- `src/tatbot/`: main package (bot, cam, gen, mcp, tools, utils, viz); entry `tatbot.main`.
- `src/conf/`: Hydra configs (e.g., `mcp/<node>.yaml`).
- `config/`: external service/tool configs (dbv3, network, trossen, vla).
- `scripts/`: developer and ops helpers (`mcp_run.sh`, `lint.sh`, `setup_env.sh`).
- `tests/`: Python tests. Add new tests here.
- `docs/`: Sphinx documentation.

## Build, Test, and Development Commands
- Setup env: `source scripts/setup_env.sh` (creates `.venv`, installs extras from node config).
- Install dev tools: `uv pip install .[dev,docs]`.
- Lint/format/type-check: `./scripts/lint.sh` (runs isort, ruff, mypy on `src/tatbot`).
- Run tests: `uv run pytest -q`.
- Run MCP server: `./scripts/mcp_run.sh <node>` (logs in `/nfs/tatbot/mcp-logs/<node>.log`).
- Local viz examples:
  - `uv run python -m tatbot.viz.stroke --scene=tatbotlogo`
  - `uv run python -m tatbot.viz.teleop --enable-robot --enable-depth`

## Coding Style & Naming Conventions
- Python 3.11; line length 110 (`tool.ruff.line-length`).
- Imports: `isort` (profile `black`). Linting: `ruff` (PEP8 + naming + pydocstyle Google). Types: `mypy` (strict on `src/tatbot`).
- Naming: snake_case for modules/functions/vars; PascalCase for classes; UPPER_SNAKE for constants.
- Docstrings: Google style; prefer explicit types over `Any`.

## Testing Guidelines
- Framework: `pytest`.
- Location: `tests/` mirroring package paths; name files `test_*.py` and tests `test_*`.
- Fast unit tests for `utils/` and pure logic in `gen/`; use marks to isolate hardware/integration.
- Run locally with `uv run pytest -q`; add fixtures for Hydra configs as needed.

## Commit & Pull Request Guidelines
- History shows no strict convention; use clear, imperative subject lines. Prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) when possible.
- Before opening a PR: run `./scripts/lint.sh` and `pytest`; update docs/configs if behavior changes.
- PRs should include: purpose, scope, linked issues, test notes, and screenshots/logs for tooling or viz.

## Security & Configuration Tips
- Secrets: use `.env` (see `.env.example`). Never commit credentials.
- Shared paths: NFS at `/nfs/tatbot` for logs, designs, recordings.
- Extras: install only needed extras (`.[bot,cam,gen,gpu,img,viz]`).
- MCP nodes and ports are Hydra-driven (`src/conf/mcp/`); avoid hardcoding.
