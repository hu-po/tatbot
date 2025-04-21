#!/bin/bash
source ".env"
source "scripts/util/validate_backend.sh"
source ".venv/bin/activate"
uv run python "tatbot/util/copy_to_nodes.py" ".env"
uv run python "tatbot/util/copy_to_nodes.py" "assets"