"""``python3 scripts/lib/tatbot_cli …`` — the entry the shim execs."""

from __future__ import annotations

import os
import sys

# Running the package directory directly (`python3 scripts/lib/tatbot_cli`)
# puts that directory on sys.path, not its parent; make the package importable
# by its name so `from tatbot_cli import …` works without an install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()  # every script root, once: verbs import lib/vision/train modules by bare name
from tatbot_cli.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
