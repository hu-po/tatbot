"""``python3 scripts/lib/tatbot_cli …`` — the entry the shim execs."""

from __future__ import annotations

import os
import sys

# Running the package directory directly (`python3 scripts/lib/tatbot_cli`)
# puts that directory on sys.path, not its parent; make the package importable
# by its name so `from tatbot_cli import …` works without an install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.dirname(_HERE)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from tatbot_cli.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
