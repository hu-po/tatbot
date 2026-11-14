#!/usr/bin/env python3
"""The lab's commands, run by the shared offline drawing interpreter."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# tatbot_ink (with tatbot_description and tatbot_motion) compiles a plan's runs in process, as `tatbot ros compile`
# does; scripts/vision's fiducials reads the print's markers in camera frames.
sys.path[:0] = [str(REPO / "python/tatbot_lab/src"), str(REPO / "python/tatbot_contracts/src"),
                str(REPO / "scripts/vision"),
                *(str(REPO / "ros" / name) for name in ("tatbot_ink", "tatbot_motion", "tatbot_description"))]

from tatbot_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
