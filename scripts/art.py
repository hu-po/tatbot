#!/usr/bin/env python3
"""The art studio's commands, run by the shared offline drawing interpreter."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# tatbot_motion: `art shade` times its strokes by the planner's own estimate (pure numpy, no ROS).
# tatbot_ink, which reads tatbot_description: `art separate` checks a design against the compile's page before DBV3.
sys.path[:0] = [str(REPO / "scripts/lib"), str(REPO / "python/tatbot_art/src"), str(REPO / "python/tatbot_contracts/src"),
                *(str(REPO / "ros" / name) for name in ("tatbot_motion", "tatbot_ink", "tatbot_description"))]

from tatbot_art.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
