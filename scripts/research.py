#!/usr/bin/env python3
"""DBV3 paired research; production preparation and the ROS stack own drawing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for root in ('scripts/lib', 'python/tatbot_contracts/src', 'ros/tatbot_ink', 'ros/tatbot_motion',
             'ros/tatbot_description', 'ros/tatbot_session'):
    sys.path.insert(0, str(REPO/root))

from draw_research.cli import main  # noqa: E402 -- entry point owns its build-root paths

if __name__ == '__main__':
    try:
        raise SystemExit(main(REPO))
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'research: {exc}', file=sys.stderr)
        raise SystemExit(3) from exc
