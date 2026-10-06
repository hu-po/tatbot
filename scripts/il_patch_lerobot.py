#!/usr/bin/env python3
"""Apply the versioned Tatbot patch catalog after syncing the locked LeRobot environment.

Validate every available target before writing. The CLI records source hashes in
<venv>/share/tatbot/lerobot-patches.json; imports and test calls have no receipt side effect.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()
from lerobot_patch_engine import apply_patches  # noqa: E402
from lerobot_patches import PATCHES  # noqa: E402


def main(receipt_path: Path | None = None):
    return apply_patches(PATCHES, receipt_path=receipt_path)


if __name__ == "__main__":
    sys.exit(main(Path(sys.prefix) / "share" / "tatbot" / "lerobot-patches.json"))
