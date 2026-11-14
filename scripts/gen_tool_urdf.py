#!/usr/bin/env python3
"""Put each drawing arm's fitted tool into its model's URDF (urdf/tatbot.urdf, urdf/xarm7.urdf), where FK sees it.

    gen_tool_urdf.py [--check]

The block, what it models and why: scripts/lib/tool_urdf.py. Re-run this
after editing config/workspace.yaml or a fitted tool's datasheet; ``--check``
verifies the files are current without writing (exit 1 if one is stale), which
is what `scripts/check tools` runs (with `tool sync` and
`export_wrist_tags.py --check`). A ROS owner's fitted cartridge needs no run:
its robot description renders the block from it at every stack start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts/lib"))
from tatbot_paths import bootstrap  # noqa: E402

bootstrap()

import tool_spec  # noqa: E402
import tool_urdf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the URDF is not what this would generate")
    args = ap.parse_args()

    workspace = tool_spec.read_workspace(REPO)
    stale = []
    for path, fitted in tool_urdf.drawing_arms(REPO, workspace).items():
        try:
            rendered = tool_urdf.build(REPO, path, fitted, workspace)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        for arm in fitted:
            spec, _, measured = tool_urdf.tip(REPO, arm, workspace)
            if not measured:
                print(f"  {arm}: no touch-off in {spec.mount_frame(arm)} yet — modelling {spec.tool_id} at its "
                      "datasheet nominal", file=sys.stderr)
            print(f"  {arm}: {spec.summary()}")
        name = path.relative_to(REPO)
        if rendered == path.read_text():
            print(f"{name} {'is' if args.check else 'already'} current")
        elif args.check:
            stale.append(name)
        else:
            path.write_text(rendered)
            print(f"wrote {name}")
    for name in stale:
        print(f"{name} is stale — run scripts/gen_tool_urdf.py", file=sys.stderr)
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
