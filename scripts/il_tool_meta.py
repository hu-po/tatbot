#!/usr/bin/env python3
"""Stamp a recorded dataset with the tool that made it.

    il_tool_meta.py --ee-tool <id> --repo-id <user>/<name> [--push]
    il_tool_meta.py --ee-tool <id> --root ~/.cache/huggingface/lerobot/<user>/<name>
    il_tool_meta.py --ee-tool <id> --root <dir> --recorded-with-past-tool \
        [--workspace <era workspace.yaml>]

LeRobot's own metadata says only ``robot_type: tatbot_follower``, which does
not distinguish a 60 mm rotary pen from anything else that could be in the
gripper. This writes ``meta/tool.json`` next to it: the fitted tool's whole
datasheet INLINED, plus the measured tip offset and the touch-off it came
from.

Inlined, not referenced, on purpose. The datasheet will be edited and this
pen will eventually be retired, but a dataset recorded today has to stay
readable in a year — so it carries the geometry itself, with the hash of the
file it came from for anyone who wants to check it against a living datasheet.

Stamping an ARCHIVE is a different job and says so. By default the tool must
match the one config/workspace.yaml is calibrated for, because a live
recording that disagrees with the live calibration is an error. An old dataset
disagrees by construction — it was made under a tool that has since been
swapped — so --recorded-with-past-tool lifts that check. It also refuses to
borrow TODAY's geometry for it: the measured tip offset either comes from an
era-correct --workspace or is recorded as null. Writing the laser's 136 mm tip
onto a dataset drawn with a 63.7 mm ballpoint would be worse than leaving it
unstamped, and the file records that the tool was asserted, not verified.

Nothing here moves the arm; it is safe to re-run on an existing dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "lib"))

import ink_session  # noqa: E402
import ink_spec  # noqa: E402
import tool_spec  # noqa: E402

LEROBOT_CACHE = Path(os.environ.get(
    "HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")).expanduser()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", help="<user>/<name>; resolved under the LeRobot cache")
    ap.add_argument("--root", type=Path, help="dataset directory (overrides --repo-id)")
    ap.add_argument("--arm", default="right")
    ap.add_argument("--ee-tool", "--tool-id", dest="tool_id", required=True,
                    help="REQUIRED: the tool in the gripper for this dataset. "
                         "Not defaulted to workspace.yaml — a dataset stamped "
                         "with the previous tool's geometry is silently wrong "
                         "for every episode in it.")
    ap.add_argument("--push", action="store_true",
                    help="also upload the file to the hub dataset repo")
    ap.add_argument("--recorded-with-past-tool", action="store_true",
                    help="stamp an archive: the dataset was recorded under a "
                         "tool that is no longer fitted. Lifts the "
                         "workspace.yaml cross-check and does NOT borrow "
                         "today's geometry — pass --workspace for the era's, "
                         "or the tip offset is recorded as null.")
    ap.add_argument("--workspace", type=Path,
                    help="era-correct workspace.yaml for a "
                         "--recorded-with-past-tool stamp; recover one with "
                         "`git show <commit>:config/workspace.yaml`")
    args = ap.parse_args()

    root = args.root or (LEROBOT_CACHE / args.repo_id if args.repo_id else None)
    if root is None:
        return ap.error("pass --root or --repo-id")
    root = Path(root).expanduser()
    if not (root / "meta").is_dir():
        print(f"no dataset at {root} (expected a meta/ directory)", file=sys.stderr)
        return 1

    extra = None
    if args.recorded_with_past_tool:
        # Do not borrow the live workspace: its geometry belongs to whatever is
        # fitted NOW, which by definition is not what recorded this.
        if args.workspace:
            workspace = tool_spec.parse_simple_yaml(
                args.workspace.expanduser().read_text())
            provenance = f"geometry from {args.workspace}"
        else:
            workspace = {}
            provenance = "no era workspace supplied; tip offset left null"
        spec = tool_spec.load_tool(args.tool_id, REPO)
        extra = {"tool_id_source": "asserted retrospectively, not verified "
                                   "against a live calibration",
                 "retrospective_note": provenance}
    else:
        workspace = tool_spec.read_workspace(REPO)
        # Stated and cross-checked: this stamps every episode in the dataset
        # with the tool's geometry, and a wrong stamp is unrecoverable later.
        spec = tool_spec.require_stated_tool(
            args.tool_id, REPO, args.arm, workspace,
            context="dataset tool metadata")
    path = tool_spec.write_dataset_tool_metadata(
        root, spec, workspace, args.arm, extra)
    payload = json.loads(path.read_text())
    tip = payload.get("tip_offset_m")
    print(f"wrote {path}")
    # The ink story next to the tool story, for the same reason: the palette
    # load and the policy constants will change, and the dataset has to stay
    # readable after they do. A tool with ink.mode none gets a stamp that
    # says so, which is still an answer.
    ink_meta = ink_spec.dataset_ink_metadata(spec, REPO, args.arm)
    ink_meta["source"] = "archive" if args.recorded_with_past_tool else "robot"
    # Whether this recording took part in ink accounting at all: a launcher
    # run with --no-ink (TATBOT_INK=0) did not, and says so; otherwise the
    # open session on this node is the one the episodes drew from.
    tracking = os.environ.get("TATBOT_INK", "1") != "0" and not args.recorded_with_past_tool
    ink_meta["tracking"] = tracking
    sess = ink_session.current() if tracking else None
    ink_meta["session"] = (
        {"session_id": sess.session_id, "node": sess.node, "charge_ul": sess.charge_ul,
         "capacity_ul": sess.capacity_ul, "ink": sess.ink_id, "dips": sess.dips}
        if sess is not None else None)
    ink_path = Path(path).with_name("ink.json")
    ink_path.write_text(json.dumps(ink_meta, indent=2) + "\n")
    print(f"wrote {ink_path} (ink.mode {ink_meta['policy']['mode']}, tracking {tracking}"
          f"{', session ' + sess.session_id if sess else ''})")
    print(f"  tool {spec.summary()}")
    print(f"  tip offset {'uncalibrated' if tip is None else [round(v * 1000, 2) for v in tip]} mm")

    if args.push:
        if not args.repo_id:
            print("--push needs --repo-id", file=sys.stderr)
            return 1
        # The dataset itself was pushed by lerobot-record; this is one extra
        # file into the same repo, so a hub copy is not missing its tool.
        result = subprocess.run(
            ["hf", "upload", args.repo_id, str(path), "meta/tool.json",
             "--repo-type", "dataset"],
            check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: could not upload meta/tool.json ({result.stderr.strip()});"
                  " the local copy is written", file=sys.stderr)
        else:
            print(f"  uploaded to {args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
