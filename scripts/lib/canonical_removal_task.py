#!/usr/bin/env python3
"""The task string the sim would use for removal with the fitted tool.

The sim builds its prompt from the tool's datasheet and the substrate registry.
A task typed by hand at recording time can drift from it, and a
language-conditioned policy needs only that difference to tell the two domains
apart -- so il_record.sh prints this alongside a task that disagrees.

Silent on a RULED substrate: there the motif is printed and legible, the
operator can name what they are removing, and the sim names it too.

    canonical_removal_task.py <repo> [tool-id]
"""

import sys
from pathlib import Path


def main() -> int:
    repo = sys.argv[1]
    sys.path.insert(0, str(Path(repo) / "scripts" / "lib"))
    import tool_spec

    try:
        workspace = tool_spec.read_workspace(repo)
        tool_id = (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2]
                   else (workspace.get("right") or {}).get("tool_id"))
        spec = tool_spec.load_tool(tool_id, repo)
        substrate = tool_spec.substrate_for(spec, repo)
    except Exception:
        return 0  # nothing to say is always safe here; this only ever advises
    if substrate.ruled:
        return 0
    print(f"remove the ink {substrate.surface_phrase} {spec.prompt_phrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
