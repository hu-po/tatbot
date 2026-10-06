"""Tatbot simulation helpers. Importing this package never builds a world.

Engine construction is explicit in ``tatbot_sim.env.TatbotDrawEnv``; pure
configuration and artifact preparation remain usable without an engine.
"""

import sys
from pathlib import Path

# The simulator imports the script tree by bare module name (arm_kinematics,
# ee_fiducial, ...). One bootstrap here replaces the sys.path
# edits every such module used to carry. The editable tatbot-scriptlib install
# exports tatbot_paths; an environment without it (the body-model lock, a bare
# checkout on PYTHONPATH) reaches it by position.
try:
    from tatbot_paths import bootstrap
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/lib"))
    from tatbot_paths import bootstrap
from tatbot_sim.repo import repo_root

# TATBOT_REPO, not this file's checkout: the public simulator profile points
# it at a copy of the tree and binds every module it imports to that copy.
bootstrap(repo_root())

__all__: list[str] = []
