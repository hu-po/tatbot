"""Shared constants for tatbot."""

from pathlib import Path

# NFS mount point - canonical path for shared storage across all nodes
NFS_DIR = Path("/nfs/tatbot")

# Common subdirectories
NFS_RECORDINGS_DIR = NFS_DIR / "recordings"
NFS_DESIGNS_DIR = NFS_DIR / "designs"
NFS_LOGS_DIR = NFS_DIR / "mcp-logs"


def resolve_design_dir(design_name: str) -> Path:
    """Return the absolute path to a design directory by name."""
    return (NFS_DESIGNS_DIR / design_name)

# Local pens config directory (per-user workspace)
PENS_CONFIGS_DIR = Path.home() / "tatbot/config/dbv3/pens"


def resolve_pens_config_path(pens_config_name: str) -> Path:
    """Return the absolute path to a pens config JSON by name."""
    return (PENS_CONFIGS_DIR / f"{pens_config_name}.json")

# Local repo configuration directories
CONF_DIR = Path.home() / "tatbot/src/conf"
CONF_POSES_DIR = CONF_DIR / "poses"
CONF_CAMS_DIR = CONF_DIR / "cams"
CONF_SCENES_DIR = CONF_DIR / "scenes"
CONF_NODES_PATH = CONF_DIR / "nodes.yaml"

# Vendor/tooling configuration directories
TROSSEN_CONFIG_DIR = Path.home() / "tatbot/config/trossen"

# DrawingBotV3 local configuration directories
DBV3_DIR = Path.home() / "tatbot/config/dbv3"
DBV3_GCODE_DIR = DBV3_DIR / "gcode"
DBV3_AREAS_DIR = DBV3_DIR / "areas"

# NFS 3D assets directory (for meshes/pointclouds)
NFS_3D_DIR = NFS_DIR / "3d"

# Environment file on shared storage
ENV_FILE_PATH = NFS_DIR / ".env"