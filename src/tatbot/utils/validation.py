"""Validation utilities for tatbot data models."""

from pathlib import Path
from typing import Optional

from tatbot.utils.constants import resolve_design_dir, resolve_pens_config_path


def expand_user_path(path_str: Optional[str]) -> Optional[Path]:
    """Centralized path expansion utility."""
    if path_str:
        return Path(path_str).expanduser()
    return None


def validate_files_exist(scene_config: dict, runtime_validate: bool = True) -> None:
    """
    Validate that files referenced in scene config exist.
    
    This should be called once at program start, not during model construction
    to avoid heavy I/O in validators.
    
    Args:
        scene_config: Scene configuration dictionary
        runtime_validate: Whether to perform file existence checks
    """
    if not runtime_validate:
        return
        
    # Check pens config by name
    if 'pens_config_name' in scene_config and scene_config['pens_config_name']:
        pens_path = resolve_pens_config_path(scene_config['pens_config_name'])
        if not pens_path.exists():
            raise ValueError(f"Pens config does not exist: {pens_path}")
    
    # Check design directory by name
    if 'design_name' in scene_config and scene_config['design_name']:
        design_path = resolve_design_dir(scene_config['design_name'])
        if not design_path.exists():
            raise ValueError(f"Design directory does not exist: {design_path}")