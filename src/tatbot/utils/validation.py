"""Validation utilities for tatbot data models."""

from pathlib import Path
from typing import Optional


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
        
    # Check pens config path
    if 'pens_config_path' in scene_config and scene_config['pens_config_path']:
        pens_path = Path(scene_config['pens_config_path']).expanduser()
        if not pens_path.exists():
            raise ValueError(f"Pens config path does not exist: {pens_path}")
    
    # Check design directory path
    if 'design_dir_path' in scene_config and scene_config['design_dir_path']:
        design_path = Path(scene_config['design_dir_path']).expanduser()
        if not design_path.exists():
            raise ValueError(f"Design directory path does not exist: {design_path}")