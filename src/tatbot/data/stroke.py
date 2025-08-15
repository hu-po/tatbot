from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import jax_dataclasses as jdc
import numpy as np
from jaxtyping import Array, Float
from pydantic import Field, field_validator
from safetensors.flax import load_file, save_file

from tatbot.data.base import BaseCfg
from tatbot.utils.log import get_logger

log = get_logger("data.stroke", "🔳")


class Stroke(BaseCfg):
    """A stroke in the tatbot system."""
    
    model_config = {'arbitrary_types_allowed': True}
    
    description: str
    """Natural language description of the path."""
    arm: str
    """Arm that will execute the path, either left or right."""

    # Large numpy arrays - stored as separate files
    meter_coords: Optional[np.ndarray] = None  # (N, 3)
    """Numpy array of meter coordinates for each pose in path <x, y, z> (lasercross frame)."""
    pixel_coords: Optional[np.ndarray] = None  # (N, 2)
    """Numpy array of pixel coordinates for each pose in path <x (0-width), y (0-height)> (lasercross frame)."""
    ee_pos: Optional[np.ndarray] = None  # (N, 3)
    """End effector position in meters <x, y, z> (world frame)."""
    ee_rot: Optional[np.ndarray] = None  # (N, 4)
    """End effector orientation in quaternion <x, y, z, w> (world frame)."""
    normals: Optional[np.ndarray] = None  # (N, 3)
    """Surface normals for each pose in the stroke <x, y, z> (world frame)."""
    
    # File references for arrays (populated during save/load)
    meter_coords_file: Optional[str] = Field(None, exclude=True)
    """File path for meter_coords array."""
    pixel_coords_file: Optional[str] = Field(None, exclude=True)
    """File path for pixel_coords array."""
    ee_pos_file: Optional[str] = Field(None, exclude=True)
    """File path for ee_pos array."""
    ee_rot_file: Optional[str] = Field(None, exclude=True)
    """File path for ee_rot array."""
    normals_file: Optional[str] = Field(None, exclude=True)
    """File path for normals array."""
    
    gcode_text: Optional[str] = None
    """G-code text for the stroke."""
    frame_path: Optional[str] = None
    """Relative path to the frame image for this stroke, or None if not applicable."""

    is_rest: bool = False
    """Whether the stroke is a rest stroke."""
    is_inkdip: bool = False
    """Whether the stroke is an inkdip stroke."""

    inkcap: Optional[str] = None
    """Name of the inkcap which provided the ink for the stroke."""
    color: Optional[str] = None
    """Color of the stroke."""
    
    @field_validator('arm')
    def validate_arm(cls, v: str) -> str:  # noqa: N805
        if v not in ['left', 'right']:
            raise ValueError("arm must be 'left' or 'right'")
        return v
    
    @field_validator('meter_coords', 'pixel_coords', 'ee_pos', 'ee_rot', 'normals', mode='before')
    def convert_numpy_arrays(cls, v: Any) -> Any:  # noqa: N805
        """Convert lists to numpy arrays for validation, keep arrays as-is."""
        if isinstance(v, list):
            return np.array(v, dtype=np.float32)
        return v
    
    def save_arrays(self, base_dir: Union[str, Path], stroke_id: str) -> None:
        """Save numpy arrays to separate .npy files and update file references."""
        base_dir = Path(base_dir)
        arrays_dir = base_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        
        array_fields = ['meter_coords', 'pixel_coords', 'ee_pos', 'ee_rot', 'normals']
        for field_name in array_fields:
            array_data = getattr(self, field_name)
            if array_data is not None:
                file_path = arrays_dir / f"{stroke_id}_{field_name}.npy"
                np.save(file_path, array_data)
                # Store relative path for portability
                setattr(self, f"{field_name}_file", f"arrays/{stroke_id}_{field_name}.npy")
    
    def load_arrays(self, base_dir: Union[str, Path]) -> None:
        """Load numpy arrays from .npy files using stored file references."""
        base_dir = Path(base_dir)
        
        array_fields = ['meter_coords', 'pixel_coords', 'ee_pos', 'ee_rot', 'normals']
        for field_name in array_fields:
            file_ref = getattr(self, f"{field_name}_file")
            if file_ref is not None:
                file_path = base_dir / file_ref
                if file_path.exists():
                    array_data = np.load(file_path)
                    setattr(self, field_name, array_data)
                    
    def model_dump_for_yaml(self) -> dict:
        """Dump model excluding numpy arrays, keeping only file references."""
        data = self.model_dump()
        # Remove numpy arrays from the dump (they're saved separately)
        array_fields = ['meter_coords', 'pixel_coords', 'ee_pos', 'ee_rot', 'normals']
        for field_name in array_fields:
            if field_name in data:
                del data[field_name]
        # Include file references
        for field_name in array_fields:
            file_ref = getattr(self, f"{field_name}_file")
            if file_ref is not None:
                data[f"{field_name}_file"] = file_ref
        return data    


class StrokeList(BaseCfg):
    """List of stroke pairs."""
    
    strokes: list[tuple[Stroke, Stroke]]
    """List of stroke pairs."""
    
    def to_yaml_with_arrays(self, filepath: Union[str, Path]) -> None:
        """Save stroke list to YAML with arrays stored separately."""
        filepath = Path(filepath)
        base_dir = filepath.parent
        
        # Save arrays for each stroke
        for stroke_idx, (stroke_l, stroke_r) in enumerate(self.strokes):
            stroke_l.save_arrays(base_dir, f"stroke_{stroke_idx}_left")
            stroke_r.save_arrays(base_dir, f"stroke_{stroke_idx}_right")
        
        # Prepare data for YAML (without numpy arrays)
        yaml_data = {
            "strokes": [
                [stroke_l.model_dump_for_yaml(), stroke_r.model_dump_for_yaml()]
                for stroke_l, stroke_r in self.strokes
            ]
        }
        
        # Save YAML
        import yaml
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
    @classmethod
    def from_yaml_with_arrays(cls, filepath: Union[str, Path]) -> 'StrokeList':
        """Load stroke list from YAML and restore arrays from separate files."""
        filepath = Path(filepath)
        base_dir = filepath.parent
        
        # Load YAML data
        import yaml
        with open(filepath) as f:
            yaml_data = yaml.safe_load(f)
        
        # Reconstruct strokes
        strokes = []
        for stroke_data_pair in yaml_data["strokes"]:
            stroke_l_data, stroke_r_data = stroke_data_pair
            
            # Create stroke objects
            stroke_l = Stroke(**stroke_l_data)
            stroke_r = Stroke(**stroke_r_data)
            
            # Load arrays
            stroke_l.load_arrays(base_dir)
            stroke_r.load_arrays(base_dir)
            
            strokes.append((stroke_l, stroke_r))
        
        return cls(strokes=strokes)




@jdc.pytree_dataclass
class StrokeBatch:
    """
    batch of strokes:
    b = batch size
    l = stroke length
    o = offset num
    """

    ee_pos_l: Float[Array, "b l o 3"]  # type: ignore[misc]
    """End effector frame position in meters (x, y, z) for left arm."""
    ee_pos_r: Float[Array, "b l o 3"]  # type: ignore[misc]
    """End effector frame position in meters (x, y, z) for right arm."""
    ee_rot_l: Float[Array, "b l o 4"]  # type: ignore[misc]
    """End effector frame orientation as quaternion (w, x, y, z) for left arm."""
    ee_rot_r: Float[Array, "b l o 4"]  # type: ignore[misc]
    """End effector frame orientation as quaternion (w, x, y, z) for right arm."""
    joints: Float[Array, "b l o 14"]  # type: ignore[misc]
    """Joint positions in radians (URDF convention)."""

    def save(self, filepath: str) -> None:
        log.debug(f"💾 Saving StrokeBatch to {filepath}")
        save_file({k: getattr(self, k) for k in self.__dataclass_fields__}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "StrokeBatch":
        filepath = os.path.expanduser(filepath)
        assert os.path.exists(filepath), f"❌ File {filepath} does not exist"
        log.debug(f"💾 Loading StrokeBatch from {filepath}")
        data = load_file(filepath)
        return cls(**data)

    def offset_joints(
        self, stroke_idx: int, pose_idx: int, offset_idx_l: int, offset_idx_r: int
    ) -> Float[Array, "14"]:  # type: ignore[misc]
        left_joints = self.joints[stroke_idx, pose_idx, offset_idx_l][:7]
        right_joints = self.joints[stroke_idx, pose_idx, offset_idx_r][7:]
        return np.concatenate([left_joints, right_joints])
