"""Base classes for tatbot data models."""

import reprlib
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel


class BaseCfg(BaseModel):
    """Base configuration class with utility methods."""
    

    
    def to_yaml(self, filepath: str = None) -> str:
        """Convert model to YAML string or save to file."""
        def numpy_representer(dumper, data):
            """Custom representer for numpy arrays."""
            if isinstance(data, np.ndarray):
                return dumper.represent_list(data.tolist())
            return dumper.represent_data(data)
        
        # Add custom representer for numpy arrays
        yaml.add_representer(np.ndarray, numpy_representer)
        
        # Use model_dump without mode='json' to avoid JSON serialization issues with numpy
        try:
            data = self.model_dump()
        except Exception as e:
            # If direct model_dump fails, try converting numpy arrays to lists first
            data = self._model_dump_with_numpy_conversion()
        
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def _model_dump_with_numpy_conversion(self) -> dict:
        """Helper method to dump model with numpy arrays converted to lists."""
        import json
        
        def convert_numpy_to_lists(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_lists(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Get the model data and convert numpy arrays
        data = self.model_dump()
        return convert_numpy_to_lists(data)
    
    def __str__(self) -> str:
        """Pretty YAML representation, truncated for large arrays."""
        yaml_str = self.to_yaml()
        # Truncate very long output for readability
        if len(yaml_str) > 1000:
            return reprlib.repr(yaml_str)
        return yaml_str
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        class_name = self.__class__.__name__
        fields = ', '.join(f'{k}={v!r}' for k, v in self.model_dump().items())
        return f'{class_name}({fields})'
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'BaseCfg':
        """Load model from YAML file."""
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)