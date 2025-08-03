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
        
        data = self.model_dump(mode='json')
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
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