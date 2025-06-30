import os
import numpy as np
from dataclasses import asdict, is_dataclass
from typing import Any, Type, TypeVar

import yaml

from tatbot.utils.log import get_logger

log = get_logger('data', '🗃️')

T = TypeVar('T', bound='Yaml')

FLOAT_TYPE = np.float32
log.debug(f"using {FLOAT_TYPE} for numpy arrays")

def dataclass_to_dict(obj):
    """Recursively convert dataclass to dict, converting np.ndarray to list."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif hasattr(obj, '__dataclass_fields__'):
        return {k: dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    else:
        return obj

class Yaml:
    """
    Shared dataclass mixin for YAML (de)serialization.
    Inherit this in your @dataclass for load/save methods.
    """
    yaml_dir: str = os.path.expanduser("~/tatbot/config")
    default: str = os.path.join(yaml_dir, "default.yaml")

    @classmethod
    def get_yaml_dir(cls) -> str:
        """
        Returns the directory where YAML files for this class are stored.
        Subclasses can override yaml_dir or this method for custom logic.
        """
        return cls.yaml_dir

    @classmethod
    def from_name(cls: Type[T], name: str) -> T:
        """
        Loads an instance from a YAML file in the class's yaml_dir, given the base name (without .yaml).
        """
        filepath = os.path.join(cls.get_yaml_dir(), f"{name}.yaml")
        return cls.from_yaml(filepath)

    @classmethod
    def from_yaml(cls: Type[T], filepath: str) -> T:
        filepath = os.path.expanduser(filepath)
        assert os.path.exists(filepath), f"❌ File {filepath} does not exist"
        log.debug(f"💾 Loading {cls.__name__} from {filepath}...")
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        _output = cls._fromdict(data)
        log.debug(f"✅ Loaded {cls.__name__}: {_output}")
        return _output

    def to_yaml(self, filepath: str) -> None:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass to use to_yaml.")
        log.debug(f"💾 Saving {self.__class__.__name__} to {filepath}...")
        with open(filepath, 'w') as f:
            yaml.safe_dump(dataclass_to_dict(self), f)

    @classmethod
    def _fromdict(cls: Type[T], data: Any) -> T:
        def convert(fieldtype, value):
            origin = getattr(fieldtype, '__origin__', None)
            args = getattr(fieldtype, '__args__', None)
            if origin is tuple and args:
                subtype = args[0]
                return tuple(convert(subtype, v) for v in value)
            elif origin is list and args:
                subtype = args[0]
                return [convert(subtype, v) for v in value]
            if fieldtype is np.ndarray:
                return np.array(value, dtype=FLOAT_TYPE)
            if hasattr(fieldtype, '__dataclass_fields__') and isinstance(value, dict):
                fromdict = getattr(fieldtype, '_fromdict', None)
                if callable(fromdict):
                    return fromdict(value)
                else:
                    return fieldtype(**value)
            return value
        if hasattr(cls, '__dataclass_fields__'):
            fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            for k, v in data.items():
                if k in fieldtypes:
                    data[k] = convert(fieldtypes[k], v)
        return cls(**data)

    def __str__(self) -> str:
        return yaml.safe_dump(dataclass_to_dict(self), sort_keys=False, allow_unicode=True)

    def to_dict(self):
        return dataclass_to_dict(self)
