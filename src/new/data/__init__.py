import yaml
from dataclasses import asdict, is_dataclass
from typing import Type, TypeVar, Any
import os

from tatbot.utils.log import get_logger

log = get_logger('data', '🗃️')

T = TypeVar('T', bound='Yaml')

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
        log.debug(f"Loading {cls.__name__} from {filepath}...")
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls._fromdict(data)

    def to_yaml(self, filepath: str) -> None:
        if not is_dataclass(self):
            raise TypeError(f"{self.__class__.__name__} must be a dataclass to use to_yaml.")
        log.debug(f"Saving {self.__class__.__name__} to {filepath}...")
        with open(filepath, 'w') as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def _fromdict(cls: Type[T], data: Any) -> T:
        def convert(fieldtype, value):
            typ = fieldtype if isinstance(fieldtype, type) else type(fieldtype)
            if is_dataclass(typ) and isinstance(value, dict):
                fromdict = getattr(typ, '_fromdict', None)
                if callable(fromdict):
                    return fromdict(value)
                else:
                    return typ(**value)
            origin = getattr(typ, '__origin__', None)
            args = getattr(typ, '__args__', None)
            if origin in (list, tuple) and args:
                subtype = args[0]
                return type(value)(convert(subtype, v) for v in value)
            return value
        if is_dataclass(cls):
            fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
            for k, v in data.items():
                if k in fieldtypes:
                    data[k] = convert(fieldtypes[k], v)
        return cls(**data)
