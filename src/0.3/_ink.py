from dataclasses import dataclass, field
import yaml

from _log import get_logger

log = get_logger('_ink')

@dataclass
class InkCap:
    """Individual cylindrical inkcap."""
    palette_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Relative position (meters) of the inkcap in the palette frame (x, y, z)."""
    diameter_m: float = 0.008
    """Diameter of the inkcap (meters)."""
    depth_m: float = 0.01
    """Depth of the inkcap (meters)."""
    color: str = "black"
    """Natural language description of the color of the ink inside the inkcap."""

    @classmethod
    def from_dict(cls, d):
        return cls(
            palette_pos=tuple(d.get('palette_pos', (0.0, 0.0, 0.0))),
            diameter_m=d.get('diameter_m', 0.008),
            depth_m=d.get('depth_m', 0.01),
            color=d.get('color', 'black'),
        )

    def to_dict(self):
        return {
            'palette_pos': list(self.palette_pos),
            'diameter_m': self.diameter_m,
            'depth_m': self.depth_m,
            'color': self.color,
        }

@dataclass
class InkPalette:
    inkcaps: dict[str, InkCap] = field(default_factory=lambda: {
        # outer row
        # "small_0": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # "small_1": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        "large_0": InkCap(
            palette_pos=(0.0, 0.0, 0.0), # center of palette is center of big inkcap
            diameter_m=0.014,
            depth_m=0.014,
            color="black"
        ),
        # "small_2": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # "small_3": InkCap(
        #     palette_pos=(0.0, 0.0, 0.0),
        #     color=""
        # ),
        # inner row
        "medium_0": InkCap(
            palette_pos=(0.018, -0.025, 0.0),
            diameter_m=0.012,
            color="red"
        ),
        "medium_1": InkCap(
            palette_pos=(0.0, -0.02, 0.0),
            diameter_m=0.012,
            color="green"
        ),
        "medium_2": InkCap(
            palette_pos=(-0.018, -0.025, 0.0),
            diameter_m=0.012,
            color="blue"
        ),
    })

    @classmethod
    def from_yaml(cls, filepath: str) -> "InkPalette":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        inkcaps = {k: InkCap.from_dict(v) for k, v in data.get('inkcaps', {}).items()}
        return cls(inkcaps=inkcaps)

    def save_yaml(self, filepath: str):
        with open(filepath, "w") as f:
            yaml.safe_dump({'inkcaps': {k: v.to_dict() for k, v in self.inkcaps.items()}}, f)