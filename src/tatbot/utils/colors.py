from typing import Dict, Tuple

COLORS: Dict[str, Tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "blue": (82, 153, 224),
    "green": (82, 224, 105),
    "red": (224, 86, 82),
    "yellow": (224, 212, 82),
    "purple": (189, 82, 224),
    "orange": (224, 117, 82),
    "gold": (189, 224, 82),
    "chartreuse": (105, 224, 82),
    "mint": (82, 224, 177),
    "teal": (82, 224, 224),
    "cyan": (82, 189, 224),
    "indigo": (82, 82, 224),
    "violet": (129, 82, 224),
    "magenta": (224, 82, 201),
    "pink": (224, 82, 129),
}


def argb_to_bgr(argb: int) -> tuple[int, int, int]:
    """Convert ARGB color value to BGR tuple for OpenCV."""
    alpha = (argb >> 24) & 0xFF
    red = (argb >> 16) & 0xFF
    green = (argb >> 8) & 0xFF
    blue = argb & 0xFF
    
    return (blue, green, red)
