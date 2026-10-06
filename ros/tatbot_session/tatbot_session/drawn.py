"""Shared drawn evidence API; no motion or controller ownership."""
from tatbot_contracts.ros_drawn import (  # noqa: F401 -- existing native API
    _path,
    _xy,
    from_rows,
    render,
    write,
)
