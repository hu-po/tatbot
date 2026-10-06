"""Shared timing evidence API; no motion or controller ownership."""
from tatbot_contracts.ros_timing import (  # noqa: F401 -- existing native API
    SCHEMA,
    begin,
    finish,
    summarize,
    time,
)
