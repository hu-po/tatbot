"""Shared offline/native ink scoring; the native session retains all motion."""
from tatbot_contracts.ros_fidelity import (  # noqa: F401 -- existing scorer API
    _expected,
    _observed,
    analyse,
    identity,
    measure,
)
