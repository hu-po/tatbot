"""Calibration uses the shared station observation, geometry and freshness policy."""
from tatbot_motion import station as model


def __getattr__(name):
    return getattr(model, name)
