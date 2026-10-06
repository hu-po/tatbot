"""The sole Tatbot nominal-body implementation: MHR identity through SOMA-X."""

from tatbot_sim.body_models.io import BodyModelError, verify_body_cache, verify_software_lock
from tatbot_sim.body_models.mhr_soma import (
    SOMAPosedBody,
    SOMASurface,
    canonical_surface_digest,
    canonical_topology_digest,
)

__all__ = [
    "BodyModelError",
    "SOMAPosedBody",
    "SOMASurface",
    "canonical_surface_digest",
    "canonical_topology_digest",
    "verify_body_cache",
    "verify_software_lock",
]
