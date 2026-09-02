"""One contact contract shared by planning, simulation, and dataset audit.

Distances are signed along the substrate's outward normal: positive is above
the work, zero is the resolved working point on the surface, and negative is
penetration.  These are numerical tolerances, not claims about safe real-arm
force or soft-tissue deformation.
"""

INTERACTION_MODEL = "rigid-contact-v1"
KINEMATIC_INTERACTION_MODEL = "kinematic-contact-v1"
WORKING_OFFSET_M = 0.0
CONTACT_ABOVE_TOLERANCE_M = 0.0005
MAX_PENETRATION_M = 0.00025
PHYSICS_CONTACT_OFFSET_M = 0.0002
TRAVEL_FLOOR_M = 0.001


def within_contact_band(distance_m: float) -> bool:
    return -MAX_PENETRATION_M <= distance_m <= CONTACT_ABOVE_TOLERANCE_M


def model_for(*, collision: bool) -> str:
    """Name the model that actually constrained substrate interaction."""
    return INTERACTION_MODEL if collision else KINEMATIC_INTERACTION_MODEL
