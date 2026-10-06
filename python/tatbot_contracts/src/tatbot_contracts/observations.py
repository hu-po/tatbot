"""Named Tatbot follower channels and explicit simulator observation choices.

These describe the existing LeRobot features; they do not introduce a dataset
or transport format. Engine adapters and output writers share this ordering.
"""

from dataclasses import asdict, dataclass

FOLLOWER_JOINTS = tuple(f"joint_{index}" for index in range(6)) + ("left_carriage_joint",)
ACTION_NAMES = tuple(f"{joint}.pos" for joint in FOLLOWER_JOINTS)
STATE_NAMES = ACTION_NAMES + tuple(f"{joint}.ext_eff" for joint in FOLLOWER_JOINTS)


@dataclass(frozen=True)
class ObservationProfile:
    """Which effort measurements may enter the policy's existing state vector."""

    effort: str = "contact"
    effort_mask: tuple[bool, ...] = (True,) * 7

    def __post_init__(self):
        if self.effort not in ("contact", "unavailable"):
            raise ValueError("effort profile must be contact or unavailable")
        object.__setattr__(self, "effort_mask", tuple(self.effort_mask))
        if len(self.effort_mask) != 7 or any(type(value) is not bool for value in self.effort_mask):
            raise ValueError("effort mask must contain seven booleans in follower joint order")

    def metadata(self) -> dict:
        return {"schema": "tatbot.observation-profile/1", **asdict(self),
                "joints": FOLLOWER_JOINTS, "state": STATE_NAMES,
                "position_units": ("radian",) * 6 + ("meter",),
                "effort_units": ("newton-meter",) * 6 + ("newton",),
                "depth_unit": "millimeter", "invalid_depth": 0}
