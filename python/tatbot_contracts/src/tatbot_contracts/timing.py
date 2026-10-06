"""Integer sampling of a slower sensor on a controller clock."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleCadence:
    control_hz: int
    sample_hz: int

    def __post_init__(self):
        if (type(self.control_hz) is not int or type(self.sample_hz) is not int
                or not 0 < self.sample_hz <= self.control_hz):
            raise ValueError('sample frequency must be a positive integer no faster than control')

    def due(self, tick: int) -> bool:
        """Capture at the first controller tick at or after each sample deadline.

        Tick zero is initialization, not a recorded frame. At 400/30 Hz,
        captures fall on ticks 14, 27, 40, ...; no command is resampled.
        """
        if type(tick) is not int or tick <= 0:
            raise ValueError('capture tick must be a positive integer')
        return tick * self.sample_hz // self.control_hz > (tick - 1) * self.sample_hz // self.control_hz

    def metadata(self) -> dict:
        return {'control_hz': self.control_hz, 'sample_hz': self.sample_hz,
                'sampling': 'first-control-tick-at-or-after-deadline',
                'initialization_tick': 0, 'maximum_lateness_s_exclusive': 1 / self.control_hz}
