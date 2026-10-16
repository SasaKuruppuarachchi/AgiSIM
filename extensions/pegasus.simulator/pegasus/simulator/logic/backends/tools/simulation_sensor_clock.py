"""Owned HIL clock adapter; shared engine time is explicit, never inferred."""
import math
from .simulation_timestamp_policy import seconds_to_us


class SimulationSensorClock:
    def __init__(self, seconds_provider):
        if not callable(seconds_provider):
            raise ValueError('simulation_time_provider must be callable')
        self.provider = seconds_provider
        self.reset()

    def reset(self):
        # Only a supervised backend/PX4 session restart permits a clock reset.
        self.last_us = None
        self.fault = None

    def timestamp_us(self):
        if self.fault is not None:
            return None
        try:
            seconds = self.provider()
            if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('simulation sensor clock is nonfinite or negative')
            us = seconds_to_us(seconds)
            if us >= 1 << 64:
                raise ValueError('simulation sensor clock exceeds MAVLink uint64')
            if self.last_us is not None and us < self.last_us:
                raise ValueError('simulation sensor clock regressed; restart PX4/backend session')
        except Exception as error:
            self.fault = str(error)
            return None
        if us == 0 or us == self.last_us:
            return None
        self.last_us = us
        return us
