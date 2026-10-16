"""Shared simulation stamp representation using the existing HIL microsecond policy."""
import math


def seconds_to_us(seconds):
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('simulation sensor clock is nonfinite or negative')
    value = round(seconds * 1_000_000)
    if value >= 1 << 64:
        raise ValueError('simulation sensor clock exceeds MAVLink uint64')
    return value


def ros_time_parts(seconds):
    sec, us = divmod(seconds_to_us(seconds), 1_000_000)
    if sec >= 1 << 31:
        raise ValueError('simulation timestamp exceeds ROS signed int32 seconds')
    return sec, us * 1000
