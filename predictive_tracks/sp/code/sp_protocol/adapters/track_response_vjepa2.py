"""Separate response-protected calibration; original neural graph unchanged."""
from predictive_tracks.controlled_tracks.response_calibration import ResponseCalibrationMixin
from .track_moment_vjepa2 import Adapter as NativeInterceptionAdapter


class Adapter(ResponseCalibrationMixin, NativeInterceptionAdapter):
    pass
