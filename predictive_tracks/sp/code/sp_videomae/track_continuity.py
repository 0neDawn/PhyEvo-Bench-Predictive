"""Additive history-rate boundary correction, with unchanged neural calls."""
from predictive_tracks.controlled_tracks.continuity_projection import ContinuityCorrectionMixin
from .track_moment import Adapter as NativeInterceptionAdapter


class Adapter(ContinuityCorrectionMixin, NativeInterceptionAdapter):
    pass
