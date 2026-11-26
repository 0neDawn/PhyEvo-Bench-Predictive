"""Additive prediction-boundary alignment, no extra neural calls."""
from predictive_tracks.controlled_tracks.boundary_alignment import BoundaryAlignmentMixin
from .track_moment_vjepa2 import Adapter as NativeInterceptionAdapter


class Adapter(BoundaryAlignmentMixin, NativeInterceptionAdapter):
    pass
