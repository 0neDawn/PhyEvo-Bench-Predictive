"""Complementary historical-change-space boundary ablation."""
from predictive_tracks.controlled_tracks.orthogonal_boundary import ParallelBoundaryMixin
from .track_moment_vjepa2 import Adapter as NativeInterceptionAdapter


class Adapter(ParallelBoundaryMixin, NativeInterceptionAdapter):
    pass
