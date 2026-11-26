"""Complementary historical-change-space boundary ablation."""
from predictive_tracks.controlled_tracks.orthogonal_boundary import ParallelBoundaryMixin
from .track_moment_transformer import Adapter as NativeInterceptionAdapter


class Adapter(ParallelBoundaryMixin, NativeInterceptionAdapter):
    pass
