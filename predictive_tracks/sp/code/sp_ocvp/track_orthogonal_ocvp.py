"""Additive historical-subspace orthogonal boundary alignment."""
from predictive_tracks.controlled_tracks.orthogonal_boundary import OrthogonalBoundaryMixin
from .track_moment_ocvp import Adapter as NativeInterceptionAdapter


class Adapter(OrthogonalBoundaryMixin, NativeInterceptionAdapter):
    pass
