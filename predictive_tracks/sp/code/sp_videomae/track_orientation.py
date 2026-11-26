"""Additive VideoMAE-V2 orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .adapter import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
