"""Additive V-JEPA 2 ViT-L orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .adapter import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
