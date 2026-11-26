"""Additive SAVi-Transformer orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .savi_transformer import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
