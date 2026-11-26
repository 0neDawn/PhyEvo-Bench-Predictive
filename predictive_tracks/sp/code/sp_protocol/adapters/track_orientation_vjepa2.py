"""Additive V-JEPA 2 ViT-g orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .vjepa2_official_preprocess import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
