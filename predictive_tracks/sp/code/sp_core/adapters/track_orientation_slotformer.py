"""Additive SlotFormer orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .slotformer_obj3d import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
