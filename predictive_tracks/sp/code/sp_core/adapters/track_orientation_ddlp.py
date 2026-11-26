"""Additive DDLP orientation track."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .ddlp_obj3d import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
