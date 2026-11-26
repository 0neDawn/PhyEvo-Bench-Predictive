"""Additive G-SWM orientation track; retains the original eight MC samples."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .gswm_foreground_predictive_mean_robust import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
