"""Additive history-mean projection; native neural graph and score unchanged."""
from predictive_tracks.controlled_tracks.moment_projection import GSWMMomentMixin
from .gswm_foreground_predictive_mean_robust import Adapter as FrozenAdapter


class Adapter(GSWMMomentMixin, FrozenAdapter):
    pass
