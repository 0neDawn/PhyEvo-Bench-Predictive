"""Additive history-mean projection; native neural graph and score unchanged."""
from predictive_tracks.controlled_tracks.moment_projection import DINOMomentMixin
from .adapter import Adapter as FrozenAdapter


class Adapter(DINOMomentMixin, FrozenAdapter):
    pass
