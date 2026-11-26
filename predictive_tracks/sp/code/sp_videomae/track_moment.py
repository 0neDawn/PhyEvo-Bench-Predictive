"""Additive history-mean projection; native neural graph and score unchanged."""
from predictive_tracks.controlled_tracks.moment_projection import VideoMAEMomentMixin
from .adapter import Adapter as FrozenAdapter


class Adapter(VideoMAEMomentMixin, FrozenAdapter):
    pass
