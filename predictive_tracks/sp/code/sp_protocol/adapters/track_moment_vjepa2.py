"""Additive history-mean projection; native neural graph and score unchanged."""
from predictive_tracks.controlled_tracks.moment_projection import VJEPAMomentMixin
from .vjepa2_official_preprocess import Adapter as FrozenAdapter


class Adapter(VJEPAMomentMixin, FrozenAdapter):
    pass
