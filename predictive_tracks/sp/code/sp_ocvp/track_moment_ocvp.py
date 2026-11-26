"""Additive history-mean projection; native neural graph and score unchanged."""
from predictive_tracks.controlled_tracks.moment_projection import OCVPMomentMixin
from .ocvp_seq import Adapter as FrozenAdapter


class Adapter(OCVPMomentMixin, FrozenAdapter):
    pass
