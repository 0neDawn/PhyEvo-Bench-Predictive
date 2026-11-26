"""Additive OCVP-Seq orientation track; original adapter is unchanged."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .ocvp_seq import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
