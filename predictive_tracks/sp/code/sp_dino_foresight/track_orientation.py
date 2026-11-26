"""Diagnostic-only DINO orientation track; known baseline alignment is not repaired."""
from predictive_tracks.controlled_tracks.orientation import OrientationMixin
from .adapter import Adapter as FrozenAdapter


class Adapter(OrientationMixin, FrozenAdapter):
    pass
