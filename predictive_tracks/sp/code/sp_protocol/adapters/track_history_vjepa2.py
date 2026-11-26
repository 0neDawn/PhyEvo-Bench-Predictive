"""Additive fixed-budget state-anchor history entry; frozen adapter unchanged."""
from predictive_tracks.controlled_tracks.history_context import HistoryContextMixin
from .vjepa2_official_preprocess import Adapter as FrozenAdapter


class Adapter(HistoryContextMixin, FrozenAdapter):
    pass
