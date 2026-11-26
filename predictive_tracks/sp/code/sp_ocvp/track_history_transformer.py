"""Additive fixed-budget state-anchor history entry; frozen adapter unchanged."""
from predictive_tracks.controlled_tracks.history_context import HistoryContextMixin
from .savi_transformer import Adapter as FrozenAdapter


class Adapter(HistoryContextMixin, FrozenAdapter):
    pass
