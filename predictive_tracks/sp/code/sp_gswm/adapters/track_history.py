"""Additive fixed-budget state-anchor history entry; frozen adapter unchanged."""
from predictive_tracks.controlled_tracks.history_context import HistoryContextMixin
from .gswm_foreground_predictive_mean_robust import Adapter as FrozenAdapter


class Adapter(HistoryContextMixin, FrozenAdapter):
    pass
