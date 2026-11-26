"""Common contract that enforces one prediction from shared history per pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


class PredictiveAdapter(Protocol):
    """Minimal interface implemented by each model-specific benchmark adapter."""

    model_name: str

    def predict(self, history: np.ndarray) -> Any:
        """Predict from shared history only; candidate futures are unavailable here."""

    def error_to_candidate(self, prediction: Any, candidate_future: np.ndarray) -> np.ndarray:
        """Return one non-negative error for each native prediction time step."""

    def frame_spans(self, prediction: Any) -> Sequence[int] | Sequence[Sequence[int]]:
        """Map native prediction steps to inclusive original-frame spans."""


@dataclass(frozen=True)
class CandidateErrors:
    possible: np.ndarray
    impossible: np.ndarray
    frame_spans: np.ndarray

    def __post_init__(self) -> None:
        possible = np.asarray(self.possible, dtype=np.float64).reshape(-1)
        impossible = np.asarray(self.impossible, dtype=np.float64).reshape(-1)
        spans = np.asarray(self.frame_spans, dtype=np.int64)
        if spans.ndim == 1:
            spans = np.stack([spans, spans], axis=1)
        if possible.size == 0 or possible.shape != impossible.shape:
            raise ValueError("candidate errors must be equally shaped non-empty vectors")
        if not np.isfinite(possible).all() or not np.isfinite(impossible).all():
            raise ValueError("candidate errors must be finite")
        if (possible < 0).any() or (impossible < 0).any():
            raise ValueError("candidate errors must be non-negative")
        if spans.shape != (possible.size, 2):
            raise ValueError("frame_spans must have shape [T] or [T, 2]")
        if (spans[:, 1] < spans[:, 0]).any():
            raise ValueError("frame span end cannot precede its start")
        if possible.size > 1 and (spans[1:, 0] <= spans[:-1, 1]).any():
            raise ValueError("frame spans must be ordered and non-overlapping")
        object.__setattr__(self, "possible", possible.copy())
        object.__setattr__(self, "impossible", impossible.copy())
        object.__setattr__(self, "frame_spans", spans.copy())


def evaluate_pair_once(
    adapter: PredictiveAdapter,
    history: np.ndarray,
    possible_future: np.ndarray,
    impossible_future: np.ndarray,
) -> CandidateErrors:
    """Make exactly one prediction, then score both eval-only candidates against it."""
    prediction = adapter.predict(history)
    possible = adapter.error_to_candidate(prediction, possible_future)
    impossible = adapter.error_to_candidate(prediction, impossible_future)
    return CandidateErrors(possible, impossible, np.asarray(adapter.frame_spans(prediction)))

