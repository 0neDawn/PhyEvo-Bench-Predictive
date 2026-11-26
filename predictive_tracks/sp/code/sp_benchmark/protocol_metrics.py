"""Metrics for the fixed-history paired-future predictive benchmark.

This module is intentionally independent from every model repository.  Model
adapters only need to return two non-negative arrays with shape ``[N, T]``:
the per-step error to the plausible and violating candidate futures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PairwiseResult:
    mean_possible: np.ndarray
    mean_impossible: np.ndarray
    margin: np.ndarray
    correct: np.ndarray
    tied: np.ndarray
    score_possible: np.ndarray
    score_impossible: np.ndarray


def _errors(possible: np.ndarray, impossible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    possible = np.asarray(possible, dtype=np.float64)
    impossible = np.asarray(impossible, dtype=np.float64)
    if possible.ndim != 2 or impossible.ndim != 2:
        raise ValueError("candidate errors must both have shape [N, T]")
    if possible.shape != impossible.shape or possible.shape[1] == 0:
        raise ValueError("candidate errors must have the same non-empty shape")
    if not np.isfinite(possible).all() or not np.isfinite(impossible).all():
        raise ValueError("candidate errors must be finite")
    if (possible < 0).any() or (impossible < 0).any():
        raise ValueError("candidate errors must be non-negative")
    return possible, impossible


def pairwise_results(
    possible: np.ndarray,
    impossible: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> PairwiseResult:
    """Compute the paper-defined margin, strict accuracy decision, and AUC scores."""
    possible, impossible = _errors(possible, impossible)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean_possible = possible.mean(axis=1)
    mean_impossible = impossible.mean(axis=1)
    margin = mean_impossible - mean_possible
    denom = mean_possible + mean_impossible + epsilon
    return PairwiseResult(
        mean_possible=mean_possible,
        mean_impossible=mean_impossible,
        margin=margin,
        correct=margin > 0.0,
        tied=margin == 0.0,
        score_possible=mean_possible / denom,
        score_impossible=mean_impossible / denom,
    )


def pooled_auc(score_impossible: np.ndarray, score_possible: np.ndarray) -> float:
    """ROC AUC by all positive-negative comparisons, awarding 0.5 to ties."""
    positive = np.asarray(score_impossible, dtype=np.float64).reshape(-1)
    negative = np.asarray(score_possible, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("AUC requires at least one score from each class")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("AUC scores must be finite")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def relative_error(
    possible: np.ndarray,
    impossible: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return R(t) = log((E_impossible(t)+eps)/(E_possible(t)+eps))."""
    possible, impossible = _errors(possible, impossible)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return np.log((impossible + epsilon) / (possible + epsilon))


def _normalize_spans(frame_spans: Sequence[int] | Sequence[Sequence[int]], steps: int) -> np.ndarray:
    spans = np.asarray(frame_spans, dtype=np.int64)
    if spans.ndim == 1:
        spans = np.stack([spans, spans], axis=1)
    if spans.shape != (steps, 2):
        raise ValueError("frame_spans must have shape [T] or [T, 2]")
    if (spans[:, 1] < spans[:, 0]).any():
        raise ValueError("each frame span must have start <= end")
    if steps > 1 and (spans[1:, 0] <= spans[:-1, 1]).any():
        raise ValueError("frame spans must be ordered and non-overlapping")
    return spans


def localization_time(
    signal: Sequence[float],
    frame_spans: Sequence[int] | Sequence[Sequence[int]],
    *,
    threshold: float = 0.0,
    persistence_frames: int = 2,
) -> Optional[int]:
    """Find the earliest original frame starting a persistent positive run.

    A temporal token may cover more than one original frame.  For example,
    ``[[20, 21], [22, 23], [24, 25]]`` correctly treats one positive tubelet
    token as two consecutive original video frames.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("signal must be a non-empty finite vector")
    if persistence_frames < 1:
        raise ValueError("persistence_frames must be at least one")
    spans = _normalize_spans(frame_spans, signal.size)
    run_start: Optional[int] = None
    run_length = 0
    previous: Optional[int] = None
    for is_positive, (start, end) in zip(signal > threshold, spans):
        for frame in range(int(start), int(end) + 1):
            if is_positive:
                if previous is None or frame != previous + 1 or run_start is None:
                    run_start = frame
                    run_length = 1
                else:
                    run_length += 1
                if run_length >= persistence_frames:
                    return run_start
                previous = frame
            else:
                run_start = None
                run_length = 0
                previous = frame
    return None


def summarize(
    possible: np.ndarray,
    impossible: np.ndarray,
    *,
    frame_spans: Sequence[int] | Sequence[Sequence[int]],
    violation_frames: Iterable[int],
    epsilon: float = 1e-12,
    threshold: float = 0.0,
    persistence_frames: int = 2,
    joint_tolerances: Sequence[int] = (1, 2, 3),
) -> dict:
    """Compute discrimination, localization coverage/MAE, and JointHit metrics."""
    possible, impossible = _errors(possible, impossible)
    violation = np.asarray(list(violation_frames), dtype=np.int64)
    if violation.shape != (possible.shape[0],):
        raise ValueError("violation_frames must contain one original frame per pair")
    pair = pairwise_results(possible, impossible, epsilon=epsilon)
    signal = relative_error(possible, impossible, epsilon=epsilon)
    located = [
        localization_time(
            row,
            frame_spans,
            threshold=threshold,
            persistence_frames=persistence_frames,
        )
        for row in signal
    ]
    has_location = np.asarray([item is not None for item in located], dtype=bool)
    delta = np.full(possible.shape[0], np.nan, dtype=np.float64)
    for index, item in enumerate(located):
        if item is not None:
            delta[index] = int(item) - int(violation[index])
    localization_set = pair.correct & has_location
    conditional_mae = (
        float(np.abs(delta[localization_set]).mean()) if localization_set.any() else None
    )
    joint = {}
    for tolerance in joint_tolerances:
        if tolerance < 0:
            raise ValueError("joint tolerances must be non-negative")
        hits = localization_set & (np.abs(delta) <= tolerance)
        joint[str(tolerance)] = float(hits.mean())
    return {
        "num_pairs": int(possible.shape[0]),
        "paired_accuracy": float(pair.correct.mean()),
        "tie_rate": float(pair.tied.mean()),
        "auc": pooled_auc(pair.score_impossible, pair.score_possible),
        "detection_coverage": float(has_location.mean()),
        "localized_and_correct_fraction": float(localization_set.mean()),
        "localization_mae": conditional_mae,
        "localization_mae_denominator": int(localization_set.sum()),
        "joint_hit": joint,
        "margin": pair.margin.tolist(),
        "relative_error": signal.tolist(),
        "t_star": located,
        "delta_t": [None if np.isnan(value) else float(value) for value in delta],
    }
