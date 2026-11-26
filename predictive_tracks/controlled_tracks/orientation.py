"""History-only coordinate canonicalization, with zero neural forward calls."""

from __future__ import annotations

import time
import numpy as np


MODES = ("identity", "horizontal_flip", "motion_right", "motion_left")


def history_motion(history: np.ndarray) -> dict:
    """Estimate a global brightness-constancy flow, not an object/physics oracle.

    Fits two translation coefficients per adjacent observed frame pair. The
    statistic reverses horizontal sign under reflection. No learned optical
    flow, future images, sample identifiers, categories, or metadata are used.
    """
    if history.ndim != 4 or history.shape[0] != 8 or history.shape[-1] != 3:
        raise ValueError("Expected exactly eight THWC RGB history frames")
    if history.dtype != np.uint8:
        raise ValueError("Expected unmodified uint8 source pixels")
    frames = history.astype(np.float64) / 255.0
    velocities = []
    for before, after in zip(frames[:-1], frames[1:]):
        mid = (before + after) * 0.5
        gx = (mid[1:-1, 2:] - mid[1:-1, :-2]) * 0.5
        gy = (mid[2:, 1:-1] - mid[:-2, 1:-1]) * 0.5
        gt = (after - before)[1:-1, 1:-1]
        moving = np.linalg.norm(gt, axis=-1) > (2.0 / 255.0)
        textured = np.sum(gx * gx + gy * gy, axis=-1) > 1e-5
        use = moving & textured
        if int(use.sum()) < 16:
            continue
        design = np.stack((gx[use].ravel(), gy[use].ravel()), axis=1)
        temporal = gt[use].ravel()
        gram = design.T @ design
        ridge = 1e-6 * max(float(np.trace(gram)), 1e-12)
        velocity = np.linalg.solve(gram + ridge * np.eye(2), -design.T @ temporal)
        if np.isfinite(velocity).all():
            velocities.append(velocity.tolist())
    if not velocities:
        return {"vx": 0.0, "vy": 0.0, "agreement": 0.0, "direction": 0,
                "valid_intervals": 0, "interval_velocities": []}
    values = np.asarray(velocities)
    vx, vy = np.median(values, axis=0)
    active = np.abs(values[:, 0]) > 0.05
    agreement = (float(np.mean(np.sign(values[active, 0]) == np.sign(vx)))
                 if active.any() else 0.0)
    direction = int(np.sign(vx)) if abs(vx) > 0.05 and agreement >= 0.6 else 0
    return {"vx": float(vx), "vy": float(vy), "agreement": agreement,
            "direction": direction, "valid_intervals": len(velocities),
            "interval_velocities": velocities}


def decide(history: np.ndarray, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    result = history_motion(history)
    flip = (mode == "horizontal_flip" or
            (mode == "motion_right" and result["direction"] < 0) or
            (mode == "motion_left" and result["direction"] > 0))
    return {**result, "mode": mode, "flip": bool(flip)}


def transform(video: np.ndarray, flip: bool) -> np.ndarray:
    # Reindexing only: no interpolation, crop, resampling, colour change or warp.
    return np.ascontiguousarray(video[:, :, ::-1] if flip else video)


class OrientationMixin:
    """Call the inherited, unmodified adapter exactly once per candidate pair."""

    def __init__(self, *args, orientation="identity", **kwargs):
        if orientation not in MODES:
            raise ValueError(orientation)
        self.orientation = orientation
        self.last_track_audit = None
        super().__init__(*args, **kwargs)

    def score_pair(self, possible_rgb, impossible_rgb):
        if possible_rgb.shape != impossible_rgb.shape or possible_rgb.shape[0] != 14:
            raise ValueError("Expected paired eight-history/six-future videos")
        if not np.array_equal(possible_rgb[:8], impossible_rgb[:8]):
            raise ValueError("Histories differ")
        started = time.perf_counter()
        # Selection completes before either candidate future is inspected.
        decision = decide(possible_rgb[:8], self.orientation)
        possible = transform(possible_rgb, decision["flip"])
        impossible = transform(impossible_rgb, decision["flip"])
        preprocessing_seconds = time.perf_counter() - started
        previous = self.prediction_calls
        output = super().score_pair(possible, impossible)
        actual = self.prediction_calls - previous
        if actual != 1:
            raise RuntimeError(f"Shared forecast count changed: {actual}")
        self.last_track_audit = {**decision, "additional_neural_forwards": 0,
                                "shared_forecasts": actual,
                                "preprocessing_seconds": preprocessing_seconds}
        return output
