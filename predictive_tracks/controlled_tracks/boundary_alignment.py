"""Align the forecast boundary, keeping its predicted increments unchanged."""

import time
import torch


def align_boundary(prediction, history, *, rgb=False):
    if (prediction.ndim < 3 or prediction.shape[1] < 2 or
        history.ndim != prediction.ndim or prediction.shape[0] != history.shape[0]
        or prediction.shape[2:] != history.shape[2:]):
        raise ValueError("Two existing forecast steps and matching observed states required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(history).all():
        raise ValueError("Non-finite state")
    value, observed = prediction.float(), history.float()
    # This is extrapolation from the native future trajectory, NOT a model backtest.
    implied_boundary = 2*value[:, :1]-value[:, 1:2]
    residual = observed[:, -1:]-implied_boundary
    unbounded = value+residual
    result = unbounded.clamp(0,1) if rgb else unbounded
    dims = tuple(range(2,value.ndim))
    rms = lambda x: x.square().mean(dims).sqrt().flatten().tolist()
    return result.to(prediction.dtype), {
        "extrapolated_boundary_residual_rms": rms(residual),
        "native_forecast_increment_rms": rms(value[:,1:]-value[:,:-1]),
        "history_increment_rms": rms(observed[:,1:]-observed[:,:-1]),
        "aligned_increment_change_rms": rms((result[:,1:]-result[:,:-1])-(value[:,1:]-value[:,:-1])),
        "prediction_change_rms": float((result-value).square().mean().sqrt()),
        "rgb_clipped_fraction": float(((unbounded<0)|(unbounded>1)).float().mean()) if rgb else 0.,
        "not_a_historical_forecast_residual": True,
    }


class BoundaryAlignmentMixin:
    def _correct(self, prediction, history, *, rgb=False):
        if prediction.device.type == "cuda": torch.cuda.synchronize(prediction.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result, self._projection_audit = align_boundary(prediction,history,rgb=rgb)
        if prediction.device.type == "cuda": torch.cuda.synchronize(prediction.device)
        self._projection_seconds = time.perf_counter()-started
        return result

    def score_pair(self, possible_rgb, impossible_rgb):
        result = super().score_pair(possible_rgb,impossible_rgb)
        self.last_track_audit["mode"] = "boundary_alignment"
        return result
