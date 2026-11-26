"""History-rate trust region; consumes original history and forecast only."""

import time
import torch


def project_continuity(prediction, history, *, persistence=False):
    if (prediction.ndim < 3 or history.ndim != prediction.ndim or
        prediction.shape[0] != history.shape[0] or
        prediction.shape[2:] != history.shape[2:] or history.shape[1] < 2):
        raise ValueError("Expected matching [B,T,...] state sequences")
    if not torch.isfinite(prediction).all() or not torch.isfinite(history).all():
        raise ValueError("Non-finite state")
    original_dtype = prediction.dtype
    value, observed = prediction.float(), history.float()
    reduce = tuple(range(2, value.ndim))
    speed = (observed[:,1:]-observed[:,:-1]).square().mean(reduce, keepdim=True).sqrt()
    max_speed = speed.amax(1, keepdim=True)
    step_shape = (1, value.shape[1]) + (1,) * (value.ndim-2)
    steps = torch.arange(1, value.shape[1]+1, device=value.device).reshape(step_shape)
    radius = steps * max_speed
    anchor = observed[:, -1:]
    delta = value-anchor
    distance = delta.square().mean(reduce, keepdim=True).sqrt()
    alpha = torch.where(distance > radius, radius / distance.clamp_min(1e-30), torch.ones_like(distance))
    if persistence:
        alpha = torch.zeros_like(alpha)
    result = anchor + alpha * delta
    # Exact passthrough when there is no intervention.
    result = torch.where(alpha == 1, value, result)
    return result.to(original_dtype), {
        "history_step_rms": speed.flatten().tolist(),
        "max_history_step_rms": max_speed.flatten().tolist(),
        "forecast_distance_rms": distance.flatten().tolist(),
        "allowed_radius_rms": radius.flatten().tolist(),
        "retained_forecast_fraction": alpha.flatten().tolist(),
        "projection_active_fraction": float((alpha < 1).float().mean()),
        "prediction_change_rms": float((result-value).square().mean().sqrt()),
        "persistence_ablation": persistence,
    }


class ContinuityCorrectionMixin:
    """Reuse the audited native-output interception, not mean-envelope correction."""
    persistence = False

    def _correct(self, prediction, history, *, rgb=False):
        del rgb  # This convex interpolation is also RGB-box preserving.
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result, self._projection_audit = project_continuity(
                prediction, history, persistence=self.persistence)
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        self._projection_seconds = time.perf_counter()-started
        return result

    def score_pair(self, possible_rgb, impossible_rgb):
        result = super().score_pair(possible_rgb, impossible_rgb)
        self.last_track_audit["mode"] = ("history_persistence" if self.persistence else "history_rate_ball")
        return result
