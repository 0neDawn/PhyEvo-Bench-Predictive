"""Project only historical channel means; never invoke an additional model."""

import time
import numpy as np
import torch
import torch.nn.functional as F


def project_means(prediction, history, *, bounded=False):
    """Inputs [B,T,N,C]; projection onto history-mean envelope, with optional RGB box.

    For unbounded latents this is the exact Euclidean projection. For RGB,
    bisection solves the box-constrained projection to floating-point precision.
    No candidate features, labels, metadata, or model objects are accepted.
    """
    if (prediction.ndim != 4 or history.ndim != 4 or
        prediction.shape[0] != history.shape[0] or
        prediction.shape[2:] != history.shape[2:] or history.shape[1] < 2):
        raise ValueError("Expected matching [B,T,N,C] history and prediction")
    if not torch.isfinite(prediction).all() or not torch.isfinite(history).all():
        raise ValueError("Non-finite features")
    original_dtype = prediction.dtype
    values, observed = prediction.float(), history.float()
    historical_mean = observed.mean(dim=2, keepdim=True)
    lower = historical_mean.amin(dim=1, keepdim=True)
    upper = historical_mean.amax(dim=1, keepdim=True)
    mean = values.mean(dim=2, keepdim=True)
    target = mean.maximum(lower).minimum(upper)
    changed = (mean < lower) | (mean > upper)
    shift = target - mean
    projected = values + shift
    if bounded:
        if observed.min() < 0 or observed.max() > 1 or values.min() < 0 or values.max() > 1:
            raise ValueError("RGB projection requires native inputs within [0,1]")
        # KKT: x_i^*=clip(x_i+lambda,0,1), with mean equal to target.
        lo = torch.full_like(target, -1.0)
        hi = torch.full_like(target, 1.0)
        for _ in range(32):
            middle = (lo + hi) * 0.5
            current = (values + middle).clamp(0, 1).mean(dim=2, keepdim=True)
            lo = torch.where(current < target, middle, lo)
            hi = torch.where(current >= target, middle, hi)
        constrained = (values + (lo + hi) * 0.5).clamp(0, 1)
        projected = torch.where(changed, constrained, values)
    audit = {
        "outside_history_mean_fraction": float(changed.float().mean().item()),
        "mean_shift_rms": float(shift.square().mean().sqrt().item()),
        "prediction_change_rms": float((projected-values).square().mean().sqrt().item()),
        "history_mean_range_rms": float((upper-lower).square().mean().sqrt().item()),
        "bounded_rgb": bounded,
    }
    return projected.to(original_dtype), audit


def rgb_project(prediction, history):
    shape = prediction.shape
    def tokens(value):
        return value.permute(0, 1, 3, 4, 2).reshape(value.shape[0], value.shape[1], -1, value.shape[2])
    result, audit = project_means(tokens(prediction), tokens(history), bounded=True)
    return result.reshape(shape[0], shape[1], shape[3], shape[4], shape[2]).permute(0, 1, 4, 2, 3), audit


class MomentAuditMixin:
    def __init__(self, *args, orientation="identity", **kwargs):
        if orientation != "identity":
            raise ValueError("Moment projection is a separate, uncombined track")
        self.last_track_audit = None
        self._projection_audit = None
        self._projection_seconds = 0.0
        super().__init__(*args, **kwargs)

    def _correct(self, prediction, history, *, rgb=False):
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result, self._projection_audit = (rgb_project(prediction, history) if rgb
                                               else project_means(prediction, history))
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        self._projection_seconds = time.perf_counter() - started
        return result

    def score_pair(self, possible_rgb, impossible_rgb):
        if possible_rgb.shape != impossible_rgb.shape or possible_rgb.shape[0] != 14:
            raise ValueError("Expected unchanged eight-history/six-future pair")
        if not np.array_equal(possible_rgb[:8], impossible_rgb[:8]):
            raise ValueError("History differs")
        self._projection_audit = None
        previous = self.prediction_calls
        result = super().score_pair(possible_rgb, impossible_rgb)
        if self.prediction_calls - previous != 1 or self._projection_audit is None:
            raise RuntimeError("Missing correction or changed forecast count")
        self.last_track_audit = {
            "mode": "history_mean_envelope", "flip": False,
            "additional_neural_forwards": 0, "shared_forecasts": 1,
            "preprocessing_seconds": self._projection_seconds,
            "correction": self._projection_audit,
        }
        return result


class OCVPMomentMixin(MomentAuditMixin):
    def _predict_decoded_rgb(self, history):
        return self._correct(super()._predict_decoded_rgb(history), history, rgb=True)


class SlotMomentMixin(MomentAuditMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.dynamics.rollout
        def rollout(history, *args, **kwargs):
            return self._correct(original(history, *args, **kwargs), history)
        self.dynamics.rollout = rollout


class DDLPMomentMixin(MomentAuditMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.model.sample
        def sample(value, *args, **kwargs):
            result = original(value, *args, **kwargs)
            steps = kwargs["cond_steps"]
            output = result.clone()
            output[:, steps:] = self._correct(result[:, steps:], value[:, :steps], rgb=True)
            return output
        self.model.sample = sample


class GSWMMomentMixin(MomentAuditMixin):
    def _predict(self, history):
        from gswm_runtime.runtime import _as_tchw, HISTORY_STEPS
        from sp_gswm.adapters.gswm_foreground_predictive_mean import ForegroundPrediction
        native = super()._predict(history)
        prediction = torch.from_numpy(native.rgb).unsqueeze(0)
        observed = _as_tchw(history, HISTORY_STEPS).unsqueeze(0)
        corrected = self._correct(prediction, observed, rgb=True)[0].numpy()
        # Original input hash, all eight posterior draws, and alpha weights unchanged.
        return ForegroundPrediction(rgb=corrected, alpha=native.alpha)


class VJEPAMomentMixin(MomentAuditMixin):
    def _predict(self, model_input):
        captured = []
        handle = self.encoder.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            prediction = super()._predict(model_input)
        finally:
            handle.remove()
        if len(captured) != 1 or not torch.is_tensor(captured[0]):
            raise RuntimeError("Expected one existing context-encoder result")
        context = captured[0]
        # Match the target-side parameter-free normalization, not an added encoder.
        context = F.layer_norm(context, (context.shape[-1],)).reshape(1, 4, 256, -1)
        return self._correct(prediction.reshape(1, 3, 256, -1), context).reshape_as(prediction)


class VideoMAEMomentMixin(MomentAuditMixin):
    def _prediction(self, history_container):
        captured = []
        handle = self.model.decoder.norm.register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            prediction = super()._prediction(history_container)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one existing history-only decoder result")
        tokens = prediction.shape[2]
        context = captured[0][:, :4*tokens].reshape(1, 4, tokens, -1)
        return self._correct(prediction, context)


class DINOMomentMixin(MomentAuditMixin):
    """Diagnostic only: preserves the unresolved original temporal alignment."""
    def _forecast(self, history):
        captured = []
        original = self._encode_frames
        def encode(frames):
            output = original(frames)
            captured.append(output)
            return output
        self._encode_frames = encode
        try:
            prediction = super()._forecast(history)
        finally:
            self._encode_frames = original
        if len(captured) != 1:
            raise RuntimeError("Expected only the native history encoding")
        context = captured[0]
        shape = prediction.shape
        output = self._correct(prediction.reshape(1, shape[0], -1, shape[-1]),
                               context.reshape(1, context.shape[0], -1, context.shape[-1]))
        return output.reshape_as(prediction)
