"""History-scaled response protection; analytic tensors only, no model calls.

The estimated response is a representation-space velocity innovation, NOT a
contact detector or an identified physical state. See response_experiment.json.
"""

import time
import torch


def response_calibration(prediction, history, *, rgb=False, mode="response"):
    if mode not in ("response", "hard_response", "random_norm"):
        raise ValueError(mode)
    if (prediction.ndim < 3 or history.ndim != prediction.ndim
            or prediction.shape[1] < 2 or history.shape[1] < 3
            or prediction.shape[0] != history.shape[0]
            or prediction.shape[2:] != history.shape[2:]):
        raise ValueError("Matching history and native prediction tensors required")
    if not torch.isfinite(prediction).all() or not torch.isfinite(history).all():
        raise ValueError("Non-finite input")
    # Match the frozen boundary method's float32 input arithmetic, followed by
    # float64 small-matrix estimation. No candidate state is an input.
    p, h = prediction.float(), history.float()
    a = (h[:, 1:] - h[:, :-1]).flatten(2).double()
    residual = (h[:, -1:] - (2 * p[:, :1] - p[:, 1:2])).flatten(2)[:, 0].double()
    future_velocity = (p[:, 1:] - p[:, :-1]).flatten(2).double()
    corrections, details = [], []
    for batch in range(p.shape[0]):
        changes = a[batch]
        eigenvalues, eigenvectors = torch.linalg.eigh(changes @ changes.T)
        eps = torch.finfo(torch.float64).eps
        tolerance = eps * max(changes.shape) * eigenvalues.max().clamp_min(0)
        keep = eigenvalues > tolerance
        if not keep.any():
            corrections.append(torch.zeros_like(residual[batch]))
            details.append({"history_rank": 0, "response_rank": 0,
                            "history_velocity_variance": 0., "retained_correction_ratio": 0.,
                            "response_projection_before": 0., "response_projection_after": 0.})
            continue
        q = (changes.T @ eigenvectors[:, keep]) / eigenvalues[keep].sqrt()[None, :]
        c0 = q.T @ residual[batch]
        velocity = changes.mean(0)
        # Scale is fixed from historical variation; no tuned coefficient.
        sigma2 = ((changes - velocity) @ q).square().sum(1).mean()
        innovation = (future_velocity[batch] - velocity) @ q
        gram = innovation.T @ innovation / innovation.shape[0]
        lam, vectors = torch.linalg.eigh(gram)
        lam = lam.clamp_min(0)
        response_tol = eps * max(innovation.shape) * lam.max()
        active = lam > response_tol
        history_tol = eps * max(changes.shape) * changes.square().sum(1).mean()
        if mode == "hard_response" or sigma2 <= history_tol:
            gains = (~active).double()
        else:
            gains = sigma2 / (sigma2 + lam)
        c = vectors @ (gains * (vectors.T @ c0))
        # Negative control: identical pre-clipping correction norm, but a fixed
        # random direction in the SAME historical subspace. No label-dependent RNG.
        if mode == "random_norm":
            generator = torch.Generator(device="cpu").manual_seed(20261022)
            random = torch.randn(c.shape, generator=generator, dtype=torch.float64).to(c.device)
            c = random / random.norm().clamp_min(torch.finfo(torch.float64).tiny) * c.norm()
        corrections.append(q @ c)
        details.append({
            "history_rank": int(keep.sum()), "response_rank": int(active.sum()),
            "history_velocity_variance": float(sigma2),
            "response_eigenvalues": lam.tolist(), "directional_gains": gains.tolist(),
            "retained_correction_ratio": float(c.norm() / c0.norm().clamp_min(1e-30)),
            "response_projection_before": float((innovation @ c0).norm()),
            "response_projection_after": float((innovation @ c).norm()),
            "basis_orthogonality_error": float((q.T @ q - torch.eye(q.shape[1], device=q.device)).abs().max()),
        })
    correction = torch.stack(corrections).reshape_as(p[:, :1]).float()
    unconstrained = p + correction
    output = unconstrained.clamp(0, 1) if rgb else unconstrained
    return output.to(prediction.dtype), {
        "mode": mode, "interpretation": "representation-velocity innovation; not physical contact truth",
        "batch": details, "correction_rms": float(correction.square().mean().sqrt()),
        "prediction_change_rms": float((output - p).square().mean().sqrt()),
        "rgb_clipped_fraction": float(((unconstrained < 0) | (unconstrained > 1)).float().mean()) if rgb else 0.,
        "additional_neural_forwards": 0,
    }


class ResponseCalibrationMixin:
    def _correct(self, prediction, history, *, rgb=False):
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output, self._projection_audit = response_calibration(prediction, history, rgb=rgb)
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        self._projection_seconds = time.perf_counter() - started
        return output

    def score_pair(self, possible_rgb, impossible_rgb):
        result = super().score_pair(possible_rgb, impossible_rgb)
        self.last_track_audit["mode"] = "response_protected_boundary"
        return result
