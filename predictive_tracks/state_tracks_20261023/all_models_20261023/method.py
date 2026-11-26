"""Native-vector continuation; history/prediction only, no model/target arguments."""
import numpy as np
import torch

from ..trend_state_v4 import solve_trend, solve_l2_matched


def coordinates(prediction, history):
    if prediction.shape[0] != 1 or prediction.ndim != history.ndim or prediction.shape[2:] != history.shape[2:]:
        raise ValueError("Matching batch-one native tensors required")
    if history.shape[1] < 4 or prediction.shape[1] < 3:
        raise ValueError("At least4 native history and3 forecast steps required")
    p = prediction.detach().cpu().double().reshape(prediction.shape[1], -1)
    h = history.detach().cpu().double().reshape(history.shape[1], -1)
    if not torch.isfinite(p).all() or not torch.isfinite(h).all():
        raise ValueError("Nonfinite original state")
    anchor = h[-1:].clone()
    rows = torch.cat((h-anchor, p-anchor))
    gram = (rows @ rows.T / rows.shape[1]).numpy()
    values, vectors = np.linalg.eigh(gram)
    tolerance = np.finfo(float).eps * len(gram) * max(float(values[-1]), 0.)
    keep = values > tolerance
    small = vectors[:, keep] * np.sqrt(values[keep])
    inverse = (vectors[:, keep] / np.sqrt(values[keep])).T
    return p, h, rows, small, inverse, {
        "dimension": int(rows.shape[1]), "rank": int(keep.sum()),
        "rank_tolerance": float(tolerance),
        "discarded_gram_trace": float(np.maximum(values[~keep], 0.).sum()),
        "history_steps": int(len(h)), "prediction_steps": int(len(p)),
        "native_representation_not_physical_position": True,
    }


def propose(prediction, history, *, rgb=False):
    p, h, rows, small, inverse, audit = coordinates(prediction, history)
    length, horizon = len(h), len(p)
    observed, native = small[:length], small[length:]
    ts = np.arange(1-length, 1, dtype=float)
    design = np.stack((np.ones(length), ts, .5*ts**2), axis=1)
    future = np.arange(1, horizon+1, dtype=float)
    xf = np.stack((np.ones(horizon), future, .5*future**2), axis=1)
    beta = np.linalg.lstsq(design, observed, rcond=None)[0]
    residual = np.linalg.norm(observed-design@beta, axis=1)
    floor = np.sqrt(np.finfo(np.float32).eps)*float(np.linalg.norm(observed, axis=1).max())
    sigma = max(float(residual.max()), floor)
    leverage = np.einsum("hi,ij,hj->h", xf, np.linalg.inv(design.T@design), xf)
    radius = sigma*np.sqrt(1+leverage)
    reference = xf@beta
    distance = np.linalg.norm(native-reference, axis=1)
    ratio = np.minimum(1., radius/np.maximum(distance, np.finfo(float).tiny))
    tube = reference+(native-reference)*ratio[:, None]
    matrix = np.eye(horizon)
    for i in range(1, horizon): matrix[i, i-1] = -2
    for i in range(2, horizon): matrix[i, i-2] = 1
    target = np.broadcast_to(beta[2], native.shape).copy()
    target[0] = beta[1]+.5*beta[2]
    trend, solver = solve_trend(native, matrix, target, sigma)
    l2, l2_audit = solve_l2_matched(native, matrix, target, float(np.linalg.norm(trend-native)))
    outputs = {}
    for name, value in (("state_tube", tube), ("state_trend", trend), ("trend_l2_matched", l2)):
        weights = torch.from_numpy((value-native)@inverse)
        delta = weights @ rows
        raw = p+delta
        result = raw.clamp(0, 1) if rgb else raw
        outputs[name] = result.reshape_as(prediction).to(dtype=prediction.dtype)
        audit[name] = {
            "pre_box_shift_rms": float(delta.square().mean().sqrt()),
            "post_box_shift_rms": float((result-p).square().mean().sqrt()),
            "rgb_clipped_fraction": float(((raw<0)|(raw>1)).double().mean()) if rgb else 0.,
        }
    audit.update(sigma=sigma, numerical_floor=floor, history_residuals=residual.tolist(),
                 tube_radius=radius.tolist(), native_to_reference=distance.tolist(),
                 trend_solver=solver, l2_match=l2_audit,
                 history_second_difference_rms=float(np.linalg.norm(np.diff(observed,n=2,axis=0))/np.sqrt(max(length-2,1))))
    return outputs, audit


def match_direction(prediction, direction, norm, *, rgb):
    """Analytic global norm matching, including RGB bounds; no neural calls."""
    p, d = prediction.double(), direction.double()
    magnitude = float(torch.linalg.vector_norm(d))
    if norm == 0 or magnitude == 0:
        return prediction.clone(), {"desired_norm": norm, "actual_norm": 0., "matched": norm == 0,
                                    "zero_direction": magnitude == 0}
    d = d*(norm/magnitude)
    if not rgb:
        result = (p+d).to(prediction.dtype)
    else:
        def distance(factor):
            return float(torch.linalg.vector_norm((p+factor*d).clamp(0,1)-p))
        upper = 1.
        while upper < 1e6 and distance(upper) < norm:
            upper = min(upper*2, 1e6)
        lower = 0.
        for _ in range(40):
            middle = (lower+upper)/2
            if distance(middle) < norm: lower = middle
            else: upper = middle
        result = (p+upper*d).clamp(0,1).to(prediction.dtype)
    actual = float(torch.linalg.vector_norm(result.double()-p))
    return result, {"desired_norm": norm, "actual_norm": actual,
                    "matched": abs(actual-norm) <= 1e-5*max(norm,1e-12), "zero_direction": False}


def random_control(prediction, delta, seed, *, rgb):
    flat = delta.reshape(delta.shape[1], -1)
    rng = np.random.default_rng(seed)
    order = torch.from_numpy(rng.permutation(flat.shape[1]))
    signs = torch.from_numpy(rng.choice([-1.,1.], flat.shape[1]))
    direction = (flat[:, order]*signs).reshape_as(delta)
    return match_direction(prediction, direction, float(torch.linalg.vector_norm(delta.double())), rgb=rgb)


class CaptureOnlyMixin:
    """Identity at existing interception point; original network graph unchanged."""
    def _correct(self, prediction, history, *, rgb=False):
        self._projection_audit = {"identity_capture_only": True, "additional_neural_forwards": 0}
        self._projection_seconds = 0.
        return prediction
