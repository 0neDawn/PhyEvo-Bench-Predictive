"""Whole native representation extension; no attribute disentanglement claim."""
import torch
from predictive_tracks.ppr_20261028.projection import project

def reference(history):
    if history.ndim != 4 or history.shape[0] != 1 or history.shape[1] < 2:
        raise ValueError('Expected B=1, T>=2, native units, channels')
    if not torch.isfinite(history).all():
        raise ValueError('Nonfinite history')
    h = history[0].detach()
    center = h.mean(0)
    radius = torch.linalg.vector_norm(h-center, dim=-1).amax(0)
    valid = torch.ones_like(radius, dtype=torch.bool)
    return center, radius, valid

def step(value, state):
    corrected, outside = project(value, *state)
    if not torch.isfinite(corrected).all():
        raise RuntimeError('Nonfinite projection')
    norm = torch.linalg.vector_norm(corrected-value, dim=-1)
    audit = {'units': outside.numel(), 'outside_units': int(outside.sum()),
             'change_l2': float(torch.linalg.vector_norm(corrected-value)),
             'max_unit_change_l2': float(norm.max()), 'mean_unit_change_l2': float(norm.mean())}
    return corrected, audit

def trajectory(prediction, history):
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[2:] != history.shape[2:]:
        raise ValueError('Native tensor shape mismatch')
    state = reference(history)
    values, audits = [], []
    for t in range(prediction.shape[1]):
        y, audit = step(prediction[:, t], state)
        values.append(y); audits.append(audit)
    return torch.stack(values, dim=1), {'history_steps': history.shape[1],
        'radii_min_mean_max': [float(state[1].min()), float(state[1].mean()), float(state[1].max())],
        'steps': audits}

def pooled_trajectory(prediction, history):
    """Least Frobenius change constraining only the global spatial mean."""
    pooled_p = prediction.mean(dim=2, keepdim=True)
    pooled_h = history.mean(dim=2, keepdim=True)
    fixed, audit = trajectory(pooled_p, pooled_h)
    out = prediction + (fixed-pooled_p)
    if not torch.isfinite(out).all():
        raise RuntimeError('Nonfinite pooled projection')
    audit['interface'] = 'spatial_mean_same_offset_all_tokens'
    audit['spatial_units'] = prediction.shape[2]
    audit['field_change_l2'] = float(torch.linalg.vector_norm(out-prediction))
    # Quantify floating-point error in preserved relative features.
    audit['spatial_contrast_max_abs_roundoff'] = float(((out-out[:,:,:1])-(prediction-prediction[:,:,:1])).abs().max())
    return out, audit
