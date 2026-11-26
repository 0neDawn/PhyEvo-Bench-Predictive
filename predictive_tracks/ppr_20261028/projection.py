"""Tensor-only projection; no learned module, targets, or fitted readout."""
import torch


def historical_ball(features, presence, mass):
    """Input [time, object, feature], [time, object], [time, object]."""
    if features.ndim != 3 or presence.shape != features.shape[:2] or mass.shape != presence.shape:
        raise ValueError("History shape mismatch")
    if not all(torch.isfinite(x).all() for x in (features, presence, mass)):
        raise ValueError("Nonfinite history")
    keep = (presence >= .5) & (mass >= 4.) & (mass >= .5 * mass.amax(0, keepdim=True))
    counts = keep.sum(0)
    center = (features * keep[..., None]).sum(0) / counts.clamp_min(1)[..., None]
    distances = torch.linalg.vector_norm(features - center, dim=-1)
    radius = torch.where(keep, distances, torch.zeros_like(distances)).amax(0)
    return center, radius, counts >= 2, counts


def project(value, center, radius, valid):
    """Euclidean nearest point in each historical ball; invalid objects no-op."""
    delta = value - center
    distance = torch.linalg.vector_norm(delta, dim=-1)
    outside = valid & (distance > radius)
    safe_distance = torch.where(outside, distance, torch.ones_like(distance))
    ratio = torch.where(outside, radius / safe_distance, torch.ones_like(distance))
    corrected = torch.where(outside[..., None], center + ratio[..., None] * delta, value)
    return corrected, outside
