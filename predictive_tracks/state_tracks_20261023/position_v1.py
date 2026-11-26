"""History-only position tubes and a conservative projected-overlap gate.

No candidate tensors, labels, category names, contact times or metadata accepted.
All state coordinates are native 2D particle coordinates, NOT certified 3D state.
"""
from __future__ import annotations

import numpy as np


def design(times):
    times = np.asarray(times, dtype=np.float64)
    return np.stack((np.ones_like(times), times, .5 * times ** 2), axis=-1)


def project_disks(predicted, centers, radii, enabled):
    offset = predicted - centers
    norm = np.linalg.norm(offset, axis=-1)
    ratio = np.minimum(1., radii / np.maximum(norm, np.finfo(np.float64).tiny))
    projected = centers + offset * ratio[..., None]
    return np.where(enabled[..., None], projected, predicted)


def position_proposal(history_position, history_on, history_mass,
                      native_position, native_scale_logits, image_size=128):
    """Returns future positions, ablation, diagnostics. Requires 8+6 native states."""
    h = np.asarray(history_position, dtype=np.float64)
    on = np.asarray(history_on, dtype=np.float64).reshape(h.shape[:2])
    mass = np.asarray(history_mass, dtype=np.float64).reshape(h.shape[:2])
    p = np.asarray(native_position, dtype=np.float64)
    scale_logits = np.asarray(native_scale_logits, dtype=np.float64)
    if h.shape[0] != 8 or p.shape != (14, h.shape[1], 2) or scale_logits.shape != p.shape:
        raise ValueError("Original eight-history/six-future particle shapes required")
    if not all(np.isfinite(x).all() for x in (h, on, mass, p, scale_logits)):
        raise ValueError("Non-finite native state")
    particles = h.shape[1]
    active_observation = (on >= .5) & (mass >= 4.)
    active = active_observation.any(axis=0)
    reliable = active_observation & (mass >= .5 * mass.max(axis=0, keepdims=True))
    eligible = reliable.sum(axis=0) >= 4
    times = np.arange(-7, 1, dtype=float)
    future_design = design(np.arange(0, 7))
    reference = p[7:].copy()
    radii = np.zeros((7, particles), dtype=float)
    residual_scale = np.zeros(particles)
    history_rmse = np.zeros(particles)
    acceleration = np.zeros((particles, 2))
    for j in np.flatnonzero(eligible):
        x = design(times[reliable[:, j]])
        y = h[reliable[:, j], j]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = np.linalg.norm(y - x @ beta, axis=1)
        scale = max(2. / image_size, float(residual.max()))
        gram_inv = np.linalg.inv(x.T @ x)
        leverage = np.einsum("hi,ij,hj->h", future_design, gram_inv, future_design)
        reference[:, j] = future_design @ beta
        radii[:, j] = scale * np.sqrt(1. + np.maximum(leverage, 0.))
        residual_scale[j] = scale
        history_rmse[j] = np.sqrt(np.mean(residual ** 2))
        acceleration[j] = beta[2]
    scale = 1. / (1. + np.exp(-scale_logits[7:]))
    native = p[7:]
    # Union native/reference boxes, expanded by estimated position uncertainty.
    lower = np.minimum(native, reference) - scale - radii[..., None]
    upper = np.maximum(native, reference) + scale + radii[..., None]
    in_frame = (np.abs(reference) + scale + radii[..., None] <= 1.).all(axis=-1)
    safe = np.zeros((6, particles), dtype=bool)
    release = np.full(particles, -1, dtype=int)
    unsupported_scene = bool(np.any(active & ~eligible))
    for j in np.flatnonzero(eligible & active):
        released = unsupported_scene
        for step in range(1, 7):
            if not in_frame[step - 1:step + 1, j].all():
                released = True
            lo = lower[step - 1:step + 1].min(axis=0)
            hi = upper[step - 1:step + 1].max(axis=0)
            for other in np.flatnonzero(active):
                if other == j:
                    continue
                separated = np.any(hi[j] < lo[other]) or np.any(hi[other] < lo[j])
                if not separated:
                    released = True
            if released and release[j] < 0:
                release[j] = step
            safe[step - 1, j] = not released
    ungated_mask = np.broadcast_to(eligible & active, (6, particles)) & in_frame[1:]
    primary = project_disks(p[8:], reference[1:], radii[1:], safe)
    ungated = project_disks(p[8:], reference[1:], radii[1:], ungated_mask)
    delta = primary - p[8:]
    audit = {
        "active_particles": int(active.sum()), "eligible_particles": int((eligible & active).sum()),
        "reliable_history_observations": reliable.sum(axis=0).tolist(),
        "unsupported_active_particle_veto": unsupported_scene,
        "eligible_future_particle_frames": int(ungated_mask.sum()),
        "free_future_particle_frames": int(safe.sum()),
        "changed_particle_frames": int((np.linalg.norm(delta, axis=-1) > 1e-8).sum()),
        "changed_particles": int((np.linalg.norm(delta, axis=-1).max(axis=0) > 1e-8).sum()),
        "maximum_position_shift_native_pixels": float(np.linalg.norm(delta, axis=-1).max() * image_size / 2),
        "position_shift_norm": float(np.linalg.norm(delta)),
        "release_step_by_particle": release.tolist(),
        "history_residual_scale_native_pixels": (residual_scale * image_size / 2).tolist(),
        "history_fit_rmse_native_pixels": (history_rmse * image_size / 2).tolist(),
        "acceleration_native_pixels_per_step2": (acceleration * image_size / 2).tolist(),
        "future_radius_native_pixels": (radii[1:] * image_size / 2).tolist(),
        "gate_is_2d_particle_proxy_not_verified_contact": True,
    }
    return {"primary": primary, "ungated": ungated, "reference": reference[1:],
            "radii": radii[1:], "active": active, "eligible": eligible,
            "safe": safe, "delta": delta, "audit": audit}


def rotate_displacement(delta, seed):
    """Same per-particle/per-step norm; one random rotation per particle over time."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, delta.shape[1])
    cosine, sine = np.cos(angle), np.sin(angle)
    first = cosine * delta[..., 0] - sine * delta[..., 1]
    second = sine * delta[..., 0] + cosine * delta[..., 1]
    return np.stack((first, second), axis=-1)
