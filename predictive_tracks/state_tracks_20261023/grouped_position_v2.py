"""Explicit grouped image-state experiment; zero neural calls or target inputs.

Historical overlap is only an object proxy. It is deliberately not called
physical object identity. The correction preserves each group's relative layout.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .position_v1 import design, project_disks, rotate_displacement


def placed_alpha(position, scale_logits, obj_on, patches, image_size=128):
    """Original affine placement; discarded RGB never enters grouping."""
    frames, particles = position.shape[:2]
    scale = torch.sigmoid(scale_logits).reshape(-1, 2)
    pos = position.reshape(-1, 2)
    theta = torch.zeros(len(pos), 2, 3, dtype=patches.dtype, device=patches.device)
    theta[:, 0, 0] = 1 / (scale[:, 1] + 1e-9)
    theta[:, 1, 1] = 1 / (scale[:, 0] + 1e-9)
    theta[:, 0, 2] = -pos[:, 1] / (scale[:, 1] + 1e-9)
    theta[:, 1, 2] = -pos[:, 0] / (scale[:, 0] + 1e-9)
    shape = (frames * particles, 1, image_size, image_size)
    grid = F.affine_grid(theta, shape, align_corners=True)
    alpha = F.grid_sample(patches[:, :, :1].flatten(0, 1), grid,
                          mode="bilinear", align_corners=True).clamp(0, 1)
    return (alpha.reshape(frames, particles, image_size, image_size)
            * obj_on.reshape(frames, particles, 1, 1)).numpy().astype(np.float64)


def complete_link_groups(compatible, active):
    """A group is a clique, not a transitive chain of pairwise overlap."""
    groups = [[int(i)] for i in np.flatnonzero(active)]
    while True:
        pair = next(((i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))
                     if compatible[np.ix_(groups[i], groups[j])].all()), None)
        if pair is None:
            return groups
        i, j = pair
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)


def historical_groups(alpha, on):
    mass = alpha.sum(axis=(-2, -1))
    observed = (np.asarray(on).reshape(mass.shape) >= .5) & (mass >= 4.)
    reliable = observed & (mass >= .5 * mass.max(axis=0, keepdims=True))
    active = observed.any(axis=0)
    count = alpha.shape[1]
    compatible = np.eye(count, dtype=bool)
    evidence = []
    for i in range(count):
        for j in range(i + 1, count):
            joint = reliable[:, i] & reliable[:, j]
            if joint.sum() < 4:
                continue
            overlap = np.minimum(alpha[joint, i], alpha[joint, j]).sum(axis=(-2, -1))
            coefficient = overlap / np.minimum(mass[joint, i], mass[joint, j])
            compatible[i, j] = compatible[j, i] = bool((coefficient >= .5).all())
            if compatible[i, j]:
                evidence.append({"particles": [i, j], "joint_observations": int(joint.sum()),
                                 "minimum_overlap_coefficient": float(coefficient.min())})
    return complete_link_groups(compatible, active), active, evidence


def mask_geometry(masks):
    """Mass centroids and central-99%-marginal bounds, in native (y,x)."""
    size = masks.shape[-1]
    coordinates = np.linspace(-1., 1., size)
    mass = masks.sum(axis=(-2, -1))
    center = np.zeros(mass.shape + (2,))
    lower, upper = np.zeros_like(center), np.zeros_like(center)
    for axis, marginal in enumerate((masks.sum(axis=-1), masks.sum(axis=-2))):
        probability = marginal / np.maximum(mass[..., None], np.finfo(float).tiny)
        center[..., axis] = (probability * coordinates).sum(axis=-1)
        cumulative = probability.cumsum(axis=-1)
        lower[..., axis] = coordinates[np.argmax(cumulative >= .005, axis=-1)]
        upper[..., axis] = coordinates[np.argmax(cumulative >= .995, axis=-1)]
    return mass, center, lower, upper


def group_position_proposal(history_position, history_scale_logits, history_on,
                            history_patches, native_position, native_scale_logits,
                            native_on, native_patches, image_size=128):
    """Only common history and already-computed native state accepted."""
    if history_position.shape[0] != 8 or native_position.shape[0] != 14:
        raise ValueError("Original 8-history, 6-future budget required")
    if any(not torch.isfinite(v).all() for v in (history_position, history_scale_logits,
           history_on, history_patches, native_position, native_scale_logits, native_on, native_patches)):
        raise ValueError("Nonfinite native state")
    historical_alpha = placed_alpha(history_position, history_scale_logits, history_on,
                                   history_patches, image_size)
    groups, active, evidence = historical_groups(historical_alpha, history_on.numpy())
    p = native_position.numpy().astype(np.float64)
    if not groups:
        return {"primary": p[8:].copy(), "ungated": p[8:].copy(), "groups": [],
                "delta": np.zeros_like(p[8:]), "ungated_delta": np.zeros_like(p[8:]),
                "group_delta": np.zeros((6, 0, 2)), "ungated_group_delta": np.zeros((6, 0, 2)),
                "audit": {"groups": [], "group_count": 0, "active_particles": 0,
                          "eligible_groups": 0, "unsupported_active_group_veto": False,
                          "safe_group_frames": 0, "changed_group_frames": 0,
                          "ungated_changed_group_frames": 0}}
    future_alpha = placed_alpha(native_position[8:], native_scale_logits[8:], native_on[8:],
                                native_patches, image_size)
    history_masks = np.stack([historical_alpha[:, members].max(axis=1) for members in groups], axis=1)
    future_masks = np.stack([future_alpha[:, members].max(axis=1) for members in groups], axis=1)
    hm, hc, hl, hu = mask_geometry(history_masks)
    fm, fc, fl, fu = mask_geometry(future_masks)
    mass = np.concatenate((hm[-1:], fm))
    centers = np.concatenate((hc[-1:], fc))
    lower = np.concatenate((hl[-1:], fl))
    upper = np.concatenate((hu[-1:], fu))
    reliable = (hm >= 4.) & (hm >= .5 * hm.max(axis=0, keepdims=True))
    eligible = reliable.sum(axis=0) >= 4
    reference = centers.copy()
    radii = np.zeros(mass.shape)
    residual_scale = np.zeros(len(groups))
    times = np.arange(-7., 1.)
    fd = design(np.arange(7.))
    for j in np.flatnonzero(eligible):
        x = design(times[reliable[:, j]])
        y = hc[reliable[:, j], j]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = np.linalg.norm(y - x @ beta, axis=-1)
        residual_scale[j] = max(2. / image_size, float(residual.max()))
        reference[:, j] = fd @ beta
        leverage = np.einsum("hi,ij,hj->h", fd, np.linalg.inv(x.T @ x), fd)
        radii[:, j] = residual_scale[j] * np.sqrt(1 + np.maximum(leverage, 0))
    shift = reference - centers
    reference_lower = lower + shift - radii[..., None]
    reference_upper = upper + shift + radii[..., None]
    union_lower = np.minimum(lower - radii[..., None], reference_lower)
    union_upper = np.maximum(upper + radii[..., None], reference_upper)
    in_frame = (reference_lower >= -1.).all(-1) & (reference_upper <= 1.).all(-1)
    supported = eligible[None, :] & (mass >= 4.) & in_frame
    safe = np.zeros((6, len(groups)), dtype=bool)
    release = np.full(len(groups), -1, dtype=int)
    reasons = [[] for _ in groups]
    unsupported_scene = bool((~eligible).any())
    for j in range(len(groups)):
        released = unsupported_scene
        if released:
            reasons[j].append("unsupported_historical_group")
        for step in range(1, 7):
            if not supported[step - 1:step + 1, j].all():
                released = True
                if "missing_mask_or_out_of_frame" not in reasons[j]:
                    reasons[j].append("missing_mask_or_out_of_frame")
            lo = union_lower[step - 1:step + 1].min(axis=0)
            hi = union_upper[step - 1:step + 1].max(axis=0)
            for other in range(len(groups)):
                if other == j:
                    continue
                if (mass[step - 1:step + 1, other] < 4).any():
                    released = True
                    if "unsupported_other_group" not in reasons[j]:
                        reasons[j].append("unsupported_other_group")
                elif not (np.any(hi[j] < lo[other]) or np.any(hi[other] < lo[j])):
                    released = True
                    if "projected_inter_group_overlap" not in reasons[j]:
                        reasons[j].append("projected_inter_group_overlap")
            if released and release[j] < 0:
                release[j] = step
            safe[step - 1, j] = not released
    primary_centers = project_disks(fc, reference[1:], radii[1:], safe)
    ungated_centers = project_disks(fc, reference[1:], radii[1:], supported[1:])
    group_delta = primary_centers - fc
    ungated_group_delta = ungated_centers - fc
    delta = expand_groups(group_delta, groups, p.shape[1])
    ungated_delta = expand_groups(ungated_group_delta, groups, p.shape[1])
    return {"primary": p[8:] + delta, "ungated": p[8:] + ungated_delta,
            "delta": delta, "ungated_delta": ungated_delta, "groups": groups,
            "group_delta": group_delta, "ungated_group_delta": ungated_group_delta,
            "history_centers": hc, "future_centers": fc, "reference": reference[1:],
            "radii": radii[1:], "safe": safe, "active": active,
            "audit": {"groups": groups, "group_count": len(groups), "active_particles": int(active.sum()),
                      "eligible_groups": int(eligible.sum()), "grouping_edges": evidence,
                      "reliable_history_group_observations": reliable.sum(axis=0).tolist(),
                      "unsupported_active_group_veto": unsupported_scene,
                      "safe_group_frames": int(safe.sum()),
                      "changed_group_frames": int((np.linalg.norm(group_delta, axis=-1) > 1e-8).sum()),
                      "ungated_changed_group_frames": int((np.linalg.norm(ungated_group_delta, axis=-1) > 1e-8).sum()),
                      "release_steps": release.tolist(), "release_reasons": reasons,
                      "history_residual_native_pixels": (residual_scale * image_size / 2).tolist(),
                      "maximum_primary_shift_native_pixels": float(np.linalg.norm(delta, axis=-1).max() * image_size / 2),
                      "maximum_ungated_shift_native_pixels": float(np.linalg.norm(ungated_delta, axis=-1).max() * image_size / 2),
                      "group_identity_is_unverified_proxy": True}}


def expand_groups(group_delta, groups, particles):
    delta = np.zeros((group_delta.shape[0], particles, 2))
    for j, members in enumerate(groups):
        delta[:, members] = group_delta[:, j:j + 1]
    return delta


def rotated_group_delta(group_delta, groups, particles, seed):
    return expand_groups(rotate_displacement(group_delta, seed), groups, particles)


def propose_cached(data):
    """Whitelist state inputs; candidate tensors cannot reach the algorithm."""
    return group_position_proposal(data["history_z"], data["history_z_scale"], data["history_obj_on"],
                                   data["history_dec_objects_original"], data["position"], data["scale"],
                                   data["obj_on"], data["future_dec_objects"], data["future_bg"].shape[-1])
