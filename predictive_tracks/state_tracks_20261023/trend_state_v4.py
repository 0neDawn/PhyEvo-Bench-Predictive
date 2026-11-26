"""User-proposed group sparse trend prior, applied to explicit grouped 2D states."""
from __future__ import annotations

import numpy as np

from .grouped_position_v2 import propose_cached as group_state, placed_alpha, expand_groups


def difference_system(length, p0, p_minus1, acceleration):
    matrix = np.eye(length)
    for h in range(1, length):
        matrix[h, h - 1] = -2
    for h in range(2, length):
        matrix[h, h - 2] = 1
    constant = np.zeros((length, 2))
    constant[0] = -2 * p0 + p_minus1
    if length > 1:
        constant[1] = p0
    return matrix, np.broadcast_to(acceleration, (length, 2)) - constant


def solve_trend(native, matrix, target, penalty, coordinatewise=False):
    """No learned inference: convex optimization on a 6-by-2 state array."""
    native = np.asarray(native, dtype=np.float64)
    x = native.copy()
    innovation = matrix @ x - target
    auxiliary = innovation.copy()
    dual = np.zeros_like(innovation)
    inverse = np.linalg.inv(np.eye(len(native)) + matrix.T @ matrix)
    converged = False
    for iteration in range(1, 501):
        x = inverse @ (native + matrix.T @ (target + auxiliary - dual))
        innovation = matrix @ x - target
        previous = auxiliary.copy()
        value = innovation + dual
        if coordinatewise:
            auxiliary = np.sign(value) * np.maximum(np.abs(value) - penalty, 0.)
        else:
            norm = np.linalg.norm(value, axis=-1, keepdims=True)
            auxiliary = value * np.maximum(1 - penalty / np.maximum(norm, 1e-300), 0.)
        dual += innovation - auxiliary
        primal = float(np.linalg.norm(innovation - auxiliary))
        dual_residual = float(np.linalg.norm(matrix.T @ (auxiliary - previous)))
        if max(primal, dual_residual) <= 1e-8 * (1 + np.linalg.norm(x)):
            converged = True
            break
    return x, {"iterations": iteration, "converged": converged,
               "primal_residual": primal, "dual_residual": dual_residual}


def solve_l2_matched(native, matrix, target, desired_norm):
    def solve(coefficient):
        return np.linalg.solve(np.eye(len(native)) + coefficient * matrix.T @ matrix,
                               native + coefficient * matrix.T @ target)
    if desired_norm < 1e-12:
        return native.copy(), {"coefficient": 0., "norm_error": 0., "matched": True}
    upper = 1.
    while upper < 1e6 and np.linalg.norm(solve(upper) - native) < desired_norm:
        upper = min(upper * 10, 1e6)
    lower = 0.
    for _ in range(60):
        middle = (lower + upper) / 2
        if np.linalg.norm(solve(middle) - native) < desired_norm:
            lower = middle
        else:
            upper = middle
    result = solve(upper)
    error = abs(float(np.linalg.norm(result - native)) - desired_norm)
    return result, {"coefficient": upper, "norm_error": error, "matched": error <= 1e-8}


def propose_cached(data):
    state = group_state(data)
    groups = state["groups"]
    native_position = data["position"][8:].numpy().astype(np.float64)
    kinds = ("primary", "l2_matched", "coordinate_l1", "no_velocity_boundary")
    group_deltas = {name: np.zeros((6, len(groups), 2)) for name in kinds}
    group_audits = []
    if groups:
        alpha = placed_alpha(data["position"][8:], data["scale"][8:], data["obj_on"][8:],
                             data["future_dec_objects"], data["future_bg"].shape[-1])
        for j, members in enumerate(groups):
            mass = alpha[:, members].max(axis=1).sum(axis=(-2, -1))
            history_count = state["audit"]["reliable_history_group_observations"][j]
            eligible = history_count >= 4 and bool((mass >= 4).all())
            entry = {"members": members, "eligible": eligible, "reliable_history_count": history_count}
            if eligible:
                ref = state["reference"][:, j]
                acceleration = ref[2] - 2 * ref[1] + ref[0]
                p0 = 3 * ref[0] - 3 * ref[1] + ref[2]
                p_minus1 = 6 * ref[0] - 8 * ref[1] + 3 * ref[2]
                native = state["future_centers"][:, j]
                matrix, target = difference_system(6, p0, p_minus1, acceleration)
                penalty = state["audit"]["history_residual_native_pixels"][j] * 2 / data["future_bg"].shape[-1]
                primary, solver = solve_trend(native, matrix, target, penalty)
                l2, l2_audit = solve_l2_matched(native, matrix, target, float(np.linalg.norm(primary - native)))
                coordinate, coordinate_solver = solve_trend(native, matrix, target, penalty, coordinatewise=True)
                no_velocity, no_velocity_solver = solve_trend(native, matrix[1:], target[1:], penalty)
                for name, corrected in zip(kinds, (primary, l2, coordinate, no_velocity)):
                    group_deltas[name][:, j] = corrected - native
                entry.update(penalty=penalty, boundary_position=p0.tolist(), acceleration=acceleration.tolist(),
                             primary_solver=solver, coordinate_solver=coordinate_solver,
                             no_velocity_solver=no_velocity_solver, l2_match=l2_audit,
                             primary_group_shift_norm=float(np.linalg.norm(primary - native)),
                             native_innovation_norms=np.linalg.norm(matrix @ native - target, axis=-1).tolist(),
                             corrected_innovation_norms=np.linalg.norm(matrix @ primary - target, axis=-1).tolist())
            group_audits.append(entry)
    deltas = {name: expand_groups(value, groups, native_position.shape[1]) for name, value in group_deltas.items()}
    return {"positions": {name: native_position + delta for name, delta in deltas.items()},
            "delta": deltas["primary"], "group_delta": group_deltas["primary"], "groups": groups,
            "audit": {"group_count": len(groups), "eligible_groups": sum(a["eligible"] for a in group_audits),
                      "groups": group_audits, "changed_particle_frames": int((np.linalg.norm(deltas["primary"], axis=-1) > 1e-8).sum()),
                      "primary_position_shift_norm": float(np.linalg.norm(deltas["primary"])),
                      "maximum_primary_shift_native_pixels": float(np.linalg.norm(deltas["primary"], axis=-1).max() * data["future_bg"].shape[-1] / 2)}}
