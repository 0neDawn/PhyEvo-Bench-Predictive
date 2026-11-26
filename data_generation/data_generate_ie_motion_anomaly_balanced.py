"""Generate additive, motion-matched IE Motion Anomaly candidates.

The existing generators and datasets are never modified.  Both branches keep
the same per-frame speed and path length; the impossible branch differs only in
how its direction evolves relative to the curved history.  This removes the
old shortcut where the impossible branch was always the more energetic one.

Example:
    blender -b --python data_generate_ie_motion_anomaly_balanced.py -- \
      --out_root data/process_obj3d_motion_anomaly_balanced_candidates \
      --category no_contact_motion --num_pairs 30 --seed 31042 --img_size 128
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generat_process_ie_no_contact_interaction as _previous


_base = _previous._base
EVENT_FRAME = _previous.EVENT_FRAME
NUM_FRAMES = _previous.NUM_FRAMES
NO_CONTACT_CATEGORY = _previous.NO_CONTACT_CATEGORY
_ORIGINAL_SAMPLE_CONFIG = _previous._ORIGINAL_SAMPLE_CONFIG
_ORIGINAL_GET_POSITIONS = _previous._ORIGINAL_GET_POSITIONS


def _rotate_xy(vector: Vector, angle: float) -> Vector:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return Vector((
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
        0.0,
    ))


def _as_list(vector: Vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _tuple3(vector: Vector) -> tuple[float, float, float]:
    return float(vector.x), float(vector.y), float(vector.z)


def _actor_position(frame: int, cfg: dict) -> Vector:
    return Vector(cfg["actor_start"]) + Vector(cfg["motion_dir"]) * float(
        cfg["actor_speed"]
    ) * frame


def _possible_step_velocity(transition_frame: int, cfg: dict) -> Vector:
    """Velocity for the transition ending at ``transition_frame``."""

    relative = transition_frame - (EVENT_FRAME - 1)
    angle = float(cfg["history_curve_rate"]) * relative
    return _rotate_xy(Vector(cfg["target_velocity_at_event"]), angle)


def _impossible_angle_offset(relative: int, cfg: dict) -> float:
    duration = int(cfg["anomaly_duration"])
    # At frame 20 the ramp is already visible, but it takes four frames to
    # reach its full response.  This avoids the former two-frame discontinuity.
    fraction = min(1.0, (relative + 1) / duration)
    curve_rate = float(cfg["history_curve_rate"])
    side = float(cfg["anomaly_side"])
    kind = cfg["anomaly_kind"]
    if kind == "curvature_phase_error":
        return side * float(cfg["anomaly_angle"]) * fraction
    if kind == "curvature_sign_error":
        return -2.0 * curve_rate * relative * fraction + side * 0.12 * fraction
    if kind == "curvature_gain_error":
        gain = float(cfg["curvature_gain"])
        # Move away from the observed curvature.  The previous sign made this
        # term cancel the small phase offset for one deterministic subtype.
        return -curve_rate * (gain - 1.0) * relative * fraction + side * 0.10 * fraction
    raise ValueError(f"Unknown anomaly kind: {kind}")


def _step_velocity(transition_frame: int, branch: str, cfg: dict) -> Vector:
    possible = _possible_step_velocity(transition_frame, cfg)
    if transition_frame < EVENT_FRAME or branch == "possible":
        return possible
    relative = transition_frame - (EVENT_FRAME - 1)
    return _rotate_xy(possible, _impossible_angle_offset(relative, cfg))


def _target_position(frame: int, branch: str, cfg: dict) -> Vector:
    """Integrate around the shared frame-19 anchor without changing speed."""

    anchor = Vector(cfg["target_frame19_pos"])
    if frame == EVENT_FRAME - 1:
        return anchor
    position = anchor.copy()
    if frame >= EVENT_FRAME:
        for transition in range(EVENT_FRAME, frame + 1):
            position += _step_velocity(transition, branch, cfg)
        return position
    for transition in range(EVENT_FRAME - 1, frame, -1):
        position -= _step_velocity(transition, "possible", cfg)
    return position


def _positions(frame: int, branch: str, cfg: dict):
    return _tuple3(_actor_position(frame, cfg)), _tuple3(
        _target_position(frame, branch, cfg)
    )


def _minimum_clearance(cfg: dict, branch: str, last_frame: int = 25) -> float:
    minimum = float("inf")
    contact_gap = float(cfg["contact_gap"])
    for frame in range(last_frame + 1):
        clearance = (
            _actor_position(frame, cfg) - _target_position(frame, branch, cfg)
        ).length - contact_gap
        minimum = min(minimum, float(clearance))
    return minimum


def _geometry_audit(cfg: dict, branch: str) -> tuple[bool, float]:
    """Reject physical-object overlap and floor/backdrop fusion for all frames."""

    objects = [
        (cfg["actor_shape"], cfg["actor_size"]),
        (cfg["target_shape"], cfg["target_size"]),
    ]
    objects.extend(
        (item["kind"], item["size"]) for item in cfg.get("distractors", [])
    )
    extents = [_base.shape_extent(kind, size) for kind, size in objects]
    distractors = [Vector(item["pos"]) for item in cfg.get("distractors", [])]
    minimum = float("inf")
    for frame in range(NUM_FRAMES):
        actor, target = _positions(frame, branch, cfg)
        positions = [Vector(actor), Vector(target), *distractors]
        for position, extent in zip(positions, extents):
            if position.z - extent < -0.47 or position.y + extent > 3.15:
                return False, minimum
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                clearance = float(
                    (positions[first] - positions[second]).length
                    - extents[first]
                    - extents[second]
                )
                minimum = min(minimum, clearance)
                if clearance < 0.02:
                    return False, minimum
    return True, minimum


def _path_length(cfg: dict, branch: str, first: int = 20, last: int = 25) -> float:
    return float(
        sum(
            (_target_position(frame, branch, cfg) - _target_position(frame - 1, branch, cfg)).length
            for frame in range(first, last + 1)
        )
    )


def _sample_no_contact_config(seed: int) -> dict:
    requested_kind = (
        "curvature_phase_error",
        "curvature_sign_error",
        "curvature_gain_error",
    )[seed % 3]

    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
        rng = random.Random(trial_seed + 104729)

        direction = Vector(cfg["motion_dir"])
        lateral = Vector(cfg["lateral_dir"])
        curve_side = rng.choice((-1.0, 1.0))
        curve_rate = curve_side * rng.uniform(0.035, 0.060)
        target_speed = rng.uniform(0.145, 0.190)
        target_heading = rng.uniform(-0.18, 0.18)
        target_velocity = _rotate_xy(direction, target_heading) * target_speed
        target_frame19 = Vector(cfg["target_pos"]) + lateral * rng.uniform(-0.10, 0.10)

        actor_speed = rng.uniform(0.080, 0.115)
        near_miss_side = rng.choice((-1.0, 1.0))
        provisional = dict(cfg)
        provisional.update(
            target_frame19_pos=_as_list(target_frame19),
            target_velocity_at_event=_as_list(target_velocity),
            history_curve_rate=float(curve_rate),
            anomaly_kind=requested_kind,
            anomaly_duration=4,
            anomaly_side=float(-curve_side),
            anomaly_angle=float(rng.uniform(0.25, 0.48)),
            curvature_gain=float(rng.uniform(2.2, 3.2)),
        )
        target20 = _target_position(EVENT_FRAME, "possible", provisional)
        actor20 = (
            target20
            - direction * rng.uniform(0.12, 0.28)
            + lateral
            * near_miss_side
            * (float(cfg["contact_gap"]) + rng.uniform(0.22, 0.34))
        )
        actor_start = actor20 - direction * actor_speed * EVENT_FRAME

        cfg.update(
            {
                "subcategory": f"matched_{requested_kind}",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "matched no-contact dynamics violation",
                "actor_start": _as_list(actor_start),
                "actor_speed": float(actor_speed),
                "target_pos": _as_list(target_frame19),
                "target_frame19_pos": _as_list(target_frame19),
                "target_start": _as_list(_target_position(0, "possible", provisional)),
                "target_velocity": _as_list(target_velocity),
                "target_velocity_at_event": _as_list(target_velocity),
                "target_total_shift": float(target_speed * 6.0),
                "history_curve_rate": float(curve_rate),
                "motion_duration": 6,
                "contact_frame": NUM_FRAMES + 100,
                "anomaly_kind": requested_kind,
                "anomaly_duration": 4,
                "anomaly_side": float(-curve_side),
                "anomaly_angle": float(provisional["anomaly_angle"]),
                "curvature_gain": float(provisional["curvature_gain"]),
                "generator_variant": "ie_motion_anomaly_matched_dynamics",
                "motion_mode": "curved_history_equal_speed_near_miss",
                "history_target_is_moving": True,
                "history_actor_is_moving": True,
                "actor_approaches_target": True,
                "near_miss_interaction": True,
                "target_response_without_contact": True,
                "history_last_shared_frame": EVENT_FRAME - 1,
                "first_divergence_frame": EVENT_FRAME,
                "branch_position_equal_at_event": False,
                "contact_forbidden_through_frame": 25,
                "speed_is_matched_between_branches": True,
                "path_length_is_matched_between_branches": True,
            }
        )

        possible_clearance = _minimum_clearance(cfg, "possible", 25)
        impossible_clearance = _minimum_clearance(cfg, "impossible", 25)
        if min(possible_clearance, impossible_clearance) < 0.18:
            continue

        p20 = _target_position(20, "possible", cfg)
        q20 = _target_position(20, "impossible", cfg)
        p25 = _target_position(25, "possible", cfg)
        q25 = _target_position(25, "impossible", cfg)
        event_delta = float((q20 - p20).length)
        late_delta = float((q25 - p25).length)
        if not (0.012 <= event_delta <= 0.075 and 0.18 <= late_delta <= 0.70):
            continue

        possible_length = _path_length(cfg, "possible")
        impossible_length = _path_length(cfg, "impossible")
        path_mismatch = abs(possible_length - impossible_length) / max(
            possible_length, impossible_length, 1e-12
        )
        if path_mismatch > 1e-6:
            continue

        possible_ok, possible_pair_clearance = _geometry_audit(cfg, "possible")
        impossible_ok, impossible_pair_clearance = _geometry_audit(cfg, "impossible")
        if not possible_ok or not impossible_ok:
            continue

        cfg["no_contact_min_clearance_possible"] = possible_clearance
        cfg["no_contact_min_clearance_impossible"] = impossible_clearance
        cfg["all_object_min_clearance_possible"] = possible_pair_clearance
        cfg["all_object_min_clearance_impossible"] = impossible_pair_clearance
        cfg["event_branch_delta"] = event_delta
        cfg["frame25_branch_delta"] = late_delta
        cfg["possible_path_length_20_25"] = possible_length
        cfg["impossible_path_length_20_25"] = impossible_length
        cfg["relative_path_length_mismatch"] = path_mismatch
        return cfg

    raise RuntimeError(f"Could not sample a matched Motion Anomaly pair: seed={seed}")


def _sample_config(category: str, seed: int) -> dict:
    if category == NO_CONTACT_CATEGORY:
        return _sample_no_contact_config(seed)
    return _ORIGINAL_SAMPLE_CONFIG(category, seed)


def _get_positions(category: str, frame: int, branch: str, cfg: dict):
    if (
        category == NO_CONTACT_CATEGORY
        and cfg.get("generator_variant") == "ie_motion_anomaly_matched_dynamics"
    ):
        return _positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def main() -> None:
    _previous._v3._v2._refuse_existing_output_root()
    _base.sample_config = _sample_config
    _base.get_positions = _get_positions
    _base.main()


if __name__ == "__main__":
    main()
