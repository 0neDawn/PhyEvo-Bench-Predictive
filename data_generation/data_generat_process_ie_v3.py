"""Additive IE-v3 generator for a dynamic no-contact near-miss track.

The frozen IE generator and the earlier v2 experiment are left untouched.
This entry point reuses the original scene/rendering code and replaces only
``no_contact_motion`` with a harder, motion-matched construction:

* actor and target both move during frames 0..19;
* the actor approaches the target on a guaranteed near-miss path;
* possible continues the observed target velocity;
* impossible changes target direction or speed smoothly from frame 20;
* neither branch contacts through the scored horizon.

Run with a new output root, for example::

    blender -b --python data_generat_process_ie_v3.py -- \
      --out_root data/process_obj3d_ie_v3_no_contact_001 \
      --category no_contact_motion --num_pairs 100 --seed 42 --img_size 128
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generat_process_ie_v2 as _v2


_base = _v2._base
EVENT_FRAME = _base.EVENT_FRAME
NUM_FRAMES = _base.NUM_FRAMES
NO_CONTACT_CATEGORY = "no_contact_motion"
_ORIGINAL_SAMPLE_CONFIG = _v2._ORIGINAL_SAMPLE_CONFIG
_ORIGINAL_GET_POSITIONS = _v2._ORIGINAL_GET_POSITIONS


def _rotate_xy(vector: Vector, angle: float) -> Vector:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return Vector((
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
        0.0,
    ))


def _as_list(vector: Vector) -> list[float]:
    return [float(vector.x), float(vector.y), 0.0]


def _actor_position(frame: int, cfg: dict) -> Vector:
    direction = Vector(cfg["motion_dir"])
    return Vector(cfg["actor_start"]) + direction * cfg["actor_speed"] * frame


def _target_position(frame: int, branch: str, cfg: dict) -> Vector:
    event_position = Vector(cfg["target_event_pos"])
    velocity = Vector(cfg["target_velocity"])

    if frame < EVENT_FRAME:
        # The target is already moving in the shared history.  Its position
        # at EVENT_FRAME-1 is event_position.
        return event_position + velocity * (frame - (EVENT_FRAME - 1))

    steps = frame - EVENT_FRAME + 1
    if branch == "possible":
        return event_position + velocity * steps

    duration = max(1, int(cfg["anomaly_duration"]))
    position = event_position.copy()
    kind = cfg["anomaly_kind"]
    for step in range(1, steps + 1):
        fraction = min(1.0, step / duration)
        if kind == "direction_turn":
            step_velocity = _rotate_xy(
                velocity, float(cfg["anomaly_angle"]) * fraction
            )
        else:
            step_velocity = velocity * (
                1.0 + float(cfg["speed_change_fraction"]) * fraction
            )
        position += step_velocity
    return position


def _v3_positions(frame: int, branch: str, cfg: dict):
    return (
        _base.xy_tuple(_actor_position(frame, cfg)),
        _base.xy_tuple(_target_position(frame, branch, cfg)),
    )


def _minimum_clearance(cfg: dict, branch: str, last_frame: int = 25) -> float:
    contact_gap = float(cfg["contact_gap"])
    minimum = float("inf")
    for frame in range(last_frame + 1):
        distance = (
            _actor_position(frame, cfg) - _target_position(frame, branch, cfg)
        ).length
        minimum = min(minimum, float(distance - contact_gap))
    return minimum


def _sample_no_contact_config(seed: int) -> dict:
    for attempt in range(2048):
        trial_seed = seed + attempt * 7919
        cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
        rng = random.Random(trial_seed + 104729)

        direction = Vector(cfg["motion_dir"])
        lateral = Vector(cfg["lateral_dir"])
        target_event = Vector(cfg["target_pos"]) + lateral * rng.uniform(-0.12, 0.12)

        # Both objects move in history.  The actor approaches from behind,
        # with a lateral offset that guarantees a near miss instead of contact.
        actor_speed = rng.uniform(0.075, 0.105)
        side = rng.choice([-1.0, 1.0])
        actor_event = (
            target_event
            - direction * (cfg["contact_gap"] + rng.uniform(0.45, 0.70))
            + lateral * side * (cfg["contact_gap"] + rng.uniform(0.35, 0.60))
        )
        actor_start = actor_event - direction * actor_speed * (EVENT_FRAME - 1)

        target_speed = rng.uniform(0.145, 0.205)
        target_angle = rng.uniform(-0.16, 0.16)
        target_velocity = _rotate_xy(direction, target_angle) * target_speed
        anomaly_kind = rng.choice(["direction_turn", "speed_change"])

        cfg.update(
            {
                "subcategory": f"moving_near_miss_{anomaly_kind}",
                "actor_start": _as_list(actor_start),
                "actor_speed": float(actor_speed),
                "target_pos": _as_list(target_event),
                "target_event_pos": _as_list(target_event),
                "target_start": _as_list(
                    target_event - target_velocity * (EVENT_FRAME - 1)
                ),
                "target_velocity": _as_list(target_velocity),
                "target_total_shift": float(target_speed * rng.uniform(4.0, 6.0)),
                "motion_duration": int(rng.randint(4, 6)),
                "contact_frame": NUM_FRAMES + 100,
                "anomaly_kind": anomaly_kind,
                "anomaly_duration": 2,
                "anomaly_angle": float(
                    rng.choice([-1.0, 1.0]) * rng.uniform(0.95, 1.25)
                ),
                "speed_change_fraction": float(rng.uniform(1.20, 1.80)),
                "generator_variant": "ie_v3_dynamic_near_miss",
                "motion_mode": "moving_actor_moving_target",
                "history_target_is_moving": True,
                "history_actor_is_moving": True,
                "actor_approaches_target": True,
                "history_last_shared_frame": EVENT_FRAME - 1,
                "first_divergence_frame": EVENT_FRAME,
                "branch_position_equal_at_event": False,
                "contact_forbidden_through_frame": 25,
            }
        )

        possible_clearance = _minimum_clearance(cfg, "possible")
        impossible_clearance = _minimum_clearance(cfg, "impossible")
        if min(possible_clearance, impossible_clearance) < 0.18:
            continue

        p20 = _target_position(EVENT_FRAME, "possible", cfg)
        q20 = _target_position(EVENT_FRAME, "impossible", cfg)
        p25 = _target_position(25, "possible", cfg)
        q25 = _target_position(25, "impossible", cfg)
        event_delta = float((q20 - p20).length)
        late_delta = float((q25 - p25).length)
        # Enforce a visible, reproducible anomaly rather than relying on a
        # model or a viewer to detect a sub-pixel difference.
        if event_delta < 0.08 or late_delta < 0.60:
            continue

        all_positions = []
        for frame in range(26):
            actor, possible = _v3_positions(frame, "possible", cfg)
            _, impossible = _v3_positions(frame, "impossible", cfg)
            all_positions.extend((actor, possible, impossible))
        if any(abs(value) > 3.25 for position in all_positions for value in position[:2]):
            continue

        cfg["no_contact_min_clearance_possible"] = possible_clearance
        cfg["no_contact_min_clearance_impossible"] = impossible_clearance
        cfg["event_branch_delta"] = event_delta
        cfg["frame25_branch_delta"] = late_delta
        return cfg

    raise RuntimeError(f"Could not sample a valid IE-v3 scene: seed={seed}")


def _sample_config_v3(category: str, seed: int) -> dict:
    if category == NO_CONTACT_CATEGORY:
        return _sample_no_contact_config(seed)
    return _ORIGINAL_SAMPLE_CONFIG(category, seed)


def _get_positions_v3(category: str, frame: int, branch: str, cfg: dict):
    if category == NO_CONTACT_CATEGORY and cfg.get("generator_variant") == "ie_v3_dynamic_near_miss":
        return _v3_positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def main() -> None:
    _v2._refuse_existing_output_root()
    _base.sample_config = _sample_config_v3
    _base.get_positions = _get_positions_v3
    _base.main()


if __name__ == "__main__":
    main()
