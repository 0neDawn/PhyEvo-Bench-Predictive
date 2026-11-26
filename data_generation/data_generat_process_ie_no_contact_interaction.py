"""IE no-contact interaction benchmark: near miss with phantom contact.

This additive generator keeps the original IE generator and earlier drafts
unchanged.  It targets the semantic boundary that separates IE from SP:
two objects approach one another, pass with a measured positive clearance,
and the impossible branch exhibits a collision-like response despite the
absence of physical contact.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generat_process_ie_v3 as _v3


_base = _v3._base
EVENT_FRAME = _base.EVENT_FRAME
NUM_FRAMES = _base.NUM_FRAMES
NO_CONTACT_CATEGORY = "no_contact_motion"
_ORIGINAL_SAMPLE_CONFIG = _v3._ORIGINAL_SAMPLE_CONFIG
_ORIGINAL_GET_POSITIONS = _v3._ORIGINAL_GET_POSITIONS


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
    # Motion continues for the complete video; reaching the image edge is
    # allowed and naturally makes the object leave the frame.
    return Vector(cfg["actor_start"]) + direction * cfg["actor_speed"] * frame


def _tuple3(vector: Vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def _target_position(frame: int, branch: str, cfg: dict) -> Vector:
    event_position = Vector(cfg["target_event_pos"])
    velocity = Vector(cfg["target_velocity"])

    if frame < EVENT_FRAME:
        return event_position + velocity * (frame - (EVENT_FRAME - 1))

    # The future trajectory is not truncated at frame 25.  The anomaly ramp
    # may saturate, but the object keeps moving until it exits the image.
    steps = frame - EVENT_FRAME + 1
    if branch == "possible":
        return event_position + velocity * steps

    if cfg["anomaly_kind"] == "phantom_parabolic":
        progress = min(1.0, steps / max(1, int(cfg["motion_duration"])))
        base = event_position + velocity * steps
        lateral = Vector(cfg["lateral_dir"])
        return base + lateral * float(cfg["parabola_lateral_amplitude"]) * progress + Vector((
            0.0,
            0.0,
            float(cfg["parabola_height"]) * 4.0 * progress * (1.0 - progress),
        ))

    duration = max(1, int(cfg["anomaly_duration"]))
    position = event_position.copy()
    for step in range(1, steps + 1):
        fraction = min(1.0, step / duration)
        if cfg["anomaly_kind"] == "phantom_turn":
            step_velocity = _rotate_xy(
                velocity, float(cfg["anomaly_angle"]) * fraction
            )
        elif cfg["anomaly_kind"] == "phantom_bounce":
            # A collision-like reversal without a collision.  The ramp keeps
            # the position continuous while making the response unambiguous.
            step_velocity = velocity * (
                1.0 - (1.0 + float(cfg["bounce_strength"])) * fraction
            )
        elif cfg["anomaly_kind"] == "phantom_upward":
            # A vertical response without contact.  This is deliberately a
            # small minority subtype so the benchmark remains interaction-
            # centered while covering 3-D motion.
            step_velocity = Vector((0.0, 0.0, float(cfg["upward_speed"]))) * fraction
        else:
            step_velocity = velocity * fraction
        position += step_velocity
    return position


def _positions(frame: int, branch: str, cfg: dict):
    return (
        _tuple3(_actor_position(frame, cfg)),
        _tuple3(_target_position(frame, branch, cfg)),
    )


def _minimum_clearance(cfg: dict, branch: str, last_frame: int = 25) -> float:
    contact_gap = float(cfg["contact_gap"])
    minimum = float("inf")
    for frame in range(last_frame + 1):
        minimum = min(
            minimum,
            float(
                (
                    _actor_position(frame, cfg)
                    - _target_position(frame, branch, cfg)
                ).length
                - contact_gap
            ),
        )
    return minimum


def _geometry_audit(cfg: dict, branch: str) -> tuple[bool, float]:
    """Reject interpenetration and internal floor/wall fusion.

    The camera frame is intentionally *not* a boundary: an object may leave
    the image naturally.  Only the physical floor, backdrop wall, and other
    scene objects are treated as collision constraints.
    """

    objects = [
        ("actor", cfg["actor_shape"], cfg["actor_size"]),
        ("target", cfg["target_shape"], cfg["target_size"]),
    ]
    for distractor in cfg.get("distractors", []):
        objects.append(("distractor", distractor["kind"], distractor["size"]))
    extents = [_base.shape_extent(kind, size) for _, kind, size in objects]
    distractor_positions = [
        Vector(d["pos"]) for d in cfg.get("distractors", [])
    ]
    min_pair_clearance = float("inf")

    for frame in range(NUM_FRAMES):
        actor, target = _positions(frame, branch, cfg)
        positions = [Vector(actor), Vector(target)] + distractor_positions

        # Grounded objects may overlap the original floor plane by at most the
        # renderer's existing 0.05-unit tolerance; new downward motion cannot.
        for position, extent in zip(positions, extents):
            if position.z - extent < -0.47:
                return False, min_pair_clearance
            if position.y + extent > 3.15:
                return False, min_pair_clearance

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                clearance = float(
                    (positions[i] - positions[j]).length
                    - extents[i]
                    - extents[j]
                )
                min_pair_clearance = min(min_pair_clearance, clearance)
                if clearance < 0.02:
                    return False, min_pair_clearance

    return True, min_pair_clearance


def _sample_no_contact_config(seed: int) -> dict:
    kind_rng = random.Random(seed + 271828)
    # Use a deterministic 1-in-10 schedule for the vertical subtype, so a
    # 100-pair evaluation contains exactly 10 upward cases (apart from an
    # intentionally different starting seed).
    if seed % 10 == 0:
        requested_anomaly_kind = "phantom_upward"
    elif seed % 10 in (1, 2):
        requested_anomaly_kind = "phantom_parabolic"
    elif kind_rng.random() < 0.5:
        requested_anomaly_kind = "phantom_turn"
    else:
        requested_anomaly_kind = "phantom_bounce"

    for attempt in range(2048):
        trial_seed = seed + attempt * 7919
        cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
        rng = random.Random(trial_seed + 104729)

        direction = Vector(cfg["motion_dir"])
        lateral = Vector(cfg["lateral_dir"])
        target_event = Vector(cfg["target_pos"]) + lateral * rng.uniform(-0.10, 0.10)

        # The actor passes close to the target at EVENT_FRAME.  The lateral
        # offset is just beyond the contact gap, so the interaction is visible
        # but geometrically remains a true near miss.
        actor_speed = rng.uniform(0.075, 0.105)
        side = rng.choice([-1.0, 1.0])
        actor_event = (
            target_event
            - direction * rng.uniform(0.15, 0.30)
            + lateral * side * (cfg["contact_gap"] + rng.uniform(0.20, 0.30))
        )
        actor_start = actor_event - direction * actor_speed * (EVENT_FRAME - 1)

        target_speed = rng.uniform(0.145, 0.205)
        target_velocity = _rotate_xy(direction, rng.uniform(-0.14, 0.14)) * target_speed
        # Keep the 10% vertical subtype fixed across geometric retries so
        # rejection sampling cannot change its intended prevalence.
        anomaly_kind = requested_anomaly_kind

        cfg.update(
            {
                "subcategory": f"near_miss_{anomaly_kind}",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "no-contact interaction / phantom-contact response",
                "actor_start": _as_list(actor_start),
                "actor_speed": float(actor_speed),
                "target_pos": _as_list(target_event),
                "target_event_pos": _as_list(target_event),
                "target_start": _as_list(
                    target_event - target_velocity * (EVENT_FRAME - 1)
                ),
                "target_velocity": _as_list(target_velocity),
                "target_total_shift": float(target_speed * rng.uniform(4.0, 6.0)),
                "motion_duration": int(rng.randint(6, 8)),
                "contact_frame": NUM_FRAMES + 100,
                "anomaly_kind": anomaly_kind,
                "anomaly_duration": 2,
                "anomaly_angle": float(
                    -side * rng.uniform(1.00, 1.30)
                ),
                "bounce_strength": float(rng.uniform(0.70, 1.10)),
                "upward_speed": float(rng.uniform(0.16, 0.24)),
                "parabola_height": float(rng.uniform(0.80, 1.10)),
                "parabola_lateral_amplitude": float(rng.uniform(0.60, 0.90)),
                "generator_variant": "ie_no_contact_phantom_interaction",
                "motion_mode": "moving_actor_moving_target_near_miss",
                "history_target_is_moving": True,
                "history_actor_is_moving": True,
                "actor_approaches_target": True,
                "near_miss_interaction": True,
                "target_response_without_contact": True,
                "vertical_anomaly_fraction": 0.10,
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

        event_distance = float(
            (
                _actor_position(EVENT_FRAME, cfg)
                - _target_position(EVENT_FRAME, "possible", cfg)
            ).length
        )
        p20 = _target_position(EVENT_FRAME, "possible", cfg)
        q20 = _target_position(EVENT_FRAME, "impossible", cfg)
        p25 = _target_position(25, "possible", cfg)
        q25 = _target_position(25, "impossible", cfg)
        event_delta = float((q20 - p20).length)
        late_delta = float((q25 - p25).length)
        if event_distance > cfg["contact_gap"] + 0.70:
            continue
        if event_delta < 0.10 or late_delta < 0.80:
            continue

        possible_ok, possible_pair_clearance = _geometry_audit(cfg, "possible")
        impossible_ok, impossible_pair_clearance = _geometry_audit(cfg, "impossible")
        if not possible_ok or not impossible_ok:
            continue

        cfg["near_miss_event_distance"] = event_distance
        cfg["no_contact_min_clearance_possible"] = possible_clearance
        cfg["no_contact_min_clearance_impossible"] = impossible_clearance
        cfg["all_object_min_clearance_possible"] = possible_pair_clearance
        cfg["all_object_min_clearance_impossible"] = impossible_pair_clearance
        cfg["event_branch_delta"] = event_delta
        cfg["frame25_branch_delta"] = late_delta
        return cfg

    raise RuntimeError(f"Could not sample a valid IE no-contact interaction: seed={seed}")


def _sample_config(category: str, seed: int) -> dict:
    if category == NO_CONTACT_CATEGORY:
        return _sample_no_contact_config(seed)
    return _ORIGINAL_SAMPLE_CONFIG(category, seed)


def _get_positions(category: str, frame: int, branch: str, cfg: dict):
    if category == NO_CONTACT_CATEGORY and cfg.get("generator_variant") == "ie_no_contact_phantom_interaction":
        return _positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def main() -> None:
    _v3._v2._refuse_existing_output_root()
    _base.sample_config = _sample_config
    _base.get_positions = _get_positions
    _base.main()


if __name__ == "__main__":
    main()
