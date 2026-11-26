"""Additive IE-v2 generator entry point with a corrected no-contact track.

The original ``data_generat_process.py`` is intentionally left untouched.  This
entry point reuses its scene, rendering, and non-no-contact categories, while
replacing only ``no_contact_motion`` with a matched non-contact trajectory:

* one target object is static while the actor approaches it;
* frames 0..EVENT_FRAME-1 are the shared history;
* the first branch difference is visible at EVENT_FRAME;
* neither branch makes contact through the scored horizon;
* the impossible branch starts target motion smoothly without contact.

Run this file with Blender and an additive output root, e.g.::

    blender -b --python data_generat_process_ie_v2.py -- \
      --out_root data/process_obj3d_v2 --category no_contact_motion \
      --num_pairs 100 --seed 42 --img_size 128

The old generator and ``data/process_obj3d`` are never modified.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_generat_process as _base


EVENT_FRAME = _base.EVENT_FRAME
NUM_FRAMES = _base.NUM_FRAMES
NO_CONTACT_CATEGORY = "no_contact_motion"

_ORIGINAL_SAMPLE_CONFIG = _base.sample_config
_ORIGINAL_GET_POSITIONS = _base.get_positions


def _as_list(vector: Vector) -> list[float]:
    return [float(vector.x), float(vector.y), 0.0]


def _actor_position(frame: int, cfg: dict) -> Vector:
    direction = Vector(cfg["motion_dir"])
    actor_start = Vector(cfg["actor_start"])
    return actor_start + direction * cfg["actor_speed"] * frame


def _target_position(frame: int, branch: str, cfg: dict) -> Vector:
    event_position = Vector(cfg["target_event_pos"])
    velocity = Vector(cfg["target_velocity"])

    if frame < EVENT_FRAME:
        # Frames 0..EVENT_FRAME-1 are the common observed history;
        # event_position denotes the target position on its last frame.
        return event_position.copy()

    # The first future frame is EVENT_FRAME, matching the IE protocol.  The
    # possible branch keeps the static target still.  The impossible branch
    # introduces motion without contact using a short acceleration ramp.
    if branch == "possible":
        return event_position.copy()

    steps = frame - EVENT_FRAME + 1
    duration = max(1, int(cfg["deflection_duration"]))
    position = event_position.copy()
    for step in range(1, steps + 1):
        fraction = min(1.0, step / duration)
        position += velocity * fraction
    return position


def _v2_positions(frame: int, branch: str, cfg: dict):
    actor_position = _actor_position(frame, cfg)
    target_position = _target_position(frame, branch, cfg)
    return _base.xy_tuple(actor_position), _base.xy_tuple(target_position)


def _minimum_clearance(cfg: dict, branch: str, last_frame: int = 25) -> float:
    contact_gap = float(cfg["contact_gap"])
    minimum = float("inf")
    for frame in range(last_frame + 1):
        actor = _actor_position(frame, cfg)
        target = _target_position(frame, branch, cfg)
        minimum = min(minimum, float((actor - target).length - contact_gap))
    return minimum


def _sample_no_contact_config(seed: int) -> dict:
    """Create one deterministic, non-contact, motion-matched configuration."""

    for attempt in range(128):
        trial_seed = seed + attempt * 7919
        cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
        rng = random.Random(trial_seed + 104729)

        direction = Vector(cfg["motion_dir"])
        lateral = Vector(cfg["lateral_dir"])
        base_target = Vector(cfg["target_pos"])

        # Keep the target static while the actor approaches on a near-miss
        # path.  The lateral offset is larger than the contact gap, so there
        # is no accidental contact in the scored horizon.
        target_event = base_target + lateral * rng.uniform(-0.12, 0.12)
        actor_speed = rng.uniform(0.105, 0.155)
        side = rng.choice([-1.0, 1.0])
        actor_event = (
            target_event
            - direction * rng.uniform(0.15, 0.35)
            + lateral * side * (cfg["contact_gap"] + rng.uniform(0.22, 0.45))
        )
        actor_start = actor_event - direction * actor_speed * EVENT_FRAME

        target_speed = rng.uniform(0.105, 0.155)
        anomaly_kind = rng.choice(["lateral_drift", "away_drift"])
        # Move laterally away from the actor's near-miss side so the anomaly
        # remains visibly relative while respecting the no-contact constraint.
        anomaly_direction = lateral * (-side)
        if anomaly_kind == "away_drift":
            anomaly_direction = direction
        target_velocity = anomaly_direction * target_speed
        target_start = target_event

        cfg.update(
            {
                "subcategory": f"static_target_{anomaly_kind}",
                "actor_start": _as_list(actor_start),
                "actor_speed": float(actor_speed),
                "target_pos": _as_list(target_event),
                "target_event_pos": _as_list(target_event),
                "target_start": _as_list(target_start),
                "target_velocity": _as_list(target_velocity),
                "target_total_shift": float(target_speed * rng.uniform(4.0, 6.0)),
                "motion_duration": int(rng.randint(4, 6)),
                "contact_frame": NUM_FRAMES + 100,
                "deflection_duration": int(rng.randint(2, 3)),
                "deflection_angle": 0.0,
                "generator_variant": (
                    "ie_v2_static_target_relative_motion"
                ),
                "motion_mode": "static_target_moving_actor",
                "history_target_is_moving": False,
                "history_actor_is_moving": True,
                "actor_approaches_static_target": True,
                "target_motion_without_contact": True,
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

        # Keep the scene inside the existing camera envelope.
        all_positions = []
        for frame in range(26):
            actor, possible = _v2_positions(frame, "possible", cfg)
            _, impossible = _v2_positions(frame, "impossible", cfg)
            all_positions.extend((actor, possible, impossible))
        if any(abs(value) > 3.25 for position in all_positions for value in position[:2]):
            continue

        cfg["no_contact_min_clearance_possible"] = possible_clearance
        cfg["no_contact_min_clearance_impossible"] = impossible_clearance
        return cfg

    raise RuntimeError(f"Could not sample a valid IE-v2 no-contact scene: seed={seed}")


def _sample_config_v2(category: str, seed: int) -> dict:
    if category == NO_CONTACT_CATEGORY:
        return _sample_no_contact_config(seed)
    return _ORIGINAL_SAMPLE_CONFIG(category, seed)


def _get_positions_v2(category: str, frame: int, branch: str, cfg: dict):
    if category == NO_CONTACT_CATEGORY and cfg.get("generator_variant"):
        return _v2_positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def _refuse_existing_output_root() -> None:
    """Fail closed so a rerun cannot overwrite an earlier dataset root."""

    forwarded = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if "--out_root" not in forwarded:
        return
    index = forwarded.index("--out_root")
    if index + 1 >= len(forwarded):
        return
    output_root = Path(forwarded[index + 1]).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to write into existing output root: {output_root}. "
            "Choose a new additive --out_root."
        )


def main() -> None:
    # base.main() keeps the original argument contract and rendering path.  Its
    # globals are patched only inside this new process, never in the source file.
    _refuse_existing_output_root()
    _base.sample_config = _sample_config_v2
    _base.get_positions = _get_positions_v2
    _base.main()


if __name__ == "__main__":
    main()
