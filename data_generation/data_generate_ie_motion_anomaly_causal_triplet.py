"""Generate 30 additive IE Motion Anomaly candidates.

The candidates are deliberately split into three physically interpretable
subtypes (ten of each):

``remote_oblique_response``
    A faster rear object uniformly approaches a slower front object while a
    large lateral gap remains visible.  The possible branch continues
    straight; the impossible branch turns the two velocity vectors to
    opposite sides by 25--45 degrees at frame 20.  The speed magnitudes are
    unchanged and no contact occurs.

``gravity_direction_anomaly``
    Both branches begin with the target visibly supported on a raised block.
    It travels to the block's edge and leaves it at frames 7--10, then falls
    under gravity.  The possible branch keeps that release trajectory; the
    impossible branch adds an immediate upward or frontward horizontal
    deviation from frame 20.  Thus the airborne motion has a visible common
    cause and is not an unexplained launch or an actor--target collision.

``premature_rebound``
    Two objects are on a head-on trajectory and would first touch at frame 20
    in the possible branch, then rebound.  The impossible branch reverses both
    velocities one frame early while a visible gap remains.  Thus the error is
    causal timing, not a collision/penetration artifact.

Frames 0--19 are bit-for-bit shared between branches and the first decoded
future difference is frame 20.  The original generator, source data, and all
previous candidate roots are never modified.  The output root is fail-closed:
it must be new.

Example::

    blender -b --python data_generate_ie_motion_anomaly_causal_triplet.py -- \
      --out_root data/process_obj3d_motion_anomaly_causal_candidates \
      --category no_contact_motion --num_pairs 30 --seed 41042 --img_size 128

Preflight without rendering::

    blender -b --python data_generate_ie_motion_anomaly_causal_triplet.py -- \
      --preflight --num_pairs 30 --seed 41042
"""

from __future__ import annotations

import argparse
import json
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
_ORIGINAL_CREATE_SCENE_OBJECTS = _base.create_scene_objects

_KINDS = (
    "remote_oblique_response",
    "gravity_direction_anomaly",
    "premature_rebound",
)


def _vector(value) -> Vector:
    return Vector((float(value[0]), float(value[1]), float(value[2])))


def _as_list(value: Vector) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _tuple3(value: Vector) -> tuple[float, float, float]:
    return float(value.x), float(value.y), float(value.z)


def _rotate_xy(value: Vector, angle: float) -> Vector:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return Vector(
        (
            cosine * value.x - sine * value.y,
            sine * value.x + cosine * value.y,
            value.z,
        )
    )


def _event_number(frame: int) -> int:
    """Number of transitions after the shared frame-19 anchor."""

    return frame - (EVENT_FRAME - 1)


def _oblique_positions(frame: int, branch: str, cfg: dict):
    actor19 = _vector(cfg["actor_frame19_pos"])
    target19 = _vector(cfg["target_frame19_pos"])
    actor_velocity = _vector(cfg["actor_velocity"])
    target_velocity = _vector(cfg["target_velocity"])

    if frame < EVENT_FRAME:
        # Both objects approach uniformly during the observed history.
        delta = frame - (EVENT_FRAME - 1)
        actor = actor19 + actor_velocity * delta
        target = target19 + target_velocity * delta
        return _tuple3(actor), _tuple3(target)

    delta = _event_number(frame)
    if branch == "possible":
        actor_step = actor_velocity
        target_step = target_velocity
    else:
        angle = float(cfg["deflection_angle"])
        side = float(cfg["deflection_side"])
        # Actor and target turn outwards from their shared approach corridor.
        actor_step = _rotate_xy(actor_velocity, -side * angle)
        target_step = _rotate_xy(target_velocity, side * angle)

    actor = actor19 + actor_step * delta
    target = target19 + target_step * delta
    return _tuple3(actor), _tuple3(target)


def _gravity_target(frame: int, branch: str, cfg: dict) -> Vector:
    target0 = _vector(cfg["target_frame0_pos"])
    horizontal_velocity = _vector(cfg["target_horizontal_velocity"])
    gravity = float(cfg["gravity"])
    launch_frame = int(cfg["launch_frame"])

    if frame < launch_frame:
        # The target rolls/slides across the raised block during the common
        # history.  It reaches the edge exactly at ``launch_frame``.
        target = target0 + horizontal_velocity * float(frame)
    else:
        elapsed = float(frame - launch_frame)
        launch_position = target0 + horizontal_velocity * float(launch_frame)
        target = launch_position + horizontal_velocity * elapsed
        # The object leaves a horizontal ledge with no artificial upward
        # impulse.  Its vertical motion is the ordinary free-fall solution.
        target.z -= 0.5 * gravity * elapsed * elapsed

    if frame >= EVENT_FRAME and branch == "impossible":
        anomaly_delta = _event_number(frame)
        impulse = float(cfg["anomaly_impulse_per_frame"])
        if cfg["gravity_anomaly_mode"] == "upward":
            target.z += impulse * anomaly_delta
        else:
            target += _vector(cfg["gravity_lateral_dir"]) * impulse * anomaly_delta
    return target


def _gravity_actor_position(frame: int, cfg: dict) -> Vector:
    # The actor is a separate floor object.  Keeping it independent removes
    # the physically inconsistent "one object rebounds while the other is
    # launched" construction used by the rejected candidate generator.
    return _vector(cfg["actor_frame0_pos"]) + _vector(cfg["actor_velocity"]) * float(frame)


def _gravity_positions(frame: int, branch: str, cfg: dict):
    actor = _gravity_actor_position(frame, cfg)
    return _tuple3(actor), _tuple3(_gravity_target(frame, branch, cfg))


def _rebound_positions(frame: int, branch: str, cfg: dict):
    actor19 = _vector(cfg["actor_frame19_pos"])
    target19 = _vector(cfg["target_frame19_pos"])
    direction = _vector(cfg["motion_dir"])
    actor_speed = float(cfg["actor_speed"])
    target_speed = float(cfg["target_speed"])

    if frame <= EVENT_FRAME - 1:
        delta = frame - (EVENT_FRAME - 1)
        actor = actor19 + direction * actor_speed * delta
        target = target19 - direction * target_speed * delta
        return _tuple3(actor), _tuple3(target)

    if branch == "possible":
        # Contact is first reached at frame 20; separation starts at frame 21.
        contact_actor = actor19 + direction * actor_speed
        contact_target = target19 - direction * target_speed
        separation_steps = frame - EVENT_FRAME
        actor = contact_actor - direction * float(cfg["rebound_actor_speed"]) * separation_steps
        target = contact_target + direction * float(cfg["rebound_target_speed"]) * separation_steps
    else:
        # The impossible branch rebounds at frame 20, one transition before
        # the visible contact that would occur in the possible branch.
        early_steps = frame - (EVENT_FRAME - 1)
        actor = actor19 - direction * float(cfg["rebound_actor_speed"]) * early_steps
        target = target19 + direction * float(cfg["rebound_target_speed"]) * early_steps
    return _tuple3(actor), _tuple3(target)


def _positions(frame: int, branch: str, cfg: dict):
    kind = cfg.get("triplet_kind")
    if kind == "remote_oblique_response":
        return _oblique_positions(frame, branch, cfg)
    if kind == "gravity_direction_anomaly":
        return _gravity_positions(frame, branch, cfg)
    if kind == "premature_rebound":
        return _rebound_positions(frame, branch, cfg)
    raise ValueError(f"Unknown triplet kind: {kind}")


def _mesh_clearance(cfg: dict, branch: str, frame: int) -> float:
    actor, target = _positions(frame, branch, cfg)
    actor_extent = float(cfg["actor_extent"])
    target_extent = float(cfg["target_extent"])
    return float((_vector(actor) - _vector(target)).length - actor_extent - target_extent)


def _geometry_audit(cfg: dict, branch: str) -> tuple[bool, float]:
    """Check floor/backdrop and pair geometry for every rendered frame."""

    actor_extent = float(cfg["actor_extent"])
    target_extent = float(cfg["target_extent"])
    minimum = float("inf")
    allowed_contact_clearance = 0.006
    for frame in range(NUM_FRAMES):
        actor, target = (_vector(value) for value in _positions(frame, branch, cfg))
        for position, extent in ((actor, actor_extent), (target, target_extent)):
            # The original renderer uses a floor plane at z=-0.42.  Keep the
            # same small tolerance as prior IE generators and never fuse with
            # the internal backdrop plane at y=3.2.
            if position.z - extent < -0.47:
                return False, minimum
            if position.y + extent > 3.15:
                return False, minimum
        clearance = float((actor - target).length - actor_extent - target_extent)
        minimum = min(minimum, clearance)
        threshold = allowed_contact_clearance
        physical_touch_frame = (
            (cfg["triplet_kind"] == "premature_rebound" and branch == "possible" and frame == EVENT_FRAME)
            or (cfg["triplet_kind"] == "gravity_direction_anomaly" and frame == int(cfg.get("launch_frame", -1)))
        )
        if not physical_touch_frame and clearance < 0.02:
            return False, minimum
        if clearance < threshold:
            return False, minimum
        if cfg["triplet_kind"] == "gravity_direction_anomaly":
            # Before release the target must remain above the visible block;
            # after release it must already be beyond the block's edge.
            platform_center = _vector(cfg["platform_center"])
            axis = _vector(cfg["motion_dir"]).normalized()
            side = _vector(cfg["lateral_dir"]).normalized()
            relative = target - platform_center
            along = float(relative.dot(axis))
            across = abs(float(relative.dot(side)))
            edge = float(cfg["platform_edge_distance"])
            if frame <= int(cfg["launch_frame"]):
                if along > edge + 1e-4 or across > float(cfg["platform_width"]) / 2.0:
                    return False, minimum
                expected_bottom = float(cfg["platform_top_z"]) + 0.012
                if abs(float(target.z - float(cfg["target_extent"]) - expected_bottom)) > 1e-4:
                    return False, minimum
            elif along <= edge - 0.02:
                return False, minimum
    return True, minimum


def _visible_window_audit(cfg: dict) -> bool:
    """Keep the scored event window inside a reproducible camera envelope."""

    frames = list(range(EVENT_FRAME - 1, min(NUM_FRAMES, EVENT_FRAME + 6)))
    if cfg.get("triplet_kind") == "gravity_direction_anomaly":
        frames = [0, 5, 10, 15, *frames]
    for frame in frames:
        for branch in ("possible", "impossible"):
            for position in (_vector(value) for value in _positions(frame, branch, cfg)):
                x_limit = 1.90 if cfg.get("triplet_kind") == "remote_oblique_response" else 1.72
                if abs(position.x) > x_limit:
                    return False
                y_lower = -3.50 if cfg.get("triplet_kind") == "remote_oblique_response" else -1.75
                if position.y < y_lower or position.y > 2.55:
                    return False
                if position.z < -0.15 or position.z > 2.55:
                    return False
    return True


def _shared_history_audit(cfg: dict) -> bool:
    for frame in range(EVENT_FRAME):
        possible = _positions(frame, "possible", cfg)
        impossible = _positions(frame, "impossible", cfg)
        if possible != impossible:
            return False
    return _positions(EVENT_FRAME, "possible", cfg) != _positions(EVENT_FRAME, "impossible", cfg)


def _base_cfg(seed: int, trial_seed: int) -> tuple[dict, random.Random]:
    cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
    cfg["distractors"] = []
    return cfg, random.Random(trial_seed + 104729)


def _sample_oblique(seed: int) -> dict:
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        # Both objects move toward the camera/front; the faster rear object
        # approaches the slower one while a large lateral gap remains.  The
        # impossible branch then turns them to opposite sides.  The larger
        # object sizes and the constant-speed, near-45-degree response make
        # the first divergent frame unambiguously visible at 128 px.
        direction = Vector((0.0, -1.0, 0.0))
        lateral = Vector((1.0, 0.0, 0.0))
        cfg["motion_dir"] = _as_list(direction)
        cfg["lateral_dir"] = _as_list(lateral)
        # Keep the geometry within the original OBJ3D size range while
        # avoiding tiny silhouettes that make a one-frame turn ambiguous.
        cfg["actor_size"] = rng.uniform(0.80, 0.88)
        cfg["target_size"] = rng.uniform(0.80, 0.88)
        cfg["actor_extent"] = float(_base.shape_extent(cfg["actor_shape"], cfg["actor_size"]))
        cfg["target_extent"] = float(_base.shape_extent(cfg["target_shape"], cfg["target_size"]))
        cfg["contact_gap"] = float(cfg["actor_extent"] + cfg["target_extent"] + 0.05)
        center = (
            _vector(cfg["target_pos"])
            + lateral * rng.uniform(-0.12, 0.12)
            + Vector((0.0, -2.30, 0.0))
        )
        side = float(rng.choice((-1, 1)))
        # A genuinely remote near-miss: the lateral clearance is large enough
        # to be obvious in the rendered frame, so the response cannot be read
        # as an ordinary contact-and-rebound event.
        # The gap is deliberately larger than a contact-plus-rebound reading;
        # the two silhouettes are visibly separated in frame 19 and remain so
        # in both branches.
        # Use a modest screen-horizontal gap together with a larger
        # front/back gap.  The Euclidean clearance is therefore unmistakable,
        # while both silhouettes stay away from the image boundaries after the
        # 45-degree turn.
        lateral_gap = float(cfg["contact_gap"]) + rng.uniform(0.45, 0.65)
        longitudinal_gap = rng.uniform(0.85, 0.98)
        # The larger, still constant-speed step makes the direction change
        # visible already at frame 20; the objects remain remote because the
        # lateral clearance is substantially larger than their extents.
        actor_speed = rng.uniform(0.235, 0.245)
        target_speed = rng.uniform(0.100, 0.130)
        actor19 = center - direction * (longitudinal_gap / 2.0) - lateral * side * (lateral_gap / 2.0)
        target19 = center + direction * (longitudinal_gap / 2.0) + lateral * side * (lateral_gap / 2.0)
        actor_velocity = direction * actor_speed
        target_velocity = direction * target_speed
        angle = math.radians(rng.uniform(43.5, 45.0))
        cfg.update(
            {
                "triplet_kind": "remote_oblique_response",
                "subcategory": "remote_oblique_collision_response",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "remote oblique response",
                "generator_variant": "ie_motion_anomaly_causal_triplet_v1",
                "motion_mode": "two_uniformly_approaching_objects_with_lateral_gap",
                "actor_frame19_pos": _as_list(actor19),
                "target_frame19_pos": _as_list(target19),
                "actor_velocity": _as_list(actor_velocity),
                "target_velocity": _as_list(target_velocity),
                "actor_start": _as_list(actor19 - actor_velocity * (EVENT_FRAME - 1)),
                "target_start": _as_list(target19 - target_velocity * (EVENT_FRAME - 1)),
                "actor_speed": float(actor_speed),
                "target_speed": float(target_speed),
                "deflection_angle": float(angle),
                "deflection_angle_degrees": float(math.degrees(angle)),
                "deflection_side": side,
                "history_objects_approach_uniformly": True,
                "history_visible_lateral_gap": float(lateral_gap - cfg["actor_extent"] - cfg["target_extent"]),
                "speed_magnitude_matched": True,
                "history_last_shared_frame": EVENT_FRAME - 1,
                "first_divergence_frame": EVENT_FRAME,
                "branch_position_equal_at_event": False,
                "contact_forbidden_through_frame": NUM_FRAMES - 1,
            }
        )
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        event_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(EVENT_FRAME, "possible", cfg), _positions(EVENT_FRAME, "impossible", cfg))
        )
        late_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(25, "possible", cfg), _positions(25, "impossible", cfg))
        )
        if event_delta < 0.075 or late_delta < 0.55:
            continue
        cfg.update(
            {
                "event_branch_delta": float(event_delta),
                "frame25_branch_delta": float(late_delta),
                "minimum_mesh_clearance_possible": float(min_p),
                "minimum_mesh_clearance_impossible": float(min_i),
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample remote oblique candidate: seed={seed}")


def _sample_gravity(seed: int) -> dict:
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        # A raised block provides a visible, common physical cause for the
        # airborne trajectory.  The target moves along the block and leaves
        # its edge at frame 7--10; it then follows ordinary free fall.  This
        # replaces the rejected collision-launch construction: the actor does
        # not collide with, or mysteriously throw, the target.
        direction = Vector((0.0, -1.0, 0.0))
        lateral = Vector((1.0, 0.0, 0.0))
        cfg["motion_dir"] = _as_list(direction)
        cfg["lateral_dir"] = _as_list(lateral)
        platform_center = _vector(cfg["target_pos"]) + lateral * rng.uniform(-0.08, 0.08)
        platform_length = 2.40
        platform_width = 0.82
        platform_height = 0.80
        platform_bottom_z = -0.42
        platform_top_z = platform_bottom_z + platform_height
        launch_frame = rng.randint(7, 10)
        horizontal_speed = rng.uniform(0.070, 0.095)
        horizontal_velocity = direction * horizontal_speed
        edge_distance = platform_length / 2.0 - cfg["target_extent"] - 0.045
        target0 = platform_center + direction * (edge_distance - horizontal_speed * launch_frame)
        target0.z = platform_top_z + cfg["target_extent"] + 0.012

        # Keep a second object on the floor, visibly separated from the block;
        # it has a small, uniform history motion but no contact with the target.
        actor_offset = platform_width / 2.0 + cfg["actor_extent"] + 0.28
        actor0 = platform_center + lateral * actor_offset - direction * 0.55
        actor0.z = 0.0
        actor_velocity = direction * rng.uniform(0.018, 0.030)

        gravity = rng.uniform(0.00085, 0.00120)
        anomaly_mode = rng.choice(("upward", "lateral"))
        # The branch impulse is an explicit post-event violation.  It is large
        # enough to be visible in frame 20 and grows again in frame 21, while
        # the possible branch remains a textbook release-and-fall trajectory.
        impulse = rng.uniform(0.105, 0.135)
        cfg.update(
            {
                "triplet_kind": "gravity_direction_anomaly",
                "subcategory": f"gravity_direction_{anomaly_mode}",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "gravity direction anomaly",
                "generator_variant": "ie_motion_anomaly_causal_triplet_v2_platform_release",
                "motion_mode": "visible_platform_release_then_gravity_violation",
                "launch_frame": int(launch_frame),
                "platform_center": _as_list(platform_center),
                "platform_length": float(platform_length),
                "platform_width": float(platform_width),
                "platform_height": float(platform_height),
                "platform_bottom_z": float(platform_bottom_z),
                "platform_top_z": float(platform_top_z),
                "platform_edge_distance": float(edge_distance),
                "platform_axis_angle_degrees": -90.0,
                "actor_frame0_pos": _as_list(actor0),
                "actor_velocity": _as_list(actor_velocity),
                "actor_start": _as_list(actor0),
                "actor_frame19_pos": _as_list(actor0 + actor_velocity * (EVENT_FRAME - 1)),
                "target_frame0_pos": _as_list(target0),
                "target_start": _as_list(target0),
                "target_frame19_pos": _as_list(target0 + horizontal_velocity * (EVENT_FRAME - 1)),
                "target_horizontal_velocity": _as_list(horizontal_velocity),
                "target_initial_vertical_velocity": 0.0,
                "gravity": float(gravity),
                "gravity_anomaly_mode": anomaly_mode,
                "anomaly_impulse_per_frame": float(impulse),
                "gravity_lateral_dir": [0.0, -1.0, 0.0],
                "history_airborne": False,
                "history_starts_grounded": False,
                "history_target_supported_on_visible_platform": True,
                "history_platform_release_is_shared": True,
                "history_launch_frame": int(launch_frame),
                "shared_platform_release_frame": int(launch_frame),
                "event_difference_world_units": float(impulse),
                "history_last_shared_frame": EVENT_FRAME - 1,
                "first_divergence_frame": EVENT_FRAME,
                "branch_position_equal_at_event": False,
                "contact_forbidden_through_frame": NUM_FRAMES - 1,
            }
        )
        cfg["target_vertical_velocity_at_event"] = float(-gravity * (EVENT_FRAME - launch_frame))
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        event_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(EVENT_FRAME, "possible", cfg), _positions(EVENT_FRAME, "impossible", cfg))
        )
        late_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(25, "possible", cfg), _positions(25, "impossible", cfg))
        )
        if event_delta < 0.065 or late_delta < event_delta * 1.8:
            continue
        cfg.update(
            {
                "event_branch_delta": float(event_delta),
                "frame25_branch_delta": float(late_delta),
                "minimum_mesh_clearance_possible": float(min_p),
                "minimum_mesh_clearance_impossible": float(min_i),
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample gravity candidate: seed={seed}")


def _sample_rebound(seed: int) -> dict:
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        direction = _vector(cfg["motion_dir"]).normalized()
        center = _vector(cfg["target_pos"]) + direction * rng.uniform(-0.08, 0.08)
        touch_distance = float(cfg["actor_extent"] + cfg["target_extent"] + 0.015)
        actor_speed = rng.uniform(0.070, 0.105)
        target_speed = rng.uniform(0.065, 0.100)
        frame19_distance = touch_distance + actor_speed + target_speed
        actor19 = center - direction * (frame19_distance / 2.0)
        target19 = center + direction * (frame19_distance / 2.0)
        rebound_actor_speed = actor_speed * rng.uniform(0.88, 1.00)
        rebound_target_speed = target_speed * rng.uniform(0.88, 1.00)
        cfg.update(
            {
                "triplet_kind": "premature_rebound",
                "subcategory": "premature_rebound_before_contact",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "premature rebound before contact",
                "generator_variant": "ie_motion_anomaly_causal_triplet_v1",
                "motion_mode": "head_on_contact_timing_error",
                "actor_frame19_pos": _as_list(actor19),
                "target_frame19_pos": _as_list(target19),
                "actor_start": _as_list(actor19 - direction * actor_speed * (EVENT_FRAME - 1)),
                "target_start": _as_list(target19 + direction * target_speed * (EVENT_FRAME - 1)),
                "actor_speed": float(actor_speed),
                "target_speed": float(target_speed),
                "rebound_actor_speed": float(rebound_actor_speed),
                "rebound_target_speed": float(rebound_target_speed),
                "possible_contact_frame": EVENT_FRAME,
                "impossible_rebound_frame": EVENT_FRAME,
                "possible_touch_distance": float(touch_distance),
                "impossible_frame20_visible_gap": float(
                    touch_distance + 2.0 * (rebound_actor_speed + rebound_target_speed)
                ),
                "history_head_on_approach": True,
                "history_last_shared_frame": EVENT_FRAME - 1,
                "first_divergence_frame": EVENT_FRAME,
                "branch_position_equal_at_event": False,
                "contact_forbidden_through_frame": NUM_FRAMES - 1,
            }
        )
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        # The possible branch must touch (within the renderer-safe margin) at
        # frame 20; the impossible branch must still have a positive gap.
        possible_contact_clearance = _mesh_clearance(cfg, "possible", EVENT_FRAME)
        impossible_gap = _mesh_clearance(cfg, "impossible", EVENT_FRAME)
        if not (0.006 <= possible_contact_clearance <= 0.025 and impossible_gap >= 0.18):
            continue
        event_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(EVENT_FRAME, "possible", cfg), _positions(EVENT_FRAME, "impossible", cfg))
        )
        late_delta = max(
            (_vector(v) - _vector(w)).length
            for v, w in zip(_positions(25, "possible", cfg), _positions(25, "impossible", cfg))
        )
        # The timing error remains a constant separation after the early
        # reversal; it is intentionally not an ever-growing displacement.
        if event_delta < 0.15 or late_delta < 0.14:
            continue
        cfg.update(
            {
                "event_branch_delta": float(event_delta),
                "frame25_branch_delta": float(late_delta),
                "possible_contact_mesh_clearance_frame20": float(possible_contact_clearance),
                "impossible_gap_mesh_clearance_frame20": float(impossible_gap),
                "minimum_mesh_clearance_possible": float(min_p),
                "minimum_mesh_clearance_impossible": float(min_i),
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample premature-rebound candidate: seed={seed}")


def _sample_motion_anomaly(seed: int) -> dict:
    kind = _KINDS[seed % len(_KINDS)]
    if kind == "remote_oblique_response":
        return _sample_oblique(seed)
    if kind == "gravity_direction_anomaly":
        return _sample_gravity(seed)
    return _sample_rebound(seed)


def _sample_config(category: str, seed: int) -> dict:
    if category == NO_CONTACT_CATEGORY:
        return _sample_motion_anomaly(seed)
    return _ORIGINAL_SAMPLE_CONFIG(category, seed)


def _get_positions(category: str, frame: int, branch: str, cfg: dict):
    if category == NO_CONTACT_CATEGORY and cfg.get("generator_variant") == "ie_motion_anomaly_causal_triplet_v1":
        return _positions(frame, branch, cfg)
    if category == NO_CONTACT_CATEGORY and cfg.get("generator_variant") == "ie_motion_anomaly_causal_triplet_v2_platform_release":
        return _positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def _create_scene_objects(cfg: dict):
    """Create the original scene plus the explicitly documented release block."""

    objects = _ORIGINAL_CREATE_SCENE_OBJECTS(cfg)
    if (
        cfg.get("category") == NO_CONTACT_CATEGORY
        and cfg.get("generator_variant") == "ie_motion_anomaly_causal_triplet_v2_platform_release"
    ):
        # ``create_cube`` starts as a unit cube; setting dimensions and applying
        # the transform gives a static, visible block that supports the target
        # during the common history and has a well-defined edge at launch.
        import bpy

        center = _vector(cfg["platform_center"])
        platform = _base.create_cube(
            "gravity_release_platform",
            1.0,
            (0.22, 0.24, 0.27, 1.0),
        )
        platform.dimensions = (
            float(cfg["platform_length"]),
            float(cfg["platform_width"]),
            float(cfg["platform_height"]),
        )
        platform.location = (
            float(center.x),
            float(center.y),
            float(cfg["platform_bottom_z"] + cfg["platform_height"] / 2.0),
        )
        platform.rotation_euler[2] = math.radians(float(cfg.get("platform_axis_angle_degrees", -90.0)))
        bpy.context.view_layer.objects.active = platform
        platform.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        platform.select_set(False)
    return objects


def _forwarded_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _preflight() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=41042)
    args = parser.parse_args(_forwarded_args())
    counts = {kind: 0 for kind in _KINDS}
    configs = []
    for index in range(args.num_pairs):
        cfg = _sample_motion_anomaly(args.seed + index)
        counts[cfg["triplet_kind"]] += 1
        configs.append(cfg)
    print(
        json.dumps(
            {
                "status": "passed",
                "num_pairs": len(configs),
                "subtype_counts": counts,
                "shared_history_frames": [0, EVENT_FRAME - 1],
                "first_divergence_frame": EVENT_FRAME,
                "min_event_delta": min(c["event_branch_delta"] for c in configs),
                "min_frame25_delta": min(c["frame25_branch_delta"] for c in configs),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    forwarded = _forwarded_args()
    if "--preflight" in forwarded:
        _base.sample_config = _sample_config
        _base.get_positions = _get_positions
        _base.create_scene_objects = _create_scene_objects
        _preflight()
        return
    _v2._refuse_existing_output_root()
    _base.sample_config = _sample_config
    _base.get_positions = _get_positions
    _base.create_scene_objects = _create_scene_objects
    _base.main()


if __name__ == "__main__":
    main()
