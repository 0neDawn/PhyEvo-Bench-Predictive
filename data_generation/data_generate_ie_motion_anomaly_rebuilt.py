"""Rebuild 30 additive IE Motion Anomaly replacement candidates.

The generated set contains ten examples of each of three causal anomaly types:

``remote_oblique_response``
    Two objects remain visibly separated.  The normal branch continues its
    motion; the anomalous branch changes direction as if a collision happened
    despite a large screen-space gap.

``gravity_direction_anomaly``
    A target begins on a visible support, leaves its edge during the common
    history, and is already descending at frame 20 in the normal branch.  The
    anomaly reverses gravity or adds a lateral gravity component.  Supports,
    headings, shapes, cameras, and lighting vary across examples.

``no_contact_premature_rebound``
    Two objects approach and would contact only at frame 23 or 24 in the
    normal branch.  The anomalous branch reverses at frame 20, several frames
    before contact, while a large silhouette gap is still visible.  This is
    deliberately different from the rejected construction in which both
    branches appeared to touch at the event frame.

Frames 0--19 are shared exactly and frame 20 is the first branch divergence.
The output root must not already exist, so earlier generators and datasets are
never overwritten.

Example::

    blender -b --python data_generate_ie_motion_anomaly_rebuilt.py -- \
      --out_root data/process_obj3d/motion_anomaly_replacement_candidates_rebuilt \
      --category no_contact_motion --num_pairs 30 --seed 61042 --img_size 128

Preflight::

    blender -b --python data_generate_ie_motion_anomaly_rebuilt.py -- \
      --preflight --num_pairs 30 --seed 61042
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
import data_generate_ie_motion_anomaly_causal_corrected as _previous


_base = _previous._base
_v2 = _previous._v2
EVENT_FRAME = _previous.EVENT_FRAME
NUM_FRAMES = _previous.NUM_FRAMES
NO_CONTACT_CATEGORY = _previous.NO_CONTACT_CATEGORY
_ORIGINAL_SAMPLE_CONFIG = _previous._ORIGINAL_SAMPLE_CONFIG
_ORIGINAL_GET_POSITIONS = _previous._ORIGINAL_GET_POSITIONS
_ORIGINAL_CREATE_SCENE_OBJECTS = _previous._ORIGINAL_CREATE_SCENE_OBJECTS

_KINDS = (
    "remote_oblique_response",
    "gravity_direction_anomaly",
    "no_contact_premature_rebound",
)

_SHAPE_PAIRS = (
    ("cube", "sphere"),
    ("sphere", "cylinder"),
    ("cylinder", "cube"),
    ("sphere", "sphere"),
    ("cube", "cylinder"),
    ("cylinder", "sphere"),
    ("cube", "cube"),
    ("cylinder", "cylinder"),
    ("sphere", "cube"),
    ("cylinder", "cube"),
)

_REBOUND_DIRECTIONS_DEGREES = (
    0.0,
    12.0,
    -12.0,
    24.0,
    -24.0,
    180.0,
    168.0,
    192.0,
    156.0,
    204.0,
)

_REMOTE_DIRECTIONS_DEGREES = (
    -82.0,
    -98.0,
    -72.0,
    -108.0,
    -88.0,
    -92.0,
    -76.0,
    -104.0,
    -68.0,
    -112.0,
)

_GRAVITY_DIRECTIONS_DEGREES = (
    -90.0,
    -78.0,
    -102.0,
    -70.0,
    -110.0,
    -84.0,
    -96.0,
    -74.0,
    -106.0,
    -90.0,
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


def _direction(degrees: float) -> Vector:
    radians = math.radians(degrees)
    return Vector((math.cos(radians), math.sin(radians), 0.0)).normalized()


def _scene_index(seed: int) -> int:
    return (seed // len(_KINDS)) % 10


def _vertical_extent(kind: str, size: float) -> float:
    if kind == "cube":
        return float(size) * 0.50
    if kind == "sphere":
        return float(size) * 0.52
    if kind == "cylinder":
        return float(size) * 0.475
    raise ValueError(kind)


def _rotation_at_frame(role: str, frame: int) -> float:
    if role == "actor":
        return 0.045 * frame
    if role == "target":
        return 0.025 * frame
    return 0.0


def _axis_radius(
    kind: str,
    size: float,
    role: str,
    frame: int,
    axis: Vector,
) -> float:
    """Exact horizontal support radius for the rendered primitive."""

    axis = Vector((float(axis.x), float(axis.y), 0.0))
    if axis.length == 0:
        return 0.0
    axis.normalize()
    if kind == "cube":
        angle = _rotation_at_frame(role, frame)
        basis_x = Vector((math.cos(angle), math.sin(angle), 0.0))
        basis_y = Vector((-math.sin(angle), math.cos(angle), 0.0))
        half = float(size) * 0.50
        return half * (abs(float(axis.dot(basis_x))) + abs(float(axis.dot(basis_y))))
    if kind == "sphere":
        return float(size) * 0.52
    if kind == "cylinder":
        return float(size) * 0.42
    raise ValueError(kind)


def _role_radius(cfg: dict, role: str, frame: int, axis: Vector) -> float:
    return _axis_radius(
        cfg[f"{role}_shape"],
        float(cfg[f"{role}_size"]),
        role,
        frame,
        axis,
    )


def _floor_center_z(cfg: dict, role: str) -> float:
    return -0.42 + _vertical_extent(
        cfg[f"{role}_shape"], float(cfg[f"{role}_size"])
    ) + 0.015


def _set_scene_shapes(cfg: dict, rng: random.Random, seed: int) -> None:
    actor_shape, target_shape = _SHAPE_PAIRS[_scene_index(seed)]
    cfg["actor_shape"] = actor_shape
    cfg["target_shape"] = target_shape
    cfg["actor_size"] = rng.uniform(0.74, 0.86)
    cfg["target_size"] = rng.uniform(0.74, 0.86)
    cfg["actor_extent"] = float(
        _base.shape_extent(actor_shape, cfg["actor_size"])
    )
    cfg["target_extent"] = float(
        _base.shape_extent(target_shape, cfg["target_size"])
    )
    cfg["contact_gap"] = float(cfg["actor_extent"] + cfg["target_extent"] + 0.05)
    cfg["distractors"] = []


def _base_cfg(seed: int, trial_seed: int) -> tuple[dict, random.Random]:
    cfg = _ORIGINAL_SAMPLE_CONFIG(NO_CONTACT_CATEGORY, trial_seed)
    rng = random.Random(trial_seed + 104729)
    _set_scene_shapes(cfg, rng, seed)
    return cfg, rng


def _oblique_positions(frame: int, branch: str, cfg: dict):
    actor19 = _vector(cfg["actor_frame19_pos"])
    target19 = _vector(cfg["target_frame19_pos"])
    actor_velocity = _vector(cfg["actor_velocity"])
    target_velocity = _vector(cfg["target_velocity"])
    if frame < EVENT_FRAME:
        delta = frame - (EVENT_FRAME - 1)
        return (
            _tuple3(actor19 + actor_velocity * delta),
            _tuple3(target19 + target_velocity * delta),
        )
    delta = frame - (EVENT_FRAME - 1)
    if branch == "possible":
        actor_step = actor_velocity
        target_step = target_velocity
    else:
        angle = float(cfg["deflection_angle"])
        side = float(cfg["deflection_side"])
        actor_step = _rotate_xy(actor_velocity, -side * angle)
        target_step = _rotate_xy(target_velocity, side * angle)
    return (
        _tuple3(actor19 + actor_step * delta),
        _tuple3(target19 + target_step * delta),
    )


def _gravity_target(frame: int, branch: str, cfg: dict) -> Vector:
    target0 = _vector(cfg["target_frame0_pos"])
    velocity = _vector(cfg["target_horizontal_velocity"])
    launch_frame = int(cfg["launch_frame"])
    if frame < launch_frame:
        target = target0 + velocity * float(frame)
    else:
        elapsed = float(frame - launch_frame)
        target = target0 + velocity * float(frame)
        target.z -= 0.5 * float(cfg["gravity"]) * elapsed * elapsed
        target.z = max(float(cfg["target_floor_z"]), float(target.z))
    if frame >= EVENT_FRAME and branch == "impossible":
        steps = frame - (EVENT_FRAME - 1)
        impulse = float(cfg["anomaly_impulse_per_frame"])
        if cfg["gravity_anomaly_mode"] == "upward":
            target.z += impulse * steps
        else:
            target += _vector(cfg["gravity_lateral_dir"]) * impulse * steps
    return target


def _gravity_positions(frame: int, branch: str, cfg: dict):
    actor = _vector(cfg["actor_frame0_pos"]) + _vector(cfg["actor_velocity"]) * float(frame)
    return _tuple3(actor), _tuple3(_gravity_target(frame, branch, cfg))


def _rebound_positions(frame: int, branch: str, cfg: dict):
    """Normal contact is delayed; the anomalous reversal starts at frame 20."""

    actor19 = _vector(cfg["actor_frame19_pos"])
    target19 = _vector(cfg["target_frame19_pos"])
    direction = _vector(cfg["motion_dir"]).normalized()
    actor_speed = float(cfg["actor_speed"])
    target_speed = float(cfg["target_speed"])
    contact_frame = int(cfg["possible_contact_frame"])
    if frame <= EVENT_FRAME - 1:
        delta = frame - (EVENT_FRAME - 1)
        return (
            _tuple3(actor19 + direction * actor_speed * delta),
            _tuple3(target19 - direction * target_speed * delta),
        )
    if branch == "impossible":
        steps = frame - (EVENT_FRAME - 1)
        return (
            _tuple3(actor19 - direction * float(cfg["rebound_actor_speed"]) * steps),
            _tuple3(target19 + direction * float(cfg["rebound_target_speed"]) * steps),
        )
    approach_steps = min(frame, contact_frame) - (EVENT_FRAME - 1)
    actor = actor19 + direction * actor_speed * approach_steps
    target = target19 - direction * target_speed * approach_steps
    if frame > contact_frame:
        separation_steps = frame - contact_frame
        actor -= direction * float(cfg["rebound_actor_speed"]) * separation_steps
        target += direction * float(cfg["rebound_target_speed"]) * separation_steps
    return _tuple3(actor), _tuple3(target)


def _positions(frame: int, branch: str, cfg: dict):
    kind = cfg["triplet_kind"]
    if kind == "remote_oblique_response":
        return _oblique_positions(frame, branch, cfg)
    if kind == "gravity_direction_anomaly":
        return _gravity_positions(frame, branch, cfg)
    if kind == "no_contact_premature_rebound":
        return _rebound_positions(frame, branch, cfg)
    raise ValueError(kind)


def _pair_clearance(cfg: dict, branch: str, frame: int) -> float:
    actor_value, target_value = _positions(frame, branch, cfg)
    actor = _vector(actor_value)
    target = _vector(target_value)
    delta = target - actor
    horizontal = Vector((delta.x, delta.y, 0.0))
    if horizontal.length == 0:
        return -float("inf")
    axis = horizontal.normalized()
    return float(
        horizontal.length
        - _role_radius(cfg, "actor", frame, axis)
        - _role_radius(cfg, "target", frame, axis)
    )


def _screen_x_gap(cfg: dict, branch: str, frame: int) -> float:
    actor_value, target_value = _positions(frame, branch, cfg)
    actor = _vector(actor_value)
    target = _vector(target_value)
    x_axis = Vector((1.0, 0.0, 0.0))
    return float(
        abs(target.x - actor.x)
        - _role_radius(cfg, "actor", frame, x_axis)
        - _role_radius(cfg, "target", frame, x_axis)
    )


def _geometry_audit(cfg: dict, branch: str) -> tuple[bool, float]:
    minimum = float("inf")
    contact_frame = int(cfg.get("possible_contact_frame", -1))
    for frame in range(NUM_FRAMES):
        actor_value, target_value = _positions(frame, branch, cfg)
        actor = _vector(actor_value)
        target = _vector(target_value)
        for role, position in (("actor", actor), ("target", target)):
            vertical = _vertical_extent(
                cfg[f"{role}_shape"], float(cfg[f"{role}_size"])
            )
            if position.z - vertical < -0.425:
                return False, minimum
            y_radius = _role_radius(cfg, role, frame, Vector((0.0, 1.0, 0.0)))
            if position.y + y_radius > 3.15:
                return False, minimum
        clearance = _pair_clearance(cfg, branch, frame)
        minimum = min(minimum, clearance)
        physical_contact = (
            cfg["triplet_kind"] == "no_contact_premature_rebound"
            and branch == "possible"
            and frame == contact_frame
        )
        if physical_contact:
            if clearance < 0.008 or clearance > 0.045:
                return False, minimum
        elif clearance < 0.055:
            return False, minimum

        if cfg["triplet_kind"] == "gravity_direction_anomaly":
            support_center = _vector(cfg["support_center"])
            axis = _vector(cfg["motion_dir"]).normalized()
            lateral = _vector(cfg["lateral_dir"]).normalized()
            relative = target - support_center
            along = float(relative.dot(axis))
            across = abs(float(relative.dot(lateral)))
            if frame <= int(cfg["launch_frame"]):
                if along > float(cfg["support_edge_distance"]) + 1e-4:
                    return False, minimum
                if across > float(cfg["support_width"]) / 2.0:
                    return False, minimum
                target_vertical = _vertical_extent(
                    cfg["target_shape"], float(cfg["target_size"])
                )
                expected_bottom = float(cfg["support_top_z"]) + 0.015
                if abs(float(target.z - target_vertical - expected_bottom)) > 1e-4:
                    return False, minimum
            elif along <= float(cfg["support_edge_distance"]) - 0.02:
                return False, minimum
    return True, minimum


def _visible_window_audit(cfg: dict) -> bool:
    frames = [12, 15, 19, 20, 21, 22, 23, 24, 25]
    if cfg["triplet_kind"] == "gravity_direction_anomaly":
        frames = [0, 5, 10, 15, 19, 20, 21, 23, 25]
    for frame in frames:
        for branch in ("possible", "impossible"):
            for position_value in _positions(frame, branch, cfg):
                position = _vector(position_value)
                if abs(position.x) > 1.62:
                    return False
                if position.y < -2.75 or position.y > 2.45:
                    return False
                if position.z < -0.10 or position.z > 2.45:
                    return False
    return True


def _shared_history_audit(cfg: dict) -> bool:
    return all(
        _positions(frame, "possible", cfg) == _positions(frame, "impossible", cfg)
        for frame in range(EVENT_FRAME)
    ) and _positions(EVENT_FRAME, "possible", cfg) != _positions(EVENT_FRAME, "impossible", cfg)


def _sample_oblique(seed: int) -> dict:
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        angle_degrees = _REMOTE_DIRECTIONS_DEGREES[_scene_index(seed)]
        direction = _direction(angle_degrees)
        lateral = Vector((-direction.y, direction.x, 0.0)).normalized()
        cfg["motion_dir"] = _as_list(direction)
        cfg["lateral_dir"] = _as_list(lateral)
        z_actor = _floor_center_z(cfg, "actor")
        z_target = _floor_center_z(cfg, "target")
        center = _vector(cfg["target_pos"]) + Vector((0.0, -1.80, 0.0))
        center.x += rng.uniform(-0.08, 0.08)
        side = float(rng.choice((-1, 1)))
        radius_actor = _role_radius(cfg, "actor", EVENT_FRAME - 1, lateral)
        radius_target = _role_radius(cfg, "target", EVENT_FRAME - 1, lateral)
        lateral_distance = radius_actor + radius_target + rng.uniform(0.72, 0.92)
        longitudinal_distance = rng.uniform(0.62, 0.82)
        actor19 = center - direction * longitudinal_distance / 2.0 - lateral * side * lateral_distance / 2.0
        target19 = center + direction * longitudinal_distance / 2.0 + lateral * side * lateral_distance / 2.0
        actor19.z = z_actor
        target19.z = z_target
        actor_speed = rng.uniform(0.190, 0.215)
        target_speed = rng.uniform(0.105, 0.135)
        actor_velocity = direction * actor_speed
        target_velocity = direction * target_speed
        deflection = math.radians(rng.uniform(45.0, 50.0))
        cfg.update(
            {
                "triplet_kind": "remote_oblique_response",
                "subcategory": "remote_oblique_response_without_contact",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "remote oblique response",
                "generator_variant": "ie_motion_anomaly_rebuilt_remote",
                "motion_mode": "varied_remote_parallel_approach_with_screen_gap",
                "actor_frame19_pos": _as_list(actor19),
                "target_frame19_pos": _as_list(target19),
                "actor_velocity": _as_list(actor_velocity),
                "target_velocity": _as_list(target_velocity),
                "actor_start": _as_list(actor19 - actor_velocity * 19.0),
                "target_start": _as_list(target19 - target_velocity * 19.0),
                "actor_speed": float(actor_speed),
                "target_speed": float(target_speed),
                "deflection_angle": float(deflection),
                "deflection_angle_degrees": float(math.degrees(deflection)),
                "deflection_side": side,
                "approach_direction_degrees": float(angle_degrees),
                "history_last_shared_frame": 19,
                "first_divergence_frame": 20,
                "contact_forbidden_through_frame": 47,
                "speed_magnitude_matched": True,
            }
        )
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        event_window_gap = min(
            _screen_x_gap(cfg, branch, frame)
            for frame in range(19, 26)
            for branch in ("possible", "impossible")
        )
        if event_window_gap < 0.55:
            continue
        event_delta = max(
            (_vector(a) - _vector(b)).length
            for a, b in zip(_positions(20, "possible", cfg), _positions(20, "impossible", cfg))
        )
        frame25_delta = max(
            (_vector(a) - _vector(b)).length
            for a, b in zip(_positions(25, "possible", cfg), _positions(25, "impossible", cfg))
        )
        if event_delta < 0.14 or frame25_delta < 0.78:
            continue
        cfg.update(
            {
                "event_branch_delta": float(event_delta),
                "frame25_branch_delta": float(frame25_delta),
                "minimum_screen_edge_gap_frames19_25_world": float(event_window_gap),
                "minimum_screen_edge_gap_frames19_25_pixels_estimated": float(event_window_gap * 128.0 / 3.65),
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample rebuilt remote candidate: seed={seed}")


def _sample_gravity(seed: int) -> dict:
    support_kinds = ("block", "cylinder", "slab")
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        angle_degrees = _GRAVITY_DIRECTIONS_DEGREES[_scene_index(seed)]
        direction = _direction(angle_degrees)
        lateral = Vector((-direction.y, direction.x, 0.0)).normalized()
        cfg["motion_dir"] = _as_list(direction)
        cfg["lateral_dir"] = _as_list(lateral)
        support_kind = support_kinds[_scene_index(seed) % len(support_kinds)]
        support_center = _vector(cfg["target_pos"])
        support_center += lateral * rng.uniform(-0.08, 0.08)
        support_center += direction * rng.uniform(-0.06, 0.06)
        support_bottom_z = -0.42
        if support_kind == "slab":
            support_height = rng.uniform(0.62, 0.74)
            support_length = rng.uniform(2.45, 2.75)
            support_width = rng.uniform(1.12, 1.36)
            support_radius = None
        elif support_kind == "block":
            support_height = rng.uniform(0.78, 0.96)
            support_length = rng.uniform(2.05, 2.38)
            support_width = rng.uniform(1.00, 1.24)
            support_radius = None
        else:
            support_height = rng.uniform(0.70, 0.90)
            support_radius = rng.uniform(0.90, 1.04)
            support_length = 2.0 * support_radius
            support_width = 2.0 * support_radius
        support_top_z = support_bottom_z + support_height
        target_motion_radius = _role_radius(cfg, "target", 0, direction)
        if support_kind == "cylinder":
            support_edge_distance = float(support_radius - target_motion_radius - 0.06)
        else:
            support_edge_distance = support_length / 2.0 - target_motion_radius - 0.06
        if support_edge_distance < 0.38:
            continue
        launch_frame = rng.randint(8, 11)
        horizontal_speed = rng.uniform(0.067, 0.084)
        horizontal_velocity = direction * horizontal_speed
        target0 = support_center + direction * (support_edge_distance - horizontal_speed * launch_frame)
        target0.z = support_top_z + _vertical_extent(cfg["target_shape"], cfg["target_size"]) + 0.015
        actor_radius = _role_radius(cfg, "actor", 0, lateral)
        actor0 = support_center + lateral * (support_width / 2.0 + actor_radius + 0.32)
        actor0 -= direction * rng.uniform(0.35, 0.55)
        actor0.z = _floor_center_z(cfg, "actor")
        actor_velocity = direction * rng.uniform(0.014, 0.024)
        gravity = rng.uniform(0.0062, 0.0078)
        anomaly_mode = "upward" if _scene_index(seed) % 2 == 0 else "lateral"
        anomaly_impulse = rng.uniform(0.160, 0.190)
        target_floor_z = _floor_center_z(cfg, "target")
        cfg.update(
            {
                "triplet_kind": "gravity_direction_anomaly",
                "subcategory": f"gravity_direction_{anomaly_mode}",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "gravity direction anomaly",
                "generator_variant": "ie_motion_anomaly_rebuilt_support_release",
                "motion_mode": "varied_support_release_with_normal_descent",
                "support_kind": support_kind,
                "support_center": _as_list(support_center),
                "support_length": float(support_length),
                "support_width": float(support_width),
                "support_height": float(support_height),
                "support_radius": None if support_radius is None else float(support_radius),
                "support_bottom_z": float(support_bottom_z),
                "support_top_z": float(support_top_z),
                "support_edge_distance": float(support_edge_distance),
                "support_axis_angle_degrees": float(angle_degrees),
                "platform_center": _as_list(support_center),
                "platform_length": float(support_length),
                "platform_width": float(support_width),
                "platform_height": float(support_height),
                "platform_bottom_z": float(support_bottom_z),
                "platform_top_z": float(support_top_z),
                "platform_edge_distance": float(support_edge_distance),
                "actor_frame0_pos": _as_list(actor0),
                "actor_velocity": _as_list(actor_velocity),
                "actor_start": _as_list(actor0),
                "target_frame0_pos": _as_list(target0),
                "target_start": _as_list(target0),
                "target_horizontal_velocity": _as_list(horizontal_velocity),
                "target_floor_z": float(target_floor_z),
                "launch_frame": int(launch_frame),
                "gravity": float(gravity),
                "gravity_anomaly_mode": anomaly_mode,
                "anomaly_impulse_per_frame": float(anomaly_impulse),
                "gravity_lateral_dir": [1.0, 0.0, 0.0],
                "approach_direction_degrees": float(angle_degrees),
                "history_target_supported_on_varied_support": True,
                "history_support_release_is_shared": True,
                "possible_branch_descends_at_event": True,
                "history_last_shared_frame": 19,
                "first_divergence_frame": 20,
                "contact_forbidden_through_frame": 47,
            }
        )
        normal19 = _gravity_target(19, "possible", cfg)
        normal20 = _gravity_target(20, "possible", cfg)
        if not (
            normal20.z < normal19.z - 0.055
            and normal20.z > target_floor_z + 0.055
            and normal20.z < target0.z - 0.18
        ):
            continue
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        event_delta = float((_gravity_target(20, "possible", cfg) - _gravity_target(20, "impossible", cfg)).length)
        frame25_delta = float((_gravity_target(25, "possible", cfg) - _gravity_target(25, "impossible", cfg)).length)
        if event_delta < 0.155 or frame25_delta < 0.93:
            continue
        cfg.update(
            {
                "target_vertical_velocity_at_event": float(-gravity * (20 - launch_frame)),
                "normal_frame19_z": float(normal19.z),
                "normal_frame20_z": float(normal20.z),
                "normal_event_descent_world": float(normal19.z - normal20.z),
                "event_branch_delta": event_delta,
                "frame25_branch_delta": frame25_delta,
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample rebuilt gravity candidate: seed={seed}")


def _sample_rebound(seed: int) -> dict:
    for attempt in range(4096):
        trial_seed = seed + attempt * 7919
        cfg, rng = _base_cfg(seed, trial_seed)
        angle_degrees = _REBOUND_DIRECTIONS_DEGREES[_scene_index(seed)]
        direction = _direction(angle_degrees)
        lateral = Vector((-direction.y, direction.x, 0.0)).normalized()
        cfg["motion_dir"] = _as_list(direction)
        cfg["lateral_dir"] = _as_list(lateral)
        contact_frame = 23 + (_scene_index(seed) % 2)
        actor_speed = rng.uniform(0.078, 0.098)
        target_speed = rng.uniform(0.074, 0.094)
        rebound_actor_speed = actor_speed * rng.uniform(0.90, 1.02)
        rebound_target_speed = target_speed * rng.uniform(0.90, 1.02)
        touch_distance = (
            _role_radius(cfg, "actor", contact_frame, direction)
            + _role_radius(cfg, "target", contact_frame, direction)
            + 0.022
        )
        frame19_distance = touch_distance + (actor_speed + target_speed) * (contact_frame - 19)
        center = _vector(cfg["target_pos"])
        center.x += rng.uniform(-0.08, 0.08)
        center.y += rng.uniform(-0.20, 0.10)
        actor19 = center - direction * frame19_distance / 2.0
        target19 = center + direction * frame19_distance / 2.0
        actor19.z = _floor_center_z(cfg, "actor")
        target19.z = _floor_center_z(cfg, "target")
        cfg.update(
            {
                "triplet_kind": "no_contact_premature_rebound",
                "subcategory": "clearly_separated_premature_rebound",
                "paper_category": "Motion Anomaly",
                "paper_subcategory": "premature rebound before contact",
                "generator_variant": "ie_motion_anomaly_rebuilt_delayed_contact",
                "motion_mode": "normal_contact_at_23_or_24_anomalous_reversal_at_20",
                "actor_frame19_pos": _as_list(actor19),
                "target_frame19_pos": _as_list(target19),
                "actor_start": _as_list(actor19 - direction * actor_speed * 19.0),
                "target_start": _as_list(target19 + direction * target_speed * 19.0),
                "actor_speed": float(actor_speed),
                "target_speed": float(target_speed),
                "rebound_actor_speed": float(rebound_actor_speed),
                "rebound_target_speed": float(rebound_target_speed),
                "possible_contact_frame": int(contact_frame),
                "impossible_rebound_frame": 20,
                "approach_direction_degrees": float(angle_degrees),
                "history_head_on_approach": True,
                "normal_branch_contacts_at_later_frame": True,
                "impossible_branch_never_contacts": True,
                "history_last_shared_frame": 19,
                "first_divergence_frame": 20,
                "contact_forbidden_in_impossible_through_frame": 47,
            }
        )
        ok_p, min_p = _geometry_audit(cfg, "possible")
        ok_i, min_i = _geometry_audit(cfg, "impossible")
        if not ok_p or not ok_i or not _visible_window_audit(cfg) or not _shared_history_audit(cfg):
            continue
        shared_gap19 = _screen_x_gap(cfg, "possible", 19)
        normal_gap20 = _screen_x_gap(cfg, "possible", 20)
        impossible_gap20 = _screen_x_gap(cfg, "impossible", 20)
        impossible_min_gap = min(_screen_x_gap(cfg, "impossible", frame) for frame in range(19, 26))
        contact_clearance = _pair_clearance(cfg, "possible", contact_frame)
        if (
            shared_gap19 < 0.42
            or normal_gap20 < 0.28
            or impossible_gap20 < 0.65
            or impossible_min_gap < 0.42
            or not (0.008 <= contact_clearance <= 0.045)
        ):
            continue
        event_delta = max(
            (_vector(a) - _vector(b)).length
            for a, b in zip(_positions(20, "possible", cfg), _positions(20, "impossible", cfg))
        )
        frame25_delta = max(
            (_vector(a) - _vector(b)).length
            for a, b in zip(_positions(25, "possible", cfg), _positions(25, "impossible", cfg))
        )
        if event_delta < 0.145 or frame25_delta < 0.55:
            continue
        cfg.update(
            {
                "event_branch_delta": float(event_delta),
                "frame25_branch_delta": float(frame25_delta),
                "shared_screen_edge_gap_frame19_world": float(shared_gap19),
                "normal_screen_edge_gap_frame20_world": float(normal_gap20),
                "impossible_screen_edge_gap_frame20_world": float(impossible_gap20),
                "impossible_screen_edge_gap_frame20_pixels_estimated": float(impossible_gap20 * 128.0 / 3.65),
                "impossible_min_screen_edge_gap_frames19_25_world": float(impossible_min_gap),
                "possible_contact_clearance": float(contact_clearance),
                "minimum_pair_clearance_possible": float(min_p),
                "minimum_pair_clearance_impossible": float(min_i),
            }
        )
        return cfg
    raise RuntimeError(f"Could not sample rebuilt rebound candidate: seed={seed}")


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
    if category == NO_CONTACT_CATEGORY and str(cfg.get("generator_variant", "")).startswith("ie_motion_anomaly_rebuilt_"):
        return _positions(frame, branch, cfg)
    return _ORIGINAL_GET_POSITIONS(category, frame, branch, cfg)


def _create_scene_objects(cfg: dict):
    objects = _ORIGINAL_CREATE_SCENE_OBJECTS(cfg)
    if cfg.get("generator_variant") != "ie_motion_anomaly_rebuilt_support_release":
        return objects
    import bpy

    center = _vector(cfg["support_center"])
    color = (0.22, 0.24, 0.27, 1.0)
    if cfg["support_kind"] == "cylinder":
        support = _base.create_cylinder(
            "gravity_release_support",
            float(cfg["support_radius"]),
            float(cfg["support_height"]),
            color,
        )
    else:
        support = _base.create_cube("gravity_release_support", 1.0, color)
        support.dimensions = (
            float(cfg["support_length"]),
            float(cfg["support_width"]),
            float(cfg["support_height"]),
        )
        support.rotation_euler[2] = math.radians(float(cfg["support_axis_angle_degrees"]))
    support.location = (
        float(center.x),
        float(center.y),
        float(cfg["support_bottom_z"] + cfg["support_height"] / 2.0),
    )
    bpy.context.view_layer.objects.active = support
    support.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    support.select_set(False)
    return objects


def _forwarded_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def _preflight() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--num_pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=61042)
    args = parser.parse_args(_forwarded_args())
    configs = [_sample_motion_anomaly(args.seed + index) for index in range(args.num_pairs)]
    counts = {kind: sum(cfg["triplet_kind"] == kind for cfg in configs) for kind in _KINDS}
    rebound = [cfg for cfg in configs if cfg["triplet_kind"] == "no_contact_premature_rebound"]
    gravity = [cfg for cfg in configs if cfg["triplet_kind"] == "gravity_direction_anomaly"]
    print(
        json.dumps(
            {
                "status": "passed",
                "num_pairs": len(configs),
                "subtype_counts": counts,
                "shared_history_frames": [0, 19],
                "first_divergence_frame": 20,
                "minimum_rebound_impossible_frame20_screen_gap_world": min(cfg["impossible_screen_edge_gap_frame20_world"] for cfg in rebound),
                "minimum_rebound_impossible_frame20_screen_gap_pixels_estimated": min(cfg["impossible_screen_edge_gap_frame20_pixels_estimated"] for cfg in rebound),
                "normal_contact_frames": sorted({cfg["possible_contact_frame"] for cfg in rebound}),
                "gravity_support_kinds": sorted({cfg["support_kind"] for cfg in gravity}),
                "all_gravity_normal_branches_descend_at_frame20": all(cfg["possible_branch_descends_at_event"] for cfg in gravity),
                "minimum_event_branch_delta": min(cfg["event_branch_delta"] for cfg in configs),
                "minimum_frame25_branch_delta": min(cfg["frame25_branch_delta"] for cfg in configs),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    forwarded = _forwarded_args()
    _base.sample_config = _sample_config
    _base.get_positions = _get_positions
    _base.create_scene_objects = _create_scene_objects
    if "--preflight" in forwarded:
        _preflight()
        return
    _v2._refuse_existing_output_root()
    _base.main()


if __name__ == "__main__":
    main()
