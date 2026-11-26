"""
LPWM-style 3D anomaly pair dataset generator.

Generates five anomaly categories, each with paired videos:
- disappearance
- shape_jump
- color_jump
- unexplained_turn
- position_jump

Each pair has:
- possible: physically/causally consistent sequence
- impossible: identical initial conditions, but one anomaly triggered at frame 20

Output (default):
    data/anomaly_3d/
        disappearance/
            scene_0001/
                possible/0000.png ... 0047.png
                impossible/0000.png ... 0047.png
            ...
        shape_jump/
        color_jump/
        unexplained_turn/
        position_jump/
    anomaly_pairs.csv
python data_generat.py --pairs-per-type 100 --anomaly-types position_jump
"""

import os
import csv
import argparse
from copy import deepcopy

# Headless rendering on Linux servers.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import trimesh
import pyrender


IMAGE_SIZE = 128
TOTAL_FRAMES = 48
ANOMALY_FRAME = 20
PAIRS_PER_TYPE_DEFAULT = 100

OUTPUT_ROOT = "data/anomaly_3d"
CSV_PATH = "anomaly_pairs.csv"

ANOMALY_TYPES = [
    "disappearance",
    "shape_jump",
    "color_jump",
    "unexplained_turn",
    "position_jump",
]

SHAPE_NAMES = ["sphere", "cube", "cylinder"]

COLOR_PALETTE = [
    (196, 92, 196),
    (90, 110, 198),
    (92, 178, 112),
    (204, 186, 96),
    (178, 116, 178),
    (98, 170, 170),
    (204, 110, 110),
    (112, 184, 184),
]

STYLE_PRESETS = [
    {
        "bg": (150, 150, 150),
        "floor": (142, 142, 142),
        "grid": (132, 132, 132),
        "ambient": 0.34,
        "eye": np.array([0.0, 0.95, 2.20], dtype=np.float32),
        "target": np.array([0.0, -0.22, 0.0], dtype=np.float32),
        "yfov": 46.0,
        "key_light": 3.0,
        "fill_light": 0.8,
        "rim_light": 0.9,
        "blur": 0.0,
    },
    {
        "bg": (160, 160, 160),
        "floor": (147, 147, 147),
        "grid": (136, 136, 136),
        "ambient": 0.36,
        "eye": np.array([0.20, 1.05, 2.30], dtype=np.float32),
        "target": np.array([0.0, -0.18, 0.0], dtype=np.float32),
        "yfov": 44.0,
        "key_light": 2.6,
        "fill_light": 1.0,
        "rim_light": 0.7,
        "blur": 0.0,
    },
    {
        "bg": (145, 145, 145),
        "floor": (136, 136, 136),
        "grid": (125, 125, 125),
        "ambient": 0.33,
        "eye": np.array([-0.15, 0.90, 2.10], dtype=np.float32),
        "target": np.array([0.0, -0.20, 0.0], dtype=np.float32),
        "yfov": 47.0,
        "key_light": 3.2,
        "fill_light": 0.7,
        "rim_light": 1.1,
        "blur": 0.0,
    },
]

FLOOR_TOP_Y = -0.44
SCENE_X_BOUND = 0.95
SCENE_Z_BOUND = 0.85

POSITION_JUMP_SPEED_SCALE = 2.1
POSITION_JUMP_SPIN_SCALE = 1.6
POSITION_JUMP_DISTANCE_RANGE = (0.75, 1.10)
POSITION_JUMP_FALLBACK_OFFSET = 0.88
POSITION_JUMP_POST_JUMP_SPEED_MIN = 0.10
POSITION_JUMP_POST_JUMP_SPEED_MAX = 0.16


def make_look_at(eye, target, up):
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def make_pose(position, yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rot_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    rot = rot_y @ rot_x @ rot_z

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot
    pose[:3, 3] = position
    return pose


def object_half_height(shape_name, size):
    if shape_name == "sphere":
        return size
    if shape_name == "cube":
        return size * 0.5
    if shape_name == "cylinder":
        return size * 0.75
    raise ValueError(f"Unknown shape: {shape_name}")


def object_radius_xz(shape_name, size):
    """Conservative radius on floor plane to avoid object intersections."""
    if shape_name == "sphere":
        return size
    if shape_name == "cube":
        return float(np.sqrt(2.0) * size * 0.5)
    if shape_name == "cylinder":
        return size * 0.55
    raise ValueError(f"Unknown shape: {shape_name}")


def create_mesh(shape_name, size):
    if shape_name == "sphere":
        return trimesh.creation.icosphere(radius=size, subdivisions=2)
    if shape_name == "cube":
        return trimesh.creation.box(extents=(size, size, size))
    if shape_name == "cylinder":
        return trimesh.creation.cylinder(radius=size * 0.55, height=size * 1.5, sections=24)
    raise ValueError(f"Unknown shape: {shape_name}")


def choose_unique_colors(rng, n):
    idxs = rng.choice(len(COLOR_PALETTE), size=n, replace=False)
    return [COLOR_PALETTE[int(i)] for i in idxs]


def sample_objects(rng):
    n_obj = int(rng.integers(2, 5))
    colors = choose_unique_colors(rng, n_obj)
    objects = []

    moving_count = 0
    for i in range(n_obj):
        shape = SHAPE_NAMES[int(rng.integers(len(SHAPE_NAMES)))]
        size = float(rng.uniform(0.18, 0.28))
        y = FLOOR_TOP_Y + object_half_height(shape, size)

        pos = np.array([
            float(rng.uniform(-0.75, 0.75)),
            y,
            float(rng.uniform(-0.62, 0.62)),
        ], dtype=np.float32)

        is_moving = bool(rng.random() > 0.35)
        if is_moving:
            speed = float(rng.uniform(0.007, 0.018))
            ang = float(rng.uniform(-np.pi, np.pi))
            vel = np.array([speed * np.cos(ang), 0.0, speed * np.sin(ang)], dtype=np.float32)
            moving_count += 1
        else:
            vel = np.zeros(3, dtype=np.float32)

        spin = np.array([
            float(rng.uniform(-0.05, 0.05)),
            float(rng.uniform(-0.09, 0.09)),
            float(rng.uniform(-0.05, 0.05)),
        ], dtype=np.float32)

        objects.append(
            {
                "shape": shape,
                "size": size,
                "color": colors[i],
                "pos0": pos,
                "vel0": vel,
                "spin": spin,
                "moving": is_moving,
            }
        )

    if moving_count == 0:
        # Guarantee at least one moving object.
        idx = int(rng.integers(n_obj))
        speed = float(rng.uniform(0.012, 0.022))
        ang = float(rng.uniform(-np.pi, np.pi))
        objects[idx]["vel0"] = np.array([speed * np.cos(ang), 0.0, speed * np.sin(ang)], dtype=np.float32)
        objects[idx]["moving"] = True

    return objects


def simulate_positions(objects):
    n = len(objects)
    pos_seq = np.zeros((n, TOTAL_FRAMES, 3), dtype=np.float32)
    vel_seq = np.zeros((n, TOTAL_FRAMES, 3), dtype=np.float32)

    for i, obj in enumerate(objects):
        pos_seq[i, 0] = obj["pos0"].copy()
        vel_seq[i, 0] = obj["vel0"].copy()

        for t in range(1, TOTAL_FRAMES):
            pos = obj["pos0"] + obj["vel0"] * t
            pos[1] = obj["pos0"][1]
            pos_seq[i, t] = pos
            vel_seq[i, t] = obj["vel0"]

    return pos_seq, vel_seq


def trajectories_are_valid(objects, pos_seq):
    """Reject samples with boundary exits or object intersections at any frame."""
    n = len(objects)
    radii = np.array([object_radius_xz(o["shape"], o["size"]) for o in objects], dtype=np.float32)

    for t in range(TOTAL_FRAMES):
        pts = pos_seq[:, t, :]
        if np.any(np.abs(pts[:, 0]) > SCENE_X_BOUND) or np.any(np.abs(pts[:, 2]) > SCENE_Z_BOUND):
            return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = float(pts[i, 0] - pts[j, 0])
                dz = float(pts[i, 2] - pts[j, 2])
                d = np.hypot(dx, dz)
                if d < (radii[i] + radii[j]) * 1.08:
                    return False
    return True


def sample_valid_scene(rng, max_trials=400):
    for _ in range(max_trials):
        objects = sample_objects(rng)
        pos_seq, vel_seq = simulate_positions(objects)
        if trajectories_are_valid(objects, pos_seq):
            return objects, pos_seq, vel_seq
    raise RuntimeError("Failed to sample a valid non-overlapping scene after max_trials")


def choose_target_for_anomaly(rng, objects, anomaly_type):
    if anomaly_type == "unexplained_turn":
        moving = [i for i, o in enumerate(objects) if o["moving"]]
        if moving:
            return int(rng.choice(moving))
    return int(rng.integers(len(objects)))


def apply_unexplained_turn(pos_impossible, vel_impossible, objects, target_idx, rng):
    y_fixed = objects[target_idx]["pos0"][1]

    prev_pos = pos_impossible[target_idx, ANOMALY_FRAME - 1].copy()
    prev_vel = vel_impossible[target_idx, ANOMALY_FRAME - 1].copy()
    speed = float(np.linalg.norm(prev_vel[[0, 2]]))

    if speed < 1e-6:
        speed = float(rng.uniform(0.014, 0.022))
        base_ang = float(rng.uniform(-np.pi, np.pi))
    else:
        base_ang = float(np.arctan2(prev_vel[2], prev_vel[0]))

    turn_ang = float(rng.uniform(np.deg2rad(70), np.deg2rad(140)))
    if rng.random() < 0.5:
        turn_ang *= -1.0
    new_ang = base_ang + turn_ang

    vel = np.array([speed * np.cos(new_ang), 0.0, speed * np.sin(new_ang)], dtype=np.float32)

    cur = prev_pos
    for t in range(ANOMALY_FRAME, TOTAL_FRAMES):
        cur = cur + vel
        # Keep anomaly visible but bounded; clamp only for impossible sequence continuity.
        cur[0] = float(np.clip(cur[0], -SCENE_X_BOUND, SCENE_X_BOUND))
        cur[2] = float(np.clip(cur[2], -SCENE_Z_BOUND, SCENE_Z_BOUND))
        cur[1] = y_fixed
        pos_impossible[target_idx, t] = cur
        vel_impossible[target_idx, t] = vel


def apply_position_jump(pos_impossible, vel_impossible, objects, target_idx, rng):
    y_fixed = objects[target_idx]["pos0"][1]
    prev_pos = pos_impossible[target_idx, ANOMALY_FRAME - 1].copy()
    vel = vel_impossible[target_idx, ANOMALY_FRAME - 1].copy()

    base_speed = float(np.linalg.norm(vel[[0, 2]]))
    if base_speed < 1e-6:
        base_speed = float(rng.uniform(0.014, 0.022))

    jump_speed = max(base_speed * 3.0, 0.055)

    radii = np.array([object_radius_xz(o["shape"], o["size"]) for o in objects], dtype=np.float32)
    target_radius = float(radii[target_idx])

    jump_pos = None
    for _ in range(40):
        angle = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(*POSITION_JUMP_DISTANCE_RANGE))
        candidate = prev_pos.copy()
        candidate[0] += distance * np.cos(angle)
        candidate[2] += distance * np.sin(angle)
        candidate[0] = float(np.clip(candidate[0], -SCENE_X_BOUND, SCENE_X_BOUND))
        candidate[2] = float(np.clip(candidate[2], -SCENE_Z_BOUND, SCENE_Z_BOUND))
        candidate[1] = y_fixed

        valid = True
        for i, other in enumerate(objects):
            if i == target_idx:
                continue
            other_pos = pos_impossible[i, ANOMALY_FRAME - 1]
            d = np.hypot(float(candidate[0] - other_pos[0]), float(candidate[2] - other_pos[2]))
            if d < (target_radius + float(radii[i])) * 1.05:
                valid = False
                break
        if valid:
            jump_pos = candidate
            break

    if jump_pos is None:
        jump_pos = prev_pos.copy()
        jump_pos[0] = float(np.clip(jump_pos[0] + POSITION_JUMP_FALLBACK_OFFSET, -SCENE_X_BOUND, SCENE_X_BOUND))
        jump_pos[2] = float(np.clip(jump_pos[2] - POSITION_JUMP_FALLBACK_OFFSET, -SCENE_Z_BOUND, SCENE_Z_BOUND))
        jump_pos[1] = y_fixed

    cur = jump_pos
    jump_dir = float(rng.uniform(-np.pi, np.pi))
    vel = np.array([
        float(rng.uniform(POSITION_JUMP_POST_JUMP_SPEED_MIN, POSITION_JUMP_POST_JUMP_SPEED_MAX)) * np.cos(jump_dir),
        0.0,
        float(rng.uniform(POSITION_JUMP_POST_JUMP_SPEED_MIN, POSITION_JUMP_POST_JUMP_SPEED_MAX)) * np.sin(jump_dir),
    ], dtype=np.float32)

    for t in range(ANOMALY_FRAME, TOTAL_FRAMES):
        if t > ANOMALY_FRAME:
            cur = cur + vel
            wobble = np.array([
                float(rng.uniform(-0.010, 0.010)),
                0.0,
                float(rng.uniform(-0.010, 0.010)),
            ], dtype=np.float32)
            vel = vel + wobble
            speed = float(np.linalg.norm(vel[[0, 2]]))
            if speed > 0.09:
                vel[[0, 2]] *= 0.09 / speed
            cur[0] = float(np.clip(cur[0], -SCENE_X_BOUND, SCENE_X_BOUND))
            cur[2] = float(np.clip(cur[2], -SCENE_Z_BOUND, SCENE_Z_BOUND))
            cur[1] = y_fixed
        pos_impossible[target_idx, t] = cur
        vel_impossible[target_idx, t] = vel


def boost_position_jump_motion(objects):
    boosted = deepcopy(objects)
    for obj in boosted:
        vel = obj["vel0"].copy()
        speed = float(np.linalg.norm(vel[[0, 2]]))
        if speed > 1e-6:
            vel[[0, 2]] *= POSITION_JUMP_SPEED_SCALE
            obj["vel0"] = vel
            obj["moving"] = True
        obj["spin"] = obj["spin"] * POSITION_JUMP_SPIN_SCALE
    return boosted


def pick_alternative_shape(shape_name):
    alternatives = [s for s in SHAPE_NAMES if s != shape_name]
    return alternatives[0] if len(alternatives) == 1 else alternatives[np.random.randint(len(alternatives))]


def build_frame_overrides(objects, anomaly_type, target_idx, rng):
    possible_overrides = [dict() for _ in range(TOTAL_FRAMES)]
    impossible_overrides = [dict() for _ in range(TOTAL_FRAMES)]

    if anomaly_type == "disappearance":
        for t in range(ANOMALY_FRAME, TOTAL_FRAMES):
            impossible_overrides[t][target_idx] = {"visible": False}

    elif anomaly_type == "shape_jump":
        new_shape = pick_alternative_shape(objects[target_idx]["shape"])
        for t in range(ANOMALY_FRAME, TOTAL_FRAMES):
            impossible_overrides[t][target_idx] = {"shape": new_shape}

    elif anomaly_type == "color_jump":
        old_color = objects[target_idx]["color"]
        candidates = [c for c in COLOR_PALETTE if c != old_color]
        new_color = candidates[int(rng.integers(len(candidates)))]
        for t in range(ANOMALY_FRAME, TOTAL_FRAMES):
            impossible_overrides[t][target_idx] = {"color": new_color}

    elif anomaly_type == "unexplained_turn":
        # Motion is modified in positions, no visual override needed.
        pass

    elif anomaly_type == "position_jump":
        # Motion is modified in positions, no visual override needed.
        pass

    return possible_overrides, impossible_overrides


def render_frame(renderer, style, objects, pos_t, t, frame_override, mesh_cache):
    scene = pyrender.Scene(
        bg_color=np.array(style["bg"], dtype=np.uint8),
        ambient_light=[style["ambient"], style["ambient"], style["ambient"]],
    )

    floor_mesh = trimesh.creation.box(extents=(3.2, 0.04, 2.8))
    floor_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[style["floor"][0] / 255.0, style["floor"][1] / 255.0, style["floor"][2] / 255.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    scene.add(
        pyrender.Mesh.from_trimesh(floor_mesh, material=floor_mat, smooth=False),
        pose=np.array(
            [[1, 0, 0, 0], [0, 1, 0, FLOOR_TOP_Y - 0.02], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        ),
    )

    grid_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[style["grid"][0] / 255.0, style["grid"][1] / 255.0, style["grid"][2] / 255.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    for x in np.linspace(-1.4, 1.4, 8):
        line = trimesh.creation.box(extents=(0.006, 0.002, 2.8))
        scene.add(
            pyrender.Mesh.from_trimesh(line, material=grid_mat, smooth=False),
            pose=np.array(
                [[1, 0, 0, x], [0, 1, 0, FLOOR_TOP_Y + 0.002], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            ),
        )
    for z in np.linspace(-1.2, 1.2, 7):
        line = trimesh.creation.box(extents=(3.2, 0.002, 0.006))
        scene.add(
            pyrender.Mesh.from_trimesh(line, material=grid_mat, smooth=False),
            pose=np.array(
                [[1, 0, 0, 0], [0, 1, 0, FLOOR_TOP_Y + 0.002], [0, 0, 1, z], [0, 0, 0, 1]],
                dtype=np.float32,
            ),
        )

    # Lights: key + fill + rim.
    key = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=style["key_light"])
    fill = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=style["fill_light"])
    rim = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=style["rim_light"])
    scene.add(key, pose=np.array([[1, 0, 0, 1.5], [0, 1, 0, 1.8], [0, 0, 1, 2.3], [0, 0, 0, 1]], dtype=np.float32))
    scene.add(fill, pose=np.array([[1, 0, 0, -1.2], [0, 1, 0, 0.8], [0, 0, 1, 2.0], [0, 0, 0, 1]], dtype=np.float32))
    scene.add(rim, pose=np.array([[1, 0, 0, 0.0], [0, 1, 0, 1.9], [0, 0, 1, -1.8], [0, 0, 0, 1]], dtype=np.float32))

    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(style["yfov"]))
    cam_pose = make_look_at(style["eye"], style["target"], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add(cam, pose=cam_pose)

    for i, base in enumerate(objects):
        override = frame_override.get(i, {})
        visible = override.get("visible", True)
        if not visible:
            continue

        shape = override.get("shape", base["shape"])
        color = override.get("color", base["color"])
        size = base["size"]

        cache_key = (shape, round(size, 4))
        if cache_key not in mesh_cache:
            mesh_cache[cache_key] = create_mesh(shape, size)
        tri = mesh_cache[cache_key]

        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.42,
        )
        pm = pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)

        roll = t * base["spin"][0]
        yaw = t * base["spin"][1]
        pitch = t * base["spin"][2]
        pose = make_pose(pos_t[i], yaw, pitch, roll)
        scene.add(pm, pose=pose)

        # Soft contact shadow disk.
        shadow = trimesh.creation.cylinder(radius=0.18, height=0.001, sections=28)
        shadow_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.28, 0.28, 0.28, 0.42],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
        )
        scene.add(
            pyrender.Mesh.from_trimesh(shadow, material=shadow_mat, smooth=False),
            pose=np.array(
                [
                    [1, 0, 0, pos_t[i][0]],
                    [0, 1, 0, FLOOR_TOP_Y + 0.001],
                    [0, 0, 1, pos_t[i][2]],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
        )

    rgb, _ = renderer.render(scene)

    if style["blur"] > 0:
        rgb = cv2.GaussianBlur(rgb, (3, 3), sigmaX=style["blur"])

    # Unsharp mask for clearer boundaries between object and floor.
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=0.9)
    rgb = cv2.addWeighted(rgb, 1.35, blur, -0.35, 0)

    return rgb


def save_pair(
    renderer,
    style,
    objects,
    pos_possible,
    pos_impossible,
    possible_overrides,
    impossible_overrides,
    possible_dir,
    impossible_dir,
):
    mesh_cache = {}
    for t in range(TOTAL_FRAMES):
        frame_p = render_frame(
            renderer,
            style,
            objects,
            pos_possible[:, t, :],
            t,
            possible_overrides[t],
            mesh_cache,
        )
        cv2.imwrite(os.path.join(possible_dir, f"{t:04d}.png"), cv2.cvtColor(frame_p, cv2.COLOR_RGB2BGR))

        frame_i = render_frame(
            renderer,
            style,
            objects,
            pos_impossible[:, t, :],
            t,
            impossible_overrides[t],
            mesh_cache,
        )
        cv2.imwrite(os.path.join(impossible_dir, f"{t:04d}.png"), cv2.cvtColor(frame_i, cv2.COLOR_RGB2BGR))


def generate_dataset(pairs_per_type, seed, anomaly_types=None):
    rng = np.random.default_rng(seed)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(IMAGE_SIZE, IMAGE_SIZE)

    csv_rows = [
        [
            "pair_id",
            "anomaly_type",
            "anomaly_frame",
            "target_object",
            "possible_path",
            "impossible_path",
        ]
    ]

    pair_id = 0

    selected_types = ANOMALY_TYPES if anomaly_types is None else anomaly_types

    for anomaly_type in selected_types:
        type_root = os.path.join(OUTPUT_ROOT, anomaly_type)
        os.makedirs(type_root, exist_ok=True)

        for i in range(pairs_per_type):
            pair_id += 1
            scene_name = f"scene_{i + 1:04d}"
            scene_root = os.path.join(type_root, scene_name)
            possible_dir = os.path.join(scene_root, "possible")
            impossible_dir = os.path.join(scene_root, "impossible")
            os.makedirs(possible_dir, exist_ok=True)
            os.makedirs(impossible_dir, exist_ok=True)

            style = deepcopy(STYLE_PRESETS[int(rng.integers(len(STYLE_PRESETS)))])
            objects, pos_possible, vel_possible = sample_valid_scene(rng)

            if anomaly_type == "position_jump":
                objects = boost_position_jump_motion(objects)
                pos_possible, vel_possible = simulate_positions(objects)

            pos_impossible = pos_possible.copy()
            vel_impossible = vel_possible.copy()

            target_idx = choose_target_for_anomaly(rng, objects, anomaly_type)
            if anomaly_type == "unexplained_turn":
                apply_unexplained_turn(pos_impossible, vel_impossible, objects, target_idx, rng)
            elif anomaly_type == "position_jump":
                apply_position_jump(pos_impossible, vel_impossible, objects, target_idx, rng)

            possible_overrides, impossible_overrides = build_frame_overrides(
                objects, anomaly_type, target_idx, rng
            )

            save_pair(
                renderer,
                style,
                objects,
                pos_possible,
                pos_impossible,
                possible_overrides,
                impossible_overrides,
                possible_dir,
                impossible_dir,
            )

            csv_rows.append(
                [
                    str(pair_id),
                    anomaly_type,
                    str(ANOMALY_FRAME),
                    str(target_idx),
                    os.path.abspath(possible_dir),
                    os.path.abspath(impossible_dir),
                ]
            )

            print(
                f"[{anomaly_type}] generated pair {i + 1}/{pairs_per_type} -> {scene_name}",
                flush=True,
            )

    renderer.delete()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(
        f"Generated {pair_id} pairs "
        f"({pairs_per_type} per type x {len(selected_types)} types). "
        f"CSV saved to {CSV_PATH}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LPWM-style 3D anomaly pair dataset")
    parser.add_argument("--pairs-per-type", type=int, default=PAIRS_PER_TYPE_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--anomaly-types",
        nargs="+",
        choices=ANOMALY_TYPES,
        default=None,
        help="Subset of anomaly types to generate; default is all types",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset(args.pairs_per_type, args.seed, args.anomaly_types)


if __name__ == "__main__":
    main()
