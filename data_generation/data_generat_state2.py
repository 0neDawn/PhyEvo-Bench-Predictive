"""
Blender state-persistence pair generator using the visual style of
data_generat_process.py.

Categories:
- sudden_disappearance: target is visible in history, then disappears.
- identity_change: the same target keeps moving, but its appearance changes.
- occluded_wrong_reappearance: target was seen, is occluded, then reappears
  incorrectly.
- reappearance_trajectory_anomaly: target was seen, is occluded, reappears,
  then follows an abnormal trajectory.

Run with Blender, for example:
    blender -b --python data_generat_state2.py -- \
      --out_root data/anomaly_3d_state2 \
      --category all \
      --num_pairs 100 \
      --seed 7000 \
      --img_size 128
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_generat_process import (  # noqa: E402
    COLOR_PALETTE,
    EVENT_FRAME,
    IMG_SIZE,
    NUM_FRAMES,
    SHAPES,
    create_backdrop,
    create_floor,
    create_shape,
    make_mat,
    render_png,
    reset_scene,
    setup_camera,
    setup_light,
    setup_render,
)


CATEGORIES = [
    "sudden_disappearance",
    "identity_change",
    "occluded_wrong_reappearance",
    "reappearance_trajectory_anomaly",
]

CATEGORY_LABELS = {
    "sudden_disappearance": "sudden disappearance",
    "identity_change": "identity / appearance change",
    "occluded_wrong_reappearance": "wrong reappearance after occlusion",
    "reappearance_trajectory_anomaly": "trajectory anomaly after reappearance",
}

SCENE_X_BOUND = 1.08
SCENE_Y_BOUND = 0.82
OBJECT_SIZE_MIN = 0.68
OBJECT_SIZE_MAX = 0.88
OCCLUSION_COVER_START = EVENT_FRAME - 4
OCCLUSION_REAPPEAR_MIN = EVENT_FRAME
OCCLUSION_REAPPEAR_MAX = EVENT_FRAME
OCCLUDER_WIDTH_MIN = 0.86
OCCLUDER_WIDTH_MAX = 1.04
OCCLUDER_DEPTH = 0.10
OCCLUDER_HEIGHT = 1.38
OCCLUDER_KINDS = [
    "vertical_panel",
    "low_wall",
    "wide_pillar",
    "bridge_pier",
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, alpha):
    return a * (1.0 - alpha) + b * alpha


def smoothstep(alpha):
    alpha = clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def vec_to_list(v):
    return [float(v.x), float(v.y), float(v.z)]


def unit_from_angle(theta):
    return Vector((math.cos(theta), math.sin(theta), 0.0))


def shape_extent(kind, size):
    if kind == "cube":
        return size * 0.50
    if kind == "sphere":
        return size * 0.52
    if kind == "cylinder":
        return size * 0.44
    raise ValueError(kind)


def sample_occluder_style():
    kind = random.choice(OCCLUDER_KINDS)

    if kind == "vertical_panel":
        width = random.uniform(0.86, 1.04)
        depth = random.uniform(0.08, 0.12)
        height = random.uniform(1.28, 1.46)
        label = "vertical occluder panel"
    elif kind == "low_wall":
        width = random.uniform(0.92, 1.12)
        depth = random.uniform(0.12, 0.18)
        height = random.uniform(0.78, 0.96)
        label = "low block wall"
    elif kind == "wide_pillar":
        width = random.uniform(0.82, 0.98)
        depth = random.uniform(0.16, 0.22)
        height = random.uniform(1.22, 1.42)
        label = "wide cylindrical pillar"
    elif kind == "bridge_pier":
        width = random.uniform(0.98, 1.18)
        depth = random.uniform(0.10, 0.15)
        height = random.uniform(1.18, 1.38)
        label = "bridge-pier occluder block"
    else:
        raise ValueError(kind)

    return {
        "kind": kind,
        "label": label,
        "width": width,
        "depth": depth,
        "height": height,
    }


def choose_motion(seed, category):
    random.seed(seed)
    speed_hi = 0.017 if "occluded" in category or "reappearance" in category else 0.026
    speed = random.uniform(0.008, speed_hi)
    theta = random.choice([
        0.0,
        math.pi / 7,
        -math.pi / 7,
        math.pi / 5,
        -math.pi / 5,
        math.pi,
        math.pi - math.pi / 7,
        -math.pi + math.pi / 7,
    ])
    theta += random.uniform(-0.10, 0.10)
    vel = unit_from_angle(theta) * speed

    total = vel * (NUM_FRAMES - 1)
    min_dx = min(0.0, float(total.x))
    max_dx = max(0.0, float(total.x))
    min_dy = min(0.0, float(total.y))
    max_dy = max(0.0, float(total.y))

    x0 = random.uniform(-SCENE_X_BOUND - min_dx, SCENE_X_BOUND - max_dx)
    y0 = random.uniform(-SCENE_Y_BOUND - min_dy, SCENE_Y_BOUND - max_dy)
    return Vector((x0, y0, 0.0)), vel


def sample_occlusion_path(target_size):
    reappear_frame = EVENT_FRAME
    cover_start = OCCLUSION_COVER_START
    cover_end = EVENT_FRAME - 1

    move_sign = random.choice([-1.0, 1.0])
    slope = random.uniform(-0.10, 0.10)
    direction = Vector((move_sign, slope, 0.0))
    direction.normalize()

    occ_style = sample_occluder_style()
    panel_width = occ_style["width"]
    panel_depth = occ_style["depth"]
    panel_height = occ_style["height"]
    center_x = random.uniform(0.12, 0.32) * move_sign
    panel_center = Vector((
        center_x,
        random.uniform(-0.34, -0.24),
        0.22,
    ))

    target_extent = target_size * 0.52
    behind_y_gap = target_extent + panel_depth * 0.50 + random.uniform(0.28, 0.36)
    clearance_margin = behind_y_gap - target_extent - panel_depth * 0.50
    edge_distance = panel_width * random.uniform(0.30, 0.36) + target_extent * random.uniform(0.14, 0.22)
    path_center = Vector((panel_center.x, panel_center.y + behind_y_gap, 0.0))
    pos_at_cover = path_center - direction * edge_distance
    pos_at_ref = path_center + direction * edge_distance
    velocity = (pos_at_ref - pos_at_cover) / float(EVENT_FRAME - cover_start)
    start = pos_at_cover - velocity * cover_start
    pos_at_cover = start + velocity * cover_start
    pos_at_19 = start + velocity * (EVENT_FRAME - 1)
    post_vel = velocity

    path = {
        "pos0": vec_to_list(start),
        "vel": vec_to_list(velocity),
        "pos_cover": vec_to_list(pos_at_cover),
        "pos19": vec_to_list(pos_at_19),
        "pos_ref": vec_to_list(pos_at_ref),
        "post_vel": vec_to_list(post_vel),
        "cover_start": cover_start,
        "t_ref": reappear_frame,
    }

    occluder = {
        "name": "fixed_foreground_occluder",
        "kind": occ_style["kind"],
        "kind_label": occ_style["label"],
        "size": [panel_width, panel_depth, panel_height],
        "color": (0.20, 0.20, 0.20, 1),
        "pos": vec_to_list(panel_center),
        "behind_y_gap": float(behind_y_gap),
        "clearance_margin": float(clearance_margin),
        "cover_start": cover_start,
        "cover_end": cover_end,
        "reappear_frame": reappear_frame,
        "motion_direction": vec_to_list(direction),
    }
    return start, velocity, post_vel, path, occluder, reappear_frame


def target_base_pos(cfg, frame):
    if cfg.get("target_path") is not None:
        path = cfg["target_path"]
        pos0 = Vector(path["pos0"])
        vel = Vector(path["vel"])
        return pos0 + vel * frame

    return Vector(cfg["target_start"]) + Vector(cfg["target_vel"]) * frame


def choose_new_color(old_color, colors):
    candidates = [c for c in colors if c != old_color]
    return random.choice(candidates)


def choose_new_shape(old_shape):
    return random.choice([s for s in SHAPES if s != old_shape])


def far_enough(candidate, radius, placed):
    for pos, other_radius in placed:
        if (candidate - pos).length < radius + other_radius + 0.12:
            return False
    return True


def sample_distractors(colors, target_shape, target_size, target_start, target_vel):
    num = random.choice([0, 1, 1, 2])
    candidates = [
        Vector((-1.10, 0.66, 0.0)),
        Vector((1.05, 0.62, 0.0)),
        Vector((-1.04, -0.46, 0.0)),
        Vector((1.04, -0.42, 0.0)),
        Vector((-0.42, 0.78, 0.0)),
        Vector((0.46, 0.74, 0.0)),
    ]
    random.shuffle(candidates)

    placed = []
    target_radius = shape_extent(target_shape, target_size)
    for t in range(0, NUM_FRAMES, 8):
        placed.append((target_start + target_vel * t, target_radius))

    distractors = []
    for cand in candidates:
        if len(distractors) >= num:
            break
        kind = random.choice(SHAPES)
        size = random.uniform(0.36, 0.56)
        radius = shape_extent(kind, size)
        jittered = cand + Vector((random.uniform(-0.06, 0.06), random.uniform(-0.06, 0.06), 0.0))
        if not far_enough(jittered, radius, placed):
            continue
        color = random.choice(colors)
        distractors.append({
            "kind": kind,
            "size": size,
            "color": color,
            "pos": vec_to_list(jittered),
        })
        placed.append((jittered, radius))

    return distractors


def sample_config(category, seed):
    random.seed(seed)

    colors = COLOR_PALETTE[:]
    random.shuffle(colors)

    target_shape = random.choice(SHAPES)
    target_size = random.uniform(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX)
    target_color = colors[0]

    alt_color = choose_new_color(target_color, colors)
    alt_shape = choose_new_shape(target_shape)
    identity_subtype = random.choice(["color", "shape"])
    reappearance_subtype = random.choice(["wrong_appearance", "missing"])

    occluder = None
    t_ref = EVENT_FRAME
    state_intervention_frame = EVENT_FRAME
    target_path = None
    target_post_vel = None
    if category in ["occluded_wrong_reappearance", "reappearance_trajectory_anomaly"]:
        target_start, target_vel, target_post_vel, target_path, occluder, reappear_frame = sample_occlusion_path(target_size)
    else:
        target_start, target_vel = choose_motion(seed + 41, category)
        reappear_frame = random.randint(OCCLUSION_REAPPEAR_MIN, OCCLUSION_REAPPEAR_MAX)

    wrong_offset = Vector((0.0, 0.0, 0.0))
    wrong_vel = Vector((0.0, 0.0, 0.0))

    if category in ["occluded_wrong_reappearance", "reappearance_trajectory_anomaly"]:
        state_intervention_frame = EVENT_FRAME

        if category == "occluded_wrong_reappearance":
            t_ref = reappear_frame
            if reappearance_subtype == "wrong_position":
                motion_dir = Vector(occluder["motion_direction"])
                lateral = Vector((-motion_dir.y, motion_dir.x, 0.0))
                if lateral.length < 1e-6:
                    lateral = Vector((1.0, 0.0, 0.0))
                lateral.normalize()
                wrong_offset = lateral * random.choice([-1.0, 1.0]) * random.uniform(0.54, 0.74)
            elif reappearance_subtype == "wrong_appearance":
                identity_subtype = random.choice(["color", "shape"])
        else:
            t_ref = reappear_frame
            reappearance_subtype = "wrong_trajectory"
            base_vec = target_post_vel if target_post_vel is not None else target_vel
            base_speed = max(0.013, Vector(base_vec).length)
            base_angle = math.atan2(base_vec.y, base_vec.x)
            turn = random.choice([-1.0, 1.0]) * random.uniform(math.radians(70), math.radians(125))
            wrong_vel = unit_from_angle(base_angle + turn) * random.uniform(base_speed * 1.25, base_speed * 1.75)

    if category in ["occluded_wrong_reappearance", "reappearance_trajectory_anomaly"]:
        distractors = []
    else:
        distractors = sample_distractors(
            colors[2:],
            target_shape,
            target_size,
            target_start,
            target_vel,
        )

    cfg = {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "target_shape": target_shape,
        "target_size": target_size,
        "target_color": target_color,
        "target_start": vec_to_list(target_start),
        "target_vel": vec_to_list(target_vel),
        "target_path": target_path,
        "alt_shape": alt_shape,
        "alt_color": alt_color,
        "identity_subtype": identity_subtype,
        "reappearance_subtype": reappearance_subtype,
        "wrong_offset": vec_to_list(wrong_offset),
        "wrong_vel": vec_to_list(wrong_vel),
        "occluder": occluder,
        "distractors": distractors,
        "event_frame": EVENT_FRAME,
        "state_intervention_frame": state_intervention_frame,
        "t_ref": t_ref,
        "camera_dx": random.uniform(-0.05, 0.05),
        "camera_dy": random.uniform(-0.04, 0.04),
        "light_x": random.uniform(-3.4, -2.6),
        "light_y": random.uniform(-4.4, -3.6),
    }
    return cfg


def occluder_position(cfg, frame):
    occ = cfg["occluder"]
    return Vector(occ["pos"])


def branch_target_state(cfg, frame, branch):
    pos = target_base_pos(cfg, frame)
    visible = True
    shape = cfg["target_shape"]
    color = cfg["target_color"]

    if branch == "possible":
        return pos, visible, shape, color

    category = cfg["category"]
    if category == "sudden_disappearance":
        if frame >= EVENT_FRAME:
            visible = False

    elif category == "identity_change":
        if frame >= EVENT_FRAME:
            if cfg["identity_subtype"] == "shape":
                shape = cfg["alt_shape"]
            else:
                color = cfg["alt_color"]

    elif category == "occluded_wrong_reappearance":
        if frame >= cfg["t_ref"]:
            sub = cfg["reappearance_subtype"]
            if sub == "missing":
                visible = False
            elif sub == "wrong_position":
                pos = pos + Vector(cfg["wrong_offset"])
                pos.x = clamp(pos.x, -SCENE_X_BOUND, SCENE_X_BOUND)
                pos.y = clamp(pos.y, -SCENE_Y_BOUND, SCENE_Y_BOUND)
            elif sub == "wrong_appearance":
                if cfg["identity_subtype"] == "shape":
                    shape = cfg["alt_shape"]
                else:
                    color = cfg["alt_color"]

    elif category == "reappearance_trajectory_anomaly":
        if frame >= cfg["t_ref"]:
            reappear_pos = target_base_pos(cfg, cfg["t_ref"] - 1)
            pos = reappear_pos + Vector(cfg["wrong_vel"]) * (frame - cfg["t_ref"] + 1)
            pos.x = clamp(pos.x, -SCENE_X_BOUND, SCENE_X_BOUND)
            pos.y = clamp(pos.y, -SCENE_Y_BOUND, SCENE_Y_BOUND)

    return pos, visible, shape, color


def create_panel(name, size, color):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = (size[0], size[1], size[2])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(make_mat(name + "_mat", color, roughness=0.55))
    return obj


def create_occluder_parts(occ):
    kind = occ.get("kind", "vertical_panel")
    size = occ["size"]
    color = occ["color"]

    if kind in ["vertical_panel", "low_wall"]:
        return [(create_panel(occ["name"], size, color), Vector((0.0, 0.0, 0.0)))]

    if kind == "wide_pillar":
        radius = size[0] * 0.50
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=size[2])
        obj = bpy.context.object
        obj.name = occ["name"] + "_wide_pillar"
        obj.scale.y = max(0.42, size[1] / max(0.001, size[0]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(make_mat(obj.name + "_mat", color, roughness=0.52))
        return [(obj, Vector((0.0, 0.0, 0.0)))]

    if kind == "bridge_pier":
        width, depth, height = size
        body_width = width * 0.58
        cap_height = height * 0.22
        base_height = height * 0.16
        body_height = height * 0.76
        parts = [
            (
                create_panel(occ["name"] + "_body", [body_width, depth, body_height], color),
                Vector((0.0, 0.0, -height * 0.04)),
            ),
            (
                create_panel(occ["name"] + "_cap", [width, depth, cap_height], color),
                Vector((0.0, 0.0, height * 0.38)),
            ),
            (
                create_panel(occ["name"] + "_base", [width * 0.86, depth, base_height], color),
                Vector((0.0, 0.0, -height * 0.45)),
            ),
        ]
        return parts

    raise ValueError(f"Unknown occluder kind: {kind}")


def set_visible(obj, visible):
    obj.hide_viewport = not visible
    obj.hide_render = not visible


def create_scene_objects(cfg):
    create_floor()
    create_backdrop()

    target = create_shape(
        "target",
        cfg["target_shape"],
        cfg["target_size"],
        cfg["target_color"],
    )
    alt_color_target = create_shape(
        "target_alt_color",
        cfg["target_shape"],
        cfg["target_size"],
        cfg["alt_color"],
    )
    alt_shape_target = create_shape(
        "target_alt_shape",
        cfg["alt_shape"],
        cfg["target_size"],
        cfg["target_color"],
    )
    set_visible(alt_color_target, False)
    set_visible(alt_shape_target, False)

    distractors = []
    for i, d in enumerate(cfg.get("distractors", [])):
        obj = create_shape(
            f"distractor_{i}",
            d["kind"],
            d["size"],
            d["color"],
        )
        obj.location = (d["pos"][0], d["pos"][1], d["pos"][2])
        distractors.append(obj)

    occluder = []
    if cfg.get("occluder") is not None:
        occ = cfg["occluder"]
        occluder = create_occluder_parts(occ)

    return {
        "target": target,
        "alt_color_target": alt_color_target,
        "alt_shape_target": alt_shape_target,
        "distractors": distractors,
        "occluder": occluder,
    }


def apply_frame_state(objects, cfg, frame, branch):
    target_pos, visible, shape, color = branch_target_state(cfg, frame, branch)

    active = objects["target"]
    if shape != cfg["target_shape"]:
        active = objects["alt_shape_target"]
    elif color != cfg["target_color"]:
        active = objects["alt_color_target"]

    for name in ["target", "alt_color_target", "alt_shape_target"]:
        is_active = visible and objects[name] == active
        set_visible(objects[name], is_active)
        objects[name].location = (target_pos.x, target_pos.y, target_pos.z)
        objects[name].rotation_euler[2] = 0.026 * frame

    for i, distractor in enumerate(objects["distractors"]):
        distractor.rotation_euler[2] = 0.012 * frame * (i + 1)

    if objects["occluder"]:
        occ_pos = occluder_position(cfg, frame)
        for obj, offset in objects["occluder"]:
            obj.location = (occ_pos.x + offset.x, occ_pos.y + offset.y, occ_pos.z + offset.z)
            obj.rotation_euler[2] = 0.0


def render_branch_frame(out_path, cfg, frame, branch):
    reset_scene()
    setup_render(IMG_SIZE)
    setup_camera(cfg)
    setup_light(cfg)
    objects = create_scene_objects(cfg)
    apply_frame_state(objects, cfg, frame, branch)
    render_png(out_path)


def render_pair(pair_dir, category, seed):
    cfg = sample_config(category, seed)
    possible_dir = os.path.join(pair_dir, "possible")
    impossible_dir = os.path.join(pair_dir, "impossible")
    os.makedirs(possible_dir, exist_ok=True)
    os.makedirs(impossible_dir, exist_ok=True)

    for frame in range(NUM_FRAMES):
        possible_path = os.path.join(possible_dir, f"{frame:03d}.png")
        impossible_path = os.path.join(impossible_dir, f"{frame:03d}.png")

        render_branch_frame(possible_path, cfg, frame, "possible")
        if frame < cfg["t_ref"]:
            shutil.copy2(possible_path, impossible_path)
        else:
            render_branch_frame(impossible_path, cfg, frame, "impossible")

    meta = {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "num_frames": NUM_FRAMES,
        "event_frame": EVENT_FRAME,
        "state_intervention_frame": cfg["state_intervention_frame"],
        "t_ref": cfg["t_ref"],
        "context_frames": [0, EVENT_FRAME - 1],
        "future_frames": [EVENT_FRAME, NUM_FRAMES - 1],
        "seed": seed,
        "target_object": "target",
        "config": cfg,
        "notes": {
            "possible": "state-consistent future",
            "impossible": "state-persistence violation future",
            "history": "frames before t_ref are copied from possible to impossible, so shared history and pre-difference future are byte-identical",
            "occluder": "when present, the occluder is fixed in the foreground from frame 0; the target stays behind it with positive y-clearance and is hidden only by camera projection",
        },
    }

    with open(os.path.join(pair_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    verify_pair_files(pair_dir, cfg["t_ref"])
    return meta


def verify_pair_files(pair_dir, shared_until):
    possible_dir = os.path.join(pair_dir, "possible")
    impossible_dir = os.path.join(pair_dir, "impossible")

    for frame in range(NUM_FRAMES):
        p1 = os.path.join(possible_dir, f"{frame:03d}.png")
        p2 = os.path.join(impossible_dir, f"{frame:03d}.png")
        if not os.path.exists(p1):
            raise FileNotFoundError(p1)
        if not os.path.exists(p2):
            raise FileNotFoundError(p2)

        if frame < shared_until:
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                if f1.read() != f2.read():
                    raise RuntimeError(f"Shared prefix differs at frame {frame}: {pair_dir}")


def selected_categories(category):
    if category == "all":
        return CATEGORIES
    return [category]


def write_csv(csv_path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id",
            "anomaly_type",
            "anomaly_frame",
            "state_intervention_frame",
            "t_ref",
            "target_object",
            "possible_path",
            "impossible_path",
            "meta_path",
        ])
        writer.writerows(rows)


def main():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="data/anomaly_3d_state2")
    parser.add_argument("--category", type=str, default="all", choices=["all"] + CATEGORIES)
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args(argv)

    import data_generat_process as process_style

    process_style.IMG_SIZE = args.img_size
    globals()["IMG_SIZE"] = args.img_size

    out_root = os.path.abspath(args.out_root)
    csv_path = args.csv_path or os.path.join(out_root, "anomaly_pairs.csv")
    os.makedirs(out_root, exist_ok=True)

    cats = selected_categories(args.category)
    print("======== Blender State2 OBJ3D Rendering ========")
    print(f"out_root   : {out_root}")
    print(f"category   : {args.category}")
    print(f"num_pairs  : {args.num_pairs}")
    print(f"seed       : {args.seed}")
    print(f"img_size   : {args.img_size}")
    print(f"num_frames : {NUM_FRAMES}")
    print(f"event_frame: {EVENT_FRAME}")
    print(f"csv_path   : {csv_path}")
    print("================================================")

    rows = []
    pair_counter = 0
    for cidx, category in enumerate(cats):
        category_root = os.path.join(out_root, category)
        os.makedirs(category_root, exist_ok=True)

        for i in range(args.num_pairs):
            pair_dir = os.path.join(category_root, f"pair_{i:06d}")
            os.makedirs(pair_dir, exist_ok=True)
            seed = args.seed + cidx * 100000 + i
            meta = render_pair(pair_dir, category, seed)
            pair_counter += 1

            rows.append([
                f"{category}/pair_{i:06d}",
                category,
                str(meta["event_frame"]),
                str(meta["state_intervention_frame"]),
                str(meta["t_ref"]),
                meta["target_object"],
                os.path.abspath(os.path.join(pair_dir, "possible")),
                os.path.abspath(os.path.join(pair_dir, "impossible")),
                os.path.abspath(os.path.join(pair_dir, "meta.json")),
            ])

            print(f"[{category}] rendered {i + 1}/{args.num_pairs}", flush=True)

    write_csv(csv_path, rows)
    print(f"Done. Generated {pair_counter} pairs. CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
