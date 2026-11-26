import bpy
import os
import sys
import json
import math
import random
import argparse
from mathutils import Vector

NUM_FRAMES = 48
EVENT_FRAME = 20
IMG_SIZE = 128

OBJECT_SIZE_MIN = 0.68
OBJECT_SIZE_MAX = 0.88
ORTHO_SCALE = 3.65
FLOOR_SIZE = 14.0


# ============================================================
# Blender utilities
# ============================================================

def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_render(img_size=128):
    scene = bpy.context.scene
    scene.render.resolution_x = img_size
    scene.render.resolution_y = img_size
    scene.render.film_transparent = False

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True

    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    if scene.world is not None:
        scene.world.color = (0.46, 0.46, 0.46)


def setup_camera(cfg=None):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    dx = 0.0 if cfg is None else cfg.get("camera_dx", 0.0)
    dy = 0.0 if cfg is None else cfg.get("camera_dy", 0.0)

    cam.location = (dx, -4.25 + dy, 3.05)
    cam.rotation_euler = (math.radians(56), 0, 0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ORTHO_SCALE

    bpy.context.scene.camera = cam


def setup_light(cfg=None):
    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light = bpy.data.objects.new("KeyLight", light_data)
    bpy.context.collection.objects.link(light)

    lx = -3.0 if cfg is None else cfg.get("light_x", -3.0)
    ly = -4.0 if cfg is None else cfg.get("light_y", -4.0)

    light.location = (lx, ly, 6.0)
    light.data.energy = 750
    light.data.size = 3.0


def make_mat(name, color, roughness=0.45, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic

    return mat


def create_floor():
    bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0, 0, -0.42))
    floor = bpy.context.object
    floor.name = "floor"
    floor.data.materials.append(
        make_mat("floor_mat", (0.46, 0.46, 0.46, 1), roughness=0.68)
    )
    return floor


def create_backdrop():
    bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0, 3.2, 2.25))
    wall = bpy.context.object
    wall.name = "gray_backdrop"
    wall.rotation_euler[0] = math.radians(90)
    wall.data.materials.append(
        make_mat("backdrop_mat", (0.46, 0.46, 0.46, 1), roughness=0.70)
    )
    return wall


def create_cube(name, size, color):
    bpy.ops.mesh.primitive_cube_add(size=size)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat(name + "_mat", color, roughness=0.42))
    return obj


def create_sphere(name, radius, color):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat(name + "_mat", color, roughness=0.30))
    return obj


def create_cylinder(name, radius, depth, color):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat(name + "_mat", color, roughness=0.38))
    return obj


def create_shape(name, kind, size, color):
    if kind == "cube":
        return create_cube(name, size, color)
    if kind == "sphere":
        return create_sphere(name, size * 0.52, color)
    if kind == "cylinder":
        return create_cylinder(name, size * 0.42, size * 0.95, color)
    raise ValueError(f"Unknown shape kind: {kind}")


def shape_extent(kind, size):
    if kind == "cube":
        return size * 0.50
    if kind == "sphere":
        return size * 0.52
    if kind == "cylinder":
        return size * 0.42
    raise ValueError(kind)


def render_png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ============================================================
# Geometry helpers
# ============================================================

def unit_from_angle(theta):
    return Vector((math.cos(theta), math.sin(theta), 0.0))


def perp(v):
    return Vector((-v.y, v.x, 0.0))


def xy_tuple(v, z=0.0):
    return (float(v.x), float(v.y), float(z))


def future_progress(frame, start_frame, duration):
    if frame < start_frame:
        return 0.0
    return min(1.0, (frame - start_frame + 1) / max(1, duration))


def same_history_positions(frame, cfg):
    d = Vector(cfg["motion_dir"])
    actor_start = Vector(cfg["actor_start"])
    target_pos = Vector(cfg["target_pos"])

    actor_pos = actor_start + d * cfg["actor_speed"] * frame
    return xy_tuple(actor_pos), xy_tuple(target_pos)


def actor_contact_follow_pos(target_pos, d, cfg):
    return target_pos - d * cfg["contact_gap"]


def damped_displacement(initial_speed, decel, steps):
    if steps <= 0:
        return 0.0

    if abs(initial_speed) < 1e-8:
        return 0.0

    sign = 1.0 if initial_speed >= 0.0 else -1.0
    speed0 = abs(initial_speed)

    if decel <= 1e-8:
        return initial_speed * steps

    stop_steps = speed0 / decel
    active_steps = min(float(steps), stop_steps)
    distance = speed0 * active_steps - 0.5 * decel * active_steps * active_steps
    return sign * distance


def damped_speed(initial_speed, decel, steps):
    if abs(initial_speed) < 1e-8:
        return 0.0

    sign = 1.0 if initial_speed >= 0.0 else -1.0
    speed = max(0.0, abs(initial_speed) - decel * max(0.0, float(steps)))
    return sign * speed


def penetration_possible_positions(frame, cfg):
    d = Vector(cfg["motion_dir"])
    target_base = Vector(cfg["target_pos"])
    actor_contact_pos = target_base - d * cfg["contact_gap"]
    steps = frame - EVENT_FRAME + 1

    catch_step = None
    actor_catch_shift = 0.0
    target_catch_shift = 0.0

    for step in range(1, steps + 1):
        actor_shift = damped_displacement(
            cfg["possible_actor_speed"],
            cfg["possible_actor_decel"],
            step,
        )
        target_shift = damped_displacement(
            cfg["possible_target_speed"],
            cfg["possible_target_decel"],
            step,
        )
        if actor_shift >= target_shift:
            catch_step = step
            actor_catch_shift = actor_shift
            target_catch_shift = target_shift
            break

    if catch_step is None or steps <= catch_step:
        actor_shift = damped_displacement(
            cfg["possible_actor_speed"],
            cfg["possible_actor_decel"],
            steps,
        )
        target_shift = damped_displacement(
            cfg["possible_target_speed"],
            cfg["possible_target_decel"],
            steps,
        )
        actor_pos = actor_contact_pos + d * actor_shift
        target_pos = target_base + d * target_shift
        return actor_pos, target_pos

    extra_steps = steps - catch_step
    push_shift = damped_displacement(
        cfg["possible_push_speed"],
        cfg["possible_push_decel"],
        extra_steps,
    )
    target_pos = target_base + d * (target_catch_shift + push_shift)
    actor_pos = actor_contact_follow_pos(target_pos, d, cfg)
    return actor_pos, target_pos


# ============================================================
# Trajectories
# ============================================================

def traj_no_contact_motion(frame, branch, cfg):
    pos_a, pos_b = same_history_positions(frame, cfg)

    if frame < EVENT_FRAME:
        return pos_a, pos_b

    d = Vector(cfg["motion_dir"])
    lateral = Vector(cfg["lateral_dir"])
    target_base = Vector(cfg["target_pos"])

    total = cfg["target_total_shift"]
    duration = cfg["motion_duration"]
    sub = cfg["subcategory"]

    if sub == "early_motion_far":
        if branch == "possible":
            p = future_progress(frame, cfg["contact_frame"], duration)
            shift = d * (total * p) + lateral * (0.10 * total * p)
        else:
            p = future_progress(frame, EVENT_FRAME, duration)
            shift = d * (total * p) + lateral * (0.35 * total * p)

        target_pos = target_base + shift
        return pos_a, xy_tuple(target_pos)

    if sub == "spontaneous_motion":
        if branch == "possible":
            target_pos = target_base
        else:
            p = future_progress(frame, EVENT_FRAME, duration)
            move_dir = (0.55 * d + 0.85 * lateral).normalized()
            shift = move_dir * (total * p)
            target_pos = target_base + shift

        return pos_a, xy_tuple(target_pos)

    raise ValueError(f"Unknown no_contact subtype: {sub}")


def traj_contact_response_failure(frame, branch, cfg):
    """
    contact_response_failure:
    只测试接触后的响应错误，不允许穿模。
    """
    pos_a, pos_b = same_history_positions(frame, cfg)

    if frame < EVENT_FRAME:
        return pos_a, pos_b

    d = Vector(cfg["motion_dir"])
    lateral = Vector(cfg["lateral_dir"])
    target_base = Vector(cfg["target_pos"])

    total = cfg["target_total_shift"]
    duration = cfg["motion_duration"]
    p = future_progress(frame, EVENT_FRAME, duration)
    sub = cfg["subcategory"]

    actor_contact_pos = target_base - d * cfg["contact_gap"]
    actor_pos = actor_contact_pos

    if branch == "possible":
        shift = d * (0.95 * total * p) + lateral * (0.04 * total * p)
        target_pos = target_base + shift
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "no_response":
        target_pos = target_base
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "wrong_direction":
        wrong_dir = (-0.25 * d + 1.00 * lateral).normalized()
        shift = wrong_dir * (0.90 * total * p)
        target_pos = target_base + shift
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "excessive_response":
        fast_p = min(1.0, p * 1.35)
        shift = d * (2.45 * total * fast_p) + lateral * (0.18 * total * fast_p)
        target_pos = target_base + shift
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    raise ValueError(f"Unknown contact_response subtype: {sub}")


def traj_penetration(frame, branch, cfg):
    """
    penetration 子类型：
    - actor_through_target: A 穿过 B
    - target_through_wall: B 穿过静态障碍物
    - target_through_floor: B 向下穿过地面
    - actor_through_wall: A 穿过静态障碍物
    - target_sink_then_move: B 先向下穿地，再沿平面方向移动
    """
    pos_a, pos_b = same_history_positions(frame, cfg)

    if frame < EVENT_FRAME:
        return pos_a, pos_b

    d = Vector(cfg["motion_dir"])
    lateral = Vector(cfg["lateral_dir"])
    target_base = Vector(cfg["target_pos"])
    actor_contact_pos = target_base - d * cfg["contact_gap"]

    total = cfg["target_total_shift"]
    duration = cfg["motion_duration"]
    p = future_progress(frame, EVENT_FRAME, duration)
    sub = cfg["subcategory"]

    if branch == "possible":
        actor_pos, target_pos = penetration_possible_positions(frame, cfg)
        return xy_tuple(actor_pos), xy_tuple(target_pos, z=0.0)

    if sub == "actor_through_target":
        actor_pos = actor_contact_pos + d * (2.00 * total * p)
        target_pos = target_base
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "target_through_wall":
        wall_pos = Vector(cfg["wall_pos"])
        wall_stop = wall_pos - d * cfg["wall_contact_gap"]

        target_pos = target_base + d * (1.90 * total * p)
        actor_pos = actor_contact_pos + d * (0.20 * total * p)

        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "target_through_floor":
        # 明确向下穿模：B 从第 20 帧开始快速下沉到地面以下
        target_pos = target_base + d * (0.25 * total * p)
        z = -1.45 * p
        return pos_a, xy_tuple(target_pos, z=z)

    if sub == "actor_through_wall":
        wall_pos = Vector(cfg["wall_pos"])
        wall_stop = wall_pos - d * cfg["wall_contact_gap"]

        # A 穿过静态障碍物
        actor_pos = actor_contact_pos + d * (1.85 * total * p)

        target_pos = target_base
        return xy_tuple(actor_pos), xy_tuple(target_pos)

    if sub == "target_sink_then_move":
        # 异常：B 先明显向下穿地，再继续移动
        sink_p = min(1.0, p * 1.50)
        move_p = p
        target_pos = target_base + d * (0.55 * total * move_p) + lateral * (0.08 * total * move_p)
        z = -1.25 * sink_p
        return pos_a, xy_tuple(target_pos, z=z)

    raise ValueError(f"Unknown penetration subtype: {sub}")


def get_positions(category, frame, branch, cfg):
    if category == "no_contact_motion":
        return traj_no_contact_motion(frame, branch, cfg)
    if category == "contact_response_failure":
        return traj_contact_response_failure(frame, branch, cfg)
    if category == "penetration":
        return traj_penetration(frame, branch, cfg)
    raise ValueError(f"Unknown category: {category}")


# ============================================================
# Config sampling
# ============================================================

COLOR_PALETTE = [
    (0.90, 0.12, 0.10, 1),
    (0.10, 0.35, 0.85, 1),
    (0.78, 0.28, 0.78, 1),
    (0.95, 0.60, 0.12, 1),
    (0.20, 0.65, 0.22, 1),
    (0.95, 0.90, 0.18, 1),
    (0.12, 0.75, 0.75, 1),
    (0.75, 0.18, 0.40, 1),
]

SHAPES = ["cube", "sphere", "cylinder"]


def sample_direction(seed):
    random.seed(seed)
    angles = [
        0,
        math.pi / 6,
        math.pi / 4,
        math.pi / 3,
        -math.pi / 6,
        -math.pi / 4,
        -math.pi / 3,
        math.pi,
        5 * math.pi / 6,
        3 * math.pi / 4,
    ]
    theta = random.choice(angles) + random.uniform(-0.12, 0.12)
    d = unit_from_angle(theta)
    lateral = perp(d)
    return d, lateral


def sample_subcategory(category):
    if category == "no_contact_motion":
        return random.choice([
            "early_motion_far",
            "early_motion_far",
            "spontaneous_motion",
        ])

    if category == "contact_response_failure":
        return random.choice([
            "no_response",
            "wrong_direction",
            "excessive_response",
        ])

    if category == "penetration":
        return random.choice([
            "actor_through_target",
            "target_through_floor",
            "target_through_floor",
            "actor_through_target",
            "target_sink_then_move",
            "target_sink_then_move",
        ])

    raise ValueError(category)


def sample_config(category, seed):
    random.seed(seed)

    actor_shape = random.choice(SHAPES)
    target_shape = random.choice(SHAPES)

    actor_size = random.uniform(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX)
    target_size = random.uniform(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX)

    actor_extent = shape_extent(actor_shape, actor_size)
    target_extent = shape_extent(target_shape, target_size)
    contact_gap = actor_extent + target_extent + 0.05

    colors = COLOR_PALETTE[:]
    random.shuffle(colors)
    actor_color = colors[0]
    target_color = colors[1]

    d, lateral = sample_direction(seed + 999)
    subcategory = sample_subcategory(category)

    target_pos = Vector((
        random.uniform(-0.15, 0.20),
        random.uniform(-0.08, 0.08),
        0.0
    ))

    camera_dx = random.uniform(-0.05, 0.05)
    camera_dy = random.uniform(-0.04, 0.04)

    if category == "no_contact_motion":
        if subcategory == "early_motion_far":
            no_contact_extra_gap = random.uniform(1.15, 1.55)

            actor_pos_at_event = target_pos - d * (contact_gap + no_contact_extra_gap)
            actor_speed = random.uniform(0.115, 0.165)

            actor_start = actor_pos_at_event - d * actor_speed * EVENT_FRAME

            contact_frame = int(EVENT_FRAME + no_contact_extra_gap / actor_speed)
            contact_frame = max(contact_frame, EVENT_FRAME + 8)
            contact_frame = min(contact_frame, EVENT_FRAME + 16)

        else:
            contact_frame = NUM_FRAMES + 100
            actor_speed = random.uniform(0.105, 0.155)

            side_offset = random.choice([-1.0, 1.0]) * random.uniform(1.35, 1.75)
            actor_pos_at_event = (
                target_pos
                - d * random.uniform(0.55, 0.85)
                + lateral * side_offset
            )
            actor_start = actor_pos_at_event - d * actor_speed * EVENT_FRAME

    else:
        contact_frame = EVENT_FRAME
        actor_speed = random.uniform(0.125, 0.180)

        actor_pos_at_event = target_pos - d * contact_gap
        actor_start = actor_pos_at_event - d * actor_speed * EVENT_FRAME

    if category == "contact_response_failure":
        target_total_shift = random.uniform(1.85, 2.45)
        motion_duration = random.randint(3, 5)
    elif category == "penetration":
        target_total_shift = random.uniform(2.30, 3.00)
        motion_duration = random.randint(4, 6)
    else:
        target_total_shift = random.uniform(2.00, 2.70)
        motion_duration = random.randint(4, 6)

    if category == "penetration":
        actor_mass = random.uniform(0.92, 1.08) * max(0.25, actor_size * actor_size)
        target_mass = random.uniform(0.92, 1.08) * max(0.25, target_size * target_size)
        restitution = None

        mass_advantage = actor_mass / (actor_mass + target_mass)
        possible_actor_speed = actor_speed * max(0.52, min(0.68, 0.44 + 0.22 * mass_advantage))
        possible_target_speed = actor_speed * max(0.88, min(0.98, 0.82 + 0.18 * mass_advantage))

        actor_stop_steps = random.uniform(132.0, 180.0)
        target_stop_steps = random.uniform(70.0, 96.0)
        possible_actor_decel = abs(possible_actor_speed) / actor_stop_steps
        possible_target_decel = abs(possible_target_speed) / target_stop_steps

        actor_speed_at_catch = damped_speed(possible_actor_speed, possible_actor_decel, 8.0)
        target_speed_at_catch = damped_speed(possible_target_speed, possible_target_decel, 8.0)
        possible_push_speed = max(target_speed_at_catch, 0.88 * actor_speed_at_catch)
        push_stop_steps = random.uniform(180.0, 240.0)
        possible_push_decel = abs(possible_push_speed) / push_stop_steps
    else:
        actor_mass = None
        target_mass = None
        restitution = None
        possible_actor_speed = None
        possible_target_speed = None
        possible_actor_decel = None
        possible_target_decel = None
        possible_push_speed = None
        possible_push_decel = None

    wall_pos = None
    wall_contact_gap = None

    if category == "penetration" and subcategory in ["target_through_wall", "actor_through_wall"]:
        wall_distance = random.uniform(1.00, 1.25)

        if subcategory == "target_through_wall":
            wall_pos = target_pos + d * wall_distance
            wall_contact_gap = target_extent + 0.12

        elif subcategory == "actor_through_wall":
            # wall 放在 actor 接触后继续运动的路径上
            actor_event_pos = target_pos - d * contact_gap
            wall_pos = actor_event_pos + d * wall_distance
            wall_contact_gap = actor_extent + 0.12

    if category == "contact_response_failure":
        num_distractors = random.choice([0, 0, 1])
    else:
        num_distractors = random.choice([0, 0, 1, 1, 2, 3])

    distractors = []
    candidate_positions = [
        (-1.05, 0.65),
        (0.95, 0.62),
        (-1.10, -0.35),
        (0.95, -0.32),
        (-0.35, 0.78),
        (0.35, 0.75),
    ]
    random.shuffle(candidate_positions)

    for i in range(num_distractors):
        x, y = candidate_positions[i]
        kind = random.choice(SHAPES)
        size = random.uniform(0.32, 0.56)
        color = random.choice(colors[2:])

        distractors.append({
            "kind": kind,
            "size": size,
            "color": color,
            "pos": [
                x + random.uniform(-0.08, 0.08),
                y + random.uniform(-0.08, 0.08),
                0.0
            ],
        })

    cfg = {
        "category": category,
        "subcategory": subcategory,

        "actor_shape": actor_shape,
        "target_shape": target_shape,
        "actor_size": actor_size,
        "target_size": target_size,
        "actor_extent": actor_extent,
        "target_extent": target_extent,
        "contact_gap": contact_gap,

        "actor_color": actor_color,
        "target_color": target_color,

        "motion_dir": [float(d.x), float(d.y), 0.0],
        "lateral_dir": [float(lateral.x), float(lateral.y), 0.0],

        "actor_start": [float(actor_start.x), float(actor_start.y), 0.0],
        "actor_speed": actor_speed,
        "target_pos": [float(target_pos.x), float(target_pos.y), 0.0],

        "contact_frame": contact_frame,
        "target_total_shift": target_total_shift,
        "motion_duration": motion_duration,
        "possible_actor_speed": possible_actor_speed,
        "possible_target_speed": possible_target_speed,
        "possible_actor_decel": possible_actor_decel,
        "possible_target_decel": possible_target_decel,
        "possible_push_speed": possible_push_speed,
        "possible_push_decel": possible_push_decel,
        "actor_mass": actor_mass,
        "target_mass": target_mass,
        "restitution": restitution,

        "wall_pos": None if wall_pos is None else [
            float(wall_pos.x),
            float(wall_pos.y),
            0.0
        ],
        "wall_contact_gap": wall_contact_gap,

        "distractors": distractors,

        "camera_dx": camera_dx,
        "camera_dy": camera_dy,
        "light_x": random.uniform(-3.4, -2.6),
        "light_y": random.uniform(-4.4, -3.6),
    }

    return cfg


# ============================================================
# Scene creation
# ============================================================

def create_wall_obstacle(cfg):
    if cfg["category"] != "penetration":
        return None

    if cfg["subcategory"] not in ["target_through_wall", "actor_through_wall"]:
        return None

    wall_pos = cfg["wall_pos"]
    if wall_pos is None:
        return None

    obj = create_cube(
        "wall_obstacle",
        cfg["target_size"] * 0.95,
        (0.18, 0.18, 0.18, 1)
    )
    obj.location = (wall_pos[0], wall_pos[1], 0.0)
    return obj


def create_scene_objects(cfg):
    create_floor()
    create_backdrop()

    obj_a = create_shape(
        "A_actor",
        cfg["actor_shape"],
        cfg["actor_size"],
        cfg["actor_color"]
    )

    obj_b = create_shape(
        "B_target",
        cfg["target_shape"],
        cfg["target_size"],
        cfg["target_color"]
    )

    create_wall_obstacle(cfg)

    for i, d in enumerate(cfg.get("distractors", [])):
        obj = create_shape(
            f"distractor_{i}",
            d["kind"],
            d["size"],
            d["color"]
        )
        obj.location = (d["pos"][0], d["pos"][1], d["pos"][2])

    return obj_a, obj_b


# ============================================================
# Rendering
# ============================================================

def render_pair(pair_dir, category, seed):
    cfg = sample_config(category, seed)

    for branch in ["possible", "impossible"]:
        reset_scene()
        setup_render(IMG_SIZE)
        setup_camera(cfg)
        setup_light(cfg)

        obj_a, obj_b = create_scene_objects(cfg)

        out_dir = os.path.join(pair_dir, branch)
        os.makedirs(out_dir, exist_ok=True)

        for frame in range(NUM_FRAMES):
            pos_a, pos_b = get_positions(category, frame, branch, cfg)

            obj_a.location = pos_a
            obj_b.location = pos_b

            obj_a.rotation_euler[2] = 0.045 * frame
            obj_b.rotation_euler[2] = 0.025 * frame

            out_path = os.path.join(out_dir, f"{frame:03d}.png")
            render_png(out_path)

    meta = {
        "category": category,
        "subcategory": cfg["subcategory"],
        "num_frames": NUM_FRAMES,
        "event_frame": EVENT_FRAME,
        "context_frames": [0, EVENT_FRAME - 1],
        "future_frames": [EVENT_FRAME, NUM_FRAMES - 1],
        "seed": seed,
        "config": cfg,
        "notes": {
            "possible": "physically plausible future",
            "impossible": "process violation future",
            "history": "frames 0-19 are identical between possible and impossible",
            "event": "violation starts at frame 20",
            "motion": "fast overall motion; penetration includes horizontal and downward penetration"
        }
    }

    with open(os.path.join(pair_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def verify_pair_files(pair_dir):
    possible_dir = os.path.join(pair_dir, "possible")
    impossible_dir = os.path.join(pair_dir, "impossible")

    for frame in range(EVENT_FRAME):
        p1 = os.path.join(possible_dir, f"{frame:03d}.png")
        p2 = os.path.join(impossible_dir, f"{frame:03d}.png")

        if not os.path.exists(p1):
            raise FileNotFoundError(p1)
        if not os.path.exists(p2):
            raise FileNotFoundError(p2)


# ============================================================
# Main
# ============================================================

def main():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=[
            "no_contact_motion",
            "contact_response_failure",
            "penetration",
        ],
    )
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=128)

    args = parser.parse_args(argv)

    global IMG_SIZE
    IMG_SIZE = args.img_size

    category_root = os.path.join(args.out_root, args.category)
    os.makedirs(category_root, exist_ok=True)

    print("======== Blender Process OBJ3D Rendering ========")
    print(f"out_root     : {args.out_root}")
    print(f"category     : {args.category}")
    print(f"num_pairs    : {args.num_pairs}")
    print(f"seed         : {args.seed}")
    print(f"img_size     : {args.img_size}")
    print(f"num_frames   : {NUM_FRAMES}")
    print(f"event_frame  : {EVENT_FRAME}")
    print(f"category_root: {category_root}")
    print("=================================================")

    for i in range(args.num_pairs):
        pair_dir = os.path.join(category_root, f"pair_{i:06d}")
        os.makedirs(pair_dir, exist_ok=True)

        render_pair(
            pair_dir=pair_dir,
            category=args.category,
            seed=args.seed + i,
        )

        verify_pair_files(pair_dir)

        if (i + 1) % 10 == 0 or i == args.num_pairs - 1:
            print(f"[{args.category}] rendered {i + 1}/{args.num_pairs}")

    print("Done.")


if __name__ == "__main__":
    main()