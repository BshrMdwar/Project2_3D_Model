"""
هذا السكربت يُشغَّل فقط عبر Blender:
    blender --background --python blender_worker.py -- <model_path>

لا يحتوي على أي كود Flask، ويعتمد فقط على bpy.
"""

import os
import sys
import json
import math
import time

import bpy
from mathutils import Vector


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ── قراءة مسار الموديل من الـ Arguments ──
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not args:
    fail("No model path provided after '--'")

MODEL_PATH = args[0]
if not os.path.exists(MODEL_PATH):
    fail(f"Model file not found: {MODEL_PATH}")

file_name = os.path.basename(MODEL_PATH)
model_id = os.path.splitext(file_name)[0]
extension = os.path.splitext(file_name)[1].lower()

if extension not in (".fbx", ".glb", ".gltf"):
    fail(f"Unsupported extension: {extension}")

# ── إعداد المجلدات ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "dataset")
RENDERS_ROOT = os.path.join(DATASET_ROOT, "renders")
GEOMETRY_ROOT = os.path.join(DATASET_ROOT, "geometry")
RENDERS_FOLDER = os.path.join(RENDERS_ROOT, model_id)

os.makedirs(RENDERS_FOLDER, exist_ok=True)
os.makedirs(GEOMETRY_ROOT, exist_ok=True)
GEOMETRY_PATH = os.path.join(GEOMETRY_ROOT, model_id + ".json")

# =========================================
# إعدادات الرندر
# =========================================
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01
scene.cycles.max_bounces = 12
scene.cycles.diffuse_bounces = 4
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 8
scene.cycles.transparent_max_bounces = 8
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - High Contrast'
scene.render.resolution_x = 768
scene.render.resolution_y = 768
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True

# ── تنظيف المشهد ──
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ── العالم/الإضاءة ──
world = scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 0.8


def add_light(name, light_type, location, energy, size=None):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    if light_type == 'AREA' and size is not None:
        light_data.shape = 'SQUARE'
        light_data.size = size
    elif light_type == 'SUN':
        light_data.angle = 1.0
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    return light_object


add_light("Area_Key_Soft",  'AREA', (6, -6, 8),  1600, size=10.0)
add_light("Area_Fill_Soft", 'AREA', (-6, -4, 6),  900, size=8.0)
add_light("Area_Back_Soft", 'AREA', (0, 7, 7),    600, size=6.0)
add_light("Sun_Subtle",     'SUN',  (3, -3, 8),   1.5)

# ── Target + Camera ──
target_empty = bpy.data.objects.new("Target", None)
scene.collection.objects.link(target_empty)

cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

track_node = cam.constraints.new(type='TRACK_TO')
track_node.target = target_empty
track_node.track_axis = 'TRACK_NEGATIVE_Z'
track_node.up_axis = 'UP_Y'


# =========================================
# دوال مساعدة لتحليل الـ mesh
# =========================================

def get_mesh_bounding_box(objects):
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))
    has_mesh = False
    for obj in objects:
        if obj.type == 'MESH' and len(obj.data.vertices) > 0:
            has_mesh = True
            for vertex in obj.bound_box:
                world_coord = obj.matrix_world @ Vector(vertex)
                for i in range(3):
                    if world_coord[i] < min_coords[i]:
                        min_coords[i] = world_coord[i]
                    if world_coord[i] > max_coords[i]:
                        max_coords[i] = world_coord[i]
    return (min_coords, max_coords) if has_mesh else (None, None)


def compute_surface_area(obj):
    if obj.type != 'MESH':
        return 0.0
    mesh = obj.data
    total = 0.0
    for poly in mesh.polygons:
        verts = [obj.matrix_world @ mesh.vertices[v].co for v in poly.vertices]
        if len(verts) >= 3:
            for i in range(1, len(verts) - 1):
                a = verts[i] - verts[0]
                b = verts[i + 1] - verts[0]
                total += a.cross(b).length * 0.5
    return round(total, 6)


def compute_volume_estimate(min_c, max_c):
    if min_c is None:
        return None
    dims = max_c - min_c
    return round(dims.x * dims.y * dims.z, 6)


def check_manifold(obj):
    if obj.type != 'MESH':
        return None, None
    mesh = obj.data
    edge_face_count = {}
    for poly in mesh.polygons:
        for e in poly.edge_keys:
            key = tuple(sorted(e))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1
    non_manifold = sum(1 for v in edge_face_count.values() if v != 2)
    return (non_manifold == 0), non_manifold


def count_connected_components(obj):
    if obj.type != 'MESH':
        return None
    mesh = obj.data
    if not mesh.vertices:
        return 0
    n = len(mesh.vertices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in mesh.edges:
        union(edge.vertices[0], edge.vertices[1])

    return len(set(find(i) for i in range(n)))


def estimate_symmetry(min_c, max_c):
    if min_c is None:
        return None
    dims = max_c - min_c
    axes = {'X': dims.x, 'Y': dims.y, 'Z': dims.z}
    return sorted(axes.items(), key=lambda kv: kv[1])[0][0]


def compute_compactness(surface_area, volume):
    if not surface_area or not volume or volume == 0:
        return None
    try:
        return round((surface_area ** 3) / (36 * math.pi * (volume ** 2)), 6)
    except Exception:
        return None


def compute_elongation(min_c, max_c):
    if min_c is None:
        return None
    dims = sorted([
        abs(max_c.x - min_c.x),
        abs(max_c.y - min_c.y),
        abs(max_c.z - min_c.z)
    ])
    return round(dims[2] / dims[0], 4) if dims[0] > 0 else None


def compute_center_of_mass(mesh_objects):
    weighted_sum = Vector((0, 0, 0))
    total_area = 0.0
    for obj in mesh_objects:
        area = compute_surface_area(obj)
        weighted_sum += obj.matrix_world.translation * area
        total_area += area
    if total_area == 0:
        return None
    com = weighted_sum / total_area
    return [round(com.x, 4), round(com.y, 4), round(com.z, 4)]


def stability_score(min_c, max_c, com):
    if min_c is None or com is None:
        return None
    base_half_x = (max_c.x - min_c.x) / 2
    base_half_y = (max_c.y - min_c.y) / 2
    if base_half_x == 0 or base_half_y == 0:
        return None
    offset_x = abs(com[0] - ((max_c.x + min_c.x) / 2)) / base_half_x
    offset_y = abs(com[1] - ((max_c.y + min_c.y) / 2)) / base_half_y
    return max(0.0, min(1.0, round(1.0 - max(offset_x, offset_y), 4)))


# =========================================
# استيراد الموديل
# =========================================
if extension == ".fbx":
    bpy.ops.import_scene.fbx(filepath=MODEL_PATH)
elif extension in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=MODEL_PATH)

imported = bpy.context.selected_objects
if not imported:
    fail("Import produced no objects (file may be empty or corrupted)")

mesh_objects = [o for o in imported if o.type == 'MESH']
if not mesh_objects:
    fail("No mesh objects found in imported file")

# تنعيم الأسطح + تعديل بسيط على الخامات
for obj in mesh_objects:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    r = node.inputs.get('Roughness')
                    if r:
                        r.default_value = 0.6
                    m = node.inputs.get('Metallic')
                    if m and m.default_value > 0.8:
                        m.default_value = 0.5

# =========================================
# الحسابات
# =========================================
min_c, max_c = get_mesh_bounding_box(imported)
if min_c is None:
    fail("Could not compute bounding box (no valid mesh geometry)")

total_verts = sum(len(o.data.vertices) for o in mesh_objects)
total_faces = sum(len(o.data.polygons) for o in mesh_objects)
total_edges = sum(len(o.data.edges) for o in mesh_objects)
total_surface = sum(compute_surface_area(o) for o in mesh_objects)
volume_est = compute_volume_estimate(min_c, max_c)

all_manifold = True
total_holes = 0
for o in mesh_objects:
    is_m, holes = check_manifold(o)
    if is_m is not None and not is_m:
        all_manifold = False
    if holes:
        total_holes += holes

total_components = sum(count_connected_components(o) or 0 for o in mesh_objects)

symmetry = estimate_symmetry(min_c, max_c)
compactness = compute_compactness(total_surface, volume_est)
elongation = compute_elongation(min_c, max_c)
com = compute_center_of_mass(mesh_objects)
stab = stability_score(min_c, max_c, com)

size = (max_c - min_c).length
distance = max(size * 1.8, 4.0)
iso_dist = distance * 0.75
center = (min_c + max_c) / 2
target_empty.location = center

# =========================================
# الزوايا والرندر
# =========================================
views = [
    ("front",            Vector((0, -distance, center.z))),
    ("back",              Vector((0,  distance, center.z))),
    ("top",               Vector((0, 0, center.z + distance))),
    ("bottom",            Vector((0, 0, center.z - distance))),
    ("right",             Vector((distance, 0, center.z))),
    ("left",              Vector((-distance, 0, center.z))),
    ("front_right_high",  Vector((iso_dist, -iso_dist, center.z + iso_dist * 0.8))),
    ("front_left_high",   Vector((-iso_dist, -iso_dist, center.z + iso_dist * 0.8))),
    ("back_right_high",   Vector((iso_dist,  iso_dist, center.z + iso_dist * 0.8))),
    ("back_left_high",    Vector((-iso_dist,  iso_dist, center.z + iso_dist * 0.8))),
    ("front_right_low",   Vector((iso_dist, -iso_dist, center.z - iso_dist * 0.4))),
    ("back_left_low",     Vector((-iso_dist,  iso_dist, center.z - iso_dist * 0.4))),
]

render_links = []
for idx, (view_name, pos) in enumerate(views):
    cam.location = pos
    bpy.context.view_layer.update()

    image_path = os.path.join(RENDERS_FOLDER, f"view_{idx + 1:02d}.png")
    scene.render.filepath = image_path

    start = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - start

    render_links.append({
        "view": view_name,
        "file": f"view_{idx + 1:02d}.png",
        "render_time_s": round(elapsed, 2),
        "camera_position": [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)]
    })

# =========================================
# بناء وحفظ JSON النهائي
# =========================================
model_data = {
    "id": model_id,
    "mesh_stats": {
        "vertices": total_verts,
        "faces": total_faces,
        "edges": total_edges
    },
    "bounding_box": [
        round(min_c.x, 4), round(min_c.y, 4), round(min_c.z, 4),
        round(max_c.x, 4), round(max_c.y, 4), round(max_c.z, 4)
    ],
    "surface_area": round(total_surface, 4),
    "volume_estimate": volume_est,
    "topology": {
        "is_manifold": all_manifold,
        "holes_count": total_holes,
        "connected_components": total_components
    },
    "shape_descriptors": {
        "symmetry_axis": symmetry,
        "compactness": compactness,
        "elongation": elongation,
        "curvature_mean": None
    },
    "physics_proxy": {
        "center_of_mass": com,
        "stability_score": stab
    },
    "render_links": {
        "views_count": len(render_links),
        "views": [x["file"] for x in render_links]
    }
}

with open(GEOMETRY_PATH, "w", encoding="utf8") as f:
    json.dump(model_data, f, indent=4, ensure_ascii=False)

print(f"[OK] {model_id} -> {GEOMETRY_PATH}")

# ── تنظيف الذاكرة (اختياري، مفيد لو بقيت العملية شغالة) ──
protected = {cam.name, target_empty.name}
bpy.ops.object.select_all(action='DESELECT')
for obj in imported:
    if obj and obj.name not in protected:
        obj.select_set(True)
bpy.ops.object.delete()

for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
    for block in list(collection):
        if block.users == 0:
            collection.remove(block)

bpy.data.orphans_purge(do_recursive=True)

sys.exit(0)
