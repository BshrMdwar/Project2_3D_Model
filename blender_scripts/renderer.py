import os
import sys
import math
import json
import time
import shutil
import argparse

import bpy
from mathutils import Vector

# =========================================
# ARGUMENT PARSING (params come from Django)
# =========================================


if "--" in sys.argv:
    cli_args = sys.argv[sys.argv.index("--") + 1:]
else:
    cli_args = []

parser = argparse.ArgumentParser(description="Render a 3D model and extract features.")
parser.add_argument("--model", required=True,
                     help="Model filename (resolved inside temp/) or full path.")
parser.add_argument("--uid", required=True,
                     help="Unique identifier for this model — used to name the output render folder.")
parser.add_argument("--samples", type=int, default=32,
                     help="Cycles render samples (default: 32).")
parser.add_argument("--resolution", type=int, default=768,
                     help="Square render resolution in pixels (default: 768).")
parser.add_argument("--views", default="all",
                     help="Comma-separated view names to render, or 'all' (default: all).")
parser.add_argument("--gpu-backend", default="OPTIX", choices=["OPTIX", "CUDA"],
                     help="Preferred GPU backend (default: OPTIX, falls back to CUDA/CPU).")
parser.add_argument("--force", action="store_true",
                     help="Re-render even if outputs already exist.")
parser.add_argument("--keep-temp", action="store_true",
                     help="Don't delete the source model from temp/ after rendering.")

args = parser.parse_args(cli_args)

# =========================================
# PATHS
# =========================================

PUBLIC_ROOT = os.path.join("/assets", "public")
TEMP_FOLDER = os.path.join("/assets", "temp")
MODEL_ROOT = os.path.join(PUBLIC_ROOT, args.uid)
RENDERS_ROOT = os.path.join(MODEL_ROOT, "renders")
GEOMETRY_ROOT = os.path.join(MODEL_ROOT, "geometry")
MODEL_STORE_ROOT = os.path.join(MODEL_ROOT, "model")

for folder in (PUBLIC_ROOT, TEMP_FOLDER, MODEL_ROOT, RENDERS_ROOT, GEOMETRY_ROOT, MODEL_STORE_ROOT):
    os.makedirs(folder, exist_ok=True)

if os.path.isabs(args.model) or os.sep in args.model or "/" in args.model:
    MODEL_PATH = os.path.abspath(args.model)
else:
    MODEL_PATH = os.path.join(TEMP_FOLDER, args.model)

if not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    sys.exit(1)

extension = os.path.splitext(MODEL_PATH)[1].lower()
if extension not in (".glb", ".gltf"):
    print(f"[ERROR] Unsupported model format '{extension}'. Only .glb / .gltf are supported.")
    sys.exit(1)

model_id = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# Only ever delete a model that actually resolved to TEMP_FOLDER — never an
# arbitrary absolute path the caller passed in via --model.
MODEL_IN_TEMP = os.path.dirname(os.path.abspath(MODEL_PATH)) == os.path.abspath(TEMP_FOLDER)

# Where the model is persisted once rendering succeeds — lives alongside
# renders/ and geometry/ under PUBLIC_ROOT/<uid>/model/, so it survives even
# if the temp copy is cleaned up (or was never in temp/ to begin with).
STORED_MODEL_PATH = os.path.join(MODEL_STORE_ROOT, os.path.basename(MODEL_PATH))

# =========================================
# GPU ACTIVATION
# =========================================


def enable_gpu_rendering(preferred_backend="OPTIX"):
    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences

    backends_to_try = [preferred_backend]
    if preferred_backend != "CUDA":
        backends_to_try.append("CUDA")

    activated_backend = None
    for backend in backends_to_try:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.get_devices()
        gpu_devices = [d for d in cprefs.devices if d.type in ("OPTIX", "CUDA")]
        if gpu_devices:
            activated_backend = backend
            break

    if activated_backend is None:
        print("[GPU] No compatible GPU device found (OPTIX/CUDA) — falling back to CPU.")
        return None

    for device in cprefs.devices:
        device.use = device.type == activated_backend
        if device.use:
            print(f"[GPU] Enabling device: {device.name} ({device.type})")

    print(f"[GPU] GPU rendering enabled via: {activated_backend}")
    return activated_backend


gpu_backend = enable_gpu_rendering(preferred_backend=args.gpu_backend)

# =========================================
# RENDER SETTINGS
# =========================================

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU" if gpu_backend else "CPU"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX" if gpu_backend else "OPENIMAGEDENOISE"
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transmission_bounces = 8
scene.cycles.transparent_max_bounces = 8

scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - High Contrast"
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.render.image_settings.file_format = "PNG"

# =========================================
# CLEAN SCENE
# =========================================

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

#! FIXING THE MODEs
def fix_output_permissions(root_path, dir_mode=0o777, file_mode=0o666):
    """
    Recursively chmod everything under root_path so a non-root host user
    can read/write it after the container exits. Docker containers usually
    run as root by default, so files land owned by root:root with root-only
    perms from the host's point of view unless we relax them here.
    """
    try:
        os.chmod(root_path, dir_mode)
        for dirpath, dirnames, filenames in os.walk(root_path):
            for d in dirnames:
                try:
                    os.chmod(os.path.join(dirpath, d), dir_mode)
                except OSError as e:
                    print(f"[WARN] chmod failed on dir {os.path.join(dirpath, d)}: {e}")
            for f in filenames:
                try:
                    os.chmod(os.path.join(dirpath, f), file_mode)
                except OSError as e:
                    print(f"[WARN] chmod failed on file {os.path.join(dirpath, f)}: {e}")
        print(f"[PERMISSIONS FIXED] {root_path} -> dirs={oct(dir_mode)}, files={oct(file_mode)}")
    except OSError as e:
        print(f"[WARN] Failed to fix permissions on {root_path}: {e}")

# =========================================
# WORLD & LIGHTS
# =========================================

world = scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 0.8


def add_light(name, light_type, location, energy, size):
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_data.energy = energy
    if light_type == "AREA":
        light_data.shape = "SQUARE"
        light_data.size = size
    elif light_type == "SUN":
        light_data.angle = 1.0
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    return light_object


add_light("Area_Key_Soft", "AREA", (6, -6, 8), 1600, size=10.0)
add_light("Area_Fill_Soft", "AREA", (-6, -4, 6), 900, size=8.0)
add_light("Area_Back_Soft", "AREA", (0, 7, 7), 600, size=6.0)
add_light("Sun_Subtle", "SUN", (3, -3, 8), 1.5, size=1.5)

scene.render.film_transparent = True

# =========================================
# CAMERA
# =========================================

target_empty = bpy.data.objects.get("Target")
if target_empty is None:
    target_empty = bpy.data.objects.new("Target", None)
    scene.collection.objects.link(target_empty)

cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

track_node = cam.constraints.new(type="TRACK_TO")
track_node.target = target_empty
track_node.track_axis = "TRACK_NEGATIVE_Z"
track_node.up_axis = "UP_Y"

# =========================================
# HELPER FUNCTIONS
# =========================================

EPSILON = 1e-6


def get_mesh_bounding_box(objects):
    min_coords = Vector((float("inf"),) * 3)
    max_coords = Vector((float("-inf"),) * 3)
    has_mesh = False
    for obj in objects:
        if obj.type == "MESH" and len(obj.data.vertices) > 0:
            has_mesh = True
            for vertex in obj.bound_box:
                world_coord = obj.matrix_world @ Vector(vertex)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_coord[i])
                    max_coords[i] = max(max_coords[i], world_coord[i])
    return (min_coords, max_coords) if has_mesh else (None, None)


ALL_VIEW_NAMES = [
    "front", "back", "top", "bottom", "right", "left",
    "front_right_high", "front_left_high", "back_right_high", "back_left_high",
    "front_right_low", "back_left_low",
]

if args.views.strip().lower() == "all":
    requested_views = ALL_VIEW_NAMES
else:
    requested_views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = set(requested_views) - set(ALL_VIEW_NAMES)
    if unknown:
        print(f"[ERROR] Unknown view name(s): {', '.join(sorted(unknown))}")
        print(f"        Valid views: {', '.join(ALL_VIEW_NAMES)}")
        sys.exit(1)

# =========================================
# PROCESS THE MODEL
# =========================================

print(f"\nProcessing {model_id} (uid={args.uid}, {extension.upper()}) from {MODEL_PATH}")

bpy.ops.import_scene.gltf(filepath=MODEL_PATH)
imported = bpy.context.selected_objects

# original_materials_data = extract_materials(imported)

for obj in imported:
    if obj.type == "MESH":
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        if obj.data.materials:
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == "BSDF_PRINCIPLED":
                            r = node.inputs.get("Roughness") or node.inputs.get("roughness")
                            if r:
                                r.default_value = 0.6
                            m = node.inputs.get("Metallic") or node.inputs.get("metallic")
                            if m and m.default_value > 0.8:
                                m.default_value = 0.5

min_c, max_c = get_mesh_bounding_box(imported)
mesh_objects = [o for o in imported if o.type == "MESH"]

total_verts = sum(len(o.data.vertices) for o in mesh_objects)
total_faces = sum(len(o.data.polygons) for o in mesh_objects)
total_edges = sum(len(o.data.edges) for o in mesh_objects)

objects_count = len(imported)

if min_c and max_c:
    width = round(max_c.x - min_c.x, 4)
    depth = round(max_c.y - min_c.y, 4)
    height = round(max_c.z - min_c.z, 4)
    safe_width = width if width > 0 else EPSILON
    safe_depth = depth if depth > 0 else EPSILON
    aspect_hw = round(height / safe_width, 4)
    aspect_hd = round(height / safe_depth, 4)
    aspect_wd = round(width / safe_depth, 4)
    center = (min_c + max_c) / 2
    size = (max_c - min_c).length
    distance = max(size * 1.8, 4.0)
    iso_dist = distance * 0.75
else:
    width = depth = height = 0.0
    aspect_hw = aspect_hd = aspect_wd = 0.0
    center = Vector((0, 0, 0))
    distance = 5.0
    iso_dist = 3.75

view_positions = {
    "front":            Vector((0, -distance, center.z)),
    "back":             Vector((0, distance, center.z)),
    "top":              Vector((0, 0, center.z + distance)),
    "bottom":           Vector((0, 0, center.z - distance)),
    "right":            Vector((distance, 0, center.z)),
    "left":             Vector((-distance, 0, center.z)),
    "front_right_high": Vector((iso_dist, -iso_dist, center.z + iso_dist * 0.8)),
    "front_left_high":  Vector((-iso_dist, -iso_dist, center.z + iso_dist * 0.8)),
    "back_right_high":  Vector((iso_dist, iso_dist, center.z + iso_dist * 0.8)),
    "back_left_high":   Vector((-iso_dist, iso_dist, center.z + iso_dist * 0.8)),
    "front_right_low":  Vector((iso_dist, -iso_dist, center.z - iso_dist * 0.4)),
    "back_left_low":    Vector((-iso_dist, iso_dist, center.z - iso_dist * 0.4)),
}

views = [(name, view_positions[name]) for name in requested_views]

target_empty.location = center

# Renders and geometry live under PUBLIC_ROOT/<uid>/renders and
# PUBLIC_ROOT/<uid>/geometry, so re-uploads / same-named files across
# different uids never collide.
RENDERS_FOLDER = RENDERS_ROOT
os.makedirs(RENDERS_FOLDER, exist_ok=True)
GEOMETRY_PATH = os.path.join(GEOMETRY_ROOT, "geometry.json")

render_links = []
for idx, (view_name, pos) in enumerate(views):
    cam.location = pos
    bpy.context.view_layer.update()

    filename = f"{view_name}.png"
    image_path = os.path.join(RENDERS_FOLDER, filename)
    scene.render.filepath = image_path

    start = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed = time.time() - start

    render_links.append({
        "view": view_name,
        "file": filename,
        "render_time_s": round(elapsed, 2),
        "camera_position": [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)],
    })

# =========================================
# PERSIST THE SOURCE MODEL INTO THE UID DIR
# =========================================

# Copy (not move) so MODEL_PATH still exists further down for the
# temp-cleanup step below, regardless of whether it lives in temp/ or was
# passed in as an external absolute path.
try:
    shutil.copy2(MODEL_PATH, STORED_MODEL_PATH)
    print(f"[MODEL STORED] {MODEL_PATH} -> {STORED_MODEL_PATH}")
    model_stored_ok = True
except OSError as e:
    print(f"[WARN] Failed to store model at {STORED_MODEL_PATH}: {e}")
    model_stored_ok = False

model_data = {
    "model_id": model_id,
    "dimensions": {
        "width": width, "depth": depth, "height": height,
        "aspect_hw": aspect_hw, "aspect_hd": aspect_hd, "aspect_wd": aspect_wd,
    },
    "vertices": total_verts,
    "faces": total_faces,
    "render_links": {
        "views_count": len(render_links),
        "views": [x["file"] for x in render_links],
    },
    "_debug_raw": {
        "edges": total_edges,
    },
}

with open(GEOMETRY_PATH, "w", encoding="utf8") as f:
    json.dump(model_data, f, indent=4, ensure_ascii=False)

print(f"[OK] {args.uid}")

# Cleanup: remove imported model only, keep camera + target
protected = {cam.name, target_empty.name}
bpy.ops.object.select_all(action="DESELECT")
for obj in imported:
    if obj and obj.name not in protected:
        obj.select_set(True)
bpy.ops.object.delete()

for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
    for block in list(collection):
        if block.users == 0:
            collection.remove(block)

bpy.data.orphans_purge(do_recursive=True)

print("[MEMORY CLEANED]")

# Delete the source model from temp/ now that it's been persisted into the
# uid directory, unless the caller asked to keep it or it wasn't actually a
# temp file to begin with.
if not args.keep_temp and MODEL_IN_TEMP:
    try:
        os.remove(MODEL_PATH)
        print(f"[TEMP CLEANED] Removed source model: {MODEL_PATH}")
    except OSError as e:
        print(f"[WARN] Failed to remove temp model {MODEL_PATH}: {e}")

fix_output_permissions(MODEL_ROOT)
print(f"[DONE] Renders -> {RENDERS_FOLDER}")
print(f"[DONE] Geometry -> {GEOMETRY_PATH}")
print(f"[DONE] Model -> {STORED_MODEL_PATH if model_stored_ok else '(not stored)'}")