import os
import sys
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/process-3d/', methods=['POST'])
def process_3d():
    uploaded_file = request.files['file']
# ==========================================
# AUTO START THROUGH BLENDER
# ==========================================

try:

    import bpy
    from mathutils import Vector

except ImportError:

    BLENDER = (
        r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"
    )

    script = os.path.abspath(__file__)

    import subprocess

    subprocess.run([

        BLENDER,

        "--background",

        "--python",

        script

    ])

    sys.exit()

# import bpy
import os
import math
import json
import time
# from mathutils import Vector

import sys

# =========================================
# INPUT
# =========================================

# =========================================
# DATASET PATH AUTO DISCOVERY
# =========================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_ROOT = os.path.join(
    SCRIPT_DIR,
    "dataset"
)

MODELS_FOLDER = os.path.join(
    DATASET_ROOT,
    "models"
)

RENDERS_ROOT = os.path.join(
    DATASET_ROOT,
    "renders"
)

GEOMETRY_ROOT = os.path.join(
    DATASET_ROOT,
    "geometry"
)

for folder in [

    DATASET_ROOT,
    MODELS_FOLDER,
    RENDERS_ROOT,
    GEOMETRY_ROOT

]:

    os.makedirs(
        folder,
        exist_ok=True
    )
# =========================================
# RENDER SETTINGS (HIGH QUALITY REALISM)
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

# =========================================
# CLEAN SCENE
# =========================================

# تنظيف المشهد المبدئي فقط
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


# =========================================
# WORLD & ENVIRONMENT
# =========================================

world = scene.world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 0.8

# =========================================
# LIGHTS
# =========================================


def add_light(name, type, location, energy, size):
    light_data = bpy.data.lights.new(name=name, type=type)
    light_data.energy = energy
    if type == 'AREA':
        light_data.shape = 'SQUARE'
        light_data.size = size
    elif type == 'SUN':
        light_data.angle = 1.0
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    return light_object


add_light("Area_Key_Soft",  'AREA', (6, -6, 8),  1600, size=10.0)
add_light("Area_Fill_Soft", 'AREA', (-6, -4, 6),  900, size=8.0)
add_light("Area_Back_Soft", 'AREA', (0, 7, 7),    600, size=6.0)
add_light("Sun_Subtle",     'SUN',  (3, -3, 8),   1.5, size=1.5)

scene.render.film_transparent = True

# =========================================
# CAMERA
# =========================================

# ==========================
# SAFE TARGET CREATION
# ==========================

target_empty = bpy.data.objects.get(
    "Target"
)

if target_empty is None:

    target_empty = bpy.data.objects.new(
        "Target",
        None
    )

    scene.collection.objects.link(
        target_empty
    )

# target_empty.location = center
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

track_node = cam.constraints.new(type='TRACK_TO')
track_node.target = target_empty
track_node.track_axis = 'TRACK_NEGATIVE_Z'
track_node.up_axis = 'UP_Y'

# =========================================
# HELPER FUNCTIONS
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
    """حساب المساحة السطحية الكلية للـ mesh بالوحدات العالمية."""
    if obj.type != 'MESH':
        return 0.0
    mesh = obj.data
    scale = obj.matrix_world.to_scale()
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
    """تقدير الحجم من الـ bounding box (ليس الحجم الحقيقي)."""
    if min_c is None:
        return None
    dims = max_c - min_c
    return round(dims.x * dims.y * dims.z, 6)


def check_manifold(obj):
    """
    يتحقق هل الـ mesh سليم (manifold).
    يحسب الحواف التي لا تشترك بها وجهان بالضبط.
    """
    if obj.type != 'MESH':
        return None, None
    mesh = obj.data
    edge_face_count = {}
    for poly in mesh.polygons:
        edges = list(poly.edge_keys)
        for e in edges:
            key = tuple(sorted(e))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1
    non_manifold = sum(1 for v in edge_face_count.values() if v != 2)
    is_manifold = (non_manifold == 0)
    return is_manifold, non_manifold


def count_connected_components(obj):
    """
    يعد عدد المكونات المنفصلة (islands) داخل الـ mesh.
    """
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
    """
    تخمين محور التماثل الرئيسي بناءً على أبعاد الـ bounding box.
    """
    if min_c is None:
        return None
    dims = max_c - min_c
    axes = {'X': dims.x, 'Y': dims.y, 'Z': dims.z}
    sorted_axes = sorted(axes.items(), key=lambda kv: kv[1])
    # المحور ذو أصغر امتداد هو الأرجح لمحور التماثل
    return sorted_axes[0][0]


def compute_compactness(surface_area, volume):
    """
    Compactness = (surface_area^3) / (36π × volume^2)
    قيمة 1 تعني كرة مثالية — كلما كبرت القيمة كلما قل التكثيف.
    """
    if not surface_area or not volume or volume == 0:
        return None
    try:
        val = (surface_area ** 3) / (36 * math.pi * (volume ** 2))
        return round(val, 6)
    except:
        return None


def compute_elongation(min_c, max_c):
    """
    Elongation = أطول بعد / أقصر بعد
    قيمة 1 تعني شكل متكافئ، كلما كبرت كلما كان الشكل ممدوداً.
    """
    if min_c is None:
        return None
    dims = sorted([abs(max_c.x - min_c.x),
                   abs(max_c.y - min_c.y),
                   abs(max_c.z - min_c.z)])
    return round(dims[2] / dims[0], 4) if dims[0] > 0 else None


def compute_center_of_mass(objects):
    """متوسط مراكز الـ mesh المرجحة بالمساحة السطحية."""
    weighted_sum = Vector((0, 0, 0))
    total_area = 0.0
    for obj in objects:
        if obj.type == 'MESH':
            area = compute_surface_area(obj)
            center = obj.matrix_world.translation
            weighted_sum += center * area
            total_area += area
    if total_area == 0:
        return None
    com = weighted_sum / total_area
    return [round(com.x, 4), round(com.y, 4), round(com.z, 4)]


def stability_score(min_c, max_c, com):
    """
    تقدير بسيط للاستقرار:
    نسبة المسافة الأفقية لمركز الكتلة إلى نصف قاعدة الـ bounding box.
    قيمة <= 1 تعني مستقر.
    """
    if min_c is None or com is None:
        return None
    base_half_x = (max_c.x - min_c.x) / 2
    base_half_y = (max_c.y - min_c.y) / 2
    if base_half_x == 0 or base_half_y == 0:
        return None
    offset_x = abs(com[0] - ((max_c.x + min_c.x) / 2)) / base_half_x
    offset_y = abs(com[1] - ((max_c.y + min_c.y) / 2)) / base_half_y
    score = round(1.0 - max(offset_x, offset_y), 4)
    return max(0.0, min(1.0, score))


def lod_suggestions(vert_count):
    """
    اقتراح مستويات LOD بناءً على عدد الـ vertices.
    """
    if vert_count is None:
        return []
    suggestions = []
    if vert_count > 100_000:
        suggestions.append(
            {"level": "LOD0", "target_verts": vert_count,        "use": "Hero / close-up"})
        suggestions.append(
            {"level": "LOD1", "target_verts": vert_count // 4,   "use": "Mid distance"})
        suggestions.append(
            {"level": "LOD2", "target_verts": vert_count // 16,  "use": "Far distance"})
        suggestions.append(
            {"level": "LOD3", "target_verts": vert_count // 64,  "use": "Background / billboard"})
    elif vert_count > 20_000:
        suggestions.append(
            {"level": "LOD0", "target_verts": vert_count,       "use": "Hero / close-up"})
        suggestions.append(
            {"level": "LOD1", "target_verts": vert_count // 4,  "use": "Mid distance"})
        suggestions.append(
            {"level": "LOD2", "target_verts": vert_count // 10, "use": "Far distance"})
    elif vert_count > 5_000:
        suggestions.append(
            {"level": "LOD0", "target_verts": vert_count,      "use": "Hero / close-up"})
        suggestions.append(
            {"level": "LOD1", "target_verts": vert_count // 4, "use": "Far distance"})
    else:
        suggestions.append(
            {"level": "LOD0", "target_verts": vert_count, "use": "Single LOD sufficient"})
    return suggestions


def extract_materials(objects):
    """
    استخراج بيانات الخامات: الاسم، الألوان، roughness، metallic، شفافية.
    """
    mats_data = []
    seen = set()
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for mat in obj.data.materials:
            if not mat or mat.name in seen:
                continue
            seen.add(mat.name)
            entry = {
                "name": mat.name,
                "use_nodes": mat.use_nodes,
                "blend_method": mat.blend_method,
                "principled": None
            }
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        def _get(name):
                            inp = node.inputs.get(name)
                            if inp is None:
                                return None
                            val = inp.default_value
                            if hasattr(val, '__iter__'):
                                return [round(float(v), 4) for v in val]
                            return round(float(val), 4)

                        entry["principled"] = {
                            "base_color":        _get("Base Color"),
                            "roughness":         _get("Roughness"),
                            "metallic":          _get("Metallic"),
                            "specular":          _get("Specular IOR Level") or _get("Specular"),
                            "transmission":      _get("Transmission Weight") or _get("Transmission"),
                            "emission_strength": _get("Emission Strength"),
                            "alpha":             _get("Alpha"),
                            "ior":               _get("IOR"),
                        }
                        break
            mats_data.append(entry)
    return mats_data


# =========================================
# IMPORT + RENDER + DATA EXTRACTION LOOP
# =========================================


for file in os.listdir(MODELS_FOLDER):

    extension = os.path.splitext(
        file
    )[1].lower()

    if extension not in [

        ".fbx", ".glb", ".gltf"

    ]:

        continue

    MODEL_PATH = os.path.join(
        MODELS_FOLDER,
        file
    )

    model_id = os.path.splitext(
        file
    )[0]

    print(f"\nProcessing {model_id} ({extension.upper()})")
    # التعديل 2: تحديد دالة الاستيراد المناسبة بناءً على اللاحقة
    if extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=MODEL_PATH)
    elif extension in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=MODEL_PATH)

# ---- 1. استيراد الملف ----
    # bpy.ops.import_scene.fbx(
    #    filepath=MODEL_PATH
   # )
    imported = bpy.context.selected_objects

    # ---- 2. معالجة الخامات ----
    for obj in imported:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
            if obj.data.materials:
                for mat in obj.data.materials:
                    if mat and mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                r = node.inputs.get(
                                    'Roughness') or node.inputs.get('roughness')
                                if r:
                                    r.default_value = 0.6
                                m = node.inputs.get(
                                    'Metallic') or node.inputs.get('metallic')
                                if m and m.default_value > 0.8:
                                    m.default_value = 0.5

    # ---- 3. حسابات الـ mesh ----
    min_c, max_c = get_mesh_bounding_box(imported)

    mesh_objects = [o for o in imported if o.type == 'MESH']

    total_verts = sum(len(o.data.vertices) for o in mesh_objects)
    total_faces = sum(len(o.data.polygons) for o in mesh_objects)
    total_edges = sum(len(o.data.edges) for o in mesh_objects)

    total_surface = sum(compute_surface_area(o) for o in mesh_objects)
    volume_est = compute_volume_estimate(min_c, max_c)

    # Topology
    all_manifold = True
    total_holes = 0
    for o in mesh_objects:
        is_m, holes = check_manifold(o)
        if is_m is not None and not is_m:
            all_manifold = False
        if holes:
            total_holes += holes
    total_components = sum(count_connected_components(o)
                           or 0 for o in mesh_objects)

    # Shape descriptors
    symmetry = estimate_symmetry(min_c, max_c)
    compactness = compute_compactness(total_surface, volume_est)
    elongation = compute_elongation(min_c, max_c)

    # Physics
    com = compute_center_of_mass(mesh_objects)
    stab = stability_score(min_c, max_c, com)

    # LOD
    lod_recs = lod_suggestions(total_verts)

    # Materials
    materials = extract_materials(imported)

    # Bounding box
    if min_c and max_c:
        bbox_data = {
            "min": [round(min_c.x, 4), round(min_c.y, 4), round(min_c.z, 4)],
            "max": [round(max_c.x, 4), round(max_c.y, 4), round(max_c.z, 4)],
            "center": [round((min_c.x+max_c.x)/2, 4),
                       round((min_c.y+max_c.y)/2, 4),
                       round((min_c.z+max_c.z)/2, 4)],
            "dimensions": {
                "width_x":  round(max_c.x - min_c.x, 4),
                "depth_y":  round(max_c.y - min_c.y, 4),
                "height_z": round(max_c.z - min_c.z, 4),
            },
            "diagonal": round((max_c - min_c).length, 4)
        }
        center = (min_c + max_c) / 2
        size = (max_c - min_c).length
        distance = max(size * 1.8, 4.0)
        iso_dist = distance * 0.75
    else:
        bbox_data = None
        center = Vector((0, 0, 0))
        distance = 5.0
        iso_dist = 3.75

    # ---- 4. تعريف الزوايا (12 زاوية) ----
    views = [
        # الزوايا الأساسية الستة (orthographic-style)
        ("front",            Vector((0, -distance, center.z))),
        ("back",             Vector((0,  distance, center.z))),
        ("top",              Vector((0, 0, center.z + distance))),
        ("bottom",           Vector((0, 0, center.z - distance))),
        ("right",            Vector((distance, 0, center.z))),
        ("left",             Vector((-distance, 0, center.z))),
        # الزوايا الإيزومترية العالية (4 زوايا)
        ("front_right_high", Vector((iso_dist, -iso_dist, center.z + iso_dist * 0.8))),
        ("front_left_high",  Vector((-iso_dist, -iso_dist, center.z + iso_dist * 0.8))),
        ("back_right_high",  Vector((iso_dist,  iso_dist, center.z + iso_dist * 0.8))),
        ("back_left_high",   Vector((-iso_dist,  iso_dist, center.z + iso_dist * 0.8))),
        # الزاويتان الإضافيتان: إيزومتريك منخفض أمامي وخلفي
        ("front_right_low",  Vector((iso_dist, -iso_dist, center.z - iso_dist * 0.4))),
        ("back_left_low",    Vector((-iso_dist,  iso_dist, center.z - iso_dist * 0.4))),
    ]

    target_empty.location = center

    # ---- 5. إنشاء مجلد فرعي خاص بالمودل ----
    RENDERS_FOLDER = os.path.join(

        RENDERS_ROOT,

        model_id

    )

    os.makedirs(

        RENDERS_FOLDER,

        exist_ok=True

    )

    GEOMETRY_PATH = os.path.join(

        GEOMETRY_ROOT,

        model_id + ".json"

    )

    # ---- 6. رندر + تسجيل أوقات الرندر ----
    render_links = []

    for idx, (view_name, pos) in enumerate(views):

        cam.location = pos

        bpy.context.view_layer.update()

        image_path = os.path.join(

            RENDERS_FOLDER,

            f"view_{idx+1:02d}.png"

        )

        scene.render.filepath = image_path

        start = time.time()

        bpy.ops.render.render(
            write_still=True
        )

        elapsed = time.time()-start

        render_links.append({

            "view": view_name,

            "file": f"view_{idx+1:02d}.png",

            "render_time_s": round(
                elapsed,
                2
            ),

            "camera_position": [
                round(pos.x, 4),
                round(pos.y, 4),
                round(pos.z, 4)
            ]

        })
    # ---- 7. بناء JSON النهائي (مطابق تماماً للهيكل الأساسي بدون أي زيادة) ----
    model_data = {
        "id": model_id,
        "mesh_stats": {
            "vertices":        total_verts,
            "faces":           total_faces,
            "edges":           total_edges
        },
        # جعل الـ bounding box عبارة عن قائمة مستقيمة [min_x, min_y, min_z, max_x, max_y, max_z]
        "bounding_box": [round(min_c.x, 4), round(min_c.y, 4), round(min_c.z, 4),
                         round(max_c.x, 4), round(max_c.y, 4), round(max_c.z, 4)] if min_c else None,

        "surface_area":    round(total_surface, 4),
        "volume_estimate": volume_est,

        "topology": {
            "is_manifold":          all_manifold,
            "holes_count":          total_holes,          # تم تعديل الاسم هنا ليطابق طلبك
            "connected_components": total_components
        },
        "shape_descriptors": {
            "symmetry_axis":   symmetry,
            "compactness":     compactness,
            "elongation":      elongation,
            # قيمة افتراضية لتجنب الـ null إذا كانت الداتابيز تمنعه
            "curvature_mean": None
        },
        "physics_proxy": {
            "center_of_mass":  com,
            "stability_score": stab
        },
        "render_links": {

            "views_count": len(
                render_links
            ),

            "views": [

                x["file"]

                for x in render_links

            ]

        }
    }

    # ---- 8. حفظ JSON ----
    with open(

        GEOMETRY_PATH,

        "w",

        encoding="utf8"

    ) as f:

        json.dump(

            model_data,

            f,

            indent=4,

            ensure_ascii=False

        )

    print(

        f"[OK] {model_id}"

    )
    # ---- 9. تنظيف الذاكرة (الموديلات المستوردة فقط — الكاميرا والـ Target محفوظة) ----
    protected = {cam.name, target_empty.name}

    # إلغاء تحديد كل شيء أولاً
    bpy.ops.object.select_all(action='DESELECT')

    # تحديد الموديلات المستوردة فقط للحذف
    for obj in imported:
        if obj and obj.name not in protected:
            obj.select_set(True)

    bpy.ops.object.delete()

    # تنظيف البيانات اليتيمة (meshes, materials, images) مع الحفاظ على cameras وlights
    for collection in [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
    ]:
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)

    bpy.data.orphans_purge(do_recursive=True)

    print(
        "[MEMORY CLEANED]"
    )
    output_data = {
        "mesh_stats": {"vertices": 1200, "faces": 2000}, # مثال
        "ai_classification": {"label": "Chair", "confidence": 0.95}
    }
    return jsonify(output_data), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)