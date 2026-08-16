import os
import sys

# ==========================================
# AUTO START THROUGH BLENDER
# ==========================================

try:
    import bpy
    from mathutils import Vector

except ImportError:

    BLENDER = (
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
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

# IMPORTANT: convert_fbx.py (run BEFORE this script) converts any .fbx files
# in this folder into .glb files written right here, alongside native .glb/
# .gltf models. blender_worker.py just reads this one folder for anything
# with a .glb/.gltf extension — it never touches FBX or subprocess tools.
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
# GPU ACTIVATION (NVIDIA RTX 4050 / OptiX)
# =========================================


def enable_gpu_rendering(preferred_backend='OPTIX'):
    """
    يفعّل الرندر عبر كرت الشاشة (GPU) باستخدام Cycles.
    يحاول استخدام OPTIX أولاً (الأنسب لكروت RTX)، وإن لم تتوفر أجهزة
    متوافقة يرجع تلقائياً لـ CUDA. يفعّل جهاز الـ GPU فقط ويطفئ الـ CPU
    حتى لا يُستخدم بالتوازي (وهو أبطأ عادة من GPU حديث مثل RTX 4050).
    يرجع اسم الـ backend الذي تم تفعيله فعلياً، أو None إن لم يوجد أي جهاز.
    """
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    backends_to_try = [preferred_backend]
    if preferred_backend != 'CUDA':
        backends_to_try.append('CUDA')

    activated_backend = None

    for backend in backends_to_try:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            # الـ backend غير مدعوم على هذا النظام/إصدار بلندر
            continue

        # لازم نعيد استدعاء get_devices بعد كل تغيير لنوع الـ backend
        cprefs.get_devices()

        gpu_devices = [d for d in cprefs.devices if d.type in ('OPTIX', 'CUDA')]

        if gpu_devices:
            activated_backend = backend
            break

    if activated_backend is None:
        print("[GPU] لم يتم العثور على جهاز GPU متوافق (OPTIX/CUDA) — سيتم الاستمرار على CPU.")
        return None

    # تفعيل كل أجهزة الـ GPU المطابقة للـ backend المختار، وإيقاف الـ CPU
    for device in cprefs.devices:
        if device.type == activated_backend:
            device.use = True
            print(f"[GPU] تفعيل الجهاز: {device.name} ({device.type})")
        elif device.type == 'CPU':
            device.use = False
        else:
            device.use = False

    print(f"[GPU] تم تفعيل الرندر عبر: {activated_backend}")
    return activated_backend


gpu_backend = enable_gpu_rendering(preferred_backend='OPTIX')

# =========================================
# RENDER SETTINGS (HIGH QUALITY REALISM)
# =========================================

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# استخدام كرت الشاشة كجهاز معالجة، بدل الـ CPU
scene.cycles.device = 'GPU' if gpu_backend else 'CPU'

# عدد عينات مخفّض للحصول على أقصى سرعة رندر (تنازل عن بعض الجودة مقابل السرعة)
scene.cycles.samples = 32

scene.cycles.use_denoising = True
# مزيل ضوضاء OptiX (أسرع بكثير عند توفر كرت RTX، ويتطلب أن يكون OPTIX متاحاً)
scene.cycles.denoiser = 'OPTIX' if gpu_backend else 'OPENIMAGEDENOISE'

scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.01

# ارتدادات ضوء مخفّضة لتقليل زمن المعالجة لكل عينة
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
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

# قيمة إبسلون آمنة لتفادي القسمة على صفر عند حساب الكثافات النسبية
EPSILON = 1e-6


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


def safe_log(value):
    """log(value + 1) مع حماية من القيم السالبة أو None."""
    if value is None or value < 0:
        return 0.0
    return round(math.log(value + 1), 6)


def count_textures(objects):
    """
    عدد الصور/التكستشرز الفريدة المستخدمة داخل عقد الشيدر (Shader Nodes)
    لكل الخامات المرتبطة بمجسمات الموديل.
    """
    seen_images = set()
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    seen_images.add(node.image.name)
    return len(seen_images)


def check_has_uv(objects):
    """يتحقق إن كان أي mesh في الموديل يملك خريطة UV."""
    for obj in objects:
        if obj.type == 'MESH' and len(obj.data.uv_layers) > 0:
            return True
    return False


def compute_material_summary(mats_data):
    """
    متوسط قيم Principled BSDF الرئيسية (roughness, metallic) عبر كل
    الخامات المرتبطة بالموديل. يتم تجاهل الخامات التي لا تملك قيمة.
    """
    roughness_vals = []
    metallic_vals = []
    for m in mats_data:
        principled = m.get("principled")
        if not principled:
            continue
        r = principled.get("roughness")
        if r is not None:
            roughness_vals.append(r)
        met = principled.get("metallic")
        if met is not None:
            metallic_vals.append(met)

    avg_roughness = round(sum(roughness_vals) / len(roughness_vals), 4) \
        if roughness_vals else 0.0
    avg_metallic = round(sum(metallic_vals) / len(metallic_vals), 4) \
        if metallic_vals else 0.0

    return avg_roughness, avg_metallic


def compute_avg_base_color(mats_data):
    """
    متوسط الـ base_color (RGB, بدون قناة alpha) عبر كل الخامات التي تملك
    قيمة principled صالحة. مفيد كـ إشارة مباشرة للون السائد للمادة/الموديل،
    يخدم تصنيف materials.primary/secondary.
    """
    r_vals, g_vals, b_vals = [], [], []
    for m in mats_data:
        principled = m.get("principled")
        if not principled:
            continue
        color = principled.get("base_color")
        if color and len(color) >= 3:
            r_vals.append(color[0])
            g_vals.append(color[1])
            b_vals.append(color[2])

    if not r_vals:
        return [0.0, 0.0, 0.0]

    return [
        round(sum(r_vals) / len(r_vals), 4),
        round(sum(g_vals) / len(g_vals), 4),
        round(sum(b_vals) / len(b_vals), 4),
    ]


def compute_materials_breakdown(mats_data, mesh_objects):
    """
    لكل مادة: نسبة مساهمتها في المساحة السطحية الكلية للموديل + خصائصها
    (roughness, metallic, base_color). هذا أدق بكثير من متوسط عام واحد،
    لأن قطعة الأثاث الواحدة غالباً مركّبة من مواد مختلفة (خشب + قماش +
    معدن)، والمادة ذات أكبر مساهمة سطحية هي أقرب مرشح لـ materials.primary.
    """
    # نحسب المساحة السطحية لكل مادة عبر كل الأوبجكتس التي تستخدمها
    area_per_material = {}
    for obj in mesh_objects:
        if not obj.data.materials:
            continue
        obj_area = compute_surface_area(obj)
        mat_slots = [m for m in obj.data.materials if m]
        if not mat_slots or obj_area == 0:
            continue
        # تقسيم مبسّط: توزيع مساحة الأوبجكت بالتساوي على خاماته المرتبطة
        # (Blender لا يعطي مساحة لكل material slot مباشرة بدون تحليل بالـ polygon
        # material_index، وهذا تبسيط مقصود كافٍ لتحديد "المادة السائدة").
        share = obj_area / len(mat_slots)
        for mat in mat_slots:
            area_per_material[mat.name] = area_per_material.get(mat.name, 0.0) + share

    total_area = sum(area_per_material.values())

    breakdown = []
    for m in mats_data:
        name = m.get("name")
        principled = m.get("principled") or {}
        area = area_per_material.get(name, 0.0)
        share = round(area / total_area, 4) if total_area > 0 else 0.0
        breakdown.append({
            "material_name": name,
            "surface_area_share": share,
            "roughness": principled.get("roughness"),
            "metallic": principled.get("metallic"),
            "base_color_rgb": principled.get("base_color")[:3] if principled.get("base_color") else None,
        })

    # الأكثر مساهمة أولاً، يسهّل أخذ breakdown[0] كمرشح لـ materials.primary
    breakdown.sort(key=lambda x: x["surface_area_share"], reverse=True)
    return breakdown


def compute_texture_resolution(objects):
    """
    متوسط أبعاد (width, height) كل صور التكستشر الفريدة المستخدمة فعلياً
    في شيدرز الموديل. مؤشر غير مباشر على جودة/مصدر الموديل (احترافي مقابل
    هاوٍ) ومفيد كـ feature مساعد.
    """
    seen_images = {}
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    img = node.image
                    if img.name not in seen_images:
                        try:
                            w, h = img.size[0], img.size[1]
                        except Exception:
                            w, h = 0, 0
                        seen_images[img.name] = (w, h)

    if not seen_images:
        return {"avg_width": 0, "avg_height": 0}

    widths = [v[0] for v in seen_images.values() if v[0] > 0]
    heights = [v[1] for v in seen_images.values() if v[1] > 0]

    return {
        "avg_width": round(sum(widths) / len(widths)) if widths else 0,
        "avg_height": round(sum(heights) / len(heights)) if heights else 0,
    }


def is_already_rendered(model_id, renders_root, geometry_root, expected_views=12):
    """
    يتحقق إن كان الموديل قد تمت معالجته بالكامل في تشغيل سابق، عبر التأكد من:
    1. وجود مجلد الرندر الخاص به.
    2. احتوائه على كل عدد الصور المتوقع (view_01.png ... view_12.png).
    3. وجود ملف الـ JSON الخاص بجيومتريته.
    إذا تحققت الشروط الثلاثة، نعتبره مكتملاً ونتخطاه لتوفير وقت إعادة التشغيل.
    """
    renders_folder = os.path.join(renders_root, model_id)
    geometry_path = os.path.join(geometry_root, model_id + ".json")

    if not os.path.isdir(renders_folder):
        return False

    if not os.path.exists(geometry_path):
        return False

    expected_files = {f"view_{i+1:02d}.png" for i in range(expected_views)}
    existing_files = set(os.listdir(renders_folder))

    return expected_files.issubset(existing_files)


# =========================================
# IMPORT + RENDER + DATA EXTRACTION LOOP
# =========================================
# NOTE: this loop now only reads .glb / .gltf files from converted_glb/.
# All FBX conversion happens separately in convert_fbx.py, run BEFORE
# this script, with plain Python (not Blender's bundled Python).


for file in os.listdir(MODELS_FOLDER):

    extension = os.path.splitext(
        file
    )[1].lower()

    if extension not in [

        ".glb", ".gltf"

    ]:

        continue

    MODEL_PATH = os.path.join(
        MODELS_FOLDER,
        file
    )

    model_id = os.path.splitext(
        file
    )[0]

    # ---- 0. تخطي الموديلات المكتملة من تشغيل سابق (استئناف العمل) ----
    if is_already_rendered(model_id, RENDERS_ROOT, GEOMETRY_ROOT):
        print(f"[SKIP] {model_id}: already fully rendered, resuming past it")
        continue

    print(f"\nProcessing {model_id} ({extension.upper()})")

    bpy.ops.import_scene.gltf(filepath=MODEL_PATH)

    imported = bpy.context.selected_objects

    # ---- 2. معالجة الخامات ----
    # IMPORTANT: نستخرج بيانات المواد الأصلية (extract_materials) قبل أي تعديل
    # عليها لأغراض الرندرة، حتى لا "يتسرب" التعديل التجميلي (roughness=0.6
    # الثابت أدناه) إلى الـ JSON المخصص للتدريب. القيم المحفوظة بالجيومتري
    # لازم تعكس المادة الحقيقية للموديل، لا القيم المعدّلة لتفادي انعكاسات
    # الرندرة الغريبة.
    original_materials_data = extract_materials(imported)

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
    # bounding_box_volume هي نفس طريقة حساب volume_est حالياً (تقدير من الصندوق
    # المحيط)، نحتفظ بها كاسم صريح منفصل لأن volume_estimate قد يصبح لاحقاً
    # حجماً حقيقياً محسوباً من الـ mesh (mesh-based volume) بدل تقدير الصندوق،
    # وحينها occupancy_ratio يعبّر فعلاً عن "نسبة امتلاء الصندوق المحيط".
    bounding_box_volume = volume_est

    # Structure (topology مبسطة: فقط الأجزاء المنفصلة + عدد العناصر)
    total_components = sum(count_connected_components(o)
                           or 0 for o in mesh_objects)
    objects_count = len(imported)

    # Shape descriptors
    symmetry = estimate_symmetry(min_c, max_c)
    compactness = compute_compactness(total_surface, volume_est)
    elongation = compute_elongation(min_c, max_c)

    # Physics (center_of_mass تُستخدم داخلياً فقط لحساب الاستقرار ولا تُصدَّر)
    com = compute_center_of_mass(mesh_objects)
    stab = stability_score(min_c, max_c, com)

    # LOD
    lod_recs = lod_suggestions(total_verts)

    # Materials & Textures
    # نستخدم original_materials_data المحفوظة قبل تعديل الخامات لأغراض الرندرة
    # (انظر التعليق أعلاه عند استدعائها) بدل إعادة استخراجها الآن، لضمان أن
    # roughness/metallic/base_color المخزّنة تعكس المادة الأصلية للموديل.
    materials = original_materials_data
    materials_count = len(materials)
    textures_count = count_textures(imported)
    has_uv = check_has_uv(mesh_objects)
    avg_roughness, avg_metallic = compute_material_summary(materials)
    avg_base_color = compute_avg_base_color(materials)
    materials_breakdown = compute_materials_breakdown(materials, mesh_objects)
    texture_resolution = compute_texture_resolution(imported)

    # Mesh density (معالجة لوغاريتمية + كثافة نسبية بحماية من القسمة على صفر)
    log_vertices = safe_log(total_verts)
    log_faces = safe_log(total_faces)
    safe_volume = volume_est if volume_est and volume_est > 0 else EPSILON
    safe_surface = total_surface if total_surface and total_surface > 0 else EPSILON
    faces_per_volume = round(total_faces / safe_volume, 6)
    faces_per_area = round(total_faces / safe_surface, 6)

    # occupancy_ratio: نسبة الحجم المقدَّر للـ mesh إلى حجم الصندوق المحيط به.
    # قيمة قريبة من 1 = شكل مصمت يملأ صندوقه (خزانة مثلاً)، قيمة قريبة من 0 =
    # شكل مفتوح/مخرّم لا يملأ صندوقه (كرسي بهيكل نحيف مثلاً). حالياً
    # volume_estimate وbounding_box_volume محسوبتان بنفس الطريقة (تقدير
    # الصندوق)، لذلك القيمة اليوم ~1.0 دائماً؛ الحقل جاهز لتحسين volume_estimate
    # لاحقاً ليصبح حجماً حقيقياً من الـ mesh (mesh-based)، عندها occupancy_ratio
    # سيصبح فعلياً مفيداً كمؤشر شكل.
    safe_bbox_volume = bounding_box_volume if bounding_box_volume and bounding_box_volume > 0 else EPSILON
    occupancy_ratio = round((volume_est or 0.0) / safe_bbox_volume, 6)

    # Dimensions & aspect ratios (بديل الـ bounding box الخام)
    if min_c and max_c:
        width = round(max_c.x - min_c.x, 4)
        depth = round(max_c.y - min_c.y, 4)
        height = round(max_c.z - min_c.z, 4)
        safe_width = width if width > 0 else EPSILON
        safe_depth = depth if depth > 0 else EPSILON
        aspect_hw = round(height / safe_width, 4)
        aspect_hd = round(height / safe_depth, 4)
        aspect_wd = round(width / safe_depth, 4)
    else:
        width = depth = height = 0.0
        aspect_hw = aspect_hd = aspect_wd = 0.0

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
    # ---- 7. بناء JSON النهائي (ميزات ML-Ready معالجة ونسبية) ----
    model_data = {
        "model_id": model_id,

        "dimensions": {
            "width":     width,
            "depth":     depth,
            "height":    height,
            "aspect_hw": aspect_hw,
            "aspect_hd": aspect_hd,
            "aspect_wd": aspect_wd
        },

        "mesh_density": {
            "vertices":         total_verts,
            "faces":            total_faces,
            "log_vertices":     log_vertices,
            "log_faces":        log_faces,
            "faces_per_volume": faces_per_volume,
            "faces_per_area":   faces_per_area
        },

        "geometry": {
            "surface_area":       round(total_surface, 4),
            "volume_estimate":    volume_est if volume_est is not None else 0.0,
            "bounding_box_volume": bounding_box_volume if bounding_box_volume is not None else 0.0,
            "occupancy_ratio":    occupancy_ratio
        },

        "shape_descriptors": {
            "compactness":   compactness if compactness is not None else 0.0,
            "elongation":    elongation if elongation is not None else 0.0,
            "symmetry_axis": symmetry
        },

        "structure": {
            "connected_components": total_components,
            "objects_count":        objects_count
        },

        "materials_and_textures": {
            "materials_count":     materials_count,
            "textures_count":      textures_count,
            "has_uv":              has_uv,
            "avg_roughness":       avg_roughness,
            "avg_metallic":        avg_metallic,
            "avg_base_color_rgb":  avg_base_color,
            "materials_breakdown": materials_breakdown,
            "texture_resolution":  texture_resolution
        },

        "physics_proxy": {
            "stability_score": stab if stab is not None else 0.0
        },

        "render_links": {
            "views_count": len(render_links),
            "views": [x["file"] for x in render_links]
        },

        # قسم خام للـ debug فقط — vertices/faces صارت رسمية ضمن mesh_density أعلاه.
        # edges متبقية هنا فقط لأنها ليست مستخدمة حالياً بأي feature مباشر.
        "_debug_raw": {
            "edges": total_edges
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
