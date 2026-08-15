"""
GPU render test: renders a single cube with Cycles using GPU (CUDA/OptiX).

Run inside the `blender` container, e.g.:

    docker exec blender blender -b -P /scripts/render_cube_gpu.py

or set it as the container's default command / entrypoint script.
Output is written to /output/cube_gpu_test.png (mapped to ./blender_output).
"""

import bpy
import mathutils
import math

# ---------------------------------------------------------------------------
# 1. Clean slate
# ---------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene

# ---------------------------------------------------------------------------
# 2. Add a cube
# ---------------------------------------------------------------------------
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object

# Give it a simple material so it's not just flat grey/no shading info
mat = bpy.data.materials.new(name="CubeMaterial")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.8, 0.2, 0.2, 1.0)
cube.data.materials.append(mat)

# ---------------------------------------------------------------------------
# 3. Camera (positioned for a proper 3/4 view, aimed at the cube)
# ---------------------------------------------------------------------------
cam_location = mathutils.Vector((6, -6, 4))

bpy.ops.object.camera_add(location=cam_location)
camera = bpy.context.active_object
scene.camera = camera

# Point the camera at the cube's origin instead of guessing Euler angles.
direction = cube.location - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

# Pull the camera back a bit so the whole cube is comfortably in frame
camera.data.lens = 35

# ---------------------------------------------------------------------------
# 4. Light
# ---------------------------------------------------------------------------
bpy.ops.object.light_add(type='SUN', location=(4, -6, 8))
light = bpy.context.active_object
light.rotation_euler = (math.radians(45), 0, math.radians(30))
light.data.energy = 3.0

# ---------------------------------------------------------------------------
# 4b. Ground plane so the cube reads with proper context/shadow
# ---------------------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -1))
plane = bpy.context.active_object
plane_mat = bpy.data.materials.new(name="PlaneMaterial")
plane_mat.use_nodes = True
plane_bsdf = plane_mat.node_tree.nodes.get("Principled BSDF")
if plane_bsdf:
    plane_bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
plane.data.materials.append(plane_mat)

# ---------------------------------------------------------------------------
# 5. Render engine + GPU setup
# ---------------------------------------------------------------------------
scene.render.engine = 'CYCLES'
scene.cycles.samples = 64

prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences

# Try OPTIX first (best for NVIDIA + RTX), fall back to CUDA
cycles_prefs.compute_device_type = 'OPTIX'
cycles_prefs.get_devices()

gpu_found = False
for device in cycles_prefs.devices:
    if device.type in ('OPTIX', 'CUDA'):
        device.use = True
        gpu_found = True
        print(f"Enabling GPU device: {device.name} ({device.type})")
    else:
        device.use = False

if not gpu_found:
    # Fall back to CUDA compute type and re-scan
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        if device.type == 'CUDA':
            device.use = True
            gpu_found = True
            print(f"Enabling GPU device: {device.name} (CUDA)")

if not gpu_found:
    print("WARNING: No GPU device found — falling back to CPU render.")
    scene.cycles.device = 'CPU'
else:
    scene.cycles.device = 'GPU'

# ---------------------------------------------------------------------------
# 6. Output settings
# ---------------------------------------------------------------------------
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = '/output/cube_gpu_test.png'

# ---------------------------------------------------------------------------
# 7. Render
# ---------------------------------------------------------------------------
print(f"Rendering with device: {scene.cycles.device}")
bpy.ops.render.render(write_still=True)
print(f"Done. Output written to {scene.render.filepath}")