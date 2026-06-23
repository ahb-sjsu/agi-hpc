"""Blender smoke: load VRM, frame head/torso, Eevee GPU render → PNG."""

import sys

import bpy

vrm_path = "/tmp/artemis_ref.vrm"
out_png = "/tmp/blender_smoke.png"

# Clean scene.
bpy.ops.wm.read_homefile(use_empty=True)

# Import VRM.
bpy.ops.import_scene.vrm(filepath=vrm_path)
armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if armature is None:
    print("NO ARMATURE", file=sys.stderr)
    sys.exit(1)

# Camera — VRM avatars stand feet-on-Z=0, ~1.6 m tall. Face the head.
cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (0.0, -2.4, 1.5)
cam.rotation_euler = (1.4, 0.0, 0.0)  # tilt forward onto the face
cam_data.lens = 50
bpy.context.scene.camera = cam


# 3-point-ish lighting.
def add_light(name, kind, loc, energy, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, type=kind)
    d.energy = energy
    d.color = color
    o = bpy.data.objects.new(name, d)
    o.location = loc
    bpy.context.scene.collection.objects.link(o)


add_light("Key", "AREA", (2.0, -2.0, 3.0), 500.0)
add_light("Fill", "AREA", (-3.0, -1.0, 2.0), 200.0, color=(0.8, 0.9, 1.0))
add_light("Rim", "AREA", (0.0, 2.0, 3.0), 400.0, color=(1.0, 0.85, 0.7))

# Background to match the table UI slate-blue-black.
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.02, 0.035, 0.08, 1.0)
bg.inputs["Strength"].default_value = 1.0

# Eevee on GPU.
scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE"
scn.eevee.taa_render_samples = 32
scn.render.resolution_x = 1280
scn.render.resolution_y = 720
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_mode = "RGBA"
scn.render.filepath = out_png

# Point Blender at the NVIDIA GPU.
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type in ("CUDA", "OPTIX")

print("RENDERING...", flush=True)
bpy.ops.render.render(write_still=True)
print("DONE:", out_png)
