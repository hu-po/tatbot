"""Export the inkmap bodies from Blender Studio's Human Base Meshes bundle (CC0).

    blender -b human_base_meshes_bundle.blend --python tools/export-hbm.py -- public/bodies [preview_dir]

Bundle: https://www.blender.org/download/demo-files/ ("Human Base Meshes",
v1.4.1, CC0). For each stylized body this script centres the mesh on the
origin (feet on z=0), paints a per-corner colour attribute — white on the
body, dark on the eyes, so the app's skin-tone tint (material colour x vertex
colour) gives one uniform skin colour and the eyes stay dark — renames the objects to the stable node names the app
expects (Body, EyeL, EyeR), and writes a GLB with no animation, no material,
no texture, no normals or UVs (the app recomputes normals). Colours ride in
COLOR_0; the app renders vertexColors.

Deterministic for a given bundle version + this file, which is what matters:
the GLB's SHA-256 is written into every placement file.
"""
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out_dir = Path(argv[0] if argv else "public/bodies")
preview_dir = Path(argv[1]) if len(argv) > 1 else None
out_dir.mkdir(parents=True, exist_ok=True)

SKIN = (1.0, 1.0, 1.0, 1.0)  # tinted at runtime by the skin-tone picker
EYE = (0.09, 0.10, 0.11, 1.0)

BODIES = {
    "hbm-male-stylized": "GEO-body_male_stylized",
    "hbm-female-stylized": "GEO-body_female_stylized",
}


def paint(obj, color):
    mesh = obj.data
    attr = mesh.color_attributes.get("Col") or mesh.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    mesh.color_attributes.active_color = attr
    for d in attr.data:
        d.color = color


def export(key, src_name):
    body = bpy.data.objects[src_name]
    eyes = [bpy.data.objects[f"{src_name}.eye.L"], bpy.data.objects[f"{src_name}.eye.R"]]
    parts = [body, *eyes]

    # Centre on the origin, feet on the floor. Move eyes by the same delta.
    world = [body.matrix_world @ v.co for v in body.data.vertices]
    lo = Vector((min(v.x for v in world), min(v.y for v in world), min(v.z for v in world)))
    hi = Vector((max(v.x for v in world), max(v.y for v in world), max(v.z for v in world)))
    delta = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    # The eyes are parented to the body in the bundle; moving the parent moves them.
    movers = [o for o in parts if o.parent not in parts]
    for o in movers:
        o.location += delta
    bpy.context.view_layer.update()
    height = hi.z - lo.z

    paint(body, SKIN)
    for e in eyes:
        paint(e, EYE)

    names = {body: "Body", eyes[0]: "EyeL", eyes[1]: "EyeR"}
    saved = {o: o.name for o in parts}
    for o, n in names.items():
        o.name = n
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = body
    path = out_dir / f"{key}.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_animations=False, export_skins=False, export_morph=False,
        export_materials="NONE", export_colors=True, export_normals=False, export_texcoords=False,
        export_yup=True, export_extras=False, export_cameras=False, export_lights=False,
    )
    print(f"wrote {path}: height {height:.3f} m, {len(body.data.polygons)} polys")

    if preview_dir:
        render_preview(parts, preview_dir / f"{key}.png", height)

    for o, n in saved.items():
        o.name = n
    for o in movers:
        o.location -= delta


def render_preview(parts, path, height):
    scene = bpy.context.scene
    for o in bpy.data.objects:
        o.hide_render = o not in parts
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (2.2, -3.2, height * 0.55)
    cam.rotation_euler = (1.45, 0, 0.6)
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "VERTEX"
    scene.render.resolution_x, scene.render.resolution_y = 600, 900
    scene.render.filepath = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(cam_data)


for key, src in BODIES.items():
    export(key, src)
