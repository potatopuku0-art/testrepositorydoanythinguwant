"""Blender-side helpers: objects, materials, studio rendering, export + reimport checks."""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector, Matrix

from .meshdata import MeshData


# ---------------------------------------------------------------------------
# Scene / objects
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    return scn


def collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    par = parent or bpy.context.scene.collection
    if col.name not in [c.name for c in par.children]:
        par.children.link(col)
    return col


def to_blender(name, md: MeshData, material=None, col=None, props=None):
    md = md.copy().compact()
    me = bpy.data.meshes.new(name)
    me.from_pydata(md.verts.tolist(), [], [list(f) for f in md.faces])
    uv = me.uv_layers.new(name="UVMap")
    flat = np.array([c for face in md.uvs for c in face], dtype=np.float32).reshape(-1)
    if len(flat) == len(uv.data) * 2:
        uv.data.foreach_set("uv", flat)
    me.polygons.foreach_set("use_smooth", [md.sharp_angle != 0.0] * len(me.polygons))
    if md.normals is not None:
        me.normals_split_custom_set_from_vertices([tuple(n) for n in md.normals])
    elif md.sharp_angle not in (None, 0.0):
        me.set_sharp_from_angle(angle=md.sharp_angle)
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (col or bpy.context.scene.collection).objects.link(ob)
    if material is not None:
        me.materials.append(material)
    for k, v in (props or {}).items():
        ob[k] = v
    return ob


def tri_count(ob):
    me = ob.data
    return sum(len(p.vertices) - 2 for p in me.polygons)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _img(path, non_color=False):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def pbr_material(name, color, normal=None, roughness=None, metallic=0.0, normal_strength=1.0,
                 rough_value=0.5, roblox=None, emission=None, emission_strength=0.0):
    """Texture-driven Principled material that exports cleanly to glTF/FBX."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    if isinstance(color, str):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(color)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    if normal:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(normal, True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if roughness:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(roughness, True)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = rough_value
    if isinstance(metallic, str):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = _img(metallic, True)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    _roblox_props(mat, roblox)
    return mat


def neon_material(name, color, strength=6.0, alpha=1.0, roblox=None):
    """Glowing part. In Roblox this maps to Material=Neon with the same colour."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Emission Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Roughness"].default_value = 0.4
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = "BLEND"
        mat.surface_render_method = "BLENDED"
    _roblox_props(mat, roblox)
    return mat


def _roblox_props(mat, roblox):
    for k, v in (roblox or {}).items():
        mat["roblox_" + k] = v


# ---------------------------------------------------------------------------
# Studio rendering
# ---------------------------------------------------------------------------

def setup_render(samples=96, res=(1920, 1080), exposure=0.0, look="AgX - Medium High Contrast", bloom=None):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.adaptive_threshold = 0.02
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPENIMAGEDENOISE"
    scn.cycles.max_bounces = 6
    scn.cycles.diffuse_bounces = 3
    scn.cycles.glossy_bounces = 3
    scn.cycles.transmission_bounces = 4
    scn.cycles.transparent_max_bounces = 12
    scn.cycles.sample_clamp_indirect = 8.0
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.film_transparent = False
    scn.view_settings.view_transform = "AgX"
    try:
        scn.view_settings.look = look
    except TypeError:
        pass
    scn.view_settings.exposure = exposure
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    _compositor(bloom)
    return scn


def _compositor(bloom):
    scn = bpy.context.scene
    scn.use_nodes = True
    nt = scn.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    if bloom:
        g = nt.nodes.new("CompositorNodeGlare")
        g.glare_type = "FOG_GLOW"
        g.quality = "HIGH"
        g.threshold = bloom.get("threshold", 1.0)
        g.size = bloom.get("size", 8)
        g.mix = bloom.get("mix", -0.75)
        nt.links.new(rl.outputs["Image"], g.inputs["Image"])
        nt.links.new(g.outputs["Image"], comp.inputs["Image"])
    else:
        nt.links.new(rl.outputs["Image"], comp.inputs["Image"])


def world_gradient(top, bottom, strength=1.0):
    w = bpy.data.worlds.get("Studio") or bpy.data.worlds.new("Studio")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs[0])
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (*bottom, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (*top, 1)
    nt.links.new(ramp.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs[0], out.inputs[0])
    return w


def cyclorama(center, size, color, height_z, rough=0.6, grid=None, name="Backdrop"):
    """Curved studio backdrop (floor + coved wall) behind the subject."""
    sx, sy = size
    prof = []
    R = sy * 0.35
    for i in range(24):
        a = (i / 23) * (math.pi / 2)
        prof.append((-sy / 2 + (sy - R) + math.sin(a) * R, R - math.cos(a) * R))
    prof = [(-sy * 2.0, 0.0)] + prof + [(sy / 2, R + sy * 1.4)]
    verts, faces = [], []
    nx = 2
    for i, (y, z) in enumerate(prof):
        for j in range(nx):
            x = -sx / 2 + sx * j / (nx - 1)
            verts.append((x, y, z))
    for i in range(len(prof) - 1):
        a = i * nx
        faces.append((a, a + 1, a + nx + 1, a + nx))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(faces))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = (center[0], center[1], height_z)
    mat = bpy.data.materials.new(name + "_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if grid:
        nt = mat.node_tree
        tc = nt.nodes.new("ShaderNodeTexCoord")
        br = nt.nodes.new("ShaderNodeTexBrick")
        br.inputs["Scale"].default_value = grid["scale"]
        br.inputs["Mortar Size"].default_value = grid.get("line", 0.01)
        br.offset = 0.0
        br.inputs["Color1"].default_value = (0, 0, 0, 1)
        br.inputs["Color2"].default_value = (0, 0, 0, 1)
        br.inputs["Mortar"].default_value = (*grid["color"], 1)
        nt.links.new(tc.outputs["Object"], br.inputs["Vector"])
        nt.links.new(br.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = grid.get("strength", 1.0)
    me.materials.append(mat)
    return ob


def area_light(name, loc, target, power, size, color=(1, 1, 1), shape="RECTANGLE", size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.color = color
    ld.shape = shape
    ld.size = size
    ld.size_y = size_y or size
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def look_at(ob, target, up=(0, 0, 1)):
    d = Vector(target) - Vector(ob.location)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def camera(name, lens=50, sensor=36):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_start = 0.05
    cd.clip_end = 1000
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    return ob


def fit_camera(cam, points, direction, margin=0.06, target=None, up=(0, 0, 1)):
    """Place cam along -direction from the subject so every point is inside the frame
    (with margin), centred on the projected bounds. Guarantees no cropping."""
    scn = bpy.context.scene
    pts = np.asarray(points, float)
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    center = pts.mean(axis=0) if target is None else np.asarray(target, float)
    aspect = scn.render.resolution_x / scn.render.resolution_y
    fx = cam.data.sensor_width / (2 * cam.data.lens)  # tan(half fov x)
    fy = fx / aspect
    upv = np.asarray(up, float)
    right = np.cross(d, upv)
    right /= np.linalg.norm(right)
    camup = np.cross(right, d)
    for _ in range(4):  # iterate: recentre then refit distance
        rel = pts - center
        x, y, z = rel @ right, rel @ camup, rel @ d
        # distance so that |x| <= (dist + z) * fx * (1 - margin)
        need_x = np.max(np.abs(x) / (fx * (1 - margin)) - z)
        need_y = np.max(np.abs(y) / (fy * (1 - margin)) - z)
        dist = max(need_x, need_y)
        # recentre on projected bounds
        px = x / (dist + z) * dist
        py = y / (dist + z) * dist
        center = center + right * (px.max() + px.min()) / 2 + camup * (py.max() + py.min()) / 2
    rel = pts - center
    x, y, z = rel @ right, rel @ camup, rel @ d
    dist = max(np.max(np.abs(x) / (fx * (1 - margin)) - z), np.max(np.abs(y) / (fy * (1 - margin)) - z))
    cam.location = Vector(center - d * dist)
    m = Matrix(((right[0], camup[0], -d[0]), (right[1], camup[1], -d[1]), (right[2], camup[2], -d[2])))
    cam.rotation_euler = m.to_euler()
    return center, dist


def render_to(path):
    scn = bpy.context.scene
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


# ---------------------------------------------------------------------------
# Export + reimport validation
# ---------------------------------------------------------------------------

def select_only(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def export_glb(path, objs):
    select_only(objs)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_apply=True, export_yup=True, export_texcoord=True,
                              export_normals=True, export_tangents=False, export_materials="EXPORT",
                              export_extras=True, export_animations=False, export_image_format="AUTO")
    return path


def export_fbx(path, objs):
    select_only(objs)
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"MESH", "EMPTY"},
                             path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y",
                             apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL",
                             mesh_smooth_type="OFF", use_mesh_modifiers=True, use_custom_props=True,
                             bake_anim=False, add_leaf_bones=False, use_tspace=True)
    return path


def inspect_file(path):
    """Reimport into an empty scene and report what actually came back."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".glb":
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        bpy.ops.import_scene.fbx(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    rep = {"file": os.path.basename(path), "bytes": os.path.getsize(path), "meshes": []}
    allpts = []
    for o in meshes:
        me = o.data
        tris = sum(len(p.vertices) - 2 for p in me.polygons)
        co = np.array([(o.matrix_world @ v.co)[:] for v in me.vertices])
        allpts.append(co)
        imgs = set()
        for m in me.materials:
            if m and m.use_nodes:
                for n in m.node_tree.nodes:
                    if n.type == "TEX_IMAGE" and n.image:
                        imgs.add((n.image.name, tuple(n.image.size)))
        rep["meshes"].append({
            "name": o.name, "tris": tris, "verts": len(me.vertices),
            "uv_layers": len(me.uv_layers), "materials": [m.name for m in me.materials if m],
            "images": sorted(f"{a} {b[0]}x{b[1]}" for a, b in imgs),
            "finite": bool(np.isfinite(co).all()),
        })
    if allpts:
        P = np.vstack(allpts)
        rep["bbox_min"] = P.min(axis=0).round(3).tolist()
        rep["bbox_max"] = P.max(axis=0).round(3).tolist()
    rep["total_tris"] = sum(m["tris"] for m in rep["meshes"])
    rep["mesh_count"] = len(rep["meshes"])
    rep["max_mesh_tris"] = max((m["tris"] for m in rep["meshes"]), default=0)
    rep["images"] = sorted({i for m in rep["meshes"] for i in m["images"]})
    return rep


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
