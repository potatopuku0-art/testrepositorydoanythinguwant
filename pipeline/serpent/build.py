"""Build a Giant Demon Serpent (numpy geometry) and realise it in Blender."""
import os

import numpy as np

from .anatomy import SerpentAnatomy
from .config import SerpentConfig, HERO, GAMEPLAY
from .parts import PartBuilder
from . import textures as tx


class Part:
    def __init__(self, name, mesh, mat, group):
        self.name, self.mesh, self.mat, self.group = name, mesh, mat, group


class SerpentBuild:
    """Geometry + landmarks + material assignments for one LOD. Mutations add/replace parts
    and override material specs; they never need to touch the generator internals."""

    def __init__(self, cfg: SerpentConfig, lod, variant="Base"):
        self.cfg, self.lod, self.variant = cfg, lod, variant
        self.anat = SerpentAnatomy(cfg)
        self.pb = PartBuilder(self.anat, lod, cfg)
        self.parts = []
        self.mat_specs = {}
        self._build()

    def add(self, name, mesh, mat, group):
        if mesh is not None and len(mesh.faces):
            self.parts.append(Part(name, mesh, mat, group))

    def remove(self, prefix):
        self.parts = [p for p in self.parts if not p.name.startswith(prefix)]

    @property
    def land(self):
        return self.pb.land

    def _build(self):
        pb = self.pb
        self.add("Body", pb.body(), "body", "Body")
        jaw, mouth, throat = pb.jaw()
        self.add("Jaw", jaw, "body", "Head")
        self.add("Mouth", mouth, "detail", "Head")
        self.add("ThroatGlow", throat, "throat_glow", "Head")
        eye, eye_detail = pb.eyes()
        self.add("Eyes", eye, "eye_glow", "Head")
        self.add("EyeDetail_Pupils", eye_detail, "detail", "Head")
        self.add("Nostrils", pb.nostrils(), "detail", "Head")
        self.add("HeadPlates", pb.head_plates(), "armor", "Head")
        self.add("Horns", pb.horns(), "armor", "Head")
        self.add("Teeth", pb.teeth(), "armor", "Head")
        self.add("DorsalSpines", pb.spines(), "armor", "Body")
        self.add("TailSpade", pb.tail_spade(), "armor", "Body")

    def tri_count(self):
        return sum(p.mesh.tri_count() for p in self.parts)


def base_material_specs(tex_dir, cfg, lod_res=1024):
    """Generate base textures and return material specs keyed by material tag."""
    pal = cfg.palette
    F = tx.scale_field(res=lod_res)
    col, rough = tx.base_body_maps(F, pal)
    body = tx.write_body_set(tex_dir, "GDS_Base", F, col, rough, metal=pal.body_metal)
    ac, ar, am, ah = tx.armor_maps(pal)
    armor = tx.write_armor_set(tex_dir, "GDS_Base", ac, ar, am, ah)
    detail = tx.detail_atlas(tex_dir, "GDS_Base", [pal.pupil, pal.mouth, pal.tongue, pal.iris_ring, pal.glint,
                                                    pal.pupil])
    return {
        "body": dict(kind="pbr", tex=body, normal_strength=1.0,
                     roblox=dict(Material="SurfaceAppearance", note="Body textures (tiling along u)")),
        "armor": dict(kind="pbr", tex=armor, normal_strength=0.6, roblox=dict(Material="SurfaceAppearance")),
        "detail": dict(kind="pbr", tex=dict(color=detail), rough=0.25, roblox=dict(Material="SmoothPlastic")),
        "eye_glow": dict(kind="neon", color=pal.eye, strength=2.4, roblox=dict(Material="Neon", Color=_hex(pal.eye))),
        "throat_glow": dict(kind="neon", color=pal.throat, strength=5.0,
                            roblox=dict(Material="Neon", Color=_hex(pal.throat))),
    }, F


def _hex(c):
    from pmcore.texkit import linear_to_srgb
    s = (np.clip(linear_to_srgb(np.asarray(c[:3])), 0, 1) * 255).round().astype(int)
    return "#%02X%02X%02X" % tuple(s)


def realise(build, specs, prefix, collection_name=None):
    """Create Blender materials + one object per part (source scene)."""
    from pmcore import bl
    mats = {}
    for tag, sp in specs.items():
        name = f"{prefix}_{tag}"
        if sp["kind"] == "pbr":
            t = sp["tex"]
            mats[tag] = bl.pbr_material(name, t["color"], t.get("normal"), t.get("roughness"),
                                        t.get("metalness", 0.0), sp.get("normal_strength", 1.0),
                                        rough_value=sp.get("rough", 0.5), roblox=sp.get("roblox"))
        else:
            mats[tag] = bl.neon_material(name, sp["color"], sp.get("strength", 6.0), sp.get("alpha", 1.0),
                                         roblox=sp.get("roblox"))
    root = bl.collection(collection_name or prefix)
    groups = {}
    objs = []
    for p in build.parts:
        g = groups.get(p.group) or bl.collection(f"{prefix}_{p.group}", root)
        groups[p.group] = g
        ob = bl.to_blender(f"{prefix}_{p.name}", p.mesh, mats[p.mat], g,
                           props={"material_tag": p.mat, "lod": build.lod.name})
        objs.append(ob)
    return objs, mats
