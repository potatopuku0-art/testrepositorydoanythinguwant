"""Procedural, tileable serpent textures (scales, belly plates, armour atlas, detail swatches).

The scale field is exposed separately so mutations can recolour the exact same scales.
"""
import os

import numpy as np

from pmcore.texkit import write_png, hash01, normal_from_height, fbm, blur
from pmcore.mathx import smoothstep

TAU = 2 * np.pi


def scale_field(res=1024, n_cols=26, n_rows=28, belly_half=0.125, n_plates=12, seed=3):
    """Geometry of the body tile. u = along body (one tile = 2*pi*r), v = around (0 dorsal, .5 belly)."""
    u = (np.arange(res) + 0.5) / res
    U, V = np.meshgrid(u, u)
    X, Y = U * TAU, V * TAU
    cw, rh = TAU / n_cols, TAU / n_rows
    col = np.floor(X / cw)
    best = np.full(X.shape, np.inf)
    F = {k: np.zeros(X.shape) for k in ("h", "ex", "q", "rand", "rand2", "cid", "rid")}
    for dc in (-1, 0, 1):
        c = col + dc
        ysh = (np.mod(c, 2)) * 0.5 * rh
        r0 = np.floor((Y - ysh) / rh)
        for dr in (-1, 0, 1):
            r = r0 + dr
            cx, cy = (c + 0.5) * cw, (r + 0.5) * rh + ysh
            ex = (X - cx - 0.12 * cw) / (0.80 * cw)
            ey = (Y - cy) / (0.64 * rh)
            q = ex ** 2 + ey ** 2
            pri = np.where(q < 1, cx + 0.001 * q, np.inf)
            take = pri < best
            best = np.where(take, pri, best)
            h = (0.42 + 0.58 * (ex + 1) / 2) * np.sqrt(np.clip(1 - q, 0, 1)) ** 0.35
            cm, rm = np.mod(c, n_cols), np.mod(r, n_rows)
            vals = dict(h=h, ex=ex, q=q, rand=hash01(cm, rm, seed=seed), rand2=hash01(cm, rm, seed=seed + 11),
                        cid=cm, rid=rm)
            for k, v in vals.items():
                F[k] = np.where(take, v, F[k])
    F["h"] = np.where(np.isinf(best), 0.0, F["h"])
    db = np.abs(V - 0.5)
    belly = 1 - smoothstep(belly_half - 0.004, belly_half + 0.004, db)
    pl = TAU / n_plates
    fp = np.mod(X / pl, 1.0)
    pid = np.floor(X / pl)
    edge = np.clip(1 - (db / belly_half) ** 6, 0, 1)
    ph = (0.35 + 0.65 * fp ** 0.7) * edge ** 0.5
    F.update(belly=belly, fp=fp, pid=np.mod(pid, n_plates), plate_h=ph, db=db, dd=np.minimum(V, 1 - V),
             X=X, Y=Y, U=U, V=V)
    F["H"] = F["h"] * (1 - belly) + ph * belly
    F["rim"] = smoothstep(0.62, 0.92, F["q"]) * smoothstep(0.0, 0.5, F["ex"]) * (1 - belly)
    return F


def _mix(a, b, t):
    t = np.asarray(t)[..., None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def base_body_maps(F, pal):
    dd = F["dd"]
    c = _mix(pal.dorsal, pal.flank, smoothstep(0.12, 0.34, dd))
    c = _mix(c, pal.flank_low, smoothstep(0.30, 0.39, dd))
    c = c * (0.80 + 0.40 * F["rand"])[..., None]
    c = _mix(c, np.asarray(pal.secondary) * (0.8 + 0.4 * F["rand"])[..., None], 0.45 * (F["rand2"] > 0.80))
    # dorsal diamond markings, two per tile
    mx = np.abs(np.mod(F["X"] / (TAU / 2), 1.0) - 0.5) * (TAU / 2)
    dia = mx / 1.0 + dd * TAU / 0.55
    mark = smoothstep(1.0, 0.86, dia)
    outline = smoothstep(1.18, 1.04, dia) * (1 - mark)
    c = _mix(c, pal.marking, 0.85 * mark)
    c = c * (1 - 0.6 * outline)[..., None]
    shade = 0.30 + 0.70 * smoothstep(0.0, 0.55, F["h"])
    c = c * shade[..., None] + np.asarray(pal.rim) * (0.55 * F["rim"] * (0.5 + 0.5 * F["rand"]))[..., None]
    # belly plates
    fp = F["fp"]
    bc = np.asarray(pal.belly) * (0.65 + 0.35 * fp)[..., None] * (0.9 + 0.2 * hash01(F["pid"], 0, seed=5))[..., None]
    bc = bc + np.asarray(pal.belly_edge) * (smoothstep(0.80, 0.97, fp) * 0.55)[..., None]
    bc = bc * (0.25 + 0.75 * smoothstep(0.0, 0.06, fp))[..., None]
    bc = bc * (0.35 + 0.65 * np.clip(1 - (F["db"] / 0.125) ** 8, 0, 1))[..., None]
    col = _mix(c, bc, F["belly"])
    rough = np.clip(pal.body_rough + 0.30 * (1 - F["H"]) - 0.06 * F["rim"], 0.05, 1)
    rough = rough * (1 - F["belly"]) + 0.48 * F["belly"]
    return col, rough


def write_body_set(out_dir, prefix, F, col, rough, metal=0.0, normal_strength=9.0):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths["color"] = write_png(os.path.join(out_dir, f"{prefix}_Body_Color.png"), col, srgb=True)
    H = blur(F["H"], 1)
    paths["normal"] = write_png(os.path.join(out_dir, f"{prefix}_Body_Normal.png"), normal_from_height(H, normal_strength))
    paths["roughness"] = write_png(os.path.join(out_dir, f"{prefix}_Body_Roughness.png"), rough)
    m = metal if np.ndim(metal) else np.full(rough.shape, float(metal))
    paths["metalness"] = write_png(os.path.join(out_dir, f"{prefix}_Body_Metalness.png"), m)
    return paths


def armor_maps(pal, res=512, seed=2, extra=None):
    """u = base->tip along a part. v<0.5 metal (horns/spines/plates), v>=0.5 bone (teeth)."""
    u = (np.arange(res) + 0.5) / res
    U, V = np.meshgrid(u, u)
    streak = fbm((res, res), (4, 24), octaves=3, seed=seed)
    metal_band = V < 0.5
    c1 = _mix(pal.horn_root, pal.horn_mid, smoothstep(0.0, 0.45, U))
    c1 = _mix(c1, pal.horn_tip, smoothstep(0.55, 1.0, U))
    c1 = c1 * (0.75 + 0.5 * streak)[..., None]
    c2 = _mix(pal.bone_root, pal.bone_tip, smoothstep(0.0, 0.7, U)) * (0.9 + 0.2 * streak)[..., None]
    col = np.where(metal_band[..., None], c1, c2)
    rough = np.where(metal_band, 0.26 + 0.22 * streak, 0.32 + 0.1 * streak)
    metal = np.where(metal_band, pal.horn_metal, 0.0)
    h = streak * 0.6 + 0.4 * np.abs(np.sin(U * 60)) ** 0.5 * 0.3
    if extra:
        col, rough, metal, h = extra(U, V, col, rough, metal, h)
    return col, rough, metal, h


def write_armor_set(out_dir, prefix, col, rough, metal, h):
    os.makedirs(out_dir, exist_ok=True)
    return {
        "color": write_png(os.path.join(out_dir, f"{prefix}_Armor_Color.png"), col, srgb=True),
        "normal": write_png(os.path.join(out_dir, f"{prefix}_Armor_Normal.png"), normal_from_height(h, 2.5)),
        "roughness": write_png(os.path.join(out_dir, f"{prefix}_Armor_Roughness.png"), rough),
        "metalness": write_png(os.path.join(out_dir, f"{prefix}_Armor_Metalness.png"), metal),
    }


def detail_atlas(out_dir, prefix, swatches):
    """64x64 atlas of flat colour swatches (4x4); swatches: list of linear RGB."""
    img = np.zeros((64, 64, 3))
    for i, c in enumerate(swatches):
        x, y = (i % 4) * 16, (i // 4) * 16
        img[y:y + 16, x:x + 16] = c
    return write_png(os.path.join(out_dir, f"{prefix}_Detail_Color.png"), img, srgb=True)
