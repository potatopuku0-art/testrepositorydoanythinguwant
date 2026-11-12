"""Studio presets for the serpent preview renders (shared by quick previews and final renders)."""
import numpy as np
import bpy

from pmcore import bl

VIEWS = {
    # direction the camera looks along (from camera toward subject)
    "three_quarter": dict(dir=(0.62, 0.70, -0.36), lens=50, margin=0.05),
    "side": dict(dir=(1.0, 0.0, -0.06), lens=50, margin=0.04),
    "head": dict(dir=None, lens=85, margin=0.10),
    "top": dict(dir=(0.0, 0.001, -1.0), lens=50, margin=0.04),
}


def subject_points(objs, step=7):
    pts = []
    for o in objs:
        if o.type != "MESH":
            continue
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)[::step]
        M = np.array(o.matrix_world)
        pts.append(co @ M[:3, :3].T + M[:3, 3])
    return np.vstack(pts)


def studio(variant, objs, build, view, out, samples=96, res=(1920, 1080), extra_objs=()):
    hacker = variant == "hacker"
    pts = subject_points(objs)
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo + hi) / 2
    size = hi - lo
    span = float(np.max(size))
    if hacker:
        bl.world_gradient(top=(0.010, 0.018, 0.016), bottom=(0.002, 0.004, 0.004), strength=1.0)
        back_col, grid = (0.010, 0.013, 0.013), dict(scale=0.9, color=(0.05, 0.6, 0.18), line=0.012, strength=0.18)
    else:
        bl.world_gradient(top=(0.05, 0.045, 0.05), bottom=(0.012, 0.010, 0.012), strength=1.0)
        back_col, grid = (0.055, 0.045, 0.048), None
    bl.cyclorama((ctr[0], ctr[1]), (span * 4, span * 3), back_col, lo[2] - 1.2, grid=grid)
    tgt = tuple(ctr)
    key_col = (1.0, 0.93, 0.86) if not hacker else (0.85, 0.95, 1.0)
    rim_col = (1.0, 0.45, 0.30) if not hacker else (0.35, 1.0, 0.55)
    s = span
    bl.area_light("Key", (ctr[0] - 0.55 * s, ctr[1] - 0.75 * s, ctr[2] + 0.7 * s), tgt, 9000 * (s / 30) ** 2,
                  0.4 * s, key_col)
    bl.area_light("Fill", (ctr[0] + 0.8 * s, ctr[1] - 0.3 * s, ctr[2] + 0.2 * s), tgt, 2200 * (s / 30) ** 2,
                  0.5 * s, (0.75, 0.8, 1.0))
    bl.area_light("RimA", (ctr[0] + 0.25 * s, ctr[1] + 0.85 * s, ctr[2] + 0.55 * s), tgt, 9000 * (s / 30) ** 2,
                  0.25 * s, rim_col)
    bl.area_light("RimB", (ctr[0] - 0.85 * s, ctr[1] + 0.35 * s, ctr[2] + 0.25 * s), tgt, 4500 * (s / 30) ** 2,
                  0.2 * s, rim_col)
    bl.area_light("Top", (ctr[0], ctr[1], ctr[2] + 0.9 * s), tgt, 3500 * (s / 30) ** 2, 0.6 * s, key_col)

    bl.setup_render(samples=samples, res=res, bloom=dict(threshold=0.9, size=8, mix=-0.72) if hacker
                    else dict(threshold=1.4, size=7, mix=-0.85))
    v = VIEWS[view]
    cam = bl.camera("Cam_" + view, lens=v["lens"])
    if view == "head":
        A = build.anat
        hp = [o for o in objs if any(k in o.name for k in ("Eyes", "Horns", "HeadPlates", "Jaw", "Teeth",
                                                          "Holo", "Emblem"))]
        hpts = subject_points(hp, step=3)
        body = subject_points([o for o in objs if o.name.endswith("_Body")], step=1)
        near = body[np.linalg.norm(body - A.head_origin, axis=1) < 3.4]
        hpts = np.vstack([hpts[np.linalg.norm(hpts - A.head_origin, axis=1) < 5.0], near])
        d = A.hT * 1.0 - A.hS * 0.62 - A.hU * 0.32
        bl.fit_camera(cam, hpts, d, margin=v["margin"])
    else:
        bl.fit_camera(cam, pts, v["dir"], margin=v["margin"])
    return bl.render_to(out)
