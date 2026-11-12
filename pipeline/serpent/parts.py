"""Serpent part builders: body sweep, jaw, mouth, eyes, armour plates, horns, teeth, spines, tail."""
import numpy as np

from pmcore.mathx import normalize, smoothstep, rot_matrix
from pmcore.meshdata import MeshData, star_plate, lens_blade, uv_sphere, merge, fix_winding_by_centroid
from pmcore.sweep import SweepSurface, SuperProfile, ring_positions, keyed

D = np.radians

# detail atlas swatches (4x4 grid)
SW_PUPIL, SW_MOUTH, SW_TONGUE, SW_IRIS, SW_GLINT, SW_NOSTRIL = 0, 1, 2, 3, 4, 5


def swatch_uv(i):
    return (((i % 4) + 0.5) / 4.0, ((i // 4) + 0.5) / 4.0)


METAL_V = (0.03, 0.47)
BONE_V = (0.53, 0.97)


# ---------------------------------------------------------------------------
def polar_decal(outline2d, mapper, rings=3, center2d=None, hole=0.0, uv=(0.5, 0.5), ref_normal=None):
    """Single-sided decal from a star-shaped 2D outline mapped onto any surface."""
    O = np.asarray(outline2d, float)
    c = O.mean(axis=0) if center2d is None else np.asarray(center2d, float)
    n = len(O)
    fr = np.linspace(hole, 1, rings + 1)[0 if hole > 0 else 1:]
    pts = [] if hole > 0 else [c]
    for f in fr:
        pts += list(c + (O - c) * f)
    pts = np.array(pts)
    V = mapper(pts[:, 0], pts[:, 1])
    faces = []
    base = 0 if hole > 0 else 1
    if hole == 0:
        for k in range(n):
            faces.append((0, 1 + k, 1 + (k + 1) % n))
    nr = len(fr)
    for r in range(nr - 1):
        for k in range(n):
            a = base + r * n
            faces.append((a + k, a + n + k, a + n + (k + 1) % n, a + (k + 1) % n))
    m = MeshData(V, faces, [tuple(uv for _ in f) for f in faces])
    if ref_normal is not None:
        f = faces[0]
        fn = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(fn, ref_normal) < 0:
            m = m.flipped()
    return m


def circle2d(r, n=16, sx=1.0, sy=1.0, c=(0, 0)):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([c[0] + np.cos(a) * r * sx, c[1] + np.sin(a) * r * sy], axis=1)


def surface_mapper(surf, s0, th0, offset):
    return lambda lx, ly: surf.map_local(s0, th0, lx, ly, offset)


def sweep_part(points, profile_fn, around, rings, up, cap_start=True, cap_end=True,
               v_range=METAL_V, spline=True, s_end=0.997):
    surf = SweepSurface(points, None, up=up, ds=0.004, spline=spline, up_mode="rmf")
    surf.profile = profile_fn(surf.L)
    s = np.linspace(0.0, surf.L * s_end, rings)
    m = surf.mesh(s, [around] * rings, cap_start=cap_start, cap_end=cap_end,
                  u_fn=lambda x: 0.03 + 0.94 * x / surf.L, v_range=v_range)
    return surf, m


def tapered(L, r0, r1=0.004, power=1.0, oval=1.0, ridges=0, ridge_amp=0.0, ridge_end=0.6):
    def r(s):
        t = np.clip(np.asarray(s, float) / L, 0, 1)
        base = r1 + (r0 - r1) * (1 - t) ** power
        if ridges:
            rib = np.maximum(0, np.sin(2 * np.pi * ridges * t + 1.2)) ** 3
            base = base * (1 + ridge_amp * rib * (1 - smoothstep(ridge_end * 0.6, ridge_end, t)))
        return base
    return SuperProfile(a=r, bt=lambda s: r(s) * oval, bb=lambda s: r(s) * oval)


# ---------------------------------------------------------------------------
class PartBuilder:
    def __init__(self, anat, lod, cfg):
        self.A, self.lod, self.cfg = anat, lod, cfg
        self.surf = anat.surf
        self.land = {}

    # ----------------------------------------------------------- body sweep
    def body(self):
        A, lod = self.A, self.lod
        H, L = A.H, A.L
        curv_s = A.surf.s
        k = np.linalg.norm(np.gradient(A.surf.T, axis=0), axis=1) / A.surf.ds
        k_fn = lambda s: np.interp(s, curv_s, k)

        def spacing(s):
            if s < H + 0.5:
                sp = lod.head_spacing * (0.45 if s < 0.25 else 1.0)
                return sp
            sp = min(lod.body_k * max(float(A.r_eff(s)), 0.40), lod.body_max)
            sp = min(sp, 0.30 / max(float(k_fn(s)), 1e-3) * (lod.body_k / 0.15) ** 0.5)
            blend = smoothstep(H + 0.5, H + 1.5, s)
            return float(lod.head_spacing * (1 - blend) + sp * blend)
        s_r = ring_positions(0.012, L - 0.02, spacing)
        n_head = lod.head_around
        counts = []
        for s in s_r:
            if s < H + 0.7:
                counts.append(n_head)
            elif A.r(s) > 0.30:
                counts.append(n_head // 2)
            else:
                counts.append(n_head // 4)
        self.body_rings = (s_r, counts)
        m = A.surf.mesh(s_r, counts, cap_start=True, cap_end=True, u_fn=A.u_coord())
        return m

    # ----------------------------------------------------------- jaw + mouth
    def jaw(self):
        A, lod, cfg = self.A, self.lod, self.cfg
        bb = A.profile.bb
        d = np.linspace(0.10, 2.98, 120)
        sj = d - 0.10
        hs = cfg.head_scale
        jk = [0, .05, .15, .4, .8, 1.2, 1.6, 2.0, 2.4, 2.7, 2.88]
        ttop = keyed(jk, hs * np.array([0, .03, .05, .06, .06, .07, .07, .08, .09, .08, .0]))
        tbot = keyed(jk, hs * np.array([0, .085, .13, .16, .18, .21, .24, .27, .29, .22, .0]))
        aj = keyed(jk, hs * np.array([0, .10, .20, .28, .33, .40, .52, .62, .63, .48, .0]))
        yc = -bb(d) - ttop(sj) - 0.004
        pts = A.head_local(d, 0, yc)
        prof = SuperProfile(a=aj, bt=ttop, bb=tbot, nt=lambda s: np.full_like(np.asarray(s, float), 3.2),
                            nb=lambda s: np.full_like(np.asarray(s, float), 2.3))
        js = SweepSurface(pts, prof, up=A.hU, ds=0.005, spline=False, up_mode="rmf")
        hinge = A.head_local(2.55, 0, -bb(2.55) - 0.02)
        R = rot_matrix(A.hS, -D(cfg.jaw_open_deg))
        js.apply_rotation(R, hinge)
        self.jaw_surf = js
        self.jaw_R, self.jaw_hinge = R, hinge
        rings = ring_positions(0.0, js.L, lambda s: lod.head_spacing * 1.2)
        rings[0] = 0.006
        u_fn = lambda s: 0.02 + s / (2 * np.pi * 0.35)
        jaw = js.mesh(rings, [lod.jaw_around] * len(rings), cap_start=True, cap_end=True, u_fn=u_fn,
                      th0=np.pi)  # seam under the chin, not on the mouth floor
        # mouth floor (on jaw top) and palate (under upper head)
        floor = self._grid_patch(js, 0.18, 2.42, lambda s: D(52) * smoothstep(0.1, 0.5, s), 0.0, 0.005, SW_TONGUE)
        palate = self._grid_patch(A.surf, 0.22, 2.42, lambda s: D(46) * smoothstep(0.15, 0.6, s), np.pi, 0.005, SW_MOUTH)
        # inner throat glow
        c = A.head_local(2.05, 0, -bb(2.05) - 0.11)
        c = (c - hinge) @ (np.eye(3) * 0.5 + R.T * 0.5).T + hinge if False else c
        throat = uv_sphere(c, 0.26, A.hT, seg=16, rings=10, scale=(1.0, 0.36, 1.35))
        return jaw, merge([floor, palate]), throat

    def _grid_patch(self, surf, s0, s1, th_half, th_c, offset, sw, ns=24, nt=8):
        s = np.linspace(s0, s1, ns)
        verts, faces = [], []
        for i, si in enumerate(s):
            h = float(th_half(si))
            th = th_c + np.linspace(-h, h, nt)
            verts.append(surf.point(np.full(nt, si), th, offset))
        V = np.vstack(verts)
        for i in range(ns - 1):
            for j in range(nt - 1):
                a = i * nt + j
                faces.append((a, a + 1, a + nt + 1, a + nt))
        m = MeshData(V, faces, [(swatch_uv(sw),) * 4 for _ in faces])
        nref = surf.normal(s[ns // 2], th_c)
        f = faces[len(faces) // 2]
        if np.dot(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]), nref) < 0:
            m = m.flipped()
        return m

    # ----------------------------------------------------------- eyes
    def eyes(self):
        A, lod = self.A, self.lod
        re = self.cfg.eye_radius
        glow, detail = [], []
        self.land["eyes"] = []
        for side in (1, -1):
            s_e, th_e = 1.70, side * D(57)
            n = A.surf.normal(s_e, th_e)
            c = A.surf.point(s_e, th_e) - n * 0.095
            g = normalize(-A.hT * 0.78 + n * 0.62 + A.hU * 0.04)
            eu = normalize(A.hU - np.dot(A.hU, g) * g)
            er = np.cross(g, eu)
            glow.append(uv_sphere(c, re, g, seg=lod.eye_seg[0], rings=lod.eye_seg[1], ref=A.hU))

            def mapper(a, b, rr=re * 1.008, c=c, g=g, er=er, eu=eu):
                d = normalize(g[None] + np.asarray(a)[:, None] * er[None] + np.asarray(b)[:, None] * eu[None])
                return c[None] + d * rr
            beta = np.linspace(-1, 1, 15)
            slit = np.concatenate([np.stack([0.15 * np.cos(beta * np.pi / 2) ** 0.9, beta * 0.62], 1),
                                   np.stack([-0.15 * np.cos(beta[::-1] * np.pi / 2) ** 0.9, beta[::-1] * 0.62], 1)[1:-1]])
            self.land["eyes"].append(dict(center=c, gaze=g, up=eu, right=er, radius=re, mapper=mapper, side=side))
            pupil = polar_decal(slit, mapper, rings=3, center2d=(0, 0), uv=swatch_uv(SW_PUPIL), ref_normal=g)
            ring = polar_decal(circle2d(0.80, 32), lambda a, b, m=mapper: m(a, b), rings=1, center2d=(0, 0),
                               hole=0.84, uv=swatch_uv(SW_IRIS), ref_normal=g)
            gl = polar_decal(circle2d(0.075, 10, c=(-0.24 * side * 0 - 0.22, 0.30)),
                             lambda a, b, m=mapper: m(a, b) + 0.001 * g, rings=1, uv=swatch_uv(SW_GLINT), ref_normal=g)
            gl2 = polar_decal(circle2d(0.035, 8, c=(0.18, -0.26)), mapper, rings=1, uv=swatch_uv(SW_GLINT), ref_normal=g)
            detail += [pupil.compact(), ring, gl, gl2]
        return merge(glow), merge(detail)

    # ----------------------------------------------------------- nostrils
    def nostrils(self):
        A = self.A
        out = []
        for side in (1, -1):
            s0, th0 = 0.215, side * D(30)
            n = A.surf.normal(s0, th0)
            ol = circle2d(0.042, 12, sx=1.25, sy=0.62)
            rot = D(35) * side
            ol = ol @ np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).T
            out.append(polar_decal(ol, surface_mapper(A.surf, s0, th0, 0.004), rings=2, center2d=(0, 0),
                                   uv=swatch_uv(SW_NOSTRIL), ref_normal=n))
        return merge(out)

    # ----------------------------------------------------------- armour plates
    def plate(self, outline_sth, thick, surf=None, rings=None, back=0.025, lift=0.0):
        """outline in (s, th) pairs; thick(f, k) thickness; returns MeshData (armour)."""
        surf = surf or self.A.surf
        O = np.asarray(outline_sth, float)
        from pmcore.mathx import catmull_rom
        closed = np.vstack([O[-1:], O, O[:2]])
        sm = catmull_rom(closed, samples=6)[6:-7]
        pts = surf.point(sm[:, 0], sm[:, 1], lift)
        cs = sm.mean(axis=0)
        ctr = surf.point(cs[0], cs[1], lift)
        nfn = lambda P: np.array([surf.normal(*self._nearest_sth(surf, p, sm, cs)) for p in P])
        m = star_plate(pts, ctr, nfn, lambda f, k: thick(f, k / len(sm)), rings=rings or self.lod.plate_rings,
                       back=back, uv_fn=lambda f, kk: (0.05 + 0.9 * f, 0.10 + 0.3 * kk))
        return m

    @staticmethod
    def _nearest_sth(surf, p, sm, cs):
        return cs  # plates are small: centre normal is a stable approximation

    def head_plates(self):
        A = self.A
        plates = []
        for side in (1, -1):
            sd = side
            # brow: thick, angled down toward the snout for an intense gaze
            brow = [(1.28, sd * D(30)), (1.40, sd * D(46)), (1.55, sd * D(57)), (1.80, sd * D(62)),
                    (2.05, sd * D(60)), (2.30, sd * D(50)), (2.05, sd * D(40)), (1.70, sd * D(33))]
            low = lambda f, k: 0.03 + 0.11 * np.clip(np.sin(np.pi * np.clip((k - 0.05) / 0.62, 0, 1)), 0, 1)
            plates.append(self.plate(brow, lambda f, k: low(f, k) * (1 - 0.35 * f ** 4) + 0.02))
            # layered cheek plates (back ones first so front ones sit on top)
            for i, (s0, th0, ln, wd, lift) in enumerate([(2.62, 100, 0.52, 17, 0.030), (2.42, 112, 0.50, 15, 0.055),
                                                        (2.20, 96, 0.46, 14, 0.080)]):
                o = [(s0 - ln * 0.55, sd * D(th0)), (s0 - ln * 0.2, sd * D(th0 - wd)), (s0 + ln * 0.35, sd * D(th0 - wd * 0.7)),
                     (s0 + ln * 0.75, sd * D(th0 + 2)), (s0 + ln * 0.3, sd * D(th0 + wd)), (s0 - ln * 0.25, sd * D(th0 + wd * 0.9))]
                plates.append(self.plate(o, lambda f, k, lift=lift: (lift + 0.035 + 0.05 * smoothstep(0.3, 0.6, k) * (1 - smoothstep(0.6, 0.9, k))) * (1 - 0.3 * f ** 3)))
            # jaw-line plates on the lower jaw
            js = self.jaw_surf
            for s0 in (1.55, 2.05):
                o = [(s0 - 0.28, sd * D(110)), (s0 + 0.0, sd * D(96)), (s0 + 0.32, sd * D(102)),
                     (s0 + 0.20, sd * D(122)), (s0 - 0.12, sd * D(126))]
                plates.append(self.plate(o, lambda f, k: 0.035 * (1 - 0.3 * f ** 3), surf=js))
        # forehead crest (diamond) and muzzle shield
        crest = [(1.18, 0.0), (1.55, D(20)), (2.15, D(18)), (2.75, D(9)), (2.95, 0.0), (2.75, -D(9)), (2.15, -D(18)), (1.55, -D(20))]
        plates.append(self.plate(crest, lambda f, k: 0.075 * (1 - 0.45 * f ** 3) + 0.01))
        muzzle = [(0.32, 0.0), (0.55, D(17)), (1.05, D(19)), (1.25, D(10)), (1.20, 0.0), (1.25, -D(10)), (1.05, -D(19)), (0.55, -D(17))]
        plates.append(self.plate(muzzle, lambda f, k: 0.04 * (1 - 0.4 * f ** 3) + 0.006))
        self.land["crest"] = dict(s=(1.25, 2.85), thick=0.085)
        return merge(plates)

    # ----------------------------------------------------------- horns
    def horn_spec(self):
        """(name, base (s, th), path offsets in head-local (back, out, up), r0, ridges)."""
        return [
            ("MainHorn", (2.28, D(31)), [(0, 0, 0), (0.35, 0.10, 0.24), (0.95, 0.30, 0.58), (1.55, 0.44, 1.02),
                                         (1.88, 0.44, 1.56), (1.78, 0.33, 2.08)], 0.175, 8),
            ("BrowHorn", (1.98, D(47)), [(0, 0, 0), (0.30, 0.13, 0.13), (0.62, 0.27, 0.22), (0.88, 0.37, 0.20)], 0.078, 4),
            ("CheekSpikeA", (2.62, D(96)), [(0, 0, 0), (0.30, 0.18, -0.02), (0.60, 0.33, -0.08), (0.82, 0.42, -0.14)], 0.070, 3),
            ("CheekSpikeB", (2.86, D(116)), [(0, 0, 0), (0.25, 0.15, -0.08), (0.50, 0.27, -0.19), (0.66, 0.33, -0.28)], 0.055, 3),
            ("NoseHorn", (0.66, 0.0), [(0, 0, 0), (0.04, 0, 0.12), (0.0, 0, 0.24), (-0.09, 0, 0.31)], 0.062, 0),
        ]

    def horns(self):
        A, lod = self.A, self.lod
        out, self.land["horns"] = [], []
        for name, (s0, th0), offs, r0, ridges in self.horn_spec():
            for side in ((1, -1) if th0 != 0 else (1,)):
                th = th0 * side
                n = A.surf.normal(s0, th)
                base = A.surf.point(s0, th) - n * (0.6 * r0)
                pts = [base + o[0] * A.hT + side * o[1] * A.hS + o[2] * A.hU for o in offs]
                pts = np.array(pts)
                big = name == "MainHorn"
                prof = lambda L, r0=r0, ridges=ridges, big=big: tapered(L, r0, 0.006, power=0.95 if big else 1.1,
                                                                         oval=1.12, ridges=ridges, ridge_amp=0.07,
                                                                         ridge_end=0.62)
                rings = lod.horn_rings if big else max(6, lod.horn_rings // 2)
                surf, m = sweep_part(pts, prof, lod.horn_around if big else max(6, lod.horn_around - 4), rings,
                                     up=A.hU if abs(np.dot(normalize(pts[1] - pts[0]), A.hU)) < 0.9 else A.hT)
                out.append(m)
                self.land["horns"].append(dict(name=name, side=side, surf=surf, r0=r0))
        # jaw spikes
        js = self.jaw_surf
        for side in (1, -1):
            for s0, ln in ((2.15, 0.36), (2.45, 0.30)):
                th = side * D(140)
                n = js.normal(s0, th)
                base = js.point(s0, th) - n * 0.03
                d1 = normalize(n + 0.9 * js.frame(s0)[1])
                pts = np.array([base, base + d1 * ln * 0.5 + n * 0.02, base + d1 * ln])
                surf, m = sweep_part(pts, lambda L: tapered(L, 0.045, 0.004), max(6, lod.horn_around - 6), 6, up=A.hS)
                out.append(m)
        return merge(out)

    # ----------------------------------------------------------- teeth
    def teeth(self):
        A, lod = self.A, self.lod
        bb = A.profile.bb
        out = []
        down = -A.hU
        for side in (1, -1):
            # upper fangs
            specs = [(0.52, 0.86, 0.36, 0.068)]
            if lod.small_teeth:
                specs += [(0.86, 0.86, 0.15, 0.034), (1.15, 0.87, 0.14, 0.032), (1.45, 0.87, 0.13, 0.03), (1.75, 0.86, 0.11, 0.028)]
            else:
                specs += [(1.05, 0.87, 0.15, 0.036)]
            for d, xf, ln, r0 in specs:
                x = side * xf * A.profile.a(d) * 0.88
                base = A.head_local(d, x, -bb(d) * 0.55)
                tipdir = normalize(down * 1.0 + A.hT * 0.25 - side * A.hS * 0.08)
                pts = np.array([base, base + down * (0.45 * ln + bb(d) * 0.45),
                                base + down * (bb(d) * 0.45) + tipdir * ln])
                _, m = sweep_part(pts, lambda L, r0=r0: tapered(L, r0, 0.003, power=1.15), 8 if lod.small_teeth else 6,
                                  7 if lod.small_teeth else 4, up=A.hT, v_range=(0.53, 0.97))
                out.append(m)
            # lower fangs on the jaw
            js = self.jaw_surf
            lspecs = [(0.40, 0.25, 0.05)] + ([(0.85, 0.12, 0.028), (1.2, 0.11, 0.026), (1.55, 0.1, 0.025)] if lod.small_teeth else [])
            for sj, ln, r0 in lspecs:
                th = side * D(48)
                P, T, U, S = js.frame(sj)
                base = js.point(sj, th) - U * 0.05
                up = normalize(U - T * 0.25)
                pts = np.array([base, base + up * (ln * 0.55 + 0.05), base + up * (ln + 0.05) - T * 0.04])
                _, m = sweep_part(pts, lambda L, r0=r0: tapered(L, r0, 0.003, power=1.1), 8 if lod.small_teeth else 6,
                                  6 if lod.small_teeth else 4, up=T, v_range=(0.53, 0.97))
                out.append(m)
        return merge(out)

    # ----------------------------------------------------------- dorsal spines
    def spine_positions(self):
        A = self.A
        out = []
        s = A.H + 0.30
        i = 0
        while s < A.L - 1.7:
            r = float(A.r(s))
            big = i % 2 == 0
            h = r * (0.60 if big else 0.40) * (1.12 - 0.12 * smoothstep(A.H, A.H + 3, s))
            out.append((s, h, big, r))
            s += 0.50 * r + 0.10
            i += 1
        return out

    def spines(self):
        A, lod = self.A, self.lod
        out = []
        self.land["spines"] = []
        for idx, (s, h, big, r) in enumerate(self.spine_positions()):
            if lod.spine_every == 2 and not big:
                continue
            n = A.surf.normal(s, 0.0)
            T = A.surf.frame(s)[1]
            e = 0.22 * r
            base = A.surf.point(s, 0.0) - n * e
            pts = np.array([base, base + n * 0.35 * (h + e) + T * 0.06 * h,
                            base + n * 0.72 * (h + e) + T * 0.30 * h, base + n * (h + e) + T * 0.66 * h])
            c = 0.62 * r * (1.0 if big else 0.8)

            def prof(L, c=c, r=r):
                t = lambda s_: np.clip(np.asarray(s_, float) / L, 0, 1)
                return SuperProfile(a=lambda s_: 0.075 * r * (1 - t(s_)) ** 0.7 + 0.004,
                                    bt=lambda s_: c * 0.32 * (1 - t(s_)) ** 1.1 + 0.004,
                                    bb=lambda s_: c * 0.58 * (1 - t(s_)) ** 0.75 + 0.004,
                                    nt=lambda s_: np.full_like(np.asarray(s_, float), 1.7))
            surf, m = sweep_part(pts, prof, lod.spine_around, lod.spine_rings, up=-T)
            out.append(m)
            self.land["spines"].append(dict(s=s, h=h, big=big, surf=surf, tip=pts[-1]))
        return merge(out)

    # ----------------------------------------------------------- tail spade
    def tail_spade(self):
        A = self.A
        s_attach = A.L - 0.62
        P, T, U, S = A.surf.frame(s_attach)
        ol = [(1.38, 0.0), (1.05, 0.13), (0.72, 0.34), (0.50, 0.50), (0.36, 0.56), (0.12, 0.52), (0.26, 0.30), (0.12, 0.12),
              (0.0, 0.07)]
        ol = ol + [(a, -b) for a, b in ol[::-1][:-1]]
        ol = np.array(ol)
        from pmcore.mathx import catmull_rom
        sm = catmull_rom(np.vstack([ol[-1:], ol, ol[:2]]), samples=4)[4:-5]
        scale = 0.85
        blade = lens_blade(sm * scale, np.array([0.48, 0.0]) * scale, P, T, S, U, thickness=0.085, rings=4,
                           uv_fn=lambda f, p: (0.05 + 0.9 * f, 0.25)).compact()
        self.land["spade"] = dict(origin=P, T=T, S=S, U=U, outline=sm * scale)
        return blade
