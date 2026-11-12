"""Continuous swept surfaces: the backbone of every creature body, horn, fin and jaw.

A SweepSurface is a centerline (spline) + a stable moving frame + a cross-section
profile evaluated as a function of arc length s and angle th (th = 0 is dorsal,
+th towards the creature's right side). It can be evaluated anywhere, so every
decoration (glyphs, circuits, plates, rings) is placed *on the same analytic
surface* and follows curvature exactly.
"""
import numpy as np

from .mathx import (normalize, catmull_rom, resample_arclength, gaussian_smooth,
                    rotation_minimizing_frames, signed_angle, rotate, pchip, smoothstep, wrap_angle)
from .meshdata import MeshData, vertex_normals


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class Bump:
    """Gaussian relief in (s, th) space. amp is absolute, or relative to rel_fn(s)."""

    def __init__(self, s0, th0, ss, sth, amp, mirror=True, rel_fn=None, s_window=None):
        self.s0, self.th0, self.ss, self.sth, self.amp = s0, th0, ss, sth, amp
        self.mirror, self.rel_fn, self.s_window = mirror, rel_fn, s_window

    def __call__(self, s, th):
        if self.s_window is not None:
            a, b, soft = self.s_window
            ws = smoothstep(a - soft, a + soft, s) * (1 - smoothstep(b - soft, b + soft, s))
        else:
            ws = np.exp(-0.5 * ((s - self.s0) / self.ss) ** 2)
        val = np.exp(-0.5 * (wrap_angle(th - self.th0) / self.sth) ** 2)
        if self.mirror and abs(self.th0) > 1e-9:
            val = val + np.exp(-0.5 * (wrap_angle(th + self.th0) / self.sth) ** 2)
        amp = self.amp * (self.rel_fn(s) if self.rel_fn else 1.0)
        return amp * ws * val


class SuperProfile:
    """Superellipse cross-section with separate top/bottom heights and exponents.

    a(s): half width, bt(s)/bb(s): height above / below the centerline,
    nt(s)/nb(s): superellipse exponents (2 = ellipse, larger = boxier).
    """

    def __init__(self, a, bt, bb, nt=None, nb=None, bumps=(), y_off=None, r_eff=None):
        self.a, self.bt, self.bb = a, bt, bb
        self.nt = nt or (lambda s: np.full_like(np.asarray(s, float), 2.0))
        self.nb = nb or self.nt
        self.bumps = list(bumps)
        self.y_off = y_off
        self._r_eff = r_eff

    def base_xy(self, s, th):
        s = np.asarray(s, dtype=float)
        th = np.asarray(th, dtype=float)
        sn, cs = np.sin(th), np.cos(th)
        top = cs >= 0
        n = np.where(top, self.nt(s), self.nb(s))
        e = 2.0 / n
        x = self.a(s) * np.sign(sn) * np.abs(sn) ** e
        y = np.where(top, self.bt(s), self.bb(s)) * np.sign(cs) * np.abs(cs) ** e
        return x, y

    def xy(self, s, th):
        x, y = self.base_xy(s, th)
        if self.bumps:
            b = sum(bp(s, th) for bp in self.bumps)
            rho = np.maximum(np.hypot(x, y), 1e-5)
            k = 1.0 + b / rho
            x, y = x * k, y * k
        if self.y_off is not None:
            y = y + self.y_off(s)
        return x, y

    def r_eff(self, s):
        if self._r_eff is not None:
            return self._r_eff(s)
        return 0.25 * (2 * self.a(s) + self.bt(s) + self.bb(s))

    def r_max(self, s):
        return np.maximum(self.a(s), np.maximum(self.bt(s), self.bb(s)))


def keyed(keys, values):
    keys, values = np.asarray(keys, float), np.asarray(values, float)
    return lambda s: pchip(keys, values, s)


class TubeProfile(SuperProfile):
    """Tapered (optionally oval / ridged) tube, s normalised by length inside the fn."""

    def __init__(self, length, r0, r1=0.0, power=1.0, oval=1.0, ridges=0, ridge_amp=0.0,
                 ridge_end=0.6, bulge=0.0):
        self.length = length  # updated by tube() to the true swept length
        self.r0, self.r1, self.power, self.ridges = r0, r1, power, ridges
        self.ridge_amp, self.ridge_end, self.bulge = ridge_amp, ridge_end, bulge
        super().__init__(a=self.r, bt=lambda s: self.r(s) * oval, bb=lambda s: self.r(s) * oval)

    def r(self, s):
        t = np.clip(np.asarray(s, float) / self.length, 0, 1)
        base = self.r1 + (self.r0 - self.r1) * (1 - t) ** self.power
        base = base * (1 + self.bulge * np.sin(np.pi * np.clip(t * 1.6, 0, 1)) * (1 - t))
        if self.ridges:
            rib = np.maximum(0, np.sin(2 * np.pi * self.ridges * t)) ** 3
            base = base * (1 + self.ridge_amp * rib * (1 - smoothstep(self.ridge_end * 0.7, self.ridge_end, t)))
        return base


# ---------------------------------------------------------------------------
# Swept surface
# ---------------------------------------------------------------------------

class SweepSurface:
    def __init__(self, points, profile, up=(0, 0, 1), ds=0.01, spline=True, smooth_sigma=0.0,
                 smooth_from=None, up_mode="world", twist_sigma=1.2, bank=0.0, bank_sigma=1.5):
        P = catmull_rom(points, samples=96) if spline else np.asarray(points, float)
        P, s = resample_arclength(P, ds)
        if smooth_sigma > 0:
            Ps = gaussian_smooth(P, smooth_sigma / ds)
            if smooth_from is not None:  # keep e.g. a straight head segment untouched
                w = smoothstep(smooth_from[0], smooth_from[1], s)[:, None]
                Ps = P + (Ps - P) * w
            P, s = resample_arclength(Ps, ds)
        self.ds = s[1] - s[0]
        self.s = s
        self.L = s[-1]
        T = normalize(np.gradient(P, axis=0))
        up = normalize(np.asarray(up, float))
        U = rotation_minimizing_frames(P, T, up if abs(np.dot(up, T[0])) < 0.99 else np.array([1.0, 0, 0]))
        if up_mode == "world":
            d = up[None] - np.sum(up[None] * T, axis=1, keepdims=True) * T
            w = np.linalg.norm(d, axis=1) ** 2
            phi = np.unwrap(signed_angle(U, normalize(d), T))
            sig = twist_sigma / self.ds
            phi_s = gaussian_smooth(phi * w, sig, odd=False) / np.maximum(gaussian_smooth(w, sig, odd=False), 1e-6)
            U = rotate(U, T, phi_s)
            if bank:
                k = np.gradient(T, axis=0) / self.ds
                S = np.cross(U, T)
                lat = np.sum(k * S, axis=1)
                ang = gaussian_smooth(np.arctan(bank * lat), bank_sigma / self.ds, odd=False)
                U = rotate(U, T, -ang)
        U = normalize(U - np.sum(U * T, axis=1, keepdims=True) * T)
        self.P, self.T, self.U = P, T, U
        self.profile = profile

    # -------------------------------------------------------------- transforms
    def apply_rotation(self, R, pivot):
        R = np.asarray(R)
        self.P = (self.P - pivot) @ R.T + pivot
        self.T = self.T @ R.T
        self.U = self.U @ R.T
        return self

    # -------------------------------------------------------------- evaluation
    def frame(self, s):
        s = np.clip(np.asarray(s, float), 0.0, self.L)
        f = s / self.ds
        i = np.clip(np.floor(f).astype(int), 0, len(self.P) - 2)
        t = (f - i)[..., None]
        P = self.P[i] * (1 - t) + self.P[i + 1] * t
        T = normalize(self.T[i] * (1 - t) + self.T[i + 1] * t)
        U = self.U[i] * (1 - t) + self.U[i + 1] * t
        U = normalize(U - np.sum(U * T, axis=-1, keepdims=True) * T)
        S = np.cross(U, T)
        return P, T, U, S

    def point(self, s, th, offset=0.0):
        s, th = np.broadcast_arrays(np.asarray(s, float), np.asarray(th, float))
        P, T, U, S = self.frame(s)
        x, y = self.profile.xy(s, th)
        p = P + x[..., None] * S + y[..., None] * U
        off = np.asarray(offset, float)
        if np.any(off != 0):
            p = p + np.broadcast_to(off, s.shape)[..., None] * self.normal(s, th)
        return p

    def normal(self, s, th, h=2e-3):
        s, th = np.broadcast_arrays(np.asarray(s, float), np.asarray(th, float))
        Ps = self.point(s + h, th) - self.point(s - h, th)
        Pt = self.point(s, th + h) - self.point(s, th - h)
        n = np.cross(Ps, Pt)
        bad = np.linalg.norm(n, axis=-1) < 1e-12
        if np.any(bad):  # degenerate tip: fall back to radial / tangent
            P, T, U, S = self.frame(s)
            n = np.where(bad[..., None], -T if np.mean(s) < self.L / 2 else T, n)
        return normalize(n)

    def metrics(self, s, th, h=2e-3):
        """World length per unit s and per radian th (for mapping world-sized decals)."""
        s, th = np.broadcast_arrays(np.asarray(s, float), np.asarray(th, float))
        ms = np.linalg.norm(self.point(s + h, th) - self.point(s - h, th), axis=-1) / (2 * h)
        mt = np.linalg.norm(self.point(s, th + h) - self.point(s, th - h), axis=-1) / (2 * h)
        return ms, np.maximum(mt, 1e-5)

    def surface_frame(self, s, th):
        """Orthonormal (along, around, normal) basis on the surface."""
        h = 2e-3
        a = normalize(self.point(s + h, th) - self.point(s - h, th))
        n = self.normal(s, th)
        b = normalize(np.cross(n, a))
        a = np.cross(b, n)
        return a, b, n

    def map_local(self, s0, th0, lx, ly, offset=0.0):
        """Map world-unit local coords (lx along s, ly around th) to the surface."""
        ms, mt = self.metrics(s0, th0)
        return self.point(s0 + np.asarray(lx) / ms, th0 + np.asarray(ly) / mt, offset)

    # -------------------------------------------------------------- meshing
    def mesh(self, s_rings, n_rings, cap_start=True, cap_end=True, u_fn=None, v_range=(0.0, 1.0),
             th0=0.0):
        """Quad mesh over rings. Consecutive ring counts must be equal or differ by 2x.
        Seam sits at th0 (dorsal by default, hidden under spines)."""
        s_rings = np.asarray(s_rings, float)
        n_rings = [int(n) for n in n_rings]
        u_fn = u_fn or (lambda s: s / self.L)
        v0, v1 = v_range
        verts, starts = [], []
        for s, n in zip(s_rings, n_rings):
            th = th0 + 2 * np.pi * np.arange(n) / n
            starts.append(sum(len(v) for v in verts))
            verts.append(self.point(np.full(n, s), th))
        V = np.vstack(verts)
        U_ring = u_fn(s_rings)
        faces, uvs = [], []

        def uv(r, j):
            n = n_rings[r]
            return (float(U_ring[r]), v0 + (v1 - v0) * j / n)

        def vid(r, j):
            return starts[r] + (j % n_rings[r])
        for r in range(len(s_rings) - 1):
            na, nb = n_rings[r], n_rings[r + 1]
            if na == nb:
                for j in range(na):
                    faces.append((vid(r, j), vid(r, j + 1), vid(r + 1, j + 1), vid(r + 1, j)))
                    uvs.append((uv(r, j), uv(r, j + 1), uv(r + 1, j + 1), uv(r + 1, j)))
            elif na == 2 * nb:
                for k in range(nb):
                    faces.append((vid(r, 2 * k), vid(r, 2 * k + 1), vid(r + 1, k)))
                    uvs.append((uv(r, 2 * k), uv(r, 2 * k + 1), uv(r + 1, k)))
                    faces.append((vid(r, 2 * k + 1), vid(r, 2 * k + 2), vid(r + 1, k + 1), vid(r + 1, k)))
                    uvs.append((uv(r, 2 * k + 1), uv(r, 2 * k + 2), uv(r + 1, k + 1), uv(r + 1, k)))
            elif nb == 2 * na:
                for k in range(na):
                    faces.append((vid(r, k), vid(r + 1, 2 * k + 1), vid(r + 1, 2 * k)))
                    uvs.append((uv(r, k), uv(r + 1, 2 * k + 1), uv(r + 1, 2 * k)))
                    faces.append((vid(r, k), vid(r, k + 1), vid(r + 1, 2 * k + 2), vid(r + 1, 2 * k + 1)))
                    uvs.append((uv(r, k), uv(r, k + 1), uv(r + 1, 2 * k + 2), uv(r + 1, 2 * k + 1)))
            else:
                raise ValueError("ring counts must match or differ by 2x")
        extra = []
        if cap_start:
            tip = len(V) + len(extra)
            st = max(s_rings[0] - 0.5 * (s_rings[1] - s_rings[0]), 0.0) if s_rings[0] > 1e-3 else 0.0
            extra.append(self.frame(st)[0])
            n = n_rings[0]
            ut = float(u_fn(np.array([st]))[0])
            for j in range(n):
                faces.append((tip, vid(0, j + 1), vid(0, j)))
                uvs.append(((ut, v0 + (v1 - v0) * (j + .5) / n), uv(0, j + 1), uv(0, j)))
        if cap_end:
            tip = len(V) + len(extra)
            r = len(s_rings) - 1
            extra.append(self.frame(min(s_rings[-1] + 0.5 * (s_rings[-1] - s_rings[-2]), self.L))[0])
            n = n_rings[r]
            ue = float(U_ring[r])
            for j in range(n):
                faces.append((tip, vid(r, j), vid(r, j + 1)))
                uvs.append(((ue, v0 + (v1 - v0) * (j + .5) / n), uv(r, j), uv(r, j + 1)))
        if extra:
            V = np.vstack([V, np.array(extra)])
        m = MeshData(V, faces, uvs)
        # outward winding check on a mid face
        f = faces[len(faces) // 2]
        a, b, c = V[f[0]], V[f[1]], V[f[2]]
        mid_r = len(s_rings) // 2
        center = self.frame(s_rings[mid_r])[0]
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - center) < 0:
            m = m.flipped()
        m.normals = vertex_normals(m.verts, m.faces)
        return m


def ring_positions(s0, s1, spacing_fn, min_rings=2):
    """March from s0 to s1 with a spacing function, then rescale so the last ring hits s1."""
    pts = [s0]
    s = s0
    while s < s1:
        s = s + max(float(spacing_fn(s)), 1e-3)
        pts.append(s)
    pts = np.array(pts)
    if len(pts) < min_rings:
        pts = np.linspace(s0, s1, min_rings)
    return s0 + (pts - s0) * (s1 - s0) / (pts[-1] - s0)


def tube(points, profile, around=8, rings=None, up=(0, 0, 1), cap_start=False, cap_end=True,
         v_range=(0.0, 1.0), u_range=(0.0, 1.0), ds=0.005, spline=True):
    """Convenience: horn / fang / spine sweep with an RMF frame."""
    surf = SweepSurface(points, profile, up=up, ds=ds, spline=spline, up_mode="rmf")
    if hasattr(profile, "length"):
        profile.length = surf.L
    n_r = rings or 8
    s = np.linspace(0, surf.L * 0.995, n_r)
    m = surf.mesh(s, [around] * n_r, cap_start=cap_start, cap_end=cap_end,
                  u_fn=lambda x: u_range[0] + (u_range[1] - u_range[0]) * x / surf.L, v_range=v_range)
    return surf, m
