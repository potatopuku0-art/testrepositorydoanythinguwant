"""Serpent anatomy: pose (centerline), cross-section profile, head sculpt and landmarks.

Arc length s runs from the snout tip (s=0) to the tail tip (s=L). th=0 is the dorsal
midline, +th turns toward the serpent's right side, th=pi is the belly.
"""
import numpy as np

from pmcore.mathx import pchip, smoothstep, normalize
from pmcore.sweep import SweepSurface, SuperProfile, Bump

D = np.radians

# Pose: head (straight segment) -> rearing S-shaped neck -> lateral S-waves -> curled tail.
POSE = [
    (-1.05, -10.15, 7.95),   # snout tip
    (-0.27, -7.27, 8.42),    # back of skull (3.0 from snout, collinear)
    (0.06, -6.04, 8.62),     # neck, still collinear with the head
    (0.85, -4.75, 7.75),
    (1.55, -3.75, 5.85),
    (1.35, -2.30, 4.05),
    (-0.55, -0.70, 3.25),
    (-3.15, 0.95, 3.55),
    (-4.35, 3.35, 4.55),
    (-3.05, 5.85, 5.35),
    (0.40, 7.20, 4.65),
    (3.60, 8.60, 3.40),
    (4.65, 11.00, 3.05),
    (3.45, 13.40, 3.80),
    (0.65, 14.65, 4.60),
    (-1.80, 16.20, 5.05),
    (-2.65, 18.40, 5.65),
    (-1.85, 20.20, 6.60),
    (-0.45, 20.85, 7.55),
    (0.35, 20.25, 8.35),
]

# Head sculpt keys (s from snout). Half width a, top bt, bottom bb, exponents.
HEAD_S = [0.0, 0.035, 0.11, 0.28, 0.6, 1.0, 1.35, 1.7, 2.05, 2.4, 2.75, 3.1, 3.6, 4.2]
HEAD_A = [0.0, 0.14, 0.235, 0.32, 0.365, 0.42, 0.55, 0.74, 0.86, 0.885, 0.83, 0.76, 0.72, 0.72]
HEAD_BT = [0.0, 0.11, 0.18, 0.225, 0.255, 0.29, 0.40, 0.555, 0.65, 0.675, 0.665, 0.655, 0.68, 0.70]
HEAD_BB = [0.0, 0.075, 0.12, 0.15, 0.17, 0.19, 0.21, 0.24, 0.28, 0.34, 0.43, 0.55, 0.62, 0.63]
HEAD_NT = [2.0, 2.0, 2.3, 2.6, 2.75, 2.75, 2.6, 2.45, 2.35, 2.3, 2.25, 2.2, 2.2, 2.2]
HEAD_NB = [2.0, 2.0, 2.6, 3.0, 3.1, 3.1, 3.0, 2.85, 2.7, 2.65, 2.6, 2.6, 2.6, 2.6]

# Body radius keys as a fraction t of the body (head end -> tail tip).
BODY_T = [0.0, 0.035, 0.08, 0.14, 0.22, 0.32, 0.44, 0.57, 0.70, 0.82, 0.92, 0.975, 1.0]
BODY_R = [0.70, 0.72, 0.86, 0.97, 1.0, 0.975, 0.86, 0.67, 0.46, 0.285, 0.15, 0.08, 0.06]


class SerpentAnatomy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.H = cfg.head_len
        self.L = 48.0  # replaced once the sweep is built

        H = self.H
        self.w_head = lambda s: 1.0 - smoothstep(H - 0.1, H + 1.2, s)

        def r(s):
            t = (np.asarray(s, float) - H) / (self.L - H)
            return pchip(BODY_T, BODY_R, t)
        self.r = r

        def mix(keys, body_fn, scaled=True):
            hs = np.asarray(keys, float) * np.where(np.asarray(HEAD_S) <= 3.1, cfg.head_scale, 1.0) if scaled \
                else np.asarray(keys, float)
            return lambda s: (self.w_head(s) * pchip(HEAD_S, hs, s)
                              + (1 - self.w_head(s)) * body_fn(s))
        a = mix(HEAD_A, lambda s: 1.04 * r(s))
        bt = mix(HEAD_BT, lambda s: 0.98 * r(s))
        bb = mix(HEAD_BB, lambda s: 0.90 * r(s))
        nt = mix(HEAD_NT, lambda s: np.full_like(np.asarray(s, float), 2.15), scaled=False)
        nb = mix(HEAD_NB, lambda s: np.full_like(np.asarray(s, float), 2.65), scaled=False)

        body_w = (H + 0.6, 1e3, 0.5)
        bumps = [
            # --- head sculpt ---
            Bump(1.70, D(58), 0.21, D(13), -0.075),           # eye sockets
            Bump(1.47, D(40), 0.24, D(11), +0.055),           # brow ridges
            Bump(2.25, D(62), 0.30, D(14), +0.040),           # temples
            Bump(2.30, D(104), 0.36, D(20), +0.060),          # cheek muscles
            Bump(0.23, D(30), 0.075, D(10), +0.040),          # nostril flare
            Bump(0.215, D(30), 0.032, D(5.5), -0.040),        # nostril pits
            Bump(0.85, D(17), 0.42, D(5), +0.022),            # muzzle ridges
            Bump(1.65, 0.0, 0.40, D(7), -0.022, mirror=False),  # forehead groove
            Bump(0.55, D(122), 0.35, D(8), -0.018),           # lip groove
            # --- body ---
            Bump(0, 0.0, 1, D(9), 0.035, mirror=False, rel_fn=r, s_window=body_w),   # spine ridge
            Bump(0, D(70), 1, D(18), 0.025, rel_fn=r, s_window=body_w),              # flank muscle
        ]
        self.profile = SuperProfile(a, bt, bb, nt, nb, bumps=bumps)
        self.surf = SweepSurface(POSE, self.profile, up=(0, 0, 1), ds=0.01, smooth_sigma=0.45,
                                 smooth_from=(H + 0.4, H + 2.2), twist_sigma=1.6, bank=0.55)
        self.L = self.surf.L
        # Head frame (head is a straight segment, so one frame describes it)
        P0, _, _, _ = self.surf.frame(0.0)
        _, T, U, S = self.surf.frame(1.5)
        self.head_origin, self.hT, self.hU, self.hS = P0, T, U, S

    # --------------------------------------------------------------- helpers
    def head_local(self, d, x, y):
        d, x, y = np.broadcast_arrays(np.asarray(d, float), np.asarray(x, float), np.asarray(y, float))
        return (self.head_origin + d[..., None] * self.hT + x[..., None] * self.hS + y[..., None] * self.hU)

    def to_local_dir(self, dd, dx, dy):
        return normalize(dd * self.hT + dx * self.hS + dy * self.hU)

    def r_eff(self, s):
        return self.profile.r_eff(s)

    def r_max(self, s):
        return self.profile.r_max(s)

    def centerline_samples(self, step=0.05):
        s = np.arange(0, self.L, step)
        P = self.surf.frame(s)[0]
        return s, P, self.r_max(s) * 1.12

    def u_coord(self):
        """Texture u along the body: one tile per 2*pi*r_eff of length (isotropic scales)."""
        s = np.linspace(0, self.L, 6000)
        inv = 1.0 / (2 * np.pi * np.maximum(self.r_eff(s), 0.22))
        cum = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
        return lambda x: np.interp(x, s, cum)

    def clearance(self, pts, skip=None):
        """Approximate signed distance from points to the body tube (positive = outside)."""
        s, P, R = self.centerline_samples(0.04)
        if skip is not None:
            keep = (s < skip[0]) | (s > skip[1])
            s, P, R = s[keep], P[keep], R[keep]
        pts = np.atleast_2d(pts)
        d = np.linalg.norm(pts[:, None, :] - P[None], axis=-1) - R[None]
        return d.min(axis=1)
