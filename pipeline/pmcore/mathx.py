"""Numeric helpers shared by every creature generator (pure numpy, no bpy)."""
import numpy as np


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def pchip(xk, yk, x):
    """Monotone cubic (Fritsch-Carlson) interpolation, clamped outside the key range."""
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), xk[0], xk[-1])
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    if len(xk) > 2:
        w1 = 2.0 * h[1:] + h[:-1]
        w2 = h[1:] + 2.0 * h[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = (w1 + w2) / (w1 / d[:-1] + w2 / d[1:])
        m[1:-1] = np.where(d[:-1] * d[1:] > 0, hm, 0.0)
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(xk, x) - 1, 0, len(xk) - 2)
    t = (x - xk[i]) / h[i]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * yk[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
            + (-2 * t3 + 3 * t2) * yk[i + 1] + (t3 - t2) * h[i] * m[i + 1])


def gaussian_kernel(sigma):
    rad = int(3 * sigma) + 1
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), rad


def gaussian_smooth(arr, sigma, odd=True):
    """Smooth along axis 0. Odd reflection keeps endpoints and end tangents fixed."""
    arr = np.asarray(arr, dtype=float)
    if sigma <= 0:
        return arr.copy()
    k, rad = gaussian_kernel(sigma)
    rad = min(rad, len(arr) - 1)
    k, _ = gaussian_kernel(min(sigma, rad / 3.0))
    rad = (len(k) - 1) // 2
    if odd:
        head = 2 * arr[0] - arr[1:rad + 1][::-1]
        tail = 2 * arr[-1] - arr[-rad - 1:-1][::-1]
    else:
        head = arr[1:rad + 1][::-1]
        tail = arr[-rad - 1:-1][::-1]
    pad = np.concatenate([head, arr, tail], axis=0)
    flat = pad.reshape(pad.shape[0], -1)
    out = np.stack([np.convolve(flat[:, j], k, mode="valid") for j in range(flat.shape[1])], axis=1)
    return out.reshape(arr.shape)


def rotate(v, axis, ang):
    """Rodrigues rotation of vectors v about unit axes by angles (all broadcastable)."""
    v = np.asarray(v, dtype=float)
    axis = normalize(axis)
    ang = np.asarray(ang, dtype=float)[..., None]
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.sum(axis * v, axis=-1, keepdims=True) * (1 - c)


def rot_matrix(axis, ang):
    axis = normalize(axis)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def catmull_rom(points, samples=64, alpha=0.5):
    """Centripetal Catmull-Rom through all points (no cusps or self-loops)."""
    P = np.asarray(points, dtype=float)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        t = np.linspace(t1, t2, samples, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-1][None])
    return np.vstack(out)


def resample_arclength(P, ds):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(cum[-1] / ds)) + 1)
    s = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(s, cum, P[:, k]) for k in range(3)], axis=1), s


def rotation_minimizing_frames(P, T, U0):
    """Double-reflection RMF (Wang et al. 2008): twist-free normals along a polyline."""
    U = np.zeros_like(P)
    U[0] = normalize(U0 - np.dot(U0, T[0]) * T[0])
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-16:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2.0 / c1) * (v1 @ U[i]) * v1
        tL = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        r = rL - (2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-16 else rL
        U[i + 1] = r / np.linalg.norm(r)
    return U


def signed_angle(a, b, axis):
    return np.arctan2(np.sum(np.cross(a, b) * axis, axis=-1), np.sum(a * b, axis=-1))


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
