"""Procedural texture helpers (numpy only). Arrays are indexed [v, u] with v=0 at the
bottom of the image (UV convention); write_png flips to PNG row order."""
import struct
import zlib

import numpy as np


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def write_png(path, arr, srgb=False):
    """arr float [0,1] (H,W[,C]) in UV orientation. srgb=True converts linear colour."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        a = a[..., None]
    if srgb:
        a = a.copy()
        a[..., :3] = linear_to_srgb(a[..., :3])
    a8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)[::-1]
    h, w, c = a8.shape
    ctype = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + a8[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)
    return path


def hash01(*ints, seed=0):
    """Deterministic per-cell random numbers in [0,1)."""
    h = np.uint64(seed * 2654435761 + 97531) * np.ones(np.broadcast(*ints).shape, dtype=np.uint64)
    for i, v in enumerate(ints):
        v = np.asarray(v).astype(np.int64).astype(np.uint64)
        h = h ^ (v + np.uint64(0x9E3779B97F4A7C15) + (h << np.uint64(6)) + (h >> np.uint64(2)))
        h = h * np.uint64(0xBF58476D1CE4E5B9 + i)
        h = h ^ (h >> np.uint64(31))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def blur(img, radius, wrap=True):
    """Separable box-ish blur (3 passes ~ gaussian), periodic for tiling textures."""
    out = np.asarray(img, dtype=float).copy()
    r = int(max(1, radius))
    for _ in range(3):
        for ax in (0, 1):
            acc = np.zeros_like(out)
            for d in range(-r, r + 1):
                acc += np.roll(out, d, axis=ax)
            out = acc / (2 * r + 1)
    return out


def normal_from_height(h, strength):
    """Tangent-space (OpenGL, +Y up) normal map from a periodic height field."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def value_noise(shape, cells, seed=0, wrap=True):
    """Tileable smooth value noise with `cells` lattice cells across each axis."""
    H, W = shape
    cy, cx = cells if isinstance(cells, tuple) else (cells, cells)
    y = (np.arange(H) + 0.5) / H * cy
    x = (np.arange(W) + 0.5) / W * cx
    X, Y = np.meshgrid(x, y)
    x0, y0 = np.floor(X).astype(int), np.floor(Y).astype(int)
    fx, fy = X - x0, Y - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def r(ix, iy):
        return hash01(ix % cx, iy % cy, seed=seed)
    a, b = r(x0, y0), r(x0 + 1, y0)
    c, d = r(x0, y0 + 1), r(x0 + 1, y0 + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(shape, cells, octaves=4, seed=0):
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * (2 ** o) if not isinstance(cells, tuple) else (cells[0] * 2 ** o, cells[1] * 2 ** o)
        out += amp * value_noise(shape, c, seed=seed + o * 17)
        tot += amp
        amp *= 0.5
    return out / tot


def draw_segments(canvas, segs, width, value=1.0, wrap=True):
    """Rasterise anti-aliased line segments (pixel coords) into a float canvas (max blend).
    segs: (n,4) x0,y0,x1,y1."""
    H, W = canvas.shape[:2]
    for x0, y0, x1, y1 in segs:
        xmin, xmax = int(np.floor(min(x0, x1) - width - 1)), int(np.ceil(max(x0, x1) + width + 1))
        ymin, ymax = int(np.floor(min(y0, y1) - width - 1)), int(np.ceil(max(y0, y1) + width + 1))
        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        X, Y = np.meshgrid(xs + 0.5, ys + 0.5)
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / max(L2, 1e-9), 0, 1)
        d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
        a = np.clip(width * 0.5 + 0.5 - d, 0, 1) * value
        yi, xi = (ys % H)[:, None], (xs % W)[None, :]
        if wrap:
            canvas[yi, xi] = np.maximum(canvas[yi, xi], a)
        else:
            ok_y = (ys >= 0) & (ys < H)
            ok_x = (xs >= 0) & (xs < W)
            sub = canvas[np.ix_(ys[ok_y], xs[ok_x])]
            canvas[np.ix_(ys[ok_y], xs[ok_x])] = np.maximum(sub, a[np.ix_(ok_y, ok_x)])
    return canvas


def draw_discs(canvas, centers, radius, value=1.0, ring=0.0):
    H, W = canvas.shape[:2]
    for cx, cy in centers:
        r = radius + 1
        xs = np.arange(int(cx - r - 1), int(cx + r + 2))
        ys = np.arange(int(cy - r - 1), int(cy + r + 2))
        X, Y = np.meshgrid(xs + 0.5, ys + 0.5)
        d = np.hypot(X - cx, Y - cy)
        a = np.clip(radius + 0.5 - d, 0, 1)
        if ring > 0:
            a = a * np.clip(d - (radius - ring) + 0.5, 0, 1)
        a = a * value
        yi, xi = (ys % H)[:, None], (xs % W)[None, :]
        canvas[yi, xi] = np.maximum(canvas[yi, xi], a)
    return canvas


# 5x7 bitmap glyphs for 0 / 1 (used by particle and decal textures)
GLYPH_BITMAPS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "10100", "00100", "00100", "00100", "11111"],
}


def glyph_atlas(cell=64, glow=True):
    """2x1 atlas with crisp readable 0 and 1 (alpha), for Roblox particle emitters."""
    W, H = cell * 2, cell
    a = np.zeros((H, W))
    px = cell / 9.0
    for gi, ch in enumerate("01"):
        bm = GLYPH_BITMAPS[ch]
        for r, row in enumerate(bm):
            for c, bit in enumerate(row):
                if bit == "1":
                    x0 = gi * cell + (2 + c) * px
                    y0 = H - (1 + r + 1) * px  # flip: row 0 at top
                    xs = slice(int(x0), int(x0 + px * 0.92))
                    ys = slice(int(y0), int(y0 + px * 0.92))
                    a[ys, xs] = 1.0
    if glow:
        a = np.maximum(a, blur(a, 2, wrap=False) * 0.6)
    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = 1.0
    rgba[..., 3] = np.clip(a, 0, 1)
    return rgba
