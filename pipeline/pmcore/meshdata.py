"""Mesh containers and generic geometry builders (pure numpy, no bpy).

Every procedural part is produced as a MeshData so it can be merged, scaled,
split for Roblox triangle limits and converted to Blender in one place.
"""
import numpy as np

from .mathx import normalize


class MeshData:
    def __init__(self, verts=None, faces=None, uvs=None, normals=None, sharp_angle=None):
        self.verts = np.zeros((0, 3)) if verts is None else np.asarray(verts, dtype=float)
        self.faces = [] if faces is None else [tuple(f) for f in faces]
        # per-corner UVs, same nesting as faces
        self.uvs = uvs if uvs is not None else [tuple((0.5, 0.5) for _ in f) for f in self.faces]
        self.normals = None if normals is None else np.asarray(normals, dtype=float)
        # None -> fully smooth, 0 -> flat, else split edges sharper than angle (radians)
        self.sharp_angle = sharp_angle

    # ------------------------------------------------------------------ basics
    def tri_count(self):
        return sum(len(f) - 2 for f in self.faces)

    def add(self, other):
        off = len(self.verts)
        if self.normals is not None or other.normals is not None:
            a = self.normals if self.normals is not None else vertex_normals(self.verts, self.faces)
            b = other.normals if other.normals is not None else vertex_normals(other.verts, other.faces)
            self.normals = np.vstack([a, b]) if len(self.verts) else b
        self.verts = np.vstack([self.verts, other.verts]) if len(self.verts) else other.verts.copy()
        self.faces += [tuple(i + off for i in f) for f in other.faces]
        self.uvs += list(other.uvs)
        return self

    def copy(self):
        m = MeshData(self.verts.copy(), list(self.faces), list(self.uvs),
                     None if self.normals is None else self.normals.copy(), self.sharp_angle)
        return m

    def flipped(self):
        m = self.copy()
        m.faces = [tuple(reversed(f)) for f in m.faces]
        m.uvs = [tuple(reversed(u)) for u in m.uvs]
        if m.normals is not None:
            m.normals = -m.normals
        return m

    def transformed(self, M3=None, offset=None, scale=1.0):
        m = self.copy()
        v = m.verts
        if M3 is not None:
            v = v @ np.asarray(M3).T
            if m.normals is not None:
                m.normals = normalize(m.normals @ np.asarray(M3).T)
        m.verts = v * scale + (0 if offset is None else np.asarray(offset))
        return m

    def compact(self):
        used = sorted({i for f in self.faces for i in f})
        if len(used) == len(self.verts):
            return self
        remap = np.full(len(self.verts), -1)
        remap[used] = np.arange(len(used))
        self.verts = self.verts[used]
        if self.normals is not None:
            self.normals = self.normals[used]
        self.faces = [tuple(int(remap[i]) for i in f) for f in self.faces]
        return self

    def bounds(self):
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def split(self, max_tris):
        """Split into chunks of <= max_tris following face order (faces are generated
        spatially ordered, so chunks stay compact). Custom normals keep seams invisible."""
        if self.tri_count() <= max_tris:
            return [self]
        normals = self.normals if self.normals is not None else (
            vertex_normals(self.verts, self.faces) if self.sharp_angle is None else None)
        chunks, cur, cur_t = [], [], 0
        for fi, f in enumerate(self.faces):
            t = len(f) - 2
            if cur_t + t > max_tris:
                chunks.append(cur)
                cur, cur_t = [], 0
            cur.append(fi)
            cur_t += t
        if cur:
            chunks.append(cur)
        out = []
        for idxs in chunks:
            used = sorted({i for fi in idxs for i in self.faces[fi]})
            remap = {o: n for n, o in enumerate(used)}
            out.append(MeshData(self.verts[used],
                                [tuple(remap[i] for i in self.faces[fi]) for fi in idxs],
                                [self.uvs[fi] for fi in idxs],
                                None if normals is None else normals[used],
                                self.sharp_angle))
        return out


def merge(meshes):
    out = MeshData()
    for m in meshes:
        if m is not None and len(m.faces):
            out.add(m)
    if meshes:
        out.sharp_angle = meshes[0].sharp_angle
    return out


def triangulate(faces):
    tris = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def vertex_normals(verts, faces):
    tris = triangulate(faces)
    n = np.zeros_like(verts)
    if len(tris) == 0:
        return n
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return normalize(n)


# ---------------------------------------------------------------------------
# Generic builders
# ---------------------------------------------------------------------------

def box(center, axes, half, uv=(0.5, 0.5)):
    """Oriented box. axes: 3x3 rows = local x,y,z directions; half: half extents."""
    c = np.asarray(center, dtype=float)
    ax = np.asarray(axes, dtype=float)
    h = np.asarray(half, dtype=float)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(c + sx * h[0] * ax[0] + sy * h[1] * ax[1] + sz * h[2] * ax[2])
    v = np.array(corners)
    # index = sx*4 + sy*2 + sz with s in {0,1}
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    m = MeshData(v, faces, [tuple(uv for _ in f) for f in faces], sharp_angle=0.0)
    # ensure outward winding regardless of axis handedness
    if np.linalg.det(ax) < 0:
        m = m.flipped()
    return m


def ribbon(points, normals, widths, uv=(0.5, 0.5), closed=False, thickness=0.0):
    """Flat strip along a polyline lying on a surface (points already offset).
    Mitred joins keep 45-degree circuit jogs crisp."""
    P = np.asarray(points, dtype=float)
    N = normalize(np.asarray(normals, dtype=float))
    W = np.broadcast_to(np.asarray(widths, dtype=float), (len(P),))
    n = len(P)
    if n < 2:
        return MeshData()
    seg = normalize(np.diff(P, axis=0))
    side_seg = normalize(np.cross(N[:-1], seg))
    side = np.zeros_like(P)
    side[0], side[-1] = side_seg[0], side_seg[-1]
    if n > 2:
        avg = normalize(side_seg[:-1] + side_seg[1:])
        cosh = np.clip(np.sum(avg * side_seg[1:], axis=1), 0.5, 1.0)
        side[1:-1] = avg / cosh[:, None]
    left = P - side * (W[:, None] * 0.5)
    right = P + side * (W[:, None] * 0.5)
    verts = np.empty((2 * n, 3))
    verts[0::2], verts[1::2] = left, right
    faces, uvs = [], []
    for i in range(n - 1):
        a, b = 2 * i, 2 * i + 2
        faces.append((a, a + 1, b + 1, b))
        uvs.append((uv, uv, uv, uv))
    m = MeshData(verts, faces, uvs, normals=np.repeat(N, 2, axis=0))
    if thickness > 0:  # tiny skirt so strips read at grazing angles
        down = verts - np.repeat(N, 2, axis=0) * thickness
        base = len(verts)
        m.verts = np.vstack([verts, down])
        m.normals = np.vstack([m.normals, m.normals])
        for i in range(n - 1):
            a, b = 2 * i, 2 * i + 2
            m.faces.append((a, b, base + b, base + a))
            m.faces.append((a + 1, base + a + 1, base + b + 1, b + 1))
            m.uvs += [(uv,) * 4, (uv,) * 4]
    # make sure the visible side faces along N
    fn = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    if np.dot(fn, N[0]) < 0:
        m = m.flipped()
        m.normals = -m.normals
    return m


def polygon_fan(outline, normal, uv=(0.5, 0.5)):
    """Convex polygon as a fan (used for pads / glyph pieces already on a surface)."""
    P = np.asarray(outline, dtype=float)
    c = P.mean(axis=0)
    verts = np.vstack([c[None], P])
    n = len(P)
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    m = MeshData(verts, faces, [(uv,) * 3 for _ in faces], normals=np.repeat(np.asarray(normal)[None], n + 1, 0))
    fn = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    if np.dot(fn, normal) < 0:
        m = m.flipped()
        m.normals = -m.normals
    return m


def annulus(center, axis, r_in, r_out, segments, uv=(0.5, 0.5), arc=(0.0, 2 * np.pi), ref=None):
    axis = normalize(np.asarray(axis, dtype=float))
    ref = np.array([0, 0, 1.0]) if ref is None else np.asarray(ref, dtype=float)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([1.0, 0, 0])
    e1 = normalize(ref - np.dot(ref, axis) * axis)
    e2 = np.cross(axis, e1)
    closed = abs(arc[1] - arc[0] - 2 * np.pi) < 1e-6
    k = segments if closed else segments + 1
    a = np.linspace(arc[0], arc[1], segments, endpoint=not closed) if not closed else \
        np.linspace(arc[0], arc[1], segments, endpoint=False)
    dirs = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    verts = np.vstack([center + dirs * r_in, center + dirs * r_out])
    faces = []
    nseg = segments if closed else segments - 1
    for i in range(nseg):
        j = (i + 1) % k
        faces.append((i, j, k + j, k + i))
    m = MeshData(verts, faces, [(uv,) * 4 for _ in faces], normals=np.repeat(axis[None], len(verts), 0))
    return m


def rect_ring(center, axis, radius, width, depth, segments, arc=(0.0, 2 * np.pi), uv=(0.5, 0.5), ref=None):
    """Torus-like ring with rectangular section: reads from every angle, no double-siding needed."""
    axis = normalize(np.asarray(axis, dtype=float))
    ref = np.array([0, 0, 1.0]) if ref is None else np.asarray(ref, dtype=float)
    if abs(np.dot(ref, axis)) > 0.95:
        ref = np.array([1.0, 0, 0])
    e1 = normalize(ref - np.dot(ref, axis) * axis)
    e2 = np.cross(axis, e1)
    closed = abs(arc[1] - arc[0] - 2 * np.pi) < 1e-6
    n = segments if closed else segments + 1
    a = np.linspace(arc[0], arc[1], n, endpoint=not closed)
    dirs = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    prof = [(-depth / 2, -width / 2), (depth / 2, -width / 2), (depth / 2, width / 2), (-depth / 2, width / 2)]
    verts = []
    for d in dirs:
        for (dr, da) in prof:
            verts.append(center + d * (radius + dr) + axis * da)
    verts = np.array(verts)
    faces = []
    nseg = n if closed else n - 1
    for i in range(nseg):
        j = (i + 1) % n
        for q in range(4):
            q2 = (q + 1) % 4
            faces.append((i * 4 + q, i * 4 + q2, j * 4 + q2, j * 4 + q))
    if not closed:
        faces.append((0, 1, 2, 3)[::-1])
        last = (n - 1) * 4
        faces.append((last, last + 1, last + 2, last + 3))
    m = MeshData(verts, faces, [(uv,) * 4 for _ in faces], sharp_angle=np.radians(50))
    am = 0.5 * (a[:-1] + a[1:]) if not closed else a + np.pi / n
    mids = np.cos(am)[:, None] * e1 + np.sin(am)[:, None] * e2
    return fix_winding_by_centroid(m, center + mids * radius)


def fix_winding_by_centroid(m, inner_points):
    """Flip faces whose normal points toward the nearest inner reference point."""
    inner = np.asarray(inner_points, dtype=float)
    faces, uvs = [], []
    for f, u in zip(m.faces, m.uvs):
        v = m.verts[list(f)]
        fn = np.cross(v[1] - v[0], v[2] - v[0])
        fc = v.mean(axis=0)
        ref = inner[np.argmin(np.linalg.norm(inner - fc, axis=1))]
        if np.dot(fn, fc - ref) < 0:
            faces.append(tuple(reversed(f)))
            uvs.append(tuple(reversed(u)))
        else:
            faces.append(f)
            uvs.append(u)
    m.faces, m.uvs = faces, uvs
    return m


def uv_sphere(center, radius, axis, seg=24, rings=16, scale=(1, 1, 1), ref=None, uv_rect=(0, 0, 1, 1)):
    """Sphere whose pole points along `axis` (handy for eyes / glints)."""
    axis = normalize(np.asarray(axis, dtype=float))
    ref = np.array([0, 0, 1.0]) if ref is None else np.asarray(ref, dtype=float)
    if abs(np.dot(ref, axis)) > 0.95:
        ref = np.array([1.0, 0, 0])
    e1 = normalize(ref - np.dot(ref, axis) * axis)
    e2 = np.cross(axis, e1)
    verts, faces, uvs = [axis * radius * scale[2]], [], []
    u0, v0, u1, v1 = uv_rect
    for r in range(1, rings):
        phi = np.pi * r / rings
        for k in range(seg):
            th = 2 * np.pi * k / seg
            d = (np.sin(phi) * np.cos(th) * e1 * scale[0] + np.sin(phi) * np.sin(th) * e2 * scale[1]
                 + np.cos(phi) * axis * scale[2])
            verts.append(d * radius)
    verts.append(-axis * radius * scale[2])
    verts = np.array(verts) + np.asarray(center)
    last = len(verts) - 1

    def vid(r, k):
        return 1 + (r - 1) * seg + (k % seg)

    def uvc(r, k):
        return (u0 + (u1 - u0) * k / seg, v1 - (v1 - v0) * r / rings)
    for k in range(seg):
        faces.append((0, vid(1, k), vid(1, k + 1)))
        uvs.append(((u0 + (u1 - u0) * (k + .5) / seg, v1), uvc(1, k), uvc(1, k + 1)))
    for r in range(1, rings - 1):
        for k in range(seg):
            faces.append((vid(r, k), vid(r + 1, k), vid(r + 1, k + 1), vid(r, k + 1)))
            uvs.append((uvc(r, k), uvc(r + 1, k), uvc(r + 1, k + 1), uvc(r, k + 1)))
    for k in range(seg):
        faces.append((vid(rings - 1, k), last, vid(rings - 1, k + 1)))
        uvs.append((uvc(rings - 1, k), (u0 + (u1 - u0) * (k + .5) / seg, v0), uvc(rings - 1, k + 1)))
    m = MeshData(verts, faces, uvs)
    return fix_winding_by_centroid(m, [np.asarray(center)])


def star_plate(outline_pts, center, normal_fn, thickness_fn, rings=3, back=0.02, uv_fn=None):
    """Thick armour plate from a star-shaped outline already projected on a surface.

    outline_pts: (n,3) on the surface; center: (3,) on the surface.
    normal_fn(points)->normals; thickness_fn(f, k)->thickness (f radial 0..1, k outline index).
    Builds top shell, a slightly embedded bottom shell and a rim wall.
    """
    O = np.asarray(outline_pts, dtype=float)
    n = len(O)
    C = np.asarray(center, dtype=float)
    fr = np.linspace(0, 1, rings + 1)[1:]
    base = [C]
    fk = [(0.0, 0)]
    for f in fr:
        for k in range(n):
            base.append(C + (O[k] - C) * f)
            fk.append((f, k))
    base = np.array(base)
    N = normalize(normal_fn(base))
    thick = np.array([thickness_fn(f, k) for f, k in fk])
    top = base + N * thick[:, None]
    bot = base - N * back
    nv = len(base)
    verts = np.vstack([top, bot])
    faces, uvs = [], []

    def uvof(i):
        f, k = fk[i]
        return uv_fn(f, k / n) if uv_fn else (0.5, 0.5)

    def ring(r, k):
        return 1 + r * n + (k % n)
    for k in range(n):
        faces.append((0, ring(0, k), ring(0, k + 1)))
    for r in range(rings - 1):
        for k in range(n):
            faces.append((ring(r, k), ring(r + 1, k), ring(r + 1, k + 1), ring(r, k + 1)))
    top_faces = list(faces)
    for f in top_faces:  # bottom shell, reversed
        faces.append(tuple(reversed([i + nv for i in f])))
    for k in range(n):  # rim
        a, b = ring(rings - 1, k), ring(rings - 1, k + 1)
        faces.append((a, a + nv, b + nv, b))
    for f in faces:
        uvs.append(tuple(uvof(i % nv) for i in f))
    m = MeshData(verts, faces, uvs, sharp_angle=np.radians(55))
    # orientation: first top face normal should agree with surface normal
    v = verts[list(faces[0])]
    if np.dot(np.cross(v[1] - v[0], v[2] - v[0]), N[0]) < 0:
        m = m.flipped()
    return m


def lens_blade(outline_2d, center_2d, origin, ax_a, ax_b, ax_n, thickness, rings=4, uv_fn=None):
    """Flat blade (tail spade, fins) from a star-shaped 2D outline with lens thickness."""
    O = np.asarray(outline_2d, dtype=float)
    c = np.asarray(center_2d, dtype=float)
    n = len(O)
    fr = np.linspace(0, 1, rings + 1)[1:]
    pts2, ff = [c], [0.0]
    for f in fr:
        for k in range(n):
            pts2.append(c + (O[k] - c) * f)
            ff.append(f)
    pts2, ff = np.array(pts2), np.array(ff)
    t = thickness * np.sqrt(np.clip(1 - ff ** 2, 0, 1)) + 1e-4
    base = origin + pts2[:, :1] * ax_a + pts2[:, 1:2] * ax_b
    top = base + t[:, None] * ax_n
    bot = base - t[:, None] * ax_n
    nv = len(base)
    # rim vertices are shared (t~0) -> weld by reusing top ring for bottom outer ring
    verts = np.vstack([top, bot])
    faces = []

    def ring(r, k):
        return 1 + r * n + (k % n)
    for k in range(n):
        faces.append((0, ring(0, k), ring(0, k + 1)))
    for r in range(rings - 1):
        for k in range(n):
            faces.append((ring(r, k), ring(r + 1, k), ring(r + 1, k + 1), ring(r, k + 1)))
    topf = list(faces)
    outer = {ring(rings - 1, k) for k in range(n)}
    for f in topf:
        faces.append(tuple(reversed([i if i in outer else i + nv for i in f])))
    uvs = []
    for f in faces:
        uvs.append(tuple((uv_fn(ff[i % nv], pts2[i % nv]) if uv_fn else (0.5, 0.5)) for i in f))
    m = MeshData(verts, faces, uvs)
    v = verts[list(faces[0])]
    if np.dot(np.cross(v[1] - v[0], v[2] - v[0]), ax_n) < 0:
        m = m.flipped()
    return m
