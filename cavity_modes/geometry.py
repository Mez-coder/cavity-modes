"""Axisymmetric cell geometry (a wall profile in the z-r half plane) and meshing.

A :class:`Profile` is the metal wall as a polyline of (z, r) points running
from upstream to downstream. The computational domain is bounded by:

* the axis r = 0                        -> tag "axis"
* the metal wall                        -> tag "metal"   (PEC, has losses)
* a vertical "port" plane at each end where the wall does not reach the axis
  (e.g. at the end of a beam pipe)      -> tag "in" / "out", with a boundary
  condition chosen per end:
      "electric"  PEC symmetry plane, lossless (tangential E = 0)
      "magnetic"  PMC symmetry plane, lossless (tangential H = 0)
      "metal"     a real lossy wall

In a multi-cell structure, electric planes at both cell ends give the 0-mode,
and electric at one end with magnetic at the other gives the pi/2-mode, etc.

Two meshers produce a plain linear :class:`TriMesh`:

* :func:`mesh_mapped` – a structured mesh for walls that are single-valued,
  r = R(z) (elliptical cells, smooth bumps, a plain pillbox). Pure numpy.
* :func:`mesh_gmsh`  – an unstructured gmsh mesh for any simple polygon
  (nose cones, beam-pipe steps, re-entrant shapes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

PORT_BCS = ("electric", "magnetic", "metal")


# --------------------------------------------------------------------- splines
def spline_points(ctrl, n: int = 200) -> np.ndarray:
    """Interpolating cubic spline through control points [(z, r), ...].

    Parametrised by cumulative chord length so uneven spacing behaves.
    Returns an (n, 2) array of (z, r) including both end points.
    """
    ctrl = np.asarray(ctrl, float)
    if len(ctrl) < 3:
        return np.linspace(ctrl[0], ctrl[-1], n)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ctrl, axis=0).T))])
    cs = CubicSpline(s, ctrl, axis=0, bc_type="natural")
    return cs(np.linspace(0, s[-1], n))


def join_segments(segments) -> np.ndarray:
    """Join [("line", pts) | ("spline", pts) | ("spline", pts, n)] into one polyline."""
    out = []
    for seg in segments:
        kind, pts = seg[0], np.asarray(seg[1], float)
        if kind == "line":
            p = pts
        elif kind == "spline":
            p = spline_points(pts, seg[2] if len(seg) > 2 else 200)
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
        if out and np.allclose(out[-1][-1], p[0]):
            p = p[1:]
        out.append(p)
    return np.vstack(out)


# --------------------------------------------------------------------- profile
@dataclass
class Profile:
    wall: np.ndarray  # (n, 2) [z, r], upstream -> downstream
    bc_in: str = "electric"
    bc_out: str = "electric"
    cell_length: float | None = None  # used for E_acc, ZT^2; default: full z extent
    name: str = "profile"

    def __post_init__(self):
        self.wall = np.asarray(self.wall, float)
        if self.wall.ndim != 2 or self.wall.shape[1] != 2 or len(self.wall) < 2:
            raise ValueError("wall must be an (n, 2) array of (z, r)")
        if np.any(self.wall[:, 1] < -1e-15):
            raise ValueError("wall has r < 0")
        if not self.wall[0, 0] < self.wall[-1, 0]:
            raise ValueError("wall must run upstream -> downstream (z increasing end to end)")
        for bc in (self.bc_in, self.bc_out):
            if bc not in PORT_BCS:
                raise ValueError(f"port bc must be one of {PORT_BCS}, got {bc!r}")

    @property
    def z_min(self):
        return float(self.wall[0, 0])

    @property
    def z_max(self):
        return float(self.wall[-1, 0])

    @property
    def r_max(self):
        return float(self.wall[:, 1].max())

    # -------- convenient constructors
    @classmethod
    def pillbox(cls, radius, length, pipe_radius=0.0, pipe_length=0.0,
                bc_in="electric", bc_out="electric") -> "Profile":
        a, d = radius, length
        if pipe_radius <= 0:
            wall = [(0, 0), (0, a), (d, a), (d, 0)]
        else:
            rp, Lp = pipe_radius, pipe_length
            wall = [(-Lp, rp), (0, rp), (0, a), (d, a), (d, rp), (d + Lp, rp)]
        return cls(np.array(wall, float), bc_in, bc_out, cell_length=d, name="pillbox")

    @classmethod
    def from_segments(cls, segments, **kw) -> "Profile":
        return cls(join_segments(segments), **kw)

    # -------- boundary polygon
    def polygon(self):
        """CCW polygon in the (z, r) plane and a tag for each edge.

        Returns (verts (n, 2) [z, r], tags list[str]); edge i joins verts[i]
        to verts[(i + 1) % n].
        """
        w = self.wall
        z0, z1 = w[0, 0], w[-1, 0]
        verts = [(z0, 0.0), (z1, 0.0)]
        tags = ["axis"]
        rev = w[::-1]
        if w[-1, 1] > 0:
            tags.append("out")  # (z1, 0) -> (z1, r_last)
            body = rev
        else:
            body = rev[1:]
        for p in body:
            verts.append(tuple(p))
        tags += ["metal"] * (len(body) - 1)
        if w[0, 1] > 0:
            tags.append("in")  # wall[0] -> (z0, 0)
        else:
            verts.pop()  # wall[0] coincides with (z0, 0)
            tags.append("metal")
        verts = np.array(verts, float)
        assert len(tags) == len(verts)
        return verts, tags

    def edge_bc(self, tag: str) -> str:
        """Physical boundary kind for an edge tag: axis/metal/electric/magnetic."""
        if tag == "in":
            return self.bc_in
        if tag == "out":
            return self.bc_out
        return tag


# --------------------------------------------------------------------- meshes
@dataclass
class TriMesh:
    """Linear triangle mesh in the (r, z) half plane with tagged boundary edges."""

    r: np.ndarray
    z: np.ndarray
    tris: np.ndarray  # (M, 3) int
    bedges: np.ndarray  # (B, 2) int
    btags: list = field(default_factory=list)  # physical kind per boundary edge

    @property
    def n_nodes(self):
        return self.r.size


def _single_valued_core(p: Profile):
    """Split the wall into (left_tag, core polyline, right_tag) for the mapped
    mesher, or raise if the wall is not single-valued in z."""
    w = p.wall
    left, right = p.bc_in, p.bc_out
    i0, i1 = 0, len(w)
    if w[0, 1] == 0:
        if not np.isclose(w[1, 0], w[0, 0]):
            raise ValueError("mapped mesher: wall starting on axis must rise vertically")
        left, i0 = "metal", 1
    if w[-1, 1] == 0:
        if not np.isclose(w[-2, 0], w[-1, 0]):
            raise ValueError("mapped mesher: wall ending on axis must drop vertically")
        right, i1 = "metal", len(w) - 1
    core = w[i0:i1]
    if np.any(np.diff(core[:, 0]) <= 0):
        raise ValueError("mapped mesher needs r = R(z) single-valued; use mesh_gmsh")
    if np.any(core[:, 1] <= 0):
        raise ValueError("mapped mesher needs R(z) > 0")
    return left, core, right


def is_single_valued(p: Profile) -> bool:
    try:
        _single_valued_core(p)
        return True
    except ValueError:
        return False


def mesh_mapped(p: Profile, nz: int = 80, nr: int = 60, grade: float = 1.0) -> TriMesh:
    """Structured mesh: column i at z_i spans 0 <= r <= R(z_i).

    ``grade`` > 1 clusters radial nodes towards the axis (r ~ eta**grade).
    """
    left, core, right = _single_valued_core(p)
    zs = np.linspace(core[0, 0], core[-1, 0], nz + 1)
    R = np.interp(zs, core[:, 0], core[:, 1])
    eta = np.linspace(0, 1, nr + 1) ** grade
    Zg, Eg = np.meshgrid(zs, eta, indexing="ij")
    rr = (Eg * R[:, None]).ravel()
    zz = Zg.ravel()
    idx = np.arange((nz + 1) * (nr + 1)).reshape(nz + 1, nr + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    tris = np.vstack([np.c_[a, b, c], np.c_[a, c, d]])
    be, bt = [], []

    def add(nodes, tag):
        for n0, n1 in zip(nodes[:-1], nodes[1:]):
            be.append((n0, n1))
            bt.append(tag)

    add(idx[:, 0], "axis")
    add(idx[:, -1], "metal")
    add(idx[0, :], left)
    add(idx[-1, :], right)
    return TriMesh(rr, zz, tris, np.array(be, int), bt)


def mesh_gmsh(p: Profile, h: float, h_axis: float | None = None,
              h_wall: float | None = None) -> TriMesh:
    """Unstructured triangular mesh of the profile polygon with gmsh.

    h: target element size [m]; h_axis / h_wall: optional finer sizes near the
    axis (better on-axis Ez) and along the metal wall (better losses).
    """
    import gmsh

    verts, tags = p.polygon()
    started = False
    if not gmsh.isInitialized():
        try:
            gmsh.initialize(interruptible=False)
        except TypeError:  # older gmsh
            gmsh.initialize()
        started = True
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cell")
        geo = gmsh.model.geo
        pts = [geo.addPoint(z, r, 0.0, h) for z, r in verts]
        n = len(pts)
        lines = [geo.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        loop = geo.addCurveLoop(lines)
        surf = geo.addPlaneSurface([loop])
        geo.synchronize()

        fields = []
        mf = gmsh.model.mesh.field
        for size, want in ((h_axis, "axis"), (h_wall, "metal")):
            if size is None:
                continue
            curves = [l for l, t in zip(lines, tags) if t == want]
            fd = mf.add("Distance")
            mf.setNumbers(fd, "CurvesList", curves)
            mf.setNumber(fd, "Sampling", 200)
            ft = mf.add("Threshold")
            mf.setNumber(ft, "InField", fd)
            mf.setNumber(ft, "SizeMin", size)
            mf.setNumber(ft, "SizeMax", h)
            mf.setNumber(ft, "DistMin", 2 * size)
            mf.setNumber(ft, "DistMax", 2 * size + 4 * h)
            fields.append(ft)
        if fields:
            fm = mf.add("Min")
            mf.setNumbers(fm, "FieldsList", fields)
            mf.setAsBackgroundMesh(fm)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)
        lookup = np.full(node_tags.max() + 1, -1, dtype=np.int64)
        lookup[node_tags] = np.arange(node_tags.size)

        etypes, _, enodes = gmsh.model.mesh.getElements(2, surf)
        tris = None
        for et, en in zip(etypes, enodes):
            if et == 2:  # 3-node triangle
                tris = lookup[np.asarray(en, dtype=np.int64)].reshape(-1, 3)
        if tris is None:
            raise RuntimeError("gmsh produced no linear triangles")

        be, bt = [], []
        for line, tag in zip(lines, tags):
            etypes, _, enodes = gmsh.model.mesh.getElements(1, line)
            for et, en in zip(etypes, enodes):
                if et == 1:  # 2-node line
                    e = lookup[np.asarray(en, dtype=np.int64)].reshape(-1, 2)
                    be.append(e)
                    bt += [p.edge_bc(tag)] * len(e)
        gmsh.model.remove()
    finally:
        if started:
            gmsh.finalize()

    # drop unused nodes (geometry points are always used, but be safe)
    used = np.unique(tris)
    remap = np.full(coords.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    tris = remap[tris]
    bedges = remap[np.vstack(be)]
    r = coords[used, 1].copy()
    z = coords[used, 0].copy()
    r[np.abs(r) < 1e-14] = 0.0
    return TriMesh(r, z, tris, bedges, bt)


def mesh_profile(p: Profile, h: float | None = None, **kw) -> TriMesh:
    """Mapped mesh when the wall is single-valued, otherwise gmsh."""
    if is_single_valued(p) and not kw.pop("force_gmsh", False):
        if h is not None:
            L = p.z_max - p.z_min
            kw.setdefault("nz", max(8, int(np.ceil(L / h))))
            kw.setdefault("nr", max(6, int(np.ceil(p.r_max / h))))
        return mesh_mapped(p, **{k: v for k, v in kw.items() if k in ("nz", "nr", "grade")})
    if h is None:
        h = min(p.z_max - p.z_min, p.r_max) / 40
    return mesh_gmsh(p, h, **{k: v for k, v in kw.items() if k in ("h_axis", "h_wall")})
