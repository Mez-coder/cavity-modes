"""Axisymmetric (m = 0) finite-element eigen-solver with quadratic triangles.

For m = 0 the modes split into TM (Er, Ez, Hphi) and TE (Ephi, Hr, Hz) and each
family is governed by one scalar, u = Hphi (TM) or u = Ephi (TE), satisfying

    curl curl (u phi_hat) = k^2 u phi_hat.

Weak form (dV = 2 pi r dr dz; the 2 pi cancels):

    int [ du/dz dw/dz + (du/dr + u/r)(dw/dr + w/r) ] r dr dz
        = k^2 int u w r dr dz

Same operator for both families; only the boundary conditions differ:

    boundary          TM (u = Hphi)          TE (u = Ephi)
    axis              u = 0                  u = 0
    PEC (metal/elec)  natural                u = 0
    PMC (magnetic)    u = 0                  natural

There are no spurious modes in this scalar m = 0 formulation. Fields are then
recovered from Maxwell's equations (see :class:`FEMMode`).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from .constants import C, EPS0, MU0
from .geometry import Profile, TriMesh, mesh_profile
from .modes import Fields, Mode, WallSamples

# Dunavant degree-5, 7-point rule (barycentric coords, weights sum to 1)
_a, _b = 0.059715871789770, 0.470142064105115
_c, _d = 0.797426985353087, 0.101286507323456
QUAD_L = np.array([[1 / 3, 1 / 3, 1 / 3],
                   [_a, _b, _b], [_b, _a, _b], [_b, _b, _a],
                   [_c, _d, _d], [_d, _c, _d], [_d, _d, _c]])
QUAD_W = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)
_EDGES = ((0, 1), (1, 2), (2, 0))  # local P2 nodes 3, 4, 5


def p2_basis(L):
    """P2 shape functions at barycentric points L (..., 3) -> (..., 6)."""
    L0, L1, L2 = L[..., 0], L[..., 1], L[..., 2]
    return np.stack([L0 * (2 * L0 - 1), L1 * (2 * L1 - 1), L2 * (2 * L2 - 1),
                     4 * L0 * L1, 4 * L1 * L2, 4 * L2 * L0], axis=-1)


def p2_grad(L, gL):
    """Gradients of P2 shape functions.

    L: (..., 3) barycentric; gL: (..., 3, 2) gradients of L wrt (r, z).
    Returns (..., 6, 2).
    """
    out = []
    for i in range(3):
        out.append((4 * L[..., i, None] - 1) * gL[..., i, :])
    for i, j in _EDGES:
        out.append(4 * (L[..., j, None] * gL[..., i, :] + L[..., i, None] * gL[..., j, :]))
    return np.stack(out, axis=-2)


class P2Mesh:
    """Quadratic-triangle mesh (straight-sided) built from a linear TriMesh."""

    def __init__(self, tm: TriMesh):
        self.tm = tm
        r, z, tris = tm.r, tm.z, np.asarray(tm.tris)
        nv = r.size
        # unique edges -> midpoint nodes
        e_all = np.vstack([tris[:, [i, j]] for i, j in _EDGES])
        e_sorted = np.sort(e_all, axis=1)
        uniq, inv = np.unique(e_sorted, axis=0, return_inverse=True)
        inv = inv.ravel()
        M = tris.shape[0]
        mid = nv + inv.reshape(3, M).T  # (M, 3)
        self.elems = np.hstack([tris, mid])  # (M, 6)
        self.r = np.concatenate([r, 0.5 * (r[uniq[:, 0]] + r[uniq[:, 1]])])
        self.z = np.concatenate([z, 0.5 * (z[uniq[:, 0]] + z[uniq[:, 1]])])
        self.n_nodes = self.r.size
        self.n_elems = M
        edge_id = {tuple(e): k for k, e in enumerate(uniq)}
        # boundary edges: (v0, v1, midpoint node, tag, element, local edge)
        owner = {}
        for k, (i, j) in enumerate(_EDGES):
            for e, key in enumerate(map(tuple, np.sort(tris[:, [i, j]], axis=1))):
                owner[key] = (e, k)
        self.bnd = []
        for (n0, n1), tag in zip(tm.bedges, tm.btags):
            key = (min(n0, n1), max(n0, n1))
            e, k = owner[key]
            self.bnd.append((int(n0), int(n1), nv + edge_id[key], tag, e, k))
        # affine maps
        P = np.stack([r[tris], z[tris]], axis=-1)  # (M, 3, 2)
        J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=-1)  # (M, 2, 2) cols
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if np.any(np.abs(det) < 1e-300):
            raise ValueError("degenerate triangle in mesh")
        self.area = 0.5 * np.abs(det)
        Jinv = np.linalg.inv(J)  # rows: grad(L1), grad(L2)
        g1, g2 = Jinv[:, 0, :], Jinv[:, 1, :]
        self.gradL = np.stack([-(g1 + g2), g1, g2], axis=1)  # (M, 3, 2)
        self.Jinv = Jinv
        self.P0 = P[:, 0]
        self.vr = r[tris]
        self.vz = z[tris]
        self._tree = cKDTree(np.c_[r[tris].mean(1), z[tris].mean(1)])
        self.h_char = float(np.sqrt(np.median(self.area) * 2))
        be = np.asarray(tm.bedges)
        self._bseg = np.stack([np.c_[r[be[:, 0]], z[be[:, 0]]], np.c_[r[be[:, 1]], z[be[:, 1]]]], 1)

    def inside(self, r, z, chunk=4096):
        """Even-odd point-in-polygon test against the mesh boundary."""
        r = np.asarray(r, float).ravel()
        z = np.asarray(z, float).ravel()
        a, b = self._bseg[:, 0], self._bseg[:, 1]  # (E, 2) as (r, z)
        out = np.zeros(r.size, bool)
        for i in range(0, r.size, chunk):
            rr, zz = r[i:i + chunk, None], z[i:i + chunk, None]
            # ray from the point towards +r; count crossings
            straddle = (a[None, :, 1] > zz) != (b[None, :, 1] > zz)
            with np.errstate(divide="ignore", invalid="ignore"):
                rx = a[None, :, 0] + (zz - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (
                    b[None, :, 1] - a[None, :, 1])
            out[i:i + chunk] = np.count_nonzero(straddle & (rx > rr), axis=1) % 2 == 1
        return out

    # ---------------------------------------------------------------- assembly
    def assemble(self):
        """Stiffness K and mass M (without the 2 pi)."""
        Ke = np.zeros((self.n_elems, 6, 6))
        Me = np.zeros((self.n_elems, 6, 6))
        for Lq, wq in zip(QUAD_L, QUAD_W):
            N = p2_basis(Lq)  # (6,)
            G = p2_grad(np.broadcast_to(Lq, (self.n_elems, 3)), self.gradL)  # (M,6,2)
            rq = self.vr @ Lq  # (M,)
            w = wq * self.area * rq
            Dr = G[:, :, 0] + N[None, :] / rq[:, None]  # du/dr + u/r
            Dz = G[:, :, 1]
            Ke += w[:, None, None] * (Dr[:, :, None] * Dr[:, None, :] + Dz[:, :, None] * Dz[:, None, :])
            Me += w[:, None, None] * (N[:, None] * N[None, :])[None]
        rows = np.repeat(self.elems, 6, axis=1).ravel()
        cols = np.tile(self.elems, (1, 6)).ravel()
        n = self.n_nodes
        K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
        M = sp.csr_matrix((Me.ravel(), (rows, cols)), shape=(n, n))
        return K, M

    def dirichlet_nodes(self, kinds) -> np.ndarray:
        s = set()
        for n0, n1, nm, tag, _, _ in self.bnd:
            if tag in kinds:
                s.update((n0, n1, nm))
        return np.array(sorted(s), dtype=int)

    # ---------------------------------------------------------------- location
    def locate(self, r, z, tol=1e-9):
        """Element index and barycentric coords for points; elem = -1 if outside."""
        r = np.asarray(r, float).ravel()
        z = np.asarray(z, float).ravel()
        elem = np.full(r.size, -1, dtype=int)
        L = np.zeros((r.size, 3))
        todo = np.arange(r.size)
        for k in (8, 32, 128):
            if k == 32 and todo.size:
                # points that missed the nearest elements: drop those clearly outside
                todo = todo[self.inside(r[todo], z[todo]) | self._near_boundary(r[todo], z[todo])]
            if todo.size == 0:
                break
            k = min(k, self.n_elems)
            _, cand = self._tree.query(np.c_[r[todo], z[todo]], k=k)
            cand = cand.reshape(todo.size, k)
            d = np.stack([r[todo, None] - self.P0[cand, 0], z[todo, None] - self.P0[cand, 1]], -1)
            l12 = np.einsum("nkij,nkj->nki", self.Jinv[cand], d)
            l0 = 1 - l12[..., 0] - l12[..., 1]
            lam = np.concatenate([l0[..., None], l12], -1)  # (n, k, 3)
            ok = lam.min(-1) >= -tol
            hit = ok.any(1)
            first = ok.argmax(1)
            idx = todo[hit]
            elem[idx] = cand[hit, first[hit]]
            L[idx] = np.clip(lam[hit, first[hit]], 0, 1)
            todo = todo[~hit]
        return elem, L

    def _near_boundary(self, r, z):
        """Points within a small distance of the boundary (where the parity test is fragile)."""
        if not hasattr(self, "_btree"):
            pts = self._bseg.reshape(-1, 2)
            self._btree = cKDTree(pts)
            self._bmaxlen = float(np.max(np.hypot(*(self._bseg[:, 1] - self._bseg[:, 0]).T)))
        d, _ = self._btree.query(np.c_[r, z], k=1)
        return d < 0.6 * self._bmaxlen

    def eval(self, coef, elem, L):
        """u, du/dr, du/dz at located points (elem >= 0)."""
        U = coef[self.elems[elem]]  # (n, 6)
        N = p2_basis(L)
        G = p2_grad(L, self.gradL[elem])  # (n, 6, 2)
        u = np.sum(N * U, -1)
        g = np.einsum("na,nad->nd", U, G)
        return u, g[:, 0], g[:, 1]


class FEMMode(Mode):
    """An m = 0 eigenmode from the FEM solver, normalised to 1 J.

    TM (u = B_phi map):  E_r = (c^2/w) du/dz,  E_z = -(c^2/w)(du/dr + u/r)
    TE (u = E_phi map):  B_r = (1/w) du/dz,    B_z = -(1/w)(du/dr + u/r)

    which follows from E = E_map cos wt, B = B_map sin wt and Maxwell's curl
    equations (same convention as the analytic pillbox modes).
    """

    def __init__(self, mesh: P2Mesh, coef, k2, kind, label, profile: Profile):
        self.mesh = mesh
        self.kind = kind
        self.label = label
        self.profile = profile
        self.omega = C * np.sqrt(k2)
        self.z_min, self.z_max = float(mesh.z.min()), float(mesh.z.max())
        self.r_max = float(mesh.r.max())
        self.cell_length = profile.cell_length
        K, M = mesh._KM
        # 1 J normalisation: TM via magnetic energy, TE via electric energy
        mass = 2 * np.pi * coef @ (M @ coef)
        # TM: (1/2mu0) int u^2 dV = 1 J;  TE: (eps0/2) int u^2 dV = 1 J
        scale = np.sqrt(2 * MU0 / mass) if kind == "TM" else np.sqrt(2 / (EPS0 * mass))
        self.coef = coef * scale
        self._wall = None
        self._axis = None

    def on_axis_Ez(self, z):
        """On-axis Ez, from a fine cached sampling (P2 gives a piecewise-linear Ez)."""
        if self.kind == "TE":
            return np.zeros_like(np.asarray(z, float))
        if self._axis is None:
            n = int(np.clip((self.z_max - self.z_min) / (0.25 * self.mesh.h_char), 2000, 40000))
            zs = np.linspace(self.z_min, self.z_max, n)
            self._axis = (zs, self.fields(np.zeros_like(zs), zs).Ez)
        zs, ez = self._axis
        z = np.asarray(z, float)
        return np.interp(z, zs, ez, left=0.0, right=0.0)

    def _fields_at(self, elem, L, r):
        u, ur, uz = self.mesh.eval(self.coef, elem, L)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_over_r = np.where(r > 1e-12, u / np.where(r > 1e-12, r, 1.0), ur)
        div = ur + u_over_r  # (1/r) d(r u)/dr
        zero = np.zeros_like(u)
        w = self.omega
        if self.kind == "TM":
            return (C**2 / w * uz, -C**2 / w * div, zero, zero, zero, u)
        return (zero, zero, u, uz / w, -div / w, zero)

    def fields(self, r, z) -> Fields:
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        r, z = np.broadcast_arrays(r, z)
        shape = r.shape
        elem, L = self.mesh.locate(r, z)
        ok = elem >= 0
        out = [np.zeros(r.size) for _ in range(6)]
        if ok.any():
            vals = self._fields_at(elem[ok], L[ok], r.ravel()[ok])
            for o, v in zip(out, vals):
                o[ok] = v
        return Fields(*(o.reshape(shape) for o in out))

    def wall_samples(self) -> WallSamples:
        if self._wall is not None:
            return self._wall
        m = self.mesh
        g = np.array([0.5 - np.sqrt(0.15), 0.5, 0.5 + np.sqrt(0.15)])
        gw = np.array([5, 8, 5]) / 18
        R, Z, NR, NZ, W = [], [], [], [], []
        for n0, n1, _, tag, e, _k in m.bnd:
            if tag != "metal":
                continue
            r0, z0, r1, z1 = m.r[n0], m.z[n0], m.r[n1], m.z[n1]
            length = np.hypot(r1 - r0, z1 - z0)
            nr, nz = (z1 - z0) / length, -(r1 - r0) / length
            # orient outward: away from element centroid
            cr, cz = m.vr[e].mean(), m.vz[e].mean()
            if (0.5 * (r0 + r1) - cr) * nr + (0.5 * (z0 + z1) - cz) * nz < 0:
                nr, nz = -nr, -nz
            rs = r0 + g * (r1 - r0)
            zs = z0 + g * (z1 - z0)
            R.append(rs); Z.append(zs)
            NR.append(np.full(3, nr)); NZ.append(np.full(3, nz))
            W.append(2 * np.pi * rs * gw * length)
        cat = np.concatenate
        self._wall = WallSamples(cat(R), cat(Z), cat(NR), cat(NZ), cat(W))
        return self._wall

    def energies(self):
        """(electric, magnetic) peak energies [J] by element quadrature."""
        m = self.mesh
        UE = UB = 0.0
        for Lq, wq in zip(QUAD_L, QUAD_W):
            L = np.broadcast_to(Lq, (m.n_elems, 3))
            elem = np.arange(m.n_elems)
            rq = m.vr @ Lq
            Er, Ez, Ep, Br, Bz, Bp = self._fields_at(elem, L, rq)
            w = 2 * np.pi * wq * m.area * rq
            UE += 0.5 * EPS0 * np.sum(w * (Er**2 + Ez**2 + Ep**2))
            UB += 0.5 / MU0 * np.sum(w * (Br**2 + Bz**2 + Bp**2))
        return UE, UB


def solve_modes(profile: Profile, f_max: float, kinds=("TM", "TE"), mesh: TriMesh | None = None,
                h: float | None = None, n_max: int = 200, **mesh_kw) -> list[FEMMode]:
    """All m = 0 modes of ``profile`` below ``f_max`` [Hz], sorted by frequency."""
    tm = mesh if mesh is not None else mesh_profile(profile, h=h, **mesh_kw)
    pm = P2Mesh(tm)
    K, M = pm.assemble()
    pm._KM = (K, M)
    k2_max = (2 * np.pi * f_max / C) ** 2
    out = []
    for kind in kinds:
        dir_kinds = {"axis", "magnetic"} if kind == "TM" else {"axis", "metal", "electric"}
        fixed = pm.dirichlet_nodes(dir_kinds)
        free = np.setdiff1d(np.arange(pm.n_nodes), fixed)
        Kf = K[free][:, free].tocsc()
        Mf = M[free][:, free].tocsc()
        nev = 8
        while True:
            nev = min(nev, free.size - 2)
            vals, vecs = eigsh(Kf, k=nev, M=Mf, sigma=0.0, which="LM")
            order = np.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
            if vals[-1] > k2_max or nev >= min(n_max, free.size - 2):
                break
            nev *= 2
        count = 0
        for lam, v in zip(vals, vecs.T):
            if lam > k2_max or lam <= 0:
                continue
            count += 1
            coef = np.zeros(pm.n_nodes)
            coef[free] = v
            # sign convention: make the largest |u| positive
            if coef[np.argmax(np.abs(coef))] < 0:
                coef = -coef
            out.append(FEMMode(pm, coef, lam, kind, f"{kind}{count}", profile))
    return sorted(out, key=lambda m: m.omega)
