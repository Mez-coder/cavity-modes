"""Common interface for azimuthally symmetric (m = 0) cavity eigenmodes.

Conventions
-----------
Every mode is stored as a pair of *real* spatial maps, normalised so that the
stored energy is exactly 1 J:

    E(r, z, t) = E_map(r, z) * cos(w t)
    B(r, z, t) = B_map(r, z) * sin(w t)

so that (eps0/2) * int |E_map|^2 dV = (1/(2 mu0)) * int |B_map|^2 dV = 1 J.

When several modes are combined (see :mod:`cavity_modes.tracking`) each one is
given a complex amplitude A and a drive frequency w_d, and contributes

    E = Re[ A E_map exp(i w_d t) ]
    B = Re[ -i A (w_n / w_d) B_map exp(i w_d t) ]

which reduces to the free eigenmode above when A = 1 and w_d = w_n.

Field components are returned in cylindrical form
(Er, Ez, Ephi, Br, Bz, Bphi); for m = 0 they do not depend on phi.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .constants import EPS0, MU0


@lru_cache(maxsize=64)
def _gl(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    x.flags.writeable = False
    w.flags.writeable = False
    return x, w


def gauss_legendre(n: int):
    """Cached Gauss-Legendre nodes/weights on [-1, 1]."""
    return _gl(int(n))


@dataclass
class Fields:
    Er: np.ndarray
    Ez: np.ndarray
    Ephi: np.ndarray
    Br: np.ndarray
    Bz: np.ndarray
    Bphi: np.ndarray

    def as_tuple(self):
        return (self.Er, self.Ez, self.Ephi, self.Br, self.Bz, self.Bphi)


@dataclass
class WallSamples:
    """Quadrature points on the cavity wall (in the r-z plane).

    ``weights`` already contain the 2*pi*r factor, so sum(weights * f) is a
    surface integral over the full (axisymmetric) wall.
    """

    r: np.ndarray
    z: np.ndarray
    nr: np.ndarray  # outward unit normal, r component
    nz: np.ndarray  # outward unit normal, z component
    weights: np.ndarray


class Mode(ABC):
    """An m = 0 cavity eigenmode normalised to 1 J stored energy."""

    m: int = 0
    kind: str  # "TM" or "TE"
    label: str
    omega: float  # rad/s
    z_min: float
    z_max: float
    r_max: float

    @property
    def frequency(self) -> float:
        return self.omega / (2 * np.pi)

    @abstractmethod
    def fields(self, r, z) -> Fields:
        """Peak field maps at (r, z); zero outside the cavity volume."""

    @abstractmethod
    def wall_samples(self) -> WallSamples:
        """Quadrature on the conducting wall for loss / surface-field integrals."""

    def on_axis_Ez(self, z):
        z = np.asarray(z, dtype=float)
        return self.fields(np.zeros_like(z), z).Ez

    # ------------------------------------------------------------------ checks
    def energies(self, n_r: int = 160, n_z: int = 160):
        """(electric, magnetic) peak energies in J by Gauss-Legendre quadrature.

        Only valid for cavities whose volume is the rectangle
        [0, r_max] x [z_min, z_max] in the r-z plane (true for the pillbox).
        Used as a self-consistency check: both numbers should be 1.
        """
        xr, wr = gauss_legendre(n_r)
        xz, wz = gauss_legendre(n_z)
        r = 0.5 * self.r_max * (xr + 1)
        wr = 0.5 * self.r_max * wr
        z = self.z_min + 0.5 * (self.z_max - self.z_min) * (xz + 1)
        wz = 0.5 * (self.z_max - self.z_min) * wz
        R, Zg = np.meshgrid(r, z, indexing="ij")
        W = np.outer(wr, wz) * 2 * np.pi * R
        f = self.fields(R, Zg)
        e2 = f.Er**2 + f.Ez**2 + f.Ephi**2
        b2 = f.Br**2 + f.Bz**2 + f.Bphi**2
        return 0.5 * EPS0 * np.sum(W * e2), 0.5 / MU0 * np.sum(W * b2)


def tangential_H2_and_E2(mode: Mode, ws: WallSamples):
    """|H_tangential|^2 and |E|^2 at the wall samples (peak values)."""
    f = mode.fields(ws.r, ws.z)
    Hr, Hz, Hphi = f.Br / MU0, f.Bz / MU0, f.Bphi / MU0
    Hn = Hr * ws.nr + Hz * ws.nz
    H2 = Hr**2 + Hz**2 + Hphi**2 - Hn**2
    E2 = f.Er**2 + f.Ez**2 + f.Ephi**2
    return H2, E2


class FieldMapMode(Mode):
    """A mode defined by sampled maps on a regular (r, z) grid.

    This is the representation a numerical (FEM) solver will hand to the
    tracker. Fields are bilinearly interpolated and are zero outside the grid
    or where ``inside`` is False.
    """

    def __init__(self, r, z, fields: Fields, omega, kind, label, wall: WallSamples,
                 inside=None):
        self.r = np.asarray(r, float)
        self.z = np.asarray(z, float)
        self.maps = fields
        self.omega = float(omega)
        self.kind = kind
        self.label = label
        self._wall = wall
        self.inside = np.ones((self.r.size, self.z.size), bool) if inside is None else inside
        self.z_min, self.z_max = self.z[0], self.z[-1]
        self.r_max = self.r[-1]

    @classmethod
    def sample(cls, mode: Mode, n_r: int = 121, n_z: int = 241) -> "FieldMapMode":
        r = np.linspace(0, mode.r_max, n_r)
        z = np.linspace(mode.z_min, mode.z_max, n_z)
        R, Zg = np.meshgrid(r, z, indexing="ij")
        return cls(r, z, mode.fields(R, Zg), mode.omega, mode.kind,
                   mode.label + " (map)", mode.wall_samples())

    def wall_samples(self) -> WallSamples:
        return self._wall

    def fields(self, r, z) -> Fields:
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        r, z = np.broadcast_arrays(r, z)
        dr = self.r[1] - self.r[0]
        dz = self.z[1] - self.z[0]
        fr = (r - self.r[0]) / dr
        fz = (z - self.z[0]) / dz
        ok = (fr >= 0) & (fr <= self.r.size - 1) & (fz >= -1e-9) & (fz <= self.z.size - 1 + 1e-9)
        i = np.clip(np.floor(fr).astype(int), 0, self.r.size - 2)
        j = np.clip(np.floor(fz).astype(int), 0, self.z.size - 2)
        tr = np.clip(fr - i, 0, 1)
        tz = np.clip(fz - j, 0, 1)
        w00 = (1 - tr) * (1 - tz)
        w10 = tr * (1 - tz)
        w01 = (1 - tr) * tz
        w11 = tr * tz
        ok = ok & self.inside[i, j]
        out = []
        for M in self.maps.as_tuple():
            v = w00 * M[i, j] + w10 * M[i + 1, j] + w01 * M[i, j + 1] + w11 * M[i + 1, j + 1]
            out.append(np.where(ok, v, 0.0))
        return Fields(*out)
