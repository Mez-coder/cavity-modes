"""Analytic m = 0 modes of a closed cylindrical pillbox (radius a, length d).

TM_0np (n >= 1, p >= 0), kc = j_{0,n}/a, kz = p*pi/d
    Ez   =  E0 J0(kc r) cos(kz z)            * cos wt
    Er   =  E0 (kz/kc) J1(kc r) sin(kz z)    * cos wt
    Bphi = -E0 (w/(c^2 kc)) J1(kc r) cos(kz z) * sin wt

TE_0np (n >= 1, p >= 1), kc = j_{1,n}/a (zeros of J0' = -J1)
    Ephi =  E0 J1(kc r) sin(kz z)            * cos wt
    Br   =  E0 (kz/w) J1(kc r) cos(kz z)     * sin wt
    Bz   = -E0 (kc/w) J0(kc r) sin(kz z)     * sin wt

w^2/c^2 = kc^2 + kz^2. The cavity occupies 0 <= r <= a, z0 <= z <= z0 + d.
E0 is fixed so the stored energy is 1 J.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

from .constants import C, EPS0
from .modes import Fields, Mode, WallSamples, gauss_legendre

_TOL = 1e-12


class PillboxMode(Mode):
    def __init__(self, kind: str, n: int, p: int, radius: float, length: float, z0: float = 0.0):
        kind = kind.upper()
        if kind not in ("TM", "TE"):
            raise ValueError("kind must be 'TM' or 'TE'")
        if n < 1 or p < 0 or (kind == "TE" and p < 1):
            raise ValueError(f"invalid indices for {kind}: n={n}, p={p}")
        self.kind, self.n, self.p = kind, n, p
        self.a, self.d = float(radius), float(length)
        self.z_min, self.z_max, self.r_max = z0, z0 + self.d, self.a
        self.label = f"{kind}0{n}{p}"
        x = jn_zeros(0 if kind == "TM" else 1, n)[-1]
        self.x = x
        self.kc = x / self.a
        self.kz = p * np.pi / self.d
        self.omega = C * np.hypot(self.kc, self.kz)
        self.E0 = 1.0 / np.sqrt(self._energy_per_E0sq())

    def _energy_per_E0sq(self) -> float:
        a, d = self.a, self.d
        if self.kind == "TM":
            Ic = d if self.p == 0 else d / 2
            Is = 0.0 if self.p == 0 else d / 2
            radial = a**2 / 2 * j1(self.x) ** 2
            return 0.5 * EPS0 * 2 * np.pi * radial * (Ic + (self.kz / self.kc) ** 2 * Is)
        radial = a**2 / 2 * j0(self.x) ** 2
        return 0.5 * EPS0 * 2 * np.pi * radial * d / 2

    def fields(self, r, z) -> Fields:
        r = np.asarray(r, float)
        z = np.asarray(z, float)
        r, z = np.broadcast_arrays(r, z)
        inside = (r <= self.a * (1 + _TOL)) & (z >= self.z_min - _TOL * self.d) & (
            z <= self.z_max + _TOL * self.d)
        zz = z - self.z_min
        kr = self.kc * r
        J0, J1 = j0(kr), j1(kr)
        c, s = np.cos(self.kz * zz), np.sin(self.kz * zz)
        zero = np.zeros_like(r)
        E0, w = self.E0, self.omega
        if self.kind == "TM":
            Ez = E0 * J0 * c
            Er = E0 * (self.kz / self.kc) * J1 * s
            Bphi = -E0 * w / (C**2 * self.kc) * J1 * c
            out = (Er, Ez, zero, zero, zero, Bphi)
        else:
            Ephi = E0 * J1 * s
            Br = E0 * (self.kz / w) * J1 * c
            Bz = -E0 * (self.kc / w) * J0 * s
            out = (zero, zero, Ephi, Br, Bz, zero)
        return Fields(*(np.where(inside, f, 0.0) for f in out))

    def wall_samples(self, n: int = 256) -> WallSamples:
        x, w = gauss_legendre(n)
        zs = self.z_min + 0.5 * self.d * (x + 1)
        wz = 0.5 * self.d * w
        rs = 0.5 * self.a * (x + 1)
        wr = 0.5 * self.a * w
        one, zer = np.ones(n), np.zeros(n)
        r = np.concatenate([np.full(n, self.a), rs, rs])
        z = np.concatenate([zs, np.full(n, self.z_min), np.full(n, self.z_max)])
        nr = np.concatenate([one, zer, zer])
        nz = np.concatenate([zer, -one, one])
        weights = np.concatenate([2 * np.pi * self.a * wz, 2 * np.pi * rs * wr, 2 * np.pi * rs * wr])
        return WallSamples(r, z, nr, nz, weights)


def pillbox_modes(radius: float, length: float, f_max: float, z0: float = 0.0,
                  kinds=("TM", "TE")) -> list[PillboxMode]:
    """All m = 0 pillbox modes below ``f_max`` (Hz), sorted by frequency."""
    k_max = 2 * np.pi * f_max / C
    out = []
    for kind in kinds:
        order = 0 if kind == "TM" else 1
        n = 1
        while True:
            kc = jn_zeros(order, n)[-1] / radius
            if kc > k_max:
                break
            p = 0 if kind == "TM" else 1
            while np.hypot(kc, p * np.pi / length) <= k_max:
                out.append(PillboxMode(kind, n, p, radius, length, z0))
                p += 1
            n += 1
    return sorted(out, key=lambda m: m.omega)
