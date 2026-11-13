"""Macroparticle bunches and phase-space statistics.

Internal coordinates (per particle), with z the independent variable:
    x, y [m], t [s] (arrival time at the current z plane),
    px, py, pz  [dimensionless, momentum / (m c)].

Particle 0 is always the reference particle (on axis, nominal energy, t = 0
at entry). Phase-space output is relative to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import ELECTRON, Species


def gamma_from_kinetic(Ek_eV: float, species: Species) -> float:
    return 1.0 + Ek_eV / species.rest_energy_eV


@dataclass
class Bunch:
    species: Species
    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray
    t: np.ndarray
    pz: np.ndarray
    z: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.x.size

    def copy(self) -> "Bunch":
        return Bunch(self.species, self.x.copy(), self.px.copy(), self.y.copy(), self.py.copy(),
                     self.t.copy(), self.pz.copy(), self.z, dict(self.meta))

    @property
    def p(self):
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def gamma(self):
        return np.sqrt(1 + self.p**2)

    @property
    def kinetic_eV(self):
        return (self.gamma - 1) * self.species.rest_energy_eV

    @classmethod
    def gaussian(cls, n: int, Ek_eV: float, species: Species = ELECTRON, *,
                 emit_nx: float = 1e-6, beta_x: float = 1.0, alpha_x: float = 0.0,
                 emit_ny: float = 1e-6, beta_y: float = 1.0, alpha_y: float = 0.0,
                 sigma_t: float = 1e-12, sigma_delta: float = 1e-3, chirp: float = 0.0,
                 z: float = 0.0, seed: int | None = 1) -> "Bunch":
        """Gaussian bunch from Twiss parameters.

        emit_n*: rms normalised emittance [m rad]; beta [m]; sigma_t [s];
        sigma_delta: rms dp/p; chirp: d(delta)/dt [1/s] (linear correlation).
        """
        rng = np.random.default_rng(seed)
        g0 = gamma_from_kinetic(Ek_eV, species)
        p0 = np.sqrt(g0**2 - 1)
        ex, ey = emit_nx / p0, emit_ny / p0  # geometric

        def twiss(eps, beta, alpha, k):
            u1, u2 = rng.standard_normal((2, k))
            pos = np.sqrt(eps * beta) * u1
            ang = np.sqrt(eps / beta) * (u2 - alpha * u1)
            return pos, ang

        k = n - 1
        x, xp = twiss(ex, beta_x, alpha_x, k)
        y, yp = twiss(ey, beta_y, alpha_y, k)
        t = sigma_t * rng.standard_normal(k)
        delta = sigma_delta * rng.standard_normal(k) + chirp * t
        x, xp, y, yp, t, delta = (np.concatenate([[0.0], a]) for a in (x, xp, y, yp, t, delta))
        p = p0 * (1 + delta)
        pz = p / np.sqrt(1 + xp**2 + yp**2)
        return cls(species, x, xp * pz, y, yp * pz, t, pz, z,
                   meta=dict(Ek_eV=Ek_eV, p0=p0))


def phase_space(b: Bunch, f_ref: float | None = None) -> dict:
    """Coordinates relative to the reference particle (index 0), in lab units.

    x, y [mm]; xp, yp [mrad]; dt [ps]; dphi [deg at f_ref]; delta = dp/p_ref;
    dE [keV].
    """
    xp = b.px / b.pz
    yp = b.py / b.pz
    p = b.p
    dt = b.t - b.t[0]
    out = dict(
        x=1e3 * (b.x - b.x[0]), xp=1e3 * (xp - xp[0]),
        y=1e3 * (b.y - b.y[0]), yp=1e3 * (yp - yp[0]),
        dt=1e12 * dt, delta=(p - p[0]) / p[0],
        dE=1e-3 * (b.kinetic_eV - b.kinetic_eV[0]),
    )
    if f_ref:
        out["dphi"] = np.degrees(2 * np.pi * f_ref * dt)
    return out


def _twiss_2d(u, up):
    u = u - u.mean()
    up = up - up.mean()
    s11, s22, s12 = np.mean(u * u), np.mean(up * up), np.mean(u * up)
    eps = np.sqrt(max(s11 * s22 - s12**2, 0.0))
    if eps == 0:
        return eps, np.nan, np.nan
    return eps, s11 / eps, -s12 / eps


def statistics(b: Bunch) -> dict:
    """rms emittances (geometric and normalised), Twiss, centroids, energy.

    Excludes nothing: the reference particle is part of the ensemble but
    centroids are reported, so a net kick is visible separately from the
    emittance.
    """
    xp = b.px / b.pz
    yp = b.py / b.pz
    p = b.p
    bg = p.mean()
    ex, bx, ax = _twiss_2d(b.x, xp)
    ey, by, ay = _twiss_2d(b.y, yp)
    E_eV = b.kinetic_eV
    dt = b.t - b.t.mean()
    dE = E_eV - E_eV.mean()
    el = np.sqrt(max(np.mean(dt**2) * np.mean(dE**2) - np.mean(dt * dE) ** 2, 0.0))
    # x-t correlation (a "tilt" / crab signature): d<x>/dt
    tilt_x = np.mean((b.x - b.x.mean()) * dt) / np.mean(dt**2) if np.mean(dt**2) > 0 else 0.0
    return dict(
        z=b.z,
        Ek_mean_MeV=E_eV.mean() / 1e6,
        Ek_ref_MeV=E_eV[0] / 1e6,
        sigma_E_keV=E_eV.std() / 1e3,
        sigma_delta=(p.std() / p.mean()),
        sigma_t_ps=1e12 * b.t.std(),
        x_mean_mm=1e3 * b.x.mean(), xp_mean_mrad=1e3 * xp.mean(),
        y_mean_mm=1e3 * b.y.mean(), yp_mean_mrad=1e3 * yp.mean(),
        sigma_x_mm=1e3 * b.x.std(), sigma_y_mm=1e3 * b.y.std(),
        emit_x_m=ex, emit_y_m=ey,
        emit_nx_m=ex * bg, emit_ny_m=ey * bg,
        beta_x_m=bx, alpha_x=ax, beta_y_m=by, alpha_y=ay,
        emit_long_eV_s=el,
        emit_long_keV_ps=el * 1e-3 * 1e12,
        tilt_x_mm_per_ps=1e3 * tilt_x * 1e-12,
    )

