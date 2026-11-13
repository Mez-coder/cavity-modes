"""Full-Lorentz-force tracking through superposed cavity modes.

The longitudinal position z is the independent variable, so every particle
crosses the same z plane at each step and the field maps are evaluated at a
single z per step. Equations of motion (p in units of m c):

    dx/dz = px/pz,   dy/dz = py/pz,   dt/dz = gamma / (pz c)
    dp/dz = q/(m c) * (E + v x B) * dt/dz

integrated with fixed-step RK4. No paraxial or rigid-beam approximation is
made: particles may change velocity and radius inside the cell, which matters
for low-beta beams. Space charge is not included.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .beam import Bunch
from .constants import C
from .figures import complex_voltage
from .modes import Mode


@dataclass
class Component:
    """One mode, driven with complex amplitude ``amplitude`` at ``omega``.

    |amplitude|^2 is the stored energy in J for a resonant (omega = w_n) drive.
    """

    mode: Mode
    amplitude: complex
    omega: float

    def cartesian_fields(self, x, y, z, t):
        r = np.hypot(x, y)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(r > 0, x / r, 1.0)
            s = np.where(r > 0, y / r, 0.0)
        f = self.mode.fields(r, np.full_like(r, z))
        ph = self.amplitude * np.exp(1j * self.omega * t)
        e = ph.real
        b = (self.mode.omega / self.omega) * ph.imag
        Ex = (f.Er * c - f.Ephi * s) * e
        Ey = (f.Er * s + f.Ephi * c) * e
        Ez = f.Ez * e
        Bx = (f.Br * c - f.Bphi * s) * b
        By = (f.Br * s + f.Bphi * c) * b
        Bz = f.Bz * b
        return Ex, Ey, Ez, Bx, By, Bz


def make_component(mode: Mode, *, phase_deg: float = 0.0, energy_J: float | None = None,
                   voltage_V: float | None = None, beta_ref: float = 1.0,
                   z_entry: float | None = None, omega: float | None = None,
                   charge_sign: int = +1) -> Component:
    """Build a component with phase measured from the crest.

    phase_deg = 0 puts the reference particle (constant velocity beta_ref,
    at t = 0 at z_entry) on crest; positive phase means the field is ahead,
    i.e. the reference particle arrives *later* than crest.
    Set either the stored energy (J) or the effective voltage |V_c| (V).
    ``charge_sign`` = -1 for electrons so that "crest" means maximum gain.
    """
    omega = mode.omega if omega is None else omega
    z_entry = mode.z_min if z_entry is None else z_entry
    Vc = complex_voltage(mode, beta_ref) * np.exp(1j * omega * (mode.z_min - z_entry) / (beta_ref * C))
    psi = np.angle(Vc) if abs(Vc) > 0 else 0.0
    if charge_sign < 0:
        psi += np.pi
    if voltage_V is not None:
        if abs(Vc) == 0:
            raise ValueError(f"{mode.label} has no on-axis voltage; set energy_J instead")
        mag = voltage_V / abs(Vc)
    else:
        mag = np.sqrt(1.0 if energy_J is None else energy_J)
    return Component(mode, mag * np.exp(1j * (np.radians(phase_deg) - psi)), omega)


def _deriv(state, z, comps, qmc):
    x, y, t, px, py, pz = state
    g = np.sqrt(1 + px * px + py * py + pz * pz)
    dtdz = g / (pz * C)
    vx, vy, vz = C * px / g, C * py / g, C * pz / g
    Ex = np.zeros_like(x); Ey = np.zeros_like(x); Ez = np.zeros_like(x)
    Bx = np.zeros_like(x); By = np.zeros_like(x); Bz = np.zeros_like(x)
    for c in comps:
        ex, ey, ez, bx, by, bz = c.cartesian_fields(x, y, z, t)
        Ex += ex; Ey += ey; Ez += ez; Bx += bx; By += by; Bz += bz
    Fx = Ex + vy * Bz - vz * By
    Fy = Ey + vz * Bx - vx * Bz
    Fz = Ez + vx * By - vy * Bx
    k = qmc * dtdz
    return np.array([px / pz, py / pz, dtdz, k * Fx, k * Fy, k * Fz])


def track(bunch: Bunch, comps: list[Component], z_end: float, n_steps: int = 400,
          record_every: int = 0):
    """Track from bunch.z to z_end. Returns (new bunch, history list).

    history: if record_every > 0, a list of (z, x, y, t, px, py, pz) snapshots.
    """
    b = bunch.copy()
    qmc = b.species.charge / (b.species.mass * C)
    s = np.array([b.x, b.y, b.t, b.px, b.py, b.pz])
    z = b.z
    h = (z_end - z) / n_steps
    hist = []
    for i in range(n_steps):
        k1 = _deriv(s, z, comps, qmc)
        k2 = _deriv(s + 0.5 * h * k1, z + 0.5 * h, comps, qmc)
        k3 = _deriv(s + 0.5 * h * k2, z + 0.5 * h, comps, qmc)
        k4 = _deriv(s + h * k3, z + h, comps, qmc)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        z += h
        if np.any(s[5] <= 0):
            raise RuntimeError("particle reflected (pz <= 0): field too strong for this energy")
        if record_every and (i % record_every == 0 or i == n_steps - 1):
            hist.append((z, *s.copy()))
    b.x, b.y, b.t, b.px, b.py, b.pz = s
    b.z = z
    return b, hist


def refine_crest(comp: Component, bunch: Bunch, z_end: float, phase_deg: float = 0.0,
                 n_steps: int = 400) -> Component:
    """Find the crest by actually tracking the reference particle (accounts for
    its velocity change inside the cell), then return ``comp`` re-phased to
    ``phase_deg`` from that crest. ``comp`` should be the analytic on-crest
    component from :func:`make_component` (phase 0)."""
    ref = bunch.copy()
    for a in ("x", "px", "y", "py", "t", "pz"):
        setattr(ref, a, getattr(ref, a)[:1].copy())
    mag = abs(comp.amplitude)
    phi0 = np.angle(comp.amplitude)

    def neg_gain(dphi):
        c = Component(comp.mode, mag * np.exp(1j * (phi0 + dphi)), comp.omega)
        out, _ = track(ref, [c], z_end, n_steps)
        return -(out.kinetic_eV[0])

    res = minimize_scalar(neg_gain, bounds=(-np.pi / 2, np.pi / 2), method="bounded",
                          options=dict(xatol=1e-7))
    return Component(comp.mode, mag * np.exp(1j * (phi0 + res.x + np.radians(phase_deg))),
                     comp.omega)
