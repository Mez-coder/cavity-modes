"""Steady-state response of the cavity to a source at an arbitrary frequency.

Yes, it is a linear superposition of the eigenmode field maps — but with
*complex, frequency-dependent* weights, and everything oscillates at the
drive frequency w (not at each w_n).

Expanding the driven field in the (orthogonal, 1 J normalised) eigenmodes,
E = sum_n a_n E_n, and projecting the driven wave equation

    curl curl E - (w/c)^2 E = -i w mu0 J

onto E_n gives, with wall losses added as a damping term,

    a_n(w) = -i w * int(J . E_n dV) / (2 J) / (w_n^2 - w^2 + i w w_n / Q_n)

(the 2 J is eps0 * int |E_n|^2 dV for a 1 J mode). So each mode is a driven
damped oscillator: its weight is the overlap of the source with the mode
pattern, times a Lorentzian of width w_n / Q_n. A copper cavity has
Q ~ 1e4, so at any one drive frequency essentially only the nearest mode(s)
are excited, and each mode's phase swings by 180 deg across its resonance.

The magnetic part of each mode is scaled by w_n / w (Faraday's law at the
drive frequency), see :mod:`cavity_modes.modes`.

Caveat: the solenoidal eigenmodes are complete away from the source; very
close to a coupler there is also a quasi-static (irrotational) field that the
mode sum does not capture. For beam dynamics on axis this is negligible.

Sources are modelled as axisymmetric current rings (the m = 0 part of any
real coupler): a ring at (r0, z0) carrying current moment I*l [A m] in
direction "r", "z" or "phi". A z- or r-ring couples to TM modes, a phi-ring
to TE modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .figures import figures_of_merit
from .modes import Mode


@dataclass
class RingSource:
    r0: float
    z0: float
    direction: str = "z"  # "r", "z" or "phi"
    moment: complex = 1.0  # A*m (total current x length element)

    def overlap(self, mode: Mode) -> complex:
        f = mode.fields(np.array([self.r0]), np.array([self.z0]))
        comp = {"r": f.Er, "z": f.Ez, "phi": f.Ephi}[self.direction][0]
        return self.moment * comp


@dataclass
class ModeResponse:
    mode: Mode
    amplitude: complex  # a_n: multiply the 1 J field maps by this
    omega_drive: float

    @property
    def stored_energy(self) -> float:
        """Time-averaged stored energy of this mode's share of the field (J)."""
        k = self.mode.omega / self.omega_drive
        return abs(self.amplitude) ** 2 * 0.5 * (1 + k**2)


def driven_response(modes: list[Mode], omega: float, sources: list[RingSource],
                    Q: dict | None = None, beta_for_Q: float = 1.0) -> list[ModeResponse]:
    """Mode amplitudes for a steady drive at angular frequency ``omega``.

    ``Q`` maps mode labels to loaded Q; if omitted the copper Q0 is used.
    """
    out = []
    for m in modes:
        q = (Q or {}).get(m.label) or figures_of_merit(m, beta_for_Q).Q0
        drive = sum(s.overlap(m) for s in sources)
        denom = m.omega**2 - omega**2 + 1j * omega * m.omega / q
        a = -1j * omega * drive / 2.0 / denom
        out.append(ModeResponse(m, complex(a), omega))
    return out


def scale_to_energy(responses: list[ModeResponse], U_total: float) -> list[ModeResponse]:
    """Rescale all amplitudes so the total time-averaged stored energy is U_total."""
    U = sum(r.stored_energy for r in responses)
    if U == 0:
        return responses
    s = np.sqrt(U_total / U)
    return [ModeResponse(r.mode, r.amplitude * s, r.omega_drive) for r in responses]


def response_spectrum(modes: list[Mode], freqs, sources: list[RingSource], Q=None):
    """Stored energy per mode (rows) vs drive frequency (columns), for plotting."""
    qs = {m.label: (Q or {}).get(m.label) or figures_of_merit(m, 1.0).Q0 for m in modes}
    out = np.zeros((len(modes), len(freqs)))
    for j, f in enumerate(freqs):
        for i, r in enumerate(driven_response(modes, 2 * np.pi * f, sources, qs)):
            out[i, j] = r.stored_energy
    return out
