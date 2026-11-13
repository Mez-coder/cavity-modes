"""Figures of merit for an m = 0 mode, all quoted at 1 J stored energy.

Definitions (linac convention, as in SUPERFISH):

* On-axis complex voltage seen by a particle crossing at constant velocity
  beta*c, entering at z_min at time t = 0 with the field Re[E_map e^{i(wt+phi)}]:

      V_c(beta) = int Ez(0, z) exp(i w (z - z_min) / (beta c)) dz

  The energy gain is  dW(phi) = q |V_c| cos(phi + arg V_c).
  So the *phase dependence is entirely in the cosine*: |V_c| is the crest
  value, and phi_crest = -arg V_c. Shifting the z origin to the "electrical
  centre" makes V_c real, which is where the SUPERFISH form
  dW = q V0 T cos(phi) (with the sine integral S = 0) comes from.

* V0 = int |Ez(0, z)| dz                 (axial voltage, no transit-time factor)
* T  = |V_c| / V0                        (transit-time factor, <= 1)
* E0 = V0 / L, E_acc = |V_c| / L         (L = cell length)
* P  = (Rs / 2) * oint |H_tan|^2 dS      (wall loss, Rs = sqrt(w mu0 / 2 sigma))
* Q0 = w U / P
* R_sh = |V_c|^2 / P                     (effective shunt impedance, includes T^2)
* R/Q  = |V_c|^2 / (w U)                 (geometry only; independent of sigma)
* Z T^2 = (E0 T)^2 / (P / L)             (effective shunt impedance per metre)

The "circuit" convention divides R_sh and R/Q by 2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import C, MU0, SIGMA_COPPER
from .modes import Mode, tangential_H2_and_E2, gauss_legendre


def complex_voltage(mode: Mode, beta: float, n: int = 2000) -> complex:
    """V_c(beta) for 1 J, reference particle enters at z_min at t = 0."""
    x, w = gauss_legendre(n)
    L = mode.z_max - mode.z_min
    z = mode.z_min + 0.5 * L * (x + 1)
    wz = 0.5 * L * w
    Ez = mode.on_axis_Ez(z)
    return np.sum(wz * Ez * np.exp(1j * mode.omega * (z - mode.z_min) / (beta * C)))


def axial_voltage(mode: Mode, n: int = 2000) -> float:
    x, w = gauss_legendre(n)
    L = mode.z_max - mode.z_min
    z = mode.z_min + 0.5 * L * (x + 1)
    return float(np.sum(0.5 * L * w * np.abs(mode.on_axis_Ez(z))))


def surface_resistance(omega: float, sigma: float = SIGMA_COPPER) -> float:
    return np.sqrt(omega * MU0 / (2 * sigma))


@dataclass
class FigureOfMerit:
    label: str
    kind: str
    frequency_Hz: float
    beta: float
    length_m: float
    Q0: float
    wall_loss_W: float  # at 1 J
    V0_V: float  # at 1 J
    V_eff_V: float  # |V_c| at 1 J
    T: float
    phase_crest_deg: float  # phi that puts the reference particle on crest
    R_over_Q_ohm: float
    R_shunt_ohm: float
    ZTT_ohm_per_m: float
    Ez_axis_peak_V_per_m: float
    E_surf_peak_V_per_m: float
    H_surf_peak_A_per_m: float
    Epk_over_Eacc: float
    Bpk_over_Eacc_mT_per_MV_m: float

    def as_dict(self):
        return asdict(self)


def figures_of_merit(mode: Mode, beta: float, sigma: float = SIGMA_COPPER,
                     length: float | None = None) -> FigureOfMerit:
    if length is None:
        length = getattr(mode, "cell_length", None)
    L = (mode.z_max - mode.z_min) if length is None else length
    ws = mode.wall_samples()
    H2, E2 = tangential_H2_and_E2(mode, ws)
    Rs = surface_resistance(mode.omega, sigma)
    P = 0.5 * Rs * np.sum(ws.weights * H2)
    U = 1.0
    Q0 = mode.omega * U / P
    Vc = complex_voltage(mode, beta)
    Veff = abs(Vc)
    V0 = axial_voltage(mode)
    T = Veff / V0 if V0 > 0 else 0.0
    zax = np.linspace(mode.z_min, mode.z_max, 2001)
    Ez_pk = float(np.max(np.abs(mode.on_axis_Ez(zax))))
    E_pk = float(np.sqrt(E2.max()))
    H_pk = float(np.sqrt(H2.max()))
    Eacc = Veff / L
    return FigureOfMerit(
        label=mode.label,
        kind=mode.kind,
        frequency_Hz=mode.frequency,
        beta=beta,
        length_m=L,
        Q0=Q0,
        wall_loss_W=P,
        V0_V=V0,
        V_eff_V=Veff,
        T=T,
        phase_crest_deg=float(np.degrees(-np.angle(Vc))) if Veff > 0 else 0.0,
        R_over_Q_ohm=Veff**2 / (mode.omega * U),
        R_shunt_ohm=Veff**2 / P,
        ZTT_ohm_per_m=(V0 * T / L) ** 2 / (P / L),
        Ez_axis_peak_V_per_m=Ez_pk,
        E_surf_peak_V_per_m=E_pk,
        H_surf_peak_A_per_m=H_pk,
        Epk_over_Eacc=E_pk / Eacc if Eacc > 0 else np.inf,
        Bpk_over_Eacc_mT_per_MV_m=(MU0 * H_pk * 1e3) / (Eacc / 1e6) if Eacc > 0 else np.inf,
    )


def transit_time_curve(mode: Mode, betas) -> np.ndarray:
    """T(beta) — useful to see which particle velocity a cell is tuned for."""
    V0 = axial_voltage(mode)
    return np.array([abs(complex_voltage(mode, b)) / V0 if V0 > 0 else 0.0 for b in betas])
