"""Physical constants and particle species."""
from dataclasses import dataclass

from scipy import constants as _c

C = _c.c
EPS0 = _c.epsilon_0
MU0 = _c.mu_0
Z0 = MU0 * C
QE = _c.e

SIGMA_COPPER = 5.8e7  # S/m, annealed copper at room temperature


@dataclass(frozen=True)
class Species:
    name: str
    charge: float  # C (signed)
    mass: float  # kg

    @property
    def rest_energy_eV(self) -> float:
        return self.mass * C**2 / QE


ELECTRON = Species("electron", -QE, _c.m_e)
POSITRON = Species("positron", QE, _c.m_e)
PROTON = Species("proton", QE, _c.m_p)

SPECIES = {s.name: s for s in (ELECTRON, POSITRON, PROTON)}
