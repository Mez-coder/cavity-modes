import numpy as np
import pytest

import cavity_modes as cm
from cavity_modes.beam import Bunch
from cavity_modes.constants import C, ELECTRON, PROTON

A, D = 0.0383, 0.0499


def single(species, Ek, x=0.0, t=0.0):
    b = Bunch.gaussian(1, Ek, species)
    b.x[:] = x
    b.t[:] = t
    return b


@pytest.mark.parametrize("phase", [0.0, 30.0, -60.0, 90.0, 180.0])
def test_energy_gain_matches_VTcos_relativistic(phase):
    """10 GeV electron: tracked gain == |q| V0 T cos(phi) (constant-velocity limit)."""
    m = cm.PillboxMode("TM", 1, 0, A, D)
    b = single(ELECTRON, 10e9)
    beta = np.sqrt(1 - 1 / b.gamma[0] ** 2)
    comp = cm.make_component(m, phase_deg=phase, energy_J=0.5, beta_ref=beta, charge_sign=-1)
    out, _ = cm.track(b, [comp], m.z_max, 400)
    gain = out.kinetic_eV[0] - b.kinetic_eV[0]
    V = abs(comp.amplitude) * cm.figures_of_merit(m, beta).V_eff_V
    assert gain == pytest.approx(V * np.cos(np.radians(phase)), abs=1e-6 * V)


def test_fieldmap_tracking_agrees_with_analytic():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    fm = cm.FieldMapMode.sample(m, 161, 321)
    b = Bunch.gaussian(200, 5e6, ELECTRON, emit_nx=2e-6, beta_x=0.3, sigma_t=2e-12)
    ca = cm.make_component(m, energy_J=0.2, beta_ref=0.996, charge_sign=-1)
    cf = cm.Component(fm, ca.amplitude, ca.omega)
    oa, _ = cm.track(b, [ca], D, 300)
    of, _ = cm.track(b, [cf], D, 300)
    assert np.max(np.abs(oa.kinetic_eV - of.kinetic_eV)) / np.ptp(oa.kinetic_eV - b.kinetic_eV) < 2e-3
    assert np.max(np.abs(oa.px - of.px)) / np.max(np.abs(oa.px)) < 2e-3


def test_panofsky_wenzel():
    """d(dp_perp)/d(tau) = -v grad_perp(dp_z) for rigid, constant-velocity particles."""
    m = cm.PillboxMode("TM", 1, 0, A, D)
    Ek = 50e9
    b0 = single(ELECTRON, Ek)
    v = C * np.sqrt(1 - 1 / b0.gamma[0] ** 2)
    comp = cm.make_component(m, phase_deg=35.0, energy_J=0.3, beta_ref=1.0, charge_sign=-1)
    x0, dx, tau, dtau = 4e-3, 1e-4, 0.0, 2e-12

    def kick(x, t):
        b = single(ELECTRON, Ek, x=x, t=t)
        o, _ = cm.track(b, [comp], D, 600)
        return o.px[0] - b.px[0], o.pz[0] - b.pz[0]

    dpx_dtau = (kick(x0, tau + dtau)[0] - kick(x0, tau - dtau)[0]) / (2 * dtau)
    dpz_dx = (kick(x0 + dx, tau)[1] - kick(x0 - dx, tau)[1]) / (2 * dx)
    assert dpx_dtau == pytest.approx(-v * dpz_dx, rel=2e-3)


def test_emittance_preserved_for_short_bunch_linear_optics():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    b = Bunch.gaussian(4000, 20e6, ELECTRON, emit_nx=1e-6, beta_x=0.5, alpha_x=0.0,
                       emit_ny=1e-6, beta_y=0.5, sigma_t=1e-15, sigma_delta=1e-6)
    comp = cm.make_component(m, phase_deg=20, energy_J=1.0, beta_ref=1.0, charge_sign=-1)
    out, _ = cm.track(b, [comp], D, 400)
    s0, s1 = cm.statistics(b), cm.statistics(out)
    assert s1["emit_nx_m"] == pytest.approx(s0["emit_nx_m"], rel=1e-3)
    assert s1["emit_ny_m"] == pytest.approx(s0["emit_ny_m"], rel=1e-3)


def test_tracked_crest_low_beta_proton():
    """At low beta the real crest differs from the constant-velocity estimate;
    refine_crest must do at least as well."""
    m = cm.PillboxMode("TM", 1, 0, A, D)
    b = single(PROTON, 5e6)
    beta = np.sqrt(1 - 1 / b.gamma[0] ** 2)
    c0 = cm.make_component(m, energy_J=4.0, beta_ref=beta)
    c1 = cm.refine_crest(c0, b, D)
    g0 = cm.track(b, [c0], D)[0].kinetic_eV[0]
    g1 = cm.track(b, [c1], D)[0].kinetic_eV[0]
    assert g1 >= g0 - 1e-6
