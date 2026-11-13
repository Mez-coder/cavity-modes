import numpy as np
import pytest

import cavity_modes as cm
from cavity_modes.constants import EPS0, MU0
from cavity_modes.fem import solve_modes
from cavity_modes.geometry import Profile, mesh_mapped

A, D = 0.0383, 0.0499


@pytest.fixture(scope="module")
def pillbox_pair():
    p = Profile.pillbox(A, D)
    fem = solve_modes(p, 7.7e9, mesh=mesh_mapped(p, nz=60, nr=45))
    ana = cm.pillbox_modes(A, D, 7.7e9)
    return fem, ana


def test_same_mode_spectrum(pillbox_pair):
    fem, ana = pillbox_pair
    assert [m.kind for m in fem] == [m.kind for m in ana]
    for m, a in zip(fem, ana):
        assert m.frequency == pytest.approx(a.frequency, rel=5e-6), a.label


def test_energy_normalisation(pillbox_pair):
    fem, _ = pillbox_pair
    for m in fem:
        ue, ub = m.energies()
        assert ue == pytest.approx(1.0, rel=1e-6)
        assert ub == pytest.approx(1.0, rel=1e-6)


def test_figures_of_merit(pillbox_pair):
    fem, ana = pillbox_pair
    for m, a in zip(fem, ana):
        f, g = cm.figures_of_merit(m, 0.99), cm.figures_of_merit(a, 0.99)
        assert f.Q0 == pytest.approx(g.Q0, rel=3e-3 if m.kind == "TE" else 1e-5), a.label
        if a.kind == "TM":
            assert f.R_over_Q_ohm == pytest.approx(g.R_over_Q_ohm, rel=3e-3), a.label
            assert f.T == pytest.approx(g.T, rel=2e-3), a.label
            assert f.H_surf_peak_A_per_m == pytest.approx(g.H_surf_peak_A_per_m, rel=1e-3)


def test_on_axis_field_converges_second_order():
    p = Profile.pillbox(A, D)
    ref = cm.figures_of_merit(cm.PillboxMode("TM", 1, 1, A, D), 1.0).R_over_Q_ohm
    errs = []
    for nz, nr in ((20, 15), (40, 30)):
        m = solve_modes(p, 4.5e9, kinds=("TM",), mesh=mesh_mapped(p, nz=nz, nr=nr))[1]
        errs.append(abs(cm.figures_of_merit(m, 1.0).R_over_Q_ohm / ref - 1))
    assert errs[0] / errs[1] > 3.0  # ~4 for O(h^2)


@pytest.mark.parametrize("bc,expected", [("electric", ["TM010", "TM012"]),
                                         ("magnetic", ["TM011"])])
def test_half_cell_symmetry_planes(bc, expected):
    """Half a pillbox with an electric (magnetic) mid-plane gives even (odd) p."""
    half = Profile(np.array([(0, 0), (0, A), (D / 2, A)]), bc_out=bc)
    fem = solve_modes(half, 6.8e9, kinds=("TM",), mesh=mesh_mapped(half, nz=40, nr=45))
    full = {m.label: m for m in cm.pillbox_modes(A, D, 6.8e9, kinds=("TM",))}
    assert len(fem) == len(expected)
    for m, lab in zip(fem, expected):
        assert m.frequency == pytest.approx(full[lab].frequency, rel=1e-5)
        # half the losses and half the energy of the full cell: same Q
        q_full = cm.figures_of_merit(full[lab], 1.0).Q0
        assert cm.figures_of_merit(m, 1.0).Q0 == pytest.approx(q_full, rel=1e-4)


@pytest.mark.parametrize("which", [0, 1])
def test_slater_perturbation(which):
    """Push a smooth bump into the side wall; compare with Slater's theorem
    Δω/ω = ∫_ΔV (μ0 H² − ε0 E²) dV / ∫_V (μ0 H² + ε0 E²) dV (peak fields)."""
    eps = 0.15e-3
    z = np.linspace(0, D, 400)
    bump = A - eps * np.sin(np.pi * z / D) ** 2
    wall = np.vstack([[(0, 0)], np.c_[z, bump], [(D, 0)]])
    p = Profile(wall)
    pert = solve_modes(p, 4.5e9, kinds=("TM",), mesh=mesh_mapped(p, nz=100, nr=60))[which]
    a = cm.PillboxMode("TM", 1, which, A, D)

    # integrate the unperturbed energy densities over the removed sliver
    xg, wg = np.polynomial.legendre.leggauss(40)
    zq = 0.5 * D * (xg + 1)
    wz = 0.5 * D * wg
    num = 0.0
    for zi, wzi in zip(zq, wz):
        r0 = A - eps * np.sin(np.pi * zi / D) ** 2
        rq = r0 + 0.5 * (A - r0) * (xg + 1)
        wr = 0.5 * (A - r0) * wg
        f = a.fields(rq, np.full_like(rq, zi))
        dens = (f.Bphi**2 / MU0) - EPS0 * (f.Er**2 + f.Ez**2)
        num += wzi * np.sum(wr * 2 * np.pi * rq * dens)
    predicted = num / 4.0  # denominator = 4 J for 1 J peak-normalised maps
    measured = pert.omega / a.omega - 1
    if which == 0:
        assert measured > 0  # TM010 has E = 0 at the side wall: H-dominated, f rises
    assert measured == pytest.approx(predicted, rel=0.03), (measured, predicted)


def test_tracking_through_fem_matches_analytic():
    p = Profile.pillbox(A, D)
    fem = solve_modes(p, 3.5e9, kinds=("TM",), mesh=mesh_mapped(p, nz=60, nr=45, grade=1.5))[0]
    ana = cm.PillboxMode("TM", 1, 0, A, D)
    fmap = cm.FieldMapMode.sample(fem, 161, 321)
    b = cm.Bunch.gaussian(300, 5e6, cm.ELECTRON, emit_nx=2e-6, beta_x=0.3, sigma_t=2e-12)
    ca = cm.make_component(ana, energy_J=0.3, beta_ref=0.996, charge_sign=-1)
    cf = cm.make_component(fmap, energy_J=0.3, beta_ref=0.996, charge_sign=-1)
    oa, _ = cm.track(b, [ca], D, 300)
    of, _ = cm.track(b, [cf], D, 300)
    gain_a = oa.kinetic_eV - b.kinetic_eV
    gain_f = of.kinetic_eV - b.kinetic_eV
    assert np.max(np.abs(gain_a - gain_f)) / np.max(np.abs(gain_a)) < 2e-3
    assert np.max(np.abs(oa.px - of.px)) / np.max(np.abs(oa.px)) < 5e-3


def test_nonmapped_profile_rejected_by_mapped_mesher():
    p = Profile.pillbox(A, D, pipe_radius=0.01, pipe_length=0.02)
    with pytest.raises(ValueError):
        mesh_mapped(p)
