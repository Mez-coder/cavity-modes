import numpy as np
import pytest
from scipy.special import j1, jn_zeros

import cavity_modes as cm
from cavity_modes.constants import C, EPS0, MU0, SIGMA_COPPER

A, D = 0.0383, 0.0499  # ~3 GHz pillbox


def test_tm010_frequency():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    assert m.frequency == pytest.approx(jn_zeros(0, 1)[0] * C / (2 * np.pi * A), rel=1e-12)


@pytest.mark.parametrize("kind,n,p", [("TM", 1, 0), ("TM", 1, 1), ("TM", 2, 3), ("TE", 1, 1), ("TE", 2, 2)])
def test_energy_normalisation_and_balance(kind, n, p):
    """E and B energies both 1 J: checks normalisation AND field consistency."""
    m = cm.PillboxMode(kind, n, p, A, D)
    ue, ub = m.energies()
    assert ue == pytest.approx(1.0, rel=1e-9)
    assert ub == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("kind,n,p", [("TM", 1, 1), ("TE", 1, 2), ("TM", 2, 0)])
def test_boundary_conditions(kind, n, p):
    m = cm.PillboxMode(kind, n, p, A, D)
    ws = m.wall_samples()
    f = m.fields(ws.r, ws.z)
    # tangential E vanishes on the wall (tangent = (-nz, nr) in r-z, plus phi)
    Et = f.Er * (-ws.nz) + f.Ez * ws.nr
    scale = m.E0  # peak interior field scale
    assert np.max(np.abs(Et)) / scale < 1e-9
    assert np.max(np.abs(f.Ephi)) / scale < 1e-9
    # normal B vanishes
    Bn = f.Br * ws.nr + f.Bz * ws.nz
    assert np.max(np.abs(Bn)) < 1e-9 * (np.abs(f.Bphi).max() + np.abs(f.Bz).max() + 1e-30) + 1e-18


def test_tm010_Q_analytic():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    delta = np.sqrt(2 / (m.omega * MU0 * SIGMA_COPPER))
    Q = A * D / (delta * (A + D))
    assert cm.figures_of_merit(m, 1.0).Q0 == pytest.approx(Q, rel=1e-6)


def test_tm010_R_over_Q_analytic():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    x = m.omega * D / (2 * C)
    T = np.sin(x) / x
    U_per_E0sq = np.pi / 2 * EPS0 * A**2 * D * j1(jn_zeros(0, 1)[0]) ** 2
    RoQ = (D * T) ** 2 / (m.omega * U_per_E0sq)
    fom = cm.figures_of_merit(m, 1.0)
    assert fom.T == pytest.approx(T, rel=1e-9)
    assert fom.R_over_Q_ohm == pytest.approx(RoQ, rel=1e-9)
    # symmetric cell: crest phase = -(half transit angle)
    assert fom.phase_crest_deg == pytest.approx(-np.degrees(x), abs=1e-6)


def test_te_modes_do_not_accelerate():
    m = cm.PillboxMode("TE", 1, 1, A, D)
    fom = cm.figures_of_merit(m, 1.0)
    assert fom.V_eff_V == 0.0 and fom.R_over_Q_ohm == 0.0


def test_mode_list_sorted_and_complete():
    ms = cm.pillbox_modes(A, D, 8e9)
    labels = [m.label for m in ms]
    assert labels[0] == "TM010"
    assert all(a.omega <= b.omega for a, b in zip(ms, ms[1:]))
    assert "TE011" in labels and "TM020" in labels


def test_fieldmap_interpolation_close_to_analytic():
    m = cm.PillboxMode("TM", 1, 1, A, D)
    fm = cm.FieldMapMode.sample(m, 201, 401)
    rng = np.random.default_rng(0)
    r = rng.uniform(0, A, 500)
    z = rng.uniform(0, D, 500)
    a, b = m.fields(r, z), fm.fields(r, z)
    assert np.max(np.abs(a.Ez - b.Ez)) / np.abs(a.Ez).max() < 1e-3
    assert np.max(np.abs(a.Bphi - b.Bphi)) / np.abs(a.Bphi).max() < 1e-3
