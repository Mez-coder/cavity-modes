import numpy as np
import pytest

import cavity_modes as cm

A, D = 0.0383, 0.0499


def test_lorentzian_half_power_width():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    Q = cm.figures_of_merit(m, 1.0).Q0
    src = [cm.RingSource(0.5 * A, 0.0, "z", 1.0)]
    U0 = cm.driven_response([m], m.omega, src)[0].stored_energy
    for s in (+1, -1):
        w = m.omega * (1 + s / (2 * Q))
        U = cm.driven_response([m], w, src)[0].stored_energy
        assert U / U0 == pytest.approx(0.5, rel=2e-3)


def test_phase_flips_through_resonance():
    m = cm.PillboxMode("TM", 1, 0, A, D)
    src = [cm.RingSource(0.5 * A, 0.0, "z", 1.0)]
    below = cm.driven_response([m], m.omega * 0.99, src)[0].amplitude
    above = cm.driven_response([m], m.omega * 1.01, src)[0].amplitude
    dphi = np.angle(above / below)
    assert abs(abs(dphi) - np.pi) < 0.05


def test_selection_rules():
    """A z-ring cannot excite TE modes; a phi-ring cannot excite TM modes."""
    tm = cm.PillboxMode("TM", 1, 1, A, D)
    te = cm.PillboxMode("TE", 1, 1, A, D)
    zr = cm.RingSource(0.3 * A, 0.2 * D, "z")
    pr = cm.RingSource(0.3 * A, 0.2 * D, "phi")
    assert zr.overlap(te) == 0 and pr.overlap(tm) == 0
    assert zr.overlap(tm) != 0 and pr.overlap(te) != 0


def test_scale_to_energy():
    ms = cm.pillbox_modes(A, D, 8e9)
    src = [cm.RingSource(0.5 * A, 0.1 * D, "z")]
    rs = cm.scale_to_energy(cm.driven_response(ms, 2 * np.pi * 3.5e9, src), 2.0)
    assert sum(r.stored_energy for r in rs) == pytest.approx(2.0)
