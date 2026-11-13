"""gmsh-based meshing tests (need gmsh installed)."""
import numpy as np
import pytest

import cavity_modes as cm
from cavity_modes.fem import solve_modes
from cavity_modes.geometry import Profile

pytest.importorskip("gmsh", reason="gmsh not installed")

A, D = 0.0383, 0.0499

def test_gmsh_pillbox_matches_analytic():
    from cavity_modes.geometry import mesh_gmsh
    p = Profile.pillbox(A, D)
    fem = solve_modes(p, 7.7e9, mesh=mesh_gmsh(p, h=1.0e-3))
    ana = cm.pillbox_modes(A, D, 7.7e9)
    assert [m.kind for m in fem] == [m.kind for m in ana]
    for m, a in zip(fem, ana):
        assert m.frequency == pytest.approx(a.frequency, rel=2e-5), a.label
    f, g = cm.figures_of_merit(fem[0], 1.0), cm.figures_of_merit(ana[0], 1.0)
    assert f.R_over_Q_ohm == pytest.approx(g.R_over_Q_ohm, rel=5e-3)


def test_gmsh_beam_pipe_raises_tm010():
    """Beam-pipe holes add volume on axis at the end walls, where TM010 is
    E-dominated (H_phi ~ J1(kr) -> 0). Slater: adding E-dominated volume
    RAISES the frequency. With pipes far below cut-off (TM01 cut-off ~14 GHz
    for r = 8 mm) the field is confined to the cell."""
    p = Profile.pillbox(A, D, pipe_radius=0.008, pipe_length=0.03)
    m = solve_modes(p, 3.2e9, kinds=("TM",), h=1.0e-3, h_axis=0.4e-3)[0]
    f0 = cm.PillboxMode("TM", 1, 0, A, D).frequency
    assert f0 < m.frequency < 1.03 * f0
    ez = np.abs(m.on_axis_Ez(np.array([-0.03 + 1e-4, D / 2])))
    assert ez[0] < 1e-3 * ez[1]

def test_gmsh_nose_cone_runs():
    noses = [("line", [(-0.02, 0.006), (0.0, 0.006)]),
             ("spline", [(0.0, 0.006), (0.004, 0.009), (0.010, 0.010), (0.014, 0.03),
                         (0.0249, 0.035)], 60),
             ("spline", [(0.0249, 0.035), (0.0358, 0.03), (0.0398, 0.010), (0.0458, 0.009),
                         (0.0498, 0.006)], 60),
             ("line", [(0.0498, 0.006), (0.0698, 0.006)])]
    p = Profile.from_segments(noses, cell_length=0.0498)
    ms = solve_modes(p, 4e9, kinds=("TM", "TE"), h=1.2e-3, h_axis=0.4e-3)
    assert ms and ms[0].kind == "TM"
    fom = cm.figures_of_merit(ms[0], 1.0)
    assert 0 < fom.T <= 1 and fom.R_over_Q_ohm > 0
