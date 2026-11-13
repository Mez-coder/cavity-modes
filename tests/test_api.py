"""The JSON functions behind the web UI (no HTTP server needed)."""
import numpy as np
import pytest

import cavity_modes as cm
from cavity_modes import api

PILLBOX = dict(points=[dict(z=0, r=0), dict(z=0, r=38.3), dict(z=49.9, r=38.3), dict(z=49.9, r=0)],
               f_max_GHz=7.7, h_mm=1.0, beta=1.0, cell_length=49.9)


@pytest.fixture(scope="module")
def solved():
    return api.solve(PILLBOX)


def test_solve_matches_analytic(solved):
    ana = cm.pillbox_modes(0.0383, 0.0499, 7.7e9)
    assert [m["kind"] for m in solved["modes"]] == [a.kind for a in ana]
    for m, a in zip(solved["modes"], ana):
        assert m["f_GHz"] == pytest.approx(a.frequency / 1e9, rel=1e-5)
    assert solved["mesher"] == "mapped"


def test_fields_payload(solved):
    d = api.mode_fields(solved["solve_id"], 0)
    assert d["kind"] == "TM" and set(d["fields"]) == {"Ez", "Er", "Bphi"}
    assert len(d["inside"]) == d["nr"] * d["nz"] == len(d["fields"]["Ez"])
    ez = np.array(d["axis"]["Ez"])
    assert np.abs(ez).max() > 0


def test_figures_follow_beta(solved):
    a = api.figures(solved["solve_id"], 1.0)["modes"][0]["T"]
    b = api.figures(solved["solve_id"], 0.5)["modes"][0]["T"]
    assert a != b


def test_tracking_on_crest_gain(solved):
    out = api.run_tracking(dict(
        solve_id=solved["solve_id"], crest="analytic",
        beam=dict(species="electron", Ek_MeV=1000.0, n=50, sigma_t_ps=0.01, sigma_delta=0.0),
        components=[dict(mode=0, set="voltage", value=0.5, phase_deg=0.0)]))
    gain = out["stats"]["out"]["Ek_ref_MeV"] - out["stats"]["in"]["Ek_ref_MeV"]
    assert gain == pytest.approx(0.5, rel=2e-3)


def test_te_mode_needs_energy(solved):
    te = next(m["index"] for m in solved["modes"] if m["kind"] == "TE")
    with pytest.raises(api.ApiError):
        api.run_tracking(dict(solve_id=solved["solve_id"], beam=dict(n=10),
                              components=[dict(mode=te, set="voltage", value=1.0)]))


def test_preview_reports_crossing_wall():
    spec = dict(points=[dict(z=0, r=10), dict(z=30, r=40), dict(z=5, r=30), dict(z=50, r=10)])
    d = api.profile_preview(spec)
    assert d["problem"] and d["cross"] is not None and len(d["wall"]) >= 4
    with pytest.raises(api.ApiError):
        api.solve(spec)


def test_expired_solution():
    with pytest.raises(api.ApiError):
        api.mode_fields("nope", 0)
