"""JSON-in / JSON-out functions behind the web UI.

Kept free of any web framework so they can be unit-tested directly.
All lengths in requests and responses are in millimetres unless the key says
otherwise; frequencies in GHz; fields at 1 J stored energy.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

from .beam import Bunch, phase_space, statistics
from .constants import C, SPECIES
from .fem import solve_modes
from .figures import figures_of_merit
from .geometry import Profile, is_single_valued
from .modes import FieldMapMode, Fields
from .modes import gauss_legendre
from .tracking import Component, make_component, refine_crest, track

MM = 1e-3
_LOCK = threading.Lock()  # gmsh is not thread-safe
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_SIZE = 8


def warm_up():
    """Precompute quadrature tables so the first request is not slow."""
    for n in (2000, 256, 160):
        gauss_legendre(n)


class ApiError(ValueError):
    """A problem with the request that the user can fix."""


# ------------------------------------------------------------------ helpers
def _round(a, digits=6):
    return np.round(np.asarray(a, float), digits).tolist()


def _sig(a, sig=5):
    """Round to significant figures (keeps payloads small)."""
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a == 0, 1.0, 10 ** np.floor(np.log10(np.abs(a))))
    return (np.round(a / mag, sig - 1) * mag).tolist()


def _finite(x):
    x = float(x)
    return x if np.isfinite(x) else None


def build_profile(spec: dict, check: bool = True) -> Profile:
    pts = spec.get("points") or []
    if len(pts) < 2:
        raise ApiError("The wall needs at least two points.")
    zr = [(float(p["z"]) * MM, float(p["r"]) * MM) for p in pts]
    smooth = [bool(p.get("smooth", False)) for p in pts]
    if any(r < 0 for _, r in zr):
        raise ApiError("A wall point is below the axis (r < 0).")
    try:
        prof = Profile.from_control_points(
            zr, smooth, bc_in=spec.get("bc_in", "electric"), bc_out=spec.get("bc_out", "electric"),
            cell_length=(float(spec["cell_length"]) * MM) if spec.get("cell_length") else None)
        if check:
            prof.check()
    except ValueError as e:
        raise ApiError(str(e)) from None
    return prof


def profile_preview(spec: dict) -> dict:
    """Sampled wall polyline for the editor (no solve).

    An invalid shape still returns its wall, plus ``problem`` and the crossing
    point, so the editor can show what is wrong.
    """
    p = build_profile(spec, check=False)
    out = dict(wall=_round(p.wall / MM, 4), single_valued=is_single_valued(p), problem=None, cross=None)
    hit = p.self_intersection()
    if hit is not None:
        out["problem"] = f"The wall crosses itself near z = {hit[0]/MM:.1f} mm, r = {hit[1]/MM:.1f} mm."
        out["cross"] = [hit[0] / MM, hit[1] / MM]
    return out


def _mode_summary(i, m, beta):
    f = figures_of_merit(m, beta)
    return dict(
        index=i, label=m.label, kind=m.kind, f_GHz=m.frequency / 1e9,
        Q0=_finite(f.Q0), T=_finite(f.T), R_over_Q=_finite(f.R_over_Q_ohm),
        ZTT_MOhm_m=_finite(f.ZTT_ohm_per_m / 1e6), Epk_Eacc=_finite(f.Epk_over_Eacc),
        Bpk_Eacc=_finite(f.Bpk_over_Eacc_mT_per_MV_m), V_eff_MV_at_1J=f.V_eff_V / 1e6,
        P_wall_W_at_1J=f.wall_loss_W,
    )


def beta_of(species: str, Ek_MeV: float) -> float:
    sp = SPECIES[species]
    g = 1 + Ek_MeV * 1e6 / sp.rest_energy_eV
    return float(np.sqrt(1 - 1 / g**2))


def solve(spec: dict) -> dict:
    prof = build_profile(spec)
    f_max = float(spec.get("f_max_GHz", 8.0)) * 1e9
    h = float(spec.get("h_mm", 0.8)) * MM
    h_axis = spec.get("h_axis_mm")
    kinds = tuple(k for k in ("TM", "TE") if spec.get("kinds", {"TM": True, "TE": True}).get(k))
    if not kinds:
        raise ApiError("Pick TM, TE or both.")
    mesh_kw = dict(h=h)
    if h_axis:
        mesh_kw["h_axis"] = float(h_axis) * MM
    mesher = "mapped" if is_single_valued(prof) else "gmsh"
    if mesher == "gmsh":
        try:
            import gmsh  # noqa: F401
        except ImportError:
            raise ApiError("This shape needs gmsh (the wall is not single-valued in z). "
                           "Install it with: uv add gmsh") from None
    t0 = time.time()
    with _LOCK:
        try:
            modes = solve_modes(prof, f_max, kinds=kinds, **mesh_kw)
        except Exception as e:  # meshing / solver failure is reported, not raised
            raise ApiError(f"Solve failed: {e}") from None
    dt = time.time() - t0
    sid = uuid.uuid4().hex[:10]
    _CACHE[sid] = dict(profile=prof, modes=modes, maps={}, grids={})
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    beta = float(spec.get("beta", 1.0))
    mesh = modes[0].mesh if modes else None
    return dict(
        solve_id=sid, time_s=dt, mesher=mesher,
        n_nodes=int(mesh.n_nodes) if mesh else 0, n_elems=int(mesh.n_elems) if mesh else 0,
        wall=_round(prof.wall / MM, 4),
        modes=[_mode_summary(i, m, beta) for i, m in enumerate(modes)],
    )


def _get(sid: str) -> dict:
    if sid not in _CACHE:
        raise ApiError("That solution has expired. Solve again.")
    return _CACHE[sid]


def figures(sid: str, beta: float) -> dict:
    s = _get(sid)
    return dict(modes=[_mode_summary(i, m, beta) for i, m in enumerate(s["modes"])])


def mode_fields(sid: str, index: int, nz: int = 320) -> dict:
    """Field maps on a regular grid for display, plus on-axis Ez."""
    s = _get(sid)
    m = s["modes"][index]
    key = (index, nz)
    if key not in s["grids"]:
        zlen = m.z_max - m.z_min
        nr = int(np.clip(round(nz * m.r_max / zlen), 24, 240))
        z = np.linspace(m.z_min, m.z_max, nz)
        r = np.linspace(0, m.r_max, nr)
        R, Z = np.meshgrid(r, z, indexing="ij")  # (nr, nz)
        elem, L = m.mesh.locate(R.ravel(), Z.ravel())
        ok = elem >= 0
        inside = ok.reshape(R.shape)
        vals = [np.zeros(R.size) for _ in range(6)]
        if ok.any():
            for v, w in zip(vals, m._fields_at(elem[ok], L[ok], R.ravel()[ok])):
                v[ok] = w
        f = Fields(*(v.reshape(R.shape) for v in vals))
        if m.kind == "TM":
            comps = dict(Ez=f.Ez, Er=f.Er, Bphi=f.Bphi)
        else:
            comps = dict(Ephi=f.Ephi, Br=f.Br, Bz=f.Bz)
        za = np.linspace(m.z_min, m.z_max, 600)
        s["grids"][key] = dict(
            label=m.label, kind=m.kind, f_GHz=m.frequency / 1e9,
            nz=nz, nr=nr, z_mm=[m.z_min / MM, m.z_max / MM], r_mm=[0.0, m.r_max / MM],
            inside=inside.astype(np.uint8).ravel().tolist(),
            fields={k: _sig(v.ravel(), 4) for k, v in comps.items()},
            axis=dict(z_mm=_round(za / MM, 4), Ez=_sig(m.on_axis_Ez(za), 5)),
        )
    return s["grids"][key]


def _fieldmap(s, index):
    if index not in s["maps"]:
        m = s["modes"][index]
        zlen = m.z_max - m.z_min
        nz = int(np.clip(zlen / (0.15 * MM), 200, 900))
        nr = int(np.clip(m.r_max / (0.15 * MM), 80, 400))
        s["maps"][index] = FieldMapMode.sample(m, nr, nz)
    return s["maps"][index]


def run_tracking(spec: dict) -> dict:
    s = _get(spec["solve_id"])
    prof = s["profile"]
    b = spec.get("beam", {})
    sp = SPECIES[b.get("species", "electron")]
    Ek = float(b.get("Ek_MeV", 4.0)) * 1e6
    n = int(np.clip(int(b.get("n", 5000)), 2, 100_000))
    bunch = Bunch.gaussian(
        n, Ek, sp,
        emit_nx=float(b.get("emit_nx_um", 1.0)) * 1e-6, beta_x=float(b.get("beta_x", 0.4)),
        alpha_x=float(b.get("alpha_x", 0.0)),
        emit_ny=float(b.get("emit_ny_um", 1.0)) * 1e-6, beta_y=float(b.get("beta_y", 0.4)),
        alpha_y=float(b.get("alpha_y", 0.0)),
        sigma_t=float(b.get("sigma_t_ps", 3.0)) * 1e-12, sigma_delta=float(b.get("sigma_delta", 1e-3)),
        z=prof.z_min, seed=int(b.get("seed", 1)))
    beta = float(np.sqrt(1 - 1 / bunch.gamma[0] ** 2))
    sign = 1 if sp.charge > 0 else -1
    comps = []
    for c in spec.get("components", []):
        i = int(c["mode"])
        m = s["modes"][i]
        fm = _fieldmap(s, i)
        kw = dict(beta_ref=beta, charge_sign=sign, z_entry=prof.z_min)
        val = float(c.get("value", 1.0))
        if c.get("set", "voltage") == "voltage":
            if figures_of_merit(m, beta).V_eff_V == 0:
                raise ApiError(f"{m.label} has no on-axis voltage; set its stored energy instead.")
            kw["voltage_V"] = val * 1e6
        else:
            kw["energy_J"] = val
        comp = make_component(fm, phase_deg=0.0, **kw)
        phase = float(c.get("phase_deg", 0.0))
        tracked = spec.get("crest", "tracked") == "tracked"
        if tracked and abs(comp.amplitude) > 0 and figures_of_merit(m, beta).V_eff_V > 0:
            key = (i, round(abs(comp.amplitude), 12), round(Ek, 3), sp.name)
            crest = s.setdefault("crest", {})
            if key not in crest:
                crest[key] = np.angle(refine_crest(comp, bunch, prof.z_max, n_steps=300).amplitude)
            amp = abs(comp.amplitude) * np.exp(1j * (crest[key] + np.radians(phase)))
        else:
            amp = comp.amplitude * np.exp(1j * np.radians(phase))
        comps.append(Component(fm, amp, comp.omega))
    if not comps:
        raise ApiError("Add at least one mode to drive.")
    f_ref = comps[0].omega / (2 * np.pi)
    zlen = prof.z_max - prof.z_min
    steps = int(np.clip(zlen / (0.2 * MM), 100, 2000))
    t0 = time.time()
    try:
        out, hist = track(bunch, comps, prof.z_max, steps, record_every=max(1, steps // 150))
    except RuntimeError as e:
        raise ApiError(str(e)) from None
    dt = time.time() - t0

    # envelope along z
    env = dict(z_mm=[], sx_mm=[], sy_mm=[], Ek_ref_MeV=[])
    for z, x, y, t, px, py, pz in hist:
        env["z_mm"].append(z / MM)
        env["sx_mm"].append(float(np.std(x)) / MM)
        env["sy_mm"].append(float(np.std(y)) / MM)
        g = np.sqrt(1 + px[0] ** 2 + py[0] ** 2 + pz[0] ** 2)
        env["Ek_ref_MeV"].append(float((g - 1) * sp.rest_energy_eV / 1e6))

    show = slice(0, min(n, 4000))
    ps = {}
    for tag, bb in (("in", bunch), ("out", out)):
        p = phase_space(bb, f_ref)
        ps[tag] = {k: _sig(p[k][show], 5) for k in ("x", "xp", "y", "yp", "dphi", "dE")}
    st_in, st_out = statistics(bunch), statistics(out)
    return dict(
        time_s=dt, f_ref_GHz=f_ref / 1e9, beta=beta, n=n,
        phase_space=ps,
        stats={"in": {k: _finite(v) for k, v in st_in.items()},
               "out": {k: _finite(v) for k, v in st_out.items()}},
        envelope={k: _sig(v, 5) for k, v in env.items()},
        components=[dict(label=c.mode.label.replace(" (map)", ""),
                         amplitude_abs=abs(c.amplitude),
                         phase_deg=float(np.degrees(np.angle(c.amplitude))),
                         f_GHz=c.omega / 2 / np.pi / 1e9) for c in comps],
    )
