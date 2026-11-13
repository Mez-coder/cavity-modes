"""FEM tour: a smooth spline-defined cell with beam pipes, all m = 0 modes,
and the phase space of a bunch through its fundamental mode.

    uv run python examples/fem_tour.py            # mapped mesher (no gmsh needed)
    uv run python examples/fem_tour.py --nose     # re-entrant nose-cone cell via gmsh
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import cavity_modes as cm

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
NOSE = "--nose" in sys.argv

# ------------------------------------------------------------------ geometry
# Control points (z, r) in metres. Drag these around to reshape the cell.
if not NOSE:
    pipe_r, L_pipe, Lc = 0.009, 0.025, 0.050
    ctrl = [(0.0, pipe_r), (0.006, 0.014), (0.014, 0.036), (0.025, 0.0415),
            (0.036, 0.036), (0.044, 0.014), (0.050, pipe_r)]
    profile = cm.Profile.from_segments(
        [("line", [(-L_pipe, pipe_r), (0.0, pipe_r)]),
         ("spline", ctrl, 300),
         ("line", [(Lc, pipe_r), (Lc + L_pipe, pipe_r)])],
        bc_in="electric", bc_out="electric", cell_length=Lc, name="smooth cell")
    mesh_kw = dict(h=0.6e-3)
else:
    pipe_r, L_pipe, Lc = 0.006, 0.02, 0.0498
    profile = cm.Profile.from_segments(
        [("line", [(-L_pipe, pipe_r), (0.0, pipe_r)]),
         ("spline", [(0.0, pipe_r), (0.004, 0.009), (0.010, 0.010), (0.014, 0.03),
                     (0.0249, 0.035)], 80),
         ("spline", [(0.0249, 0.035), (0.0358, 0.03), (0.0398, 0.010), (0.0458, 0.009),
                     (Lc, pipe_r)], 80),
         ("line", [(Lc, pipe_r), (Lc + L_pipe, pipe_r)])],
        cell_length=Lc, name="nose-cone cell")
    mesh_kw = dict(h=1.0e-3, h_axis=0.3e-3, h_wall=0.5e-3)

F_MAX = 8e9
t0 = time.time()
modes = cm.solve_modes(profile, F_MAX, **mesh_kw)
print(f"{profile.name}: {len(modes)} modes below {F_MAX/1e9:.0f} GHz, "
      f"{modes[0].mesh.n_nodes} P2 nodes, solved in {time.time() - t0:.2f} s")

EK = 4e6
beta = float(np.sqrt(1 - 1 / cm.Bunch.gaussian(1, EK).gamma[0] ** 2))
print(f"{'mode':5s} {'f [GHz]':>8s} {'Q0':>7s} {'T':>6s} {'R/Q [Ω]':>8s} {'ZT² [MΩ/m]':>11s} "
      f"{'Epk/Eacc':>9s} {'Bpk/Eacc':>9s}")
for m in modes:
    f = cm.figures_of_merit(m, beta)
    print(f"{m.label:5s} {m.frequency/1e9:8.4f} {f.Q0:7.0f} {f.T:6.3f} {f.R_over_Q_ohm:8.1f} "
          f"{f.ZTT_ohm_per_m/1e6:11.2f} {f.Epk_over_Eacc:9.2f} {f.Bpk_over_Eacc_mT_per_MV_m:9.2f}")

# ------------------------------------------------------------------ field maps
mesh = modes[0].mesh
tri = mtri.Triangulation(mesh.tm.z * 1e3, mesh.tm.r * 1e3, mesh.tm.tris)
nv = mesh.tm.n_nodes
n = len(modes)
cols = min(4, n)
rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
verts, _ = profile.polygon()
for ax, m in zip(axs.flat, modes):
    u = m.coef[:nv]  # B_phi (TM) or E_phi (TE) at vertices
    lim = np.abs(u).max()
    ax.tripcolor(tri, u, cmap="RdBu_r" if m.kind == "TM" else "PuOr_r", vmin=-lim, vmax=lim,
                 shading="gouraud")
    ax.tricontour(tri, (mesh.tm.r if m.kind == "TM" else np.ones(nv)) * u, levels=12,
                  colors="k", linewidths=0.4)
    ax.plot(verts[:, 0] * 1e3, verts[:, 1] * 1e3, "k", lw=1)
    ax.set_aspect("equal")
    what = "Bφ, E lines" if m.kind == "TM" else "Eφ"
    ax.set_title(f"{m.label}  {m.frequency/1e9:.3f} GHz  ({what})", fontsize=9)
for ax in list(axs.flat)[n:]:
    ax.axis("off")
fig.tight_layout()
tag = "nose" if NOSE else "smooth"
fig.savefig(OUT / f"fem_modes_{tag}.png", dpi=130)

# ------------------------------------------------------------------ on-axis Ez
fig, ax = plt.subplots(figsize=(7, 3))
zz = np.linspace(profile.z_min, profile.z_max, 1500)
for m in modes:
    if m.kind == "TM":
        ax.plot(zz * 1e3, m.on_axis_Ez(zz) / 1e6, label=m.label, lw=1)
ax.set_xlabel("z [mm]")
ax.set_ylabel("Ez(r=0) [MV/m] at 1 J")
ax.legend(fontsize=7, ncol=4)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"fem_on_axis_{tag}.png", dpi=130)

# ------------------------------------------------------------------ tracking
fund = cm.FieldMapMode.sample(modes[0], 161, 481)
bunch = cm.Bunch.gaussian(10000, EK, cm.ELECTRON, emit_nx=1e-6, beta_x=0.4, alpha_x=0.5,
                          emit_ny=1e-6, beta_y=0.4, alpha_y=0.5, sigma_t=3e-12,
                          sigma_delta=2e-3, z=profile.z_min)
comp = cm.make_component(fund, voltage_V=1e6, beta_ref=beta, charge_sign=-1,
                         z_entry=profile.z_min)
comp = cm.refine_crest(comp, bunch, profile.z_max, phase_deg=-20)
out, _ = cm.track(bunch, [comp], profile.z_max, 600)
s0, s1 = cm.statistics(bunch), cm.statistics(out)
for k in ("Ek_ref_MeV", "sigma_E_keV", "emit_nx_m", "beta_x_m", "alpha_x", "emit_long_keV_ps"):
    print(f"{k:18s} {s0[k]:12.5g} -> {s1[k]:12.5g}")
p0, p1 = cm.phase_space(bunch, modes[0].frequency), cm.phase_space(out, modes[0].frequency)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for (kx, ky, lx, ly), ax in zip([("x", "xp", "x [mm]", "x' [mrad]"),
                                 ("dphi", "dE", "Δφ [deg]", "ΔE [keV]")], axs):
    ax.scatter(p0[kx], p0[ky], s=0.3, alpha=0.3, label="in")
    ax.scatter(p1[kx], p1[ky], s=0.3, alpha=0.3, label="out")
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.grid(alpha=0.3)
axs[0].legend(markerscale=20)
fig.suptitle(f"{profile.name}: {modes[0].label}, V_eff = 1 MV, φ = −20°", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / f"fem_phase_space_{tag}.png", dpi=130)
print("figures written to", OUT)
