"""Tour of the m = 0 pillbox toolkit. Writes PNGs to examples/output/.

    python examples/pillbox_tour.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cavity_modes as cm
from cavity_modes.constants import C, ELECTRON

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

A, D = 0.0383, 0.0499  # ~3 GHz S-band pillbox
F_MAX = 8e9
EK = 4e6  # 4 MeV electrons
modes = cm.pillbox_modes(A, D, F_MAX)
b0 = cm.Bunch.gaussian(1, EK, ELECTRON)
BETA = float(np.sqrt(1 - 1 / b0.gamma[0] ** 2))

# ---------------------------------------------------------------- mode table
print(f"{'mode':7s} {'f [GHz]':>8s} {'Q0':>7s} {'T':>6s} {'R/Q [Ω]':>8s} "
      f"{'Rsh [MΩ]':>9s} {'ZT² [MΩ/m]':>11s} {'Epk/Eacc':>9s}   (beta = {BETA:.4f})")
for m in modes:
    f = cm.figures_of_merit(m, BETA)
    print(f"{m.label:7s} {m.frequency/1e9:8.4f} {f.Q0:7.0f} {f.T:6.3f} {f.R_over_Q_ohm:8.1f} "
          f"{f.R_shunt_ohm/1e6:9.3f} {f.ZTT_ohm_per_m/1e6:11.2f} {f.Epk_over_Eacc:9.2f}")


# ---------------------------------------------------------------- field maps
def field_map(ax, m):
    r = np.linspace(0, A, 90)
    z = np.linspace(0, D, 120)
    R, Z = np.meshgrid(r, z, indexing="ij")
    f = m.fields(R, Z)
    if m.kind == "TM":
        mag = np.hypot(f.Er, f.Ez)
        ax.pcolormesh(Z * 1e3, R * 1e3, f.Bphi * 1e3, cmap="RdBu_r", shading="auto")
        sl = (slice(None, None, 9), slice(None, None, 10))
        ax.quiver(Z[sl] * 1e3, R[sl] * 1e3, f.Ez[sl], f.Er[sl], mag[sl], cmap="Greys", scale=None)
        ax.set_title(f"{m.label}  {m.frequency/1e9:.3f} GHz\nE (arrows), Bφ (colour)", fontsize=9)
    else:
        mag = np.hypot(f.Br, f.Bz)
        ax.pcolormesh(Z * 1e3, R * 1e3, f.Ephi / 1e6, cmap="PuOr_r", shading="auto")
        sl = (slice(None, None, 9), slice(None, None, 10))
        ax.quiver(Z[sl] * 1e3, R[sl] * 1e3, f.Bz[sl], f.Br[sl], mag[sl], cmap="Greys")
        ax.set_title(f"{m.label}  {m.frequency/1e9:.3f} GHz\nB (arrows), Eφ (colour)", fontsize=9)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_aspect("equal")


n = len(modes)
fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3.4 * ((n + 1) // 2), 6.4))
for ax, m in zip(axs.flat, modes):
    field_map(ax, m)
for ax in list(axs.flat)[n:]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "field_maps.png", dpi=130)

# ---------------------------------------------------------------- T(beta)
betas = np.linspace(0.3, 1.0, 200)
fig, ax = plt.subplots(figsize=(6, 3.6))
for m in modes:
    if m.kind == "TM":
        ax.plot(betas, cm.transit_time_curve(m, betas), label=m.label)
ax.set_xlabel("β")
ax.set_ylabel("T(β) = |V_c| / ∫|Ez|dz")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "transit_time.png", dpi=130)

# ---------------------------------------------------------------- driven spectrum
src = [cm.RingSource(0.6 * A, 0.0, "z"), cm.RingSource(0.6 * A, 0.3 * D, "phi")]
freqs = np.linspace(2.5e9, F_MAX, 6000)
spec = cm.response_spectrum(modes, freqs, src)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.semilogy(freqs / 1e9, spec.sum(0), "k", lw=1, label="total")
for i, m in enumerate(modes):
    ax.semilogy(freqs / 1e9, spec[i], lw=0.8, alpha=0.7, label=m.label)
ax.set_xlabel("drive frequency [GHz]")
ax.set_ylabel("stored energy [J] (arb. drive)")
ax.set_ylim(spec.sum(0).max() * 1e-8, spec.sum(0).max() * 3)
ax.legend(fontsize=7, ncol=3)
ax.set_title("z-ring on upstream wall (couples TM), φ-ring at z = 0.3d (couples TE); r = 0.6a", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "driven_spectrum.png", dpi=130)

# ---------------------------------------------------------------- phase space through TM010
tm010 = modes[0]
bunch = cm.Bunch.gaussian(20000, EK, ELECTRON, emit_nx=1e-6, beta_x=0.4, alpha_x=0.5,
                          emit_ny=1e-6, beta_y=0.4, alpha_y=0.5,
                          sigma_t=3e-12, sigma_delta=2e-3)
comp = cm.make_component(tm010, phase_deg=0.0, voltage_V=1.0e6, beta_ref=BETA, charge_sign=-1)
comp = cm.refine_crest(comp, bunch, D, phase_deg=-20.0)
out, _ = cm.track(bunch, [comp], D, 400)
s0, s1 = cm.statistics(bunch), cm.statistics(out)
for k in ("Ek_ref_MeV", "sigma_E_keV", "emit_nx_m", "emit_ny_m", "beta_x_m", "alpha_x",
          "emit_long_keV_ps", "sigma_t_ps"):
    print(f"{k:18s} {s0[k]:12.5g} -> {s1[k]:12.5g}")

p0 = cm.phase_space(bunch, tm010.frequency)
p1 = cm.phase_space(out, tm010.frequency)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for (kx, ky, lx, ly), ax in zip(
        [("x", "xp", "x [mm]", "x' [mrad]"), ("y", "yp", "y [mm]", "y' [mrad]"),
         ("dphi", "dE", "Δφ [deg]", "ΔE [keV]")], axs):
    ax.scatter(p0[kx], p0[ky], s=0.3, alpha=0.3, label="in")
    ax.scatter(p1[kx], p1[ky], s=0.3, alpha=0.3, label="out")
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.grid(alpha=0.3)
axs[0].legend(markerscale=20)
fig.suptitle(f"TM010, V_eff = 1 MV, φ = −20° from crest, {EK/1e6:.0f} MeV electrons", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "phase_space_tm010.png", dpi=130)
print("figures written to", OUT)
