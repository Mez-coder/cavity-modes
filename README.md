# cavity-modes

Eigenmodes of axisymmetric RF cavities, and what they do to a particle bunch.

The goal: define a cell shape (eventually by dragging spline points in a
browser), solve Maxwell's equations for every mode, and track a bunch through
any combination of modes to see how the phase space changes in all planes.

**Status (v0.2):** m = 0 only. Analytic pillbox modes, plus a 2D axisymmetric
finite-element solver for arbitrary spline-defined profiles. Also figures of
merit, driven response and a full-Lorentz-force tracker, all validated against
closed-form results. Next: the browser front end (FastAPI backend, draggable
spline points).

## Quick start

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # creates .venv, installs package + dev tools
uv run pytest
uv run python examples/pillbox_tour.py   # analytic pillbox; PNGs to examples/output/
uv run python examples/fem_tour.py       # spline cell with beam pipes (FEM)
uv run python examples/fem_tour.py --nose   # re-entrant nose-cone cell (gmsh)
```

Commit `uv.lock` so every machine resolves the same versions.

## Layout

| module | what it does |
|---|---|
| `modes.py` | `Mode` interface (fields normalised to **1 J**), `FieldMapMode` (gridded maps – what the FEM solver will produce) |
| `pillbox.py` | analytic TM0np / TE0np pillbox modes, `pillbox_modes()` lists all below f_max |
| `geometry.py` | `Profile` (wall polyline in z-r, port boundary conditions), splines, mapped mesher (numpy) and gmsh mesher |
| `fem.py` | P2 axisymmetric eigen-solver for TM0/TE0, `FEMMode` with fields rebuilt from Maxwell's equations; `solve_modes()` |
| `figures.py` | Q0, wall loss, V0, T(β), R/Q, R_sh, ZT², peak surface fields |
| `driven.py` | steady-state response to a drive at any frequency (mode expansion) |
| `beam.py` | Gaussian bunches from Twiss parameters; phase-space coordinates and rms statistics |
| `tracking.py` | RK4 in z with full E and B fields; crest finding by tracking |

## FEM in one paragraph

For m = 0, TM and TE separate. Each family is one scalar: u = H_φ for TM, u = E_φ for TE.
The weak form is ∫[∂zu ∂zw + (∂ru + u/r)(∂rw + w/r)] r dr dz = k² ∫ u w r dr dz.
TM and TE share this operator and differ only in which boundaries are Dirichlet
(TM: axis and PMC planes; TE: axis and PEC). The formulation is scalar, so there are no spurious modes.
The other field components come from the curl equations, which makes E (TM) and B (TE) one order less accurate than u.
Quantities that depend on them (R/Q, TE wall loss) converge as h². Grade the mesh towards the axis
(`grade=1.5` for the mapped mesher, `h_axis=` for gmsh) when R/Q matters.

Port planes at the beam-pipe ends take `bc_in` / `bc_out` = `"electric"`, `"magnetic"` or `"metal"`.
Electric and magnetic are lossless symmetry planes (for example, half a cell with a magnetic mid-plane gives the odd modes).

## Conventions

* Each mode: `E = E_map cos(ωt)`, `B = B_map sin(ωt)`, normalised so the stored energy is 1 J.
* A driven component with complex amplitude A at drive frequency ω:
  `E = Re[A E_map e^{iωt}]`, `B = Re[-i A (ω_n/ω) B_map e^{iωt}]`.
  For a resonant drive |A|² is the stored energy in J.
* **Transit-time factor.** `V_c(β) = ∫ Ez(0,z) exp(iω(z−z_in)/βc) dz`. Energy gain = `q |V_c| cos(φ + arg V_c)`.
  T = |V_c| / ∫|Ez| dz is the crest value, so it does **not** depend on phase: all phase dependence sits in the cosine.
  Moving the origin to the electrical centre makes V_c real, which gives the familiar `ΔW = qV0T cos φ`.
* Shunt impedance uses the linac convention `R = |V_c|²/P` (the circuit convention is half of this).
* Phase φ = 0 is on crest for the reference particle. Positive φ means the reference particle arrives *after* the crest.
  `refine_crest` finds the true crest by tracking, which matters at low β.

## Validation (tests/)

* Frequencies match the Bessel-zero formula; E and B energies each equal 1 J; tangential E and normal B vanish on the walls.
* TM010 Q0 = ad/(δ(a+d)) and R/Q, T = sin x / x agree with the closed forms.
* Tracked energy gain = |q|V_c cos φ to 1e-6 (ultra-relativistic limit).
* **Panofsky–Wenzel:** ∂Δp⊥/∂τ = −v ∇⊥Δp_z, checked by finite differences on tracked particles.
* Normalised emittance is conserved for a short, monochromatic bunch.
* Driven response: Lorentzian half-power width ω/Q, 180° phase flip through resonance, TM/TE selection rules.
* FEM vs analytic pillbox: frequencies to 5e-6 on a coarse mesh, Q, R/Q, T and peak fields; h² convergence of R/Q.
* FEM half cells with electric or magnetic mid-planes recover the even- and odd-p modes, with the same Q.
* **Slater perturbation:** a smooth bump in the wall shifts TM010 (up) and TM011 (down) by the amounts Slater's theorem predicts, to < 0.3%.
* Tracking through FEM field maps reproduces tracking through the analytic modes.
* gmsh tests (`tests/test_fem_gmsh.py`, local only): pillbox, beam-pipe pillbox, nose-cone cell.
