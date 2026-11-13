# cavity-modes

Eigenmodes of axisymmetric RF cavities, and what they do to a particle bunch.

The goal: define a cell shape (eventually by dragging spline points in a
browser), solve Maxwell's equations for every mode, and track a bunch through
any combination of modes to see how the phase space changes in all planes.

**Status (v0.1):** m = 0 only, analytic pillbox modes, figures of merit,
driven response, and a full-Lorentz-force tracker, all validated against
closed-form results. Next: a 2D FEM solver for arbitrary (spline) profiles,
then the browser front end.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python examples/pillbox_tour.py        # writes PNGs to examples/output/
```

## Layout

| module | what it does |
|---|---|
| `modes.py` | `Mode` interface (fields normalised to **1 J**), `FieldMapMode` (gridded maps – what the FEM solver will produce) |
| `pillbox.py` | analytic TM0np / TE0np pillbox modes, `pillbox_modes()` lists all below f_max |
| `figures.py` | Q0, wall loss, V0, T(β), R/Q, R_sh, ZT², peak surface fields |
| `driven.py` | steady-state response to a drive at any frequency (mode expansion) |
| `beam.py` | Gaussian bunches from Twiss parameters; phase-space coordinates and rms statistics |
| `tracking.py` | RK4 in z with full E and B fields; crest finding by tracking |

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
