# GN-PPM mode-classification determinism (Fix-3) — 2026-07-15

Branch: `agent/ppm-fit-conditioning` on `sources/lorrax_D` (from `main`).
Closes the "on-pole PPM census robustness" open item from
`reports/gw_refactor_map_2026-07-01/HANDOFF.md` (Remaining #2) and
`reports/device_invariance_2026-07-08/ROOT_CAUSE.md` (AS-FIXED §, "Fix-3").

## The problem

After the 2026-07-08 padding fixes, a residual device-count sensitivity
remained: 2 of ~23M GN-PPM modes flipped valid↔invalid between 4- and 16-GPU
runs (invalid census 255,980 vs 255,982), and because the **max valid Ω feeds
the adaptive quadrature window edges** (`ppm_windows._build_single_sigma_window`,
`x_max = S_max + omega_max`), a single flip changed minimax node counts
(15/15 vs 13/14) and hence Σ_c, amplified to ~0.27 eV on on-pole bands.

## Root cause — thresholding a cancellation

The fit (`minimax_screening.fit_gn_ppm_from_wc_pair`) classified modes with

```
safe = |Wc0 − Wc_probe| > 1e-14        # absolute, on a DIFFERENCE
Ω²   = ω̄² · Wc_probe / (Wc0 − Wc_probe)
```

Within the one-pole model, `Wc(iω) = Wc0·Ω²/(Ω²+ω̄²)`, so
`denom = Wc0·ω̄²/(Ω²+ω̄²)`: **denom → 0 means Ω → ∞**. Every
dispersion-free element's denominator is a difference of near-equal numbers —
i.e. the near-threshold population sits *at the floating-point noise floor by
construction*, and reduction-order (device-count) noise flips it. Two distinct
populations live there:

1. **Dead elements** — `Wc0 ≈ Wc_probe ≈ 0` at roundoff (far-off-diagonal μν
   pairs). `Ω² = ω̄²·(noise/noise)` is sign-and-magnitude roundoff entropy;
   when it lands positive-huge, the garbage pole enters the valid census and
   can set the window max-Ω. These are the observed cross-P flippers.
2. **Stiff elements** — finite `Wc0`, genuinely no dispersion below the probe
   (pole far above 2 Ry). The fit is ill-conditioned (relative error in Ω²
   amplified by `|Wc0|/|denom|`), but the physics is clean: as Ω → ∞ the pole
   formula reduces **exactly** to the static-COHSEX treatment
   (`ppm_invalid_mode='static_limit'`, the default = BGW mode 3). Static is
   the *analytic limit*, not a fallback.

(Third failure class, unchanged: fitted Ω² ≤ 0 — the element grows toward the
probe, impossible for definite-sign spectral weight, i.e. off-diagonal pole
interference where the one-pole ansatz itself fails.)

## The fix (`src/gw/minimax_screening.py`)

Classify on **magnitudes** (relative tests), never on a cancellation:

```
dead  : |Wc0|            ≤ 1e-12 · per-q max|Wc0|   (_DEAD_REL)
stiff : |Wc0 − Wc_probe| ≤ 1e-8  · |Wc0|            (_STIFF_REL)
valid : ¬dead ∧ ¬stiff ∧ finite ∧ Ω² > 0
```

- A magnitude test only flips if the true value sits within ~1 ulp of the cut
  — measure-sparse, unlike the old cut where the flip population was dense at
  the noise floor.
- Dead + stiff route to the existing invalid class (fallback Ω, then handled
  per `ppm_invalid_mode`): exact for stiff under `static_limit`, ~0 either way
  for dead.
- Garbage-huge Ω values (up to ~ω̄/√η) no longer enter the valid census, so
  the window max-Ω becomes a physical number and node counts stabilize.
- `_STIFF_REL = 1e-8` ≈ the relative accuracy of the W build itself; below it
  the fitted pole position is meaningless anyway. Constants are module-level
  with full rationale comments, not config keys.

Also: docstring updates (`fit_gn_ppm_from_wc_pair` four-class notes;
`ppm_sigma._prepare_sigma_state` invalid-class definition), and the division
now uses a double-`where` so no inf/nan is ever materialized.

## What was NOT done (deliberate)

- **No smooth crossfade / ramp** between pole and static treatments: at the
  stiff boundary the two branches agree to O(η) already (static is the exact
  Ω→∞ limit), so a ramp adds machinery for a discontinuity of ~1e-8 relative.
- **No fix for on-pole Σ(E) ill-conditioning** (the measured 1.28 eV/ulp
  GN-PPM amplification on on-pole bands): that is inherent to evaluating Σ at
  a pole, not a classification defect. Documenting it as ill-posed remains the
  honest disposition (HANDOFF open item, physics).
- No config knobs for the thresholds (YAGNI; revisit only if a real system
  lands modes inside the bands).

## Evidence

| check | result |
|---|---|
| `tests/test_ppm_fit_classification.py` (NEW) — four-class contract + exact pole recovery + pad-birth + ±1-ulp classification stability (includes an element straddling the old 1e-14 absolute cut) | **2 passed** |
| Full suite, canonical serial 1-GPU (`LORRAX_NGPU=1 lxrun python3 -m pytest -q tests`, nid001021) | **207 passed / 24 deselected / 0 failed** (659.9 s) |
| gnppm / bispinor-gnppm golden gates (atol 1e-6 freezes) | **unchanged** — no MoS2/Si fixture mode lands in the dead/stiff bands, so outputs match the existing goldens; no re-freeze |

Commit: `218aeb8` on `agent/ppm-fit-conditioning` (from `main`).

Note the suite wall (11 min) is well above the 2026-07-09 redesign benchmark
(~220 s): the SLATE/FFI linalg contract suite added ~31 tests since. Tracked
separately as the suite-speedup initiative (compile-cache hit-rate audit,
combined restart-variant runner, bispinor fixture regen at lower ecut — the
60 Ry / 30×30×120 gate fixture is production-sized).

## Remaining / follow-ups

- Cross-P re-validation on the original 16-GPU A_charge cell
  (`tests/multi_device/eqp_invariance_cross_p.py` covers P=1 vs P=4; the
  original 2-flip signature was 4g↔16g) — needs a 16-GPU allocation, optional,
  user-triggered per the no-16-GPU-gating rule.
- The GN-PPM on-pole 1.28 eV/ulp conditioning number should be documented in
  the manual's troubleshooting chapter as inherent ill-posedness of on-pole
  Σ(E_dft) evaluation.
