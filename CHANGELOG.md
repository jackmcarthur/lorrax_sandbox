# Changelog

## 2026-04-05: Root cause of 2D GN-GPP head discrepancy identified

- **Report**: `reports/cohsex_head_investigation_2026-04-04/report.md`
- **Run**: `runs/MoS2_1x1_full_workflow/`

### The finding

Added debug prints to BGW `Sigma/mtxel_cor.f90` and `Sigma/sigma_main.f90`
and confirmed how BGW actually treats the q=0, G=G'=0 head in `freq_dep=3`:

  - `fixwings_dyn` does NOT modify the head for 2D slab (confirmed in code
    and by debug print: `epsR(G=0,G'=0)` is unchanged after fixwings)
  - The ε⁻¹ head stored in `eps0mat.h5` is **0.895** (the inverse of the
    printed ε head 1.152) — NOT 1.152 as I previously confused
  - `vcoul(G=0)` in the sigma loop is **0.449 a.u.** — the bare Coulomb at
    the shifted q-point, NOT the mini-BZ average (315 a.u.)
  - The GN pole is valid: ω̃² = +40.2 Ry², ω̃ = 6.34 Ry
  - The on-shell head contribution to Σ^c is **~0.5 meV** — essentially zero

### Why GWJAX's head correction is wrong for GN-GPP

BGW treats every (G,G') element identically. The head (G=G'=0) gets:
  - Its own GN pole from the raw ε⁻¹ at the shifted q-point
  - Multiplied by `vcoul(G=0)` = v(q₀) ≈ 0.45 a.u. (small)
  - Net contribution: negligible

GWJAX's `head_correction.py` instead:
  - Fits a GN pole from mini-BZ averaged ⟨W⟩ and ⟨v⟩ (315 a.u.)
  - Applies as a ±2.5 eV diagonal shift
  - This is ~5000× larger than what BGW computes for the same element

The mismatch is not in the GN fit formalism (poles, signs agree when using
the same inputs). It's that **BGW uses v(q₀) ≈ 0.45 while GWJAX uses
⟨v⟩ ≈ 315** for the head element. In BGW's accounting, the large q→0
Coulomb divergence enters through the mini-BZ averaged vcoul in the **bare
exchange** Σ_x (via `vcoul_generator`), not through the correlation Σ_c.

### Implications

  - `apply_head_diagonal` should remain **false** — the correction is
    physically incorrect for comparison with BGW's GN-GPP
  - The COHSEX path is different: `apply_head_correction()` adds the head
    to V and W symmetrically, then COH = ½(W−V) cancels most of it.
    With correct ⟨W⟩ overrides this gives 46 meV MAE — working correctly.
  - The 1.8 eV GN-PPM body error is the real remaining problem, likely
    from the ISDF PPM extrapolation mixing poles across G-vectors.

### Static COHSEX results (unchanged from prior session)

  | Variant | MAE | max|Δ| |
  |---------|-----|--------|
  | S-tensor default | 0.165 eV | 0.213 eV |
  | **BGW head override** | **0.046 eV** | **0.103 eV** |

### Template/infrastructure fixes

  - `use_kihdat` (was misspelled `use_kih_dat`), needs `dont_use_vxcdat`
  - `no_t_rev = .true.` added to all QE templates
  - `apply_head_diagonal` added to cohsex.in parser (was missing)

## Open questions (as of 2026-04-05)

- **GN-PPM body error (1.8 eV)**: The ISDF body PPM extrapolation fails for
  MoS2 1×1 but works for CO (0D, 50 meV). The difference: in 2D, G=0 is
  zeroed from the body, so the ISDF PPM fits poles to an incomplete W^c. In
  0D, G=0 is finite and included. Possible fix: full-frequency integration
  in the ISDF basis (static screening is accurate; the PPM extrapolation is
  the problem).
- The prior 91 meV GN-PPM result (head_fix_test) used a different WFN.h5
  (160 bands) from `tests_isdf/`. Cannot be compared to the current 90-band
  calculation.
- 0D has a minor G=0 double-count in `compute_sqrt_vcoul_0d` (CO still
  passes at 3 meV due to large vacuum cell).
- k-grid convergence study needed (3×3, 6×6).

## 2026-04-04: Head correction separated from ISDF body — MoS2 1×1 results

- **Implementation** (see `gn_bug_plan.md` for derivation):
  - `gw/head_correction.py`: New module — scalar GN fit for q=0 G=0 head.
  - `gw/ppm_sigma.py`: Head no longer added to ISDF body W in PPM extraction.
  - `gw/gw_jax.py`: Head GN diagonal correction computed; optionally applied
    via `apply_head_diagonal = true` in cohsex.in (default: false).
  - `file_io/sigma_output.py`: New `sig_c_head` column in sigma_freq_debug.dat.

- **MoS2 1×1 Γ-only result** (run: `runs/mos2_1x1/head_fix_test/`):
  - Head GN fit: Ω_h = 1.145 Ry, R_h = 141.0 Ry·a.u., shift ±2.386 eV/state.
  - **Body only (head removed from ISDF W): MAE = 0.091 eV vs BGW. max|Δ| = 0.126 eV.**
  - Body + separate head diagonal:          MAE = 2.295 eV vs BGW (worse — double-counts).
  - Note: this comparison used a different WFN.h5 (160 bands) and different BGW
    run from `tests_isdf/`. The 91 meV result cannot be directly compared to the
    fresh 90-band `MoS2_1x1_full_workflow` runs.

- **Tests pass**: `uv run python -m pytest -q` → 2 passed.

## Known issues

- **pw2bgw GPU segfault (workaround)**: `MPICH_GPU_SUPPORT_ENABLED=0` for all pw2bgw.
- **cuFFT scratch OOM on 40 GB A100 (workaround)**: `memory_per_device_gb = 28`.
- **JAX 0.9.0 multi-GPU segfault (workaround)**: Use Shifter container, not uv.

## 2026-04-02: Initial matched runs

- Set up CO (0D, Γ-only) and MoS2 (2D, 3×3) with matched BGW + gw_jax parameters.
- CO: BGW and gw_jax Cor agree to 3 meV. **PASS.**
- MoS2: constant ±2.4 eV offset in Cor. **FAIL.**
  - Offset is the same at all 9 k-points in the 3×3 grid.
  - ISDF SoS agrees with gw_jax, not BGW, in both 0D and 2D.
- Confirmed environment workarounds: pw2bgw GPU segfault, cuFFT OOM, JAX multi-GPU.
