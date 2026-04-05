# Changelog

## 2026-04-04: S-tensor head error identified — BGW heads give 46 meV COHSEX

- **Report**: `reports/cohsex_head_investigation_2026-04-04/report.md`
- **Run**: `runs/MoS2_1x1_full_workflow/` — 6 variants total.
- **Root cause**: GWJAX's S-tensor computes wcoul0 = 42.7 a.u. vs BGW's mini-BZ
  average of 55.6 a.u. (23% error). The macroscopic dielectric model
  `ε(q) = 1 - v(q) qᵀSq` breaks down when the mini-BZ covers the full BZ (1×1).
- **Fix**: Override with BGW mini-BZ heads from `frequency_dependence 3` epsilon.out
  (`vhead`, `whead_0freq`). See `PARSE_OUTPUTS.md` for parser.

  | Variant | MAE | max|Δ| |
  |---------|-----|--------|
  | S-tensor default | 0.165 eV | 0.213 eV |
  | 800 centroids | 0.188 eV | 0.389 eV |
  | **BGW head override** | **0.046 eV** | **0.103 eV** |

- **Key lessons**: (1) Always extract BGW mini-BZ heads for GWJAX comparisons.
  (2) Run freq_dep 3 epsilon even for COHSEX — only it prints head diagnostics.
  (3) Compare sigCOH against BGW CH' (primed), never CH. (4) More centroids
  don't fix head errors. (5) 0D has a minor G=0 double-count bug.
- **GN-GPP**: MAE = 1.836 eV — this is entirely in the PPM body, NOT heads.
  BGW head overrides don't change the body values at all (head is excluded from
  ISDF body in PPM path). The static decomposition (sex_0, coh_0) from the same
  PPM run matches BGW COHSEX to 165 meV (46 meV with correct heads), confirming
  the screening is fine. The error is in the PPM frequency extrapolation itself.
- **Prior 91 meV result was from a different WFN** (160 bands, different eigenvalues)
  in `tests_isdf/`. Cannot be compared to the current fresh 90-band calculation.
- **Template fixes**: `use_kihdat` (was misspelled), `dont_use_vxcdat`, `no_t_rev`.

## Open questions (as of 2026-04-04)

- **GN-PPM body error (1.8 eV)**: The PPM fit from W(0) and W(iωp) extrapolates
  poorly to real frequencies for certain bands (worst: band 25, 4.5 eV). This is
  independent of heads and needs investigation of the ISDF PPM extraction itself.
- Run 3×3 k-grid to test whether the PPM error improves with more q-points.
- Understand why body-only (head removed) matches BGW to 91 meV but adding the
  separate head diagonal back gives +2.3 eV error. The head GN correction derived
  from gn_bug_plan.md appears to double-count what BGW already includes in its
  plane-wave W^c body. Need to understand whether the "correct" physics requires
  the head correction or whether the body-only path is self-consistent.
- Set up k-grid convergence study for MoS2 (6×6, 9×9 variants).
- CO molecule comparison with head fix (should remain ~3 meV).

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
  - Conclusion: Removing the head from the ISDF body is the essential fix.
    The separate diagonal head correction overshoots, suggesting that BGW's Cor'
    already includes the head through its plane-wave W^c, and the ISDF body without
    head naturally reproduces this.
  - Plot: `runs/mos2_1x1/head_fix_test/head_fix_comparison.png`

- **Comparison to old result**: Prior GWJAX (head in ISDF body) had MAE ≈ 2.4 eV
  vs BGW for MoS2 1×1. After removing head from body: MAE = 0.091 eV. This is a
  **26× improvement** and within the 100 meV target.

- **Tests pass**: `uv run python -m pytest -q` → 2 passed.
- **Platform**: Local WSL2, RTX 5070 8 GB, JAX 0.9.1 (GPU confirmed).
  Runtime: ~311s for 1×1 with 600 centroids and 80 bands (PPM sigma 99% of time).

## Known issues

- **2D Cor offset (RESOLVED to 91 meV)**: Was ±2.4 eV for MoS2 vs BGW. Cause: the
  q=0 G=0 Coulomb head was added to the ISDF body W via `apply_head_correction()`,
  distorting the ISDF representation. Fix: remove head from ISDF body in
  `ppm_sigma.py`. Residual 91 meV may be from wings or ISDF basis completeness.
  **The separate diagonal head correction described in `gn_bug_plan.md` should NOT
  be applied by default** — it double-counts the head that BGW already has in W^c.
  Available via `apply_head_diagonal = true` for experimentation.
- **pw2bgw GPU segfault (workaround)**: `MPICH_GPU_SUPPORT_ENABLED=0` for all pw2bgw.
- **cuFFT scratch OOM on 40 GB A100 (workaround)**: `memory_per_device_gb = 28`.
- **JAX 0.9.0 multi-GPU segfault (workaround)**: Use Shifter container, not uv.

## 2026-04-02: Initial matched runs

- Set up CO (0D, Γ-only) and MoS2 (2D, 3×3) with matched BGW + gw_jax parameters.
- CO: BGW and gw_jax Cor agree to 3 meV. **PASS.**
- MoS2: constant ±2.4 eV offset in Cor. **FAIL.**
  - Offset is the same at all 9 k-points in the 3×3 grid.
  - ISDF SoS agrees with gw_jax, not BGW, in both 0D and 2D.
  - `write_no_head_vw = true`: head accounts for ~0.3 eV of the 2.4 eV. Not the
    whole story.
- Confirmed environment workarounds: pw2bgw GPU segfault, cuFFT OOM, JAX multi-GPU.
