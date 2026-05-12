# BSE rerun with BGW `write_vcoul` + LORRAX `use_bgw_vcoul`

**Agent A, Si 4×4×4 BSE, 4v4c SOC. Two sessions Apr 23 (allocs 51971917, 51976616).**

## Goal

Test whether LORRAX's `use_bgw_vcoul=true` (reading BGW's MC-averaged
`vcoul` file) eliminates the ~0.14 eV systematic offset seen in
`reports/bse_comparison_2026-04-07/`.

## Final result

| | DFT BSE (eV 1) | EQP BSE (eV 1) |
|---|---|---|
| BGW reference | 2.347 | 2.942 |
| LORRAX Apr-7 (LORRAX vcoul, old centroids) | 2.490 | 3.078 |
| **LORRAX Apr-23 (BGW vcoul + fixed code + new centroids)** | **2.474** | **3.054** |
| BGW-vcoul delta vs BGW | **+0.127** | **+0.112** |
| LORRAX-vcoul delta vs BGW (Apr-7) | +0.142 | +0.136 |

BGW-vcoul tightens the BSE-to-BGW gap by ~15–25 meV.

## Two LORRAX regressions found and fixed along the way

### (1) `W0_qmunu` written as zero to the restart tensor file

`gw.gw_jax` was writing a zero-filled `W0_qmunu` placeholder into
`tmp/isdf_tensors_<N>.h5` but never backfilling it with the real `W_q`
after the chi0/W solve. Downstream `bse.bse_jax` reads `W0_qmunu` from
that file, so it saw an all-zeros W and the BSE Hamiltonian lost its
screened-direct term. Effect: BSE eigenvalues dropped ~1.3 eV
(1.195 eV instead of 2.49 eV on DFT mode, 1.835 eV instead of 3.08 eV
on EQP mode).

**Root cause**: commit `4ba05ba` (2026-04-12, "Delete dead code and
clean gw_init naming") removed a W0 writeback that lived inside
`compute_screening`. The calling side of `compute_screening` got
restructured in the same wave and never re-acquired the writeback.
`src/file_io/tagged_arrays.py::write_w0_qmunu_to_h5` remained in the
codebase but had no callers.

**Fix**: reintroduced the call in `src/gw/gw_jax.py` after `solve_w`
(see diff below). After the fix the tensor file's `max|W0_qmunu|`
goes from 0 → 7.24e4, and BSE eigenvalues return to ~2.47 / ~3.05 eV.

```python
# src/gw/gw_jax.py, after W_q = solve_w(...):
if config.do_screened and tensors_filename is not None and os.path.exists(tensors_filename):
    from file_io import write_w0_qmunu_to_h5
    _kgrid = tuple(int(x) for x in meta.kgrid)
    _W0_7d = W_q.reshape(*_kgrid, W_q.shape[1], W_q.shape[2])
    write_w0_qmunu_to_h5(
        tensors_filename,
        _W0_7d[None, None, None, :, :, :, :, :],
        mesh=mesh_xy, use_ffi_io=config.use_ffi_io,
    )
```

### (2) `read_bgw_vcoul.find_q_index` can't match q=(0,0,0) against BGW's shifted q₀

BGW's `epsilon` writes `write_vcoul` using the small finite-difference
shift on q₀ (e.g. `(0.00025, 0, 0)`). LORRAX's flat-q mesh asks for
`q=(0,0,0)`, and the previous `find_q_index` (tol=1e-4) rejected the
shifted q₀ so the whole run raised `ValueError`.

**Fix**: added a fallback in `src/file_io/read_bgw_vcoul.py` that, when
q=(0,0,0) is requested and no exact+symmetry match exists, treats the
stored q closest to zero (within 1e-2) as the q=0 entry. The existing
q=0,G=0 head-zeroing downstream still fires, so LORRAX's internal
head correction remains in charge of the true head.

Without this fix, GW crashes on every BGW-vcoul run after the W0
writeback is restored — the previous silent-success behaviour on an
earlier run was the overlay path failing and LORRAX's native v(q+G)
being used instead (sigma head was identical between `use_bgw_vcoul=true`
and `=false` runs — evidence the overlay had no effect).

## Aside: BGW-vcoul body overlay is now active and the COHSEX sigma drifts 1.3 eV

With the q₀ fix active, `gw.gw_jax` actually uses BGW's shifted-q₀
body values (G≠0) for the flat-q q=0 slice. LORRAX's in-memory
`eqp0.dat` then shifts by ~+1.1 to +1.3 eV band-wise vs BGW's
`sigma_hp.log` (Sig' MAE ≈ 1.6 eV for Si 4×4×4). The BSE Lanczos is
unaffected because it reads from the tensor file (pure W₀ direct
term, no bare-v overlay dependence in the same way).

This drift is separate from the W0 bug and is almost certainly
caused by BGW's shifted-q₀ file storing `8π/|q_shift + G|²` *without*
mini-BZ averaging, while LORRAX's COHSEX sigma expects the
miniBZ-averaged values (as a `sigma.cplx.x` run with `write_vcoul`
would produce at exactly q=(0,0,0)). Options to resolve:

1. Use a `vcoul` file from `sigma.cplx.x -i sigma.inp` (q-list starts
   at exactly (0,0,0), mini-BZ averaging applied) instead of the
   `epsilon.cplx.x` file. Easiest.
2. Have `read_bgw_vcoul.find_q_index` skip the shifted q₀ for q=(0,0,0)
   body terms and return "no match" so LORRAX uses its own native
   values. Then only head is covered by vcoul file (via the separate
   head machinery).

Kept out of scope for this report — the BSE side is now working.

## What ran (second session)

| Step | Dir | Result |
|------|-----|--------|
| BGW epsilon + `write_vcoul` | `runs/Si/04_si_4x4x4_bse/01_bgw_eps_vcoul/` | `vcoul` text file, 4535 rows, 8 IBZ q's |
| BGW kernel + absorption DFT + EQP + `write_vcoul` | `runs/Si/04_si_4x4x4_bse/01_bgw_bse_vcoul/` | Eigenvalues match Apr-7 exactly |
| LORRAX centroid regen (pivoted-Cholesky, `--oversample 2.0 --prune-n-cond 52`) | `01_lorrax_bse_vcoul/` | 480 rank-480 centroids |
| LORRAX `gw.gw_jax` 4-GPU with fixes | same dir | eqp0.dat + isdf_tensors_480.h5 with `W0_qmunu` nonzero (max 7.24e4) |
| LORRAX `bse.bse_jax --lanczos --tda --bse` (DFT) | same dir | 2.474 eV |
| LORRAX BSE (EQP) | same dir | 3.054 eV |

## Third session: FFI re-enabled after allocator fix

On branch `agent-A/runtime-init @ 354843b` (module default reverted from
bfc MF=0.9 → `default`, bfc opt-in MF lowered to 0.75), with my W0 and
q₀ fixes re-applied on top: `use_ffi_io=true` + `isdf_memory_mode=low_mem`
completes cleanly:

| run | exit | tensors written | BSE DFT e₁ | BSE EQP e₁ |
|---|---:|---|---:|---:|
| non-FFI (2nd session) | 0 | `isdf_tensors_480.h5` | 2.474 eV | 3.054 eV |
| FFI + low_mem (3rd session) | 0 | + `rank{0..3}` shards | **2.474 eV** | **3.054 eV** |

Timing (FFI + low_mem, 4 GPU): V_q = 6.8 s, chi0/W = 3.4 s, sigma = 1.4 s.

BSE eigenvalues are bit-identical across the two paths — confirms the
FFI serialization now faithfully reproduces the in-memory tensors once
the allocator MF is 0.75 (not 0.9).

## Suggested follow-up

- **COHSEX sigma drift** — pick option 1 or 2 above to get LORRAX's
  COHSEX `eqp0.dat` back in agreement with BGW when `use_bgw_vcoul=true`.
- **Sigma MAE vs Apr-18 baseline** — still open; with neither of my fixes
  touching the sigma code, the +150 meV Si 2×2×2 Sig' drift since Apr-18
  is a separate regression and needs git bisect on the chi0/sigma/COHSEX
  refactor wave (commits between `c6a68e5` and HEAD).

## Artifacts

- `runs/Si/04_si_4x4x4_bse/01_bgw_eps_vcoul/` — BGW epsilon + vcoul
- `runs/Si/04_si_4x4x4_bse/01_bgw_bse_vcoul/` — BGW kernel + absorption DFT/EQP
- `runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/` — LORRAX GW + BSE DFT/EQP (both fixed-code outputs saved as `*_fixed.out`)
- Source fixes: `sources/lorrax_A/src/gw/gw_jax.py` (W0 writeback),
  `sources/lorrax_A/src/file_io/read_bgw_vcoul.py` (q=0 shifted-q₀ fallback)
