# Si 4x4x4 nosym Test — SymMaps Rotation Bug Confirmation

**Date**: 2026-04-05
**Run**: `runs/Si/02_si_4x4x4_nosym/`
**Purpose**: Determine whether the 300-1600 meV off-axis k-point errors in
Si 3D are caused by SymMaps wavefunction rotation or by the ISDF/PPM pipeline.

## Method

Ran the identical Si 4x4x4 calculation with `nosym = .true.` in QE, producing
all 64 k-points directly (no IBZ reduction, no symmetry unfolding by LORRAX).
Compared LORRAX vs BGW at all 64 k-points.

Previous run: `force_symmorphic = .true.` (12 symmetries, 8 IBZ k-points,
LORRAX unfolds to 64 via SymMaps).

## Results

### GN-PPM Corp (BGW vs LORRAX)

| Condition | MAE (all k) | Max error | K-points compared |
|-----------|-------------|-----------|-------------------|
| **nosym** | **12 meV** | 82 meV | 64 (all uniform) |
| sym (Gamma) | 5 meV | 10 meV | 1 |
| sym (high-sym lines) | 4-16 meV | 65 meV | 4 |
| sym (off-axis) | **300-765 meV** | 1627 meV | 3 |

### COHSEX (BGW vs LORRAX)

| Condition | Corp MAE (all k) | SX MAE | COH MAE |
|-----------|------------------|--------|---------|
| **nosym** | **54 meV** | 57 meV | 5 meV |
| sym (Gamma only) | 52 meV | — | — |
| sym (off-axis) | 200-700 meV | — | — |

### Per-k-point COHSEX Corp MAE (nosym)

All 64 k-points fall in the range 52-56 meV. No outliers. The error is
dominated by screened exchange (57 meV average), while the Coulomb hole
matches BGW to 5 meV. This is the ISDF basis approximation error, not a
code bug.

### Per-k-point GN-PPM Corp MAE (nosym)

Range: 4-24 meV across all 64 k-points. Slightly higher at zone-boundary
points (0.5, 0.5, 0.5) where the PPM pole structure is more complex, but
no catastrophic errors anywhere. Max single-band error: 82 meV.

## Conclusions

1. **The 300-1600 meV off-axis errors are entirely from SymMaps rotation.**
   With nosym (no rotation needed), all k-points show uniform ~12 meV (PPM)
   / ~54 meV (COHSEX) agreement with BGW.

2. **The baseline ISDF error is ~54 meV for COHSEX, ~12 meV for GN-PPM.**
   This is k-point independent and comes from the ISDF basis approximation.
   The COHSEX error is dominated by screened exchange; the Coulomb hole is
   accurate to 5 meV.

3. **Workaround for production**: use `nosym = .true.` in QE until the
   SymMaps G-vector rotation bug is fixed. Cost: 8x more k-points in QE
   (64 vs 8 IBZ), but LORRAX runtime is dominated by ISDF fitting and
   screening, which scale with nq (same either way).

## SymMaps bug: what to investigate

The bug is in the G-vector mapping during wavefunction rotation:
$u_{n,Sk}(G) = U_\text{spinor} \cdot u_{n,k}(S^{-1}G - G_\text{shift})$

The spinor matrices U_spinor are verified correct. The issue is likely in:
- `src/common/symmetry_maps.py`: `find_symmetry_ops_simple`, G_shift computation
- `src/common/load_wfns.py`: how `irk_to_k_map` / `irk_sym_map` apply the rotation

The error depends on the specific k-point being rotated, not the symmetry
operation itself (the same syms 0-5 produce correct results for some k-points
and wrong results for others).

## Multi-host JAX crash

Running with `-N 1 -n 4` (4 processes, 1 GPU each) crashes in
`extract_gn_ppm_parameters_from_Wc` when calling `jax.device_get()` on
sharded arrays with 64 k-points. The fix is to replace `jax.device_get()`
with `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)`,
matching the pattern in `file_io/tagged_arrays.py:_to_host()`.

Workaround: use `-N 1 -n 1 --gres=gpu:4` (single process, 4 GPUs).
