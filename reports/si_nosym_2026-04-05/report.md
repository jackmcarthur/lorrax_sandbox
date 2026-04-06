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

## SymMaps bug: diagnostic script results

**Script**: `runs/Si/02_si_4x4x4_nosym/debug_symmaps/test_wfn_rotation.py`

A standalone diagnostic compares rotated IBZ wavefunctions against
directly-computed nosym wavefunctions. For each full BZ k-point, it:
1. Rotates the IBZ wavefunction using the symmetry operation (G-vector
   rotation + spinor rotation)
2. Computes the overlap matrix O_ij = <nosym_i | rotated_j> within
   degenerate subspaces
3. Checks that ||O||^2 = ndeg (perfect subspace match)

### Results

| Metric | Value |
|--------|-------|
| Total k-points tested | 55 (of 64, 9 are identity) |
| Excluded (incompatible G-sphere) | 2 (irk=10, irk=12) |
| Testable | 53 |
| **GOOD** (err < 0.01) | **41** |
| **BAD** (err >= 0.01) | **11** |

### Key findings

1. **G-vector rotation is PERFECT.** 100% G-vector match fraction for all
   53 testable k-points. The issue is NOT in G-sphere construction or
   G-vector indexing.

2. **2 IBZ k-points excluded** (irk=10: k=0.25,0.25,-0.25; irk=12:
   k=0.25,0.25,0.25) because their G-spheres are incompatible between
   sym and nosym WFN files — different k-point representations lead to
   different G-sphere cutoff boundaries.

3. **Failures cluster at specific IBZ k-points**, not specific symmetry ops:
   - irk=1 (Gamma): some rotations fail
   - irk=2: some rotations fail
   - irk=7: some rotations fail
   - Error magnitudes: ||O||^2 = 0.667 or 1.0 (expected 2.0 for pairs)

4. **Many nontrivial rotations PASS correctly.** C2, C3, mirror operations
   all produce correct wavefunctions at other IBZ k-points.

5. **Conclusion**: The rotation formula itself is not universally broken.
   The issue is specific to certain high-symmetry IBZ k-points, possibly
   related to how degenerate subspaces are handled during rotation (band
   mixing within degenerate manifolds at Gamma, etc.).

### What to investigate next

- Whether the rotation failures at irk=1,2,7 are from degenerate band
  mixing (the diagnostic uses energy-based degeneracy grouping; higher
  degeneracy at Gamma may need larger subspace windows)
- Whether LORRAX's SymMaps G_shift computation matches the diagnostic's
  independent implementation
- Whether the wavefunction loader in `load_wfns.py` applies the rotation
  in the same order as the diagnostic

## MoS2 nosym test

**Run**: `runs/MoS2/02_mos2_3x3_nosym/`

MoS2 3x3 with `nosym=.true.` shows COHSEX 71 meV (vs 67 meV sym) and
GN-PPM 1153 meV (vs 1324 meV sym). The errors are essentially unchanged,
confirming that **MoS2 errors are NOT from symmetry rotation**. The ~70 meV
COHSEX and ~1 eV GN-PPM errors are intrinsic to the ISDF/PPM treatment of
the 2D system.

## Multi-host JAX fix

The `jax.device_get()` crash with multi-process runs has been fixed in
LORRAX (branch `agent/fix-multihost-device-get`, merged to `main`).
Replaced `jax.device_get(arr)` with
`jax.experimental.multihost_utils.process_allgather(arr, tiled=True)` in
`minimax_screening.py`, matching the existing `_to_host()` pattern.

Previous workaround: use `-N 1 -n 1 --gres=gpu:4` (single process, 4 GPUs).
