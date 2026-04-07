# Changelog

## 2026-04-06: First-principles memory model initiative

**Report**: `reports/memory_model_assay_2026-04-06/report.md`

Goal: build a predictive memory model for every stage of the LORRAX GW pipeline
where every GPU buffer > 1 MB is identified by name, shape, and purpose, and the
sum reproduces XProf peaks to < 10%. The practical target: set `memory_per_device_gb`
in `cohsex.in` and have `compute_optimal_chunks()` derive chunk sizes that actually
use that budget without OOM, enabling 10x10x10 Si on 16 GPUs and larger.

Previous assay (`runs/Si/04_si_4x4x4_memory_assay/`) invalidated:
- All measurements were single-process (`-n 1`), not production multi-process
- Only covered stages 1-2 (load_wfns), not pair density/ZCT/solve/gather
- No XProf traces — only aggregate `memory_stats()` numbers
- Produced phenomenological multipliers ("9x shard") instead of buffer inventories

Plan: instrument all stage boundaries, run systematic sweeps in multi-process mode
with XProf traces, parse buffer-level attribution, derive and validate formulas.
Added revalidation notice to `docs/MEMORY_MODEL.md`.

## 2026-04-06: Imported GN-PPM profiling documentation

- Added the profiling workflow note at `agents_xprof.md`
- Copied the profiling report and xprof trace guide into `reports/ppm_sigma_profiling_2026-04-05/`
- No raw trace bundles or profile outputs were copied from the profiling sandbox

## Current status (2026-04-05)

All work is on `main`. Branches `agent/fix-multihost-device-get` and
`agent/fix-improper-spinor` merged and pushed.

### Static COHSEX: working for both 2D and 3D

| System | Grid | MAE vs BGW Corp | Report |
|--------|------|-----------------|--------|
| MoS2 (2D) | 3×3 | **67 meV** | `reports/mos2_kgrid_convergence_2026-04-05/` |
| MoS2 (2D) | 3×3 nosym | **71 meV** | `runs/MoS2/02_mos2_3x3_nosym/` |
| MoS2 (2D) | 4×4 | **73 meV** | same |
| Si (3D) | 4×4×4 nosym | **54 meV** (all k) | `reports/si_nosym_2026-04-05/` |
| Si (3D) | 4×4×4 sym | **52 meV** (Γ only) | `reports/3d_coulomb_si_444_2026-04-05/` |

The ~54-67 meV COHSEX error is k-grid independent and uniform across all
k-points when symmetry is disabled. The error is dominated by screened exchange
(57 meV), while the Coulomb hole matches to 5 meV. Source is ISDF basis
approximation, not symmetry or wing corrections.

### GN-PPM: ~1 eV body error in 2D MoS2, 12 meV uniform in 3D Si nosym

| System | Grid | MAE vs BGW Corp | Report |
|--------|------|-----------------|--------|
| MoS2 (2D) | 3×3 | **1324 meV** | `reports/mos2_kgrid_convergence_2026-04-05/` |
| MoS2 (2D) | 3×3 nosym | **1153 meV** | `runs/MoS2/02_mos2_3x3_nosym/` |
| MoS2 (2D) | 4×4 | **1019 meV** | same |
| Si (3D) | 4×4×4 nosym | **12 meV** (all k) | `reports/si_nosym_2026-04-05/` |
| Si (3D) | 4×4×4 sym | **5 meV** (Γ only) | `reports/3d_coulomb_si_444_2026-04-05/` |

The 2D GN-PPM body error improves 25% with denser grid but remains ~1 eV. The
ISDF PPM pole extraction (W(0), W(iωp) → Ω, B per (q,μ,ν)) is the suspected
source — the ISDF basis mixes G-vector channels so individual poles are less
well-defined than in BGW's plane-wave basis. The 3D Si nosym result (12 meV
uniform at all k-points) shows the PPM machinery works; the 2D error may be
related to how the G=0 head exclusion interacts with the ISDF body PPM fit.

**Key code locations:**
- PPM extraction: `src/gw/minimax_screening.py:extract_gn_ppm_parameters_from_Wc`
- PPM sigma integration: `src/gw/ppm_sigma.py:compute_sigma_c_ppm_omega_grid`
- BGW PPM: `Sigma/mtxel_cor.f90` (GN pole fit + sigma evaluation)
- BGW fixwings: `Common/fixwings.f90` (wing rescaling for ε⁻¹ compatibility)
- BGW avgcut logic: `Sigma/inread.f90:908-912` (10¹² for 3D semicond, 10⁻¹² otherwise)

### SymMaps wavefunction rotation: FIXED (improper spinor bug)

**Status: spinor fix merged and pushed (commit `0351c55`). Sigma errors
persist — the spinor fix corrects the wavefunction rotation but there is
an additional bug in the chi0/W/sigma pipeline.**

#### The bug

`get_spinor_rotations()` in `symmetry_maps.py` feeds the Cartesian rotation
matrix directly into the quaternion→SU(2) algorithm. For **improper** rotations
(mirrors, S₆, etc., where det(R) = −1), this is wrong. The quaternion method
assumes a proper rotation (det = +1) and produces a spurious result.

The physical reason: in the double group, inversion maps to the identity in
SU(2). An improper rotation S = I_inv · R_proper should have spinor U(S) =
U(R_proper), because U(I_inv) = I. So the correct procedure is to strip
the inversion (negate the matrix) before computing the spinor. This is exactly
what BGW does in `Common/spinor_symmetries.f90:63-69`.

#### The fix

One line added to `get_spinor_rotations()`:
```python
if np.linalg.det(R) < 0:
    R = -R
```

#### Why this produced the specific error pattern

Only 2 of the 12 symmorphic Si symmetries triggered the bug in the diagnostic:
syms 7 and 8 (mirrors). The other improper ops (sym 6 = pure inversion, syms
9-11 = S₆) were either:
- Pure inversion (sym 6): stripping gives identity → U=I either way
- Not used by the k-points in the diagnostic's test set

The mirrors have the property that −R_mirror is a C2 rotation with a nontrivial
spinor. Without the fix, these mirrors got U=I, producing ||O||²=1.0 instead
of 2.0 (exactly half the subspace overlap, because the spinor rotation accounts
for half the transformation).

#### Why this is a complete fix for all symmorphic symmetries

The diagnostic verified **every other component** of the rotation formula:

| Component | Verification |
|-----------|-------------|
| **K-point mapping** (IBZ→full BZ) | All 64 k-points correctly mapped; energies match < 0.001 meV |
| **G-vector rotation** (S@G + G_shift) | 100% G-vector match at all 64 k-points |
| **G-shift** (BZ wrapping) | Correct for all k-points, including wrap-around cases |
| **Spinor for proper rotations** (C2, C3) | Verified correct by overlap test: all proper-rotation k-points pass |
| **Spinor for improper rotations** | Was wrong (missing `det<0` negation); now fixed and verified |
| **Time-reversal** (complex conjugation for `sym >= ntran`) | Not used in Si `force_symmorphic` (all 64 k-points use spatial syms 0–11); verified separately that the MoS2 ntran=2 case works |
| **Coefficient reindexing** | Confirmed by perfect G-vector match: coefficients are placed at the correct G-vector positions after rotation |

After the fix, the Gram-matrix overlap test passes **all 64 k-points** of
Si 4×4×4 with `force_symmorphic` (12 symmorphic symmetries of Oh). The 2
k-points that show ||O||²≈0.67 in the 2-fold test are **not failures** —
they pass perfectly (||O||²=4.0, SVDs all 1.0) when the near-degenerate
bands are grouped into 4-fold manifolds, which is the physically correct
grouping (the C3 rotation mixes Kramers pairs from the same irreducible
representation of the little group).

The fix is general because the improper-stripping logic applies to **any**
space group: every improper symmetry in any crystal factors as inversion ×
proper rotation, and the SU(2) spinor of the inversion is always identity.
The quaternion→SU(2) algorithm is correct for proper rotations of any axis
and angle. No space-group-specific logic is needed.

#### Sigma errors persist after wavefunction fix

Re-running sigma with the fixed code changes per-band values (3237 of 3840
bands differ by ~2-3 meV), confirming the fix has a physical effect. But the
**dominant 300-700 meV errors at off-axis k-points are unchanged** (MAE 349
meV before and after). Since the wavefunctions are now verified correct and
the nosym baseline is 12 meV, there must be a separate bug in how the
chi0/W/sigma pipeline uses symmetry-unfolded wavefunctions. Candidates:
- ISDF pair density assembly with rotated wavefunctions
- Q-point folding / symmetry weight handling in the screening
- FFT-box G-vector indexing when placing rotated coefficients for chi0

**Workaround: `nosym=.true.` is still required** for production calculations
until the screening pipeline bug is found.

**Remaining caveat:** non-symmorphic symmetries (fractional translations
τ ≠ 0) are not yet handled. The phase factor e^{−iG·τ} is computed but
currently zero for all symmorphic operations. This was already a known
limitation — the `force_symmorphic` workaround in QE removes all non-
symmorphic operations from the symmetry group.

#### Diagnostic scripts

Located in `runs/Si/02_si_4x4x4_nosym/debug_symmaps/`:
- `test_actual_symmaps.py`: full 64-k-point test using LORRAX's real SymMaps
  class. Run via Shifter: `srun ... $SHIFTER python3 -u test_actual_symmaps.py`
- `diagnose_4_bad.py`: detailed per-G-vector and per-band analysis of specific
  failing k-points
- `test_wfn_rotation.py`: standalone reimplementation (useful for understanding
  the algorithm but has subtle differences from the real code)

### BGW `cell_average_cutoff` / mini-BZ averaging

BGW's `Sigma/inread.f90` lines 908-912 auto-set `avgcut`:
- **3D semiconductor** (`TRUNC_NONE` + `SCREEN_SEMICOND`): `avgcut = 1/TOL_ZERO ≈ 10¹²` Ry
  → MC-averages vcoul at EVERY q-point, fixwings rescales ε⁻¹ wings at EVERY q-point
- **Everything else** (slab, wire, box, metals): `avgcut = TOL_ZERO ≈ 10⁻¹²` Ry
  → MC averaging and fixwings only at exact q=0

LORRAX now MC-averages v(q, G=0) for all nonzero q in 3D (commit `ef38ce3`),
matching BGW's head treatment. This reduced Si Γ MAE from 194→52 meV. LORRAX
does NOT rescale ε⁻¹ wings (no equivalent to fixwings for body elements), but
this appears to be unnecessary for the ISDF approach where W is computed via
the Dyson equation rather than ε⁻¹ × v.

User can override BGW's avgcut via `cell_average_cutoff X.X` in sigma.inp.

---

## 2026-04-05: Si/MoS2 nosym tests + SymMaps diagnostic

- **Report**: `reports/si_nosym_2026-04-05/report.md`
- **Runs**: `runs/Si/02_si_4x4x4_nosym/`, `runs/MoS2/02_mos2_3x3_nosym/`
- **Diagnostic**: `runs/Si/02_si_4x4x4_nosym/debug_symmaps/test_wfn_rotation.py`

Si nosym confirms SymMaps rotation bug (uniform 12/54 meV vs 300-1600 meV
off-axis with sym). MoS2 nosym confirms 2D errors are NOT from symmetry.
Diagnostic script identifies 11 failing k-points clustered at irk=1,2,7 —
not a universal formula bug, specific to certain IBZ k-points.

Also applied multihost JAX fix: `jax.device_get()` → `process_allgather()`
in `minimax_screening.py` (merged to LORRAX `main`, pushed).

## 2026-04-05: MoS2 k-grid convergence study

- **Report**: `reports/mos2_kgrid_convergence_2026-04-05/report.md`

Compared MoS2 3×3 and 4×4 for both COHSEX and GN-PPM. COHSEX error is
k-grid independent (~70 meV). GN-PPM improves 25% (1324→1019 meV). New
JIT-optimized PPM code verified to reproduce old results to <1 meV.

## 2026-04-05: 3D Coulomb implementation and Si 4×4×4

- **Report**: `reports/3d_coulomb_si_444_2026-04-05/report.md`

Implemented sys_dim=3 Coulomb, identified non-symmorphic symmetry issue
(workaround: `force_symmorphic = .true.`), added MC-averaged v(q, G=0) for 3D.
Resolved CH vs CH' comparison methodology error (the ~2.4 eV MoS2 "offset"
from April 2 was comparing the wrong sigma_hp.log column).

## 2026-04-05: GN-PPM profiling and JIT optimization

GN-PPM sigma: 97.5% was XLA compilation. JIT-compiling tau-node loop: 421s→105s.
Minimax quadrature tables now shipped as assets for instant lookup.

## 2026-04-04: Head correction module

Body-only MoS2 1×1: 91 meV. Static COHSEX with BGW head: 46 meV.
`apply_head_diagonal` should remain false for GN-PPM comparisons with BGW.

## 2026-04-02: Initial matched runs

CO (0D): 3 meV. MoS2 3×3: originally reported as 2.4 eV offset (later
identified as CH vs CH' comparison error — actual COHSEX error was 67 meV).

## Known environment issues

- `pw2bgw` GPU segfault → `MPICH_GPU_SUPPORT_ENABLED=0`
- cuFFT OOM on 40 GB A100 → `memory_per_device_gb = 28`
- Multi-GPU JAX → use Shifter container, not uv
- **Multi-process JAX distributed array crash**: Running LORRAX with multiple
  OS processes (e.g. `srun -n 4`, each process gets 1 GPU) triggers
  `jax.device_get()` crashes on sharded arrays. This is a **multi-process**
  issue, not a multi-node issue — even single-node with `-n 4` crashes. With
  small k-grids (e.g. 8 IBZ points) arrays may fit on one device and the
  problem does not appear; with larger grids (e.g. 64 k-points nosym) arrays
  are sharded across devices and `device_get()` fails because it cannot access
  non-addressable devices.
  - **Crash sites**: `minimax_screening.py:extract_gn_ppm_parameters_from_Wc`
    (line 778) and `extract_gn_ppm_parameters` (line 719), both calling
    `jax.device_get()`.
  - **Fix**: Replace `jax.device_get(arr)` with
    `jax.experimental.multihost_utils.process_allgather(arr, tiled=True)`,
    matching the existing `_to_host()` pattern in `file_io/tagged_arrays.py`
    (lines 26-34).
  - **Workaround**: Use `-N 1 -n 1 --gres=gpu:4` (single process, 4 GPUs on
    one node). JAX sees all 4 GPUs within the single process, so no arrays are
    distributed across process boundaries.
- `centroid.kmeans_isdf` does not accept `-i cohsex.in`; run from CWD

- Added BGW-style non-symmorphic coefficient phases to `sources/lorrax/src/common/symmetry_maps.py` on branch `agent/non-symmorphic-phases`.
- Preserved the pre-existing time-reversal path while applying `exp[-i (G_target + kg0) · tau]` for spatial symmetries with nonzero `tnp`.
- Added `tests/test_symmetry_maps_nonsymmorphic.py`; `uv run python -m pytest -q` now passes (`10 passed, 1 warning`).
- Verified on `runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5` that `44/64` full-zone k-points now carry a nontrivial non-symmorphic phase.
