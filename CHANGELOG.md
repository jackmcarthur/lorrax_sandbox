# Changelog

## Current status (2026-04-05)

All work is on `main`. No outstanding branches.

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

### CONFIRMED: k-point-dependent errors are entirely from SymMaps rotation bug

Si 3D with `force_symmorphic` (12 syms) shows **300-1600 meV errors** at
off-axis k-points requiring nontrivial symmetry rotations. Running the **same
system with `nosym=.true.`** (64 direct k-points, no unfolding) gives
**uniform 12 meV GN-PPM / 54 meV COHSEX** at ALL k-points. This conclusively
proves the off-axis errors are from the SymMaps wavefunction rotation, not
from the ISDF/PPM/Coulomb pipeline.

**What's been checked:**
- Spinor U_spinor matrices match BGW `Common/spinor_symmetries.f90` convention
  (improper → proper before SU(2); inversion → identity). Confirmed correct.
- The SAME set of symmetry operations (syms 0–5) are used for both good
  k-points (e.g. ik_ibz=1, k=(0,0,0.25)) and bad k-points (e.g. ik_ibz=3,
  k=(0,0.25,0.25)). So the operations themselves aren't broken — the error
  depends on the specific k-point being rotated.
- No time-reversed operations are used (verified: all 64 full-BZ k-points
  use spatial syms 0–11 only).

**Diagnostic script results** (`runs/Si/02_si_4x4x4_nosym/debug_symmaps/test_wfn_rotation.py`):
- G-vector rotation and G-sphere matching is PERFECT (100% match for all testable k-points)
- 2 IBZ k-points (irk=10, irk=12) excluded due to incompatible G-spheres
  (negative k-components: k=0.25,0.25,-0.25 vs 0.25,0.25,0.75)
- Of 53 testable k-points: **41 GOOD, 11 BAD** (err >= 0.01 in subspace overlap)
- The 11 failures cluster at specific IBZ k-points (irk=1,2,7) with specific
  error magnitudes (||O||^2=0.667 or 1.0 instead of 2.0)
- Many nontrivial rotations (C2, C3, mirrors) PASS correctly
- **Conclusion: NOT a universal rotation formula bug.** The issue is at specific
  high-symmetry IBZ k-points, possibly related to degenerate subspace handling

**MoS2 nosym test** (`runs/MoS2/02_mos2_3x3_nosym/`):
- COHSEX: 71 meV (vs 67 meV with sym) -- essentially unchanged
- GN-PPM: 1153 meV (vs 1324 meV with sym) -- essentially unchanged
- **MoS2 errors are NOT from symmetry rotation.** The ~70 meV COHSEX and
  ~1 eV GN-PPM errors are intrinsic to the ISDF/PPM treatment of the 2D system.

**What's NOT been checked:**
- Whether the rotation failures at irk=1,2,7 are from degenerate band mixing
  (the diagnostic uses a simple energy-based degeneracy grouping; the actual
  rotation may mix bands differently at high-symmetry points)
- Whether the G_shift computation in SymMaps handles the BZ wrapping correctly
  for all IBZ k-points (the diagnostic reimplements this independently)

**This blocks production 3D calculations with symmetry.** Workaround: use
`nosym=.true.`. See per-k breakdown in
`reports/3d_coulomb_si_444_2026-04-05/si_gnppm_per_kpoint.png`.

**Key code:** `src/common/symmetry_maps.py` (SymMaps class, particularly
`find_symmetry_ops_simple`, `create_kpoint_symmetry_map`, and how
`irk_to_k_map` / `irk_sym_map` are used by the wavefunction loader in
`src/common/load_wfns.py`).

**Key code locations:**
- SymMaps: `src/common/symmetry_maps.py` (k-point unfolding, G-shift computation)
- BGW equivalent: `Common/genwf.f90` (wavefunction generation via symmetry)
- Wavefunction rotation: u_{n,Sk}(G) = U_spinor · u_{n,k}(S⁻¹G - G_shift)

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

## 2026-04-05: MoS2 nosym test + SymMaps wavefunction rotation diagnostic

- **Run**: `runs/MoS2/02_mos2_3x3_nosym/`
- **Diagnostic**: `runs/Si/02_si_4x4x4_nosym/debug_symmaps/test_wfn_rotation.py`

MoS2 3x3 nosym test confirms MoS2 errors are NOT from symmetry (71 meV COHSEX,
1153 meV GN-PPM, essentially unchanged from sym run). The ~70 meV COHSEX and
~1 eV GN-PPM errors are intrinsic to the ISDF/PPM treatment of 2D systems.

Wrote a standalone diagnostic script that compares rotated IBZ wavefunctions
against directly-computed nosym wavefunctions for Si 4x4x4. Key findings:
- G-vector rotation is PERFECT (100% match)
- 41 GOOD, 11 BAD out of 53 testable k-points
- Failures cluster at specific IBZ k-points (irk=1,2,7), NOT specific symmetry ops
- Many nontrivial rotations (C2, C3, mirrors) pass correctly
- The issue appears to be at specific high-symmetry IBZ k-points, not a universal
  rotation formula bug

Applied multihost JAX fix (already merged to LORRAX `main` on branch
`agent/fix-multihost-device-get`): replaced `jax.device_get()` with
`process_allgather()` in `minimax_screening.py`.

## 2026-04-05: Si nosym test — confirms SymMaps rotation bug

- **Report**: `reports/si_nosym_2026-04-05/report.md`
- **Run**: `runs/Si/02_si_4x4x4_nosym/`

Ran full QE→BGW→LORRAX pipeline with `nosym=.true.` (all 64 k-points computed
directly, no symmetry unfolding). Results:

| Method | nosym MAE (all k) | sym MAE (Gamma) | sym MAE (off-axis) |
|--------|-------------------|-----------------|---------------------|
| COHSEX Corp | **54 meV** | 52 meV | 200-700 meV |
| GN-PPM Corp | **12 meV** | 5 meV | 300-1600 meV |

The nosym errors are uniform across all 64 k-points, proving that the
300-1600 meV off-axis errors are entirely from the SymMaps wavefunction
rotation bug. The COHSEX error is dominated by screened exchange (57 meV SX,
5 meV COH), likely from ISDF basis approximation. Workaround: use nosym for
production until SymMaps is fixed.

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
