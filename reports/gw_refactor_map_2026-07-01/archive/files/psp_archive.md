# src/psp/archive/ — legacy PSP helpers (refactor-map notes)

Group: `src/psp/archive/{projector_pipeline,charge_density,build_projectors}.py` plus a
one-line `__init__.py` whose docstring says: *"Archived PSP helpers kept only for
historical/debug reference."*

Git history for the directory (last 3 commits touching it):

```
fe94752 P5 final: delete legacy WFNReader / PhdfWfnReader; cache wfn_transforms
5bba57e Move build_G_cart to dft_operators; archive charge_density.py
2a9a016 Migrate active PSP callers off legacy projector pipeline
```

So the archive is an explicit graveyard created during the P-series PSP migration.
`src/psp/dev_status.md` line 23 confirms: "`archive/` | legacy | Archived CPU-heavy
compatibility helpers (`build_projectors.py`, `projector_pipeline.py`)".

Grep scope used for all caller claims below:
`grep -rn --include='*.py' <name> src tests tools scripts` at repo root
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D` (pytest `testpaths = ["tests"]`
per `pyproject.toml:59-60`, so `src/psp/tests/` is NOT collected by pytest).

---

## 1. src/psp/archive/projector_pipeline.py (747 LOC)

### Purpose
Legacy CPU/NumPy-heavy nonlocal-pseudopotential (Kleinman–Bylander) projector
pipeline: builds β-projector rows Z(K) and their K-gradients ∇Z per species/ℓ/atom,
assembles V_NL(k) matrices and the nonlocal velocity commutator i[r, V_NL] (analytic
via ∇β and finite-difference variants). Superseded by the active path in
`psp/dft_operators.py` (`build_vnl_kdata`, `vnl_matrix_from_kdata`,
`vnl_velocity_from_kdata`, `build_Z_and_dZ`) + `psp/vnl_ops.py` +
`psp/radial/build_projectors_qe.py` (commit 2a9a016 "Migrate active PSP callers off
legacy projector pipeline").

Category: **archived legacy: V_NL / nonlocal-velocity builder (dead code)**.

### Import breakage (key evidence of deadness)
Lines 9-17 do relative imports:

```python
from .build_projectors_qe import (...)
from .radial_jax import spherical_jn_deriv_jax ...
```

Neither `build_projectors_qe.py` nor `radial_jax.py` exists in `psp/archive/`; they
live at `src/psp/radial/`. The module is therefore **unimportable in place**
(`ModuleNotFoundError` at import) — nothing can call it without fixing the imports.

### Function list
| Function | Role |
|---|---|
| `_accumulate_species_channel` (jit) | Contract one species/ℓ channel over atoms: Z ⊗ phase → P → Δ via E-block einsums |
| `Species`, `PseudoBundle`, `KBundle` (dataclasses) | Species grouping, precomputed E-blocks/radial tables, K-vector bundle |
| `group_assignments_by_pseudo` | Group atom assignments by `id(pseudo)` |
| `_infer_lmax_from_pseudo` | Max ℓ over PP_BETA entries |
| `prepare_pseudo_bundle` / `prepare_pseudo_cache` | Build E-blocks + radial spline tables per pseudo |
| `prepare_k_bundle` | K_cart + |K| bundle |
| `make_projector_rows` | Wrapper over `compute_type_projectors_real` (in radial/build_projectors_qe) |
| `_evaluate_Fprime_bessel` | F'_ℓ(q) = ∫ r³ β j'_ℓ(qr) dr via JAX Bessel kernels |
| `_evaluate_F_and_Fprime_for_l` | F and F' per (ℓ, β) — spline-derivative or Bessel mode |
| `build_beta_rows_with_grad` | Z rows + ∇Z rows (total/radial/angular parts) for all species/atoms/ℓ |
| `build_beta_rows_with_grad_components` | Near-copy of the above, also returns radial/angular parts explicitly |
| `compute_V_NL_velocity_k` | Analytic i[r, V_NL]: (3, nb, nb) via ∇β contraction against E blocks |
| `build_vnl_plan` | Per-species/per-ℓ plan dict: E blocks, β ids, radial tables, 4π/√Ω prefactor |
| `compute_V_NL_k_minimal` | Vectorized V_NL(k) from prebuilt plan (the "minimal" production path of its era) |
| `compute_V_NL_velocity_k_numeric` | Finite-difference velocity via Z(K±δ), Richardson extrapolation |

### Entry points / callers
`__all__` exports 12 names. Grep for every public name
(`compute_V_NL_velocity_k|compute_V_NL_k_minimal|build_vnl_plan|build_beta_rows_with_grad|projector_pipeline|group_assignments_by_pseudo|prepare_pseudo_bundle|prepare_pseudo_cache|prepare_k_bundle|make_projector_rows`)
across src/tests/tools/scripts: **zero hits outside this file**. No `python -m` usage.

### I/O
None directly. Consumes in-memory UPF pseudo objects (`pp_nonlocal.pp_beta`,
`pp_mesh.pp_r/pp_rab`) parsed elsewhere (`psp/upf`). Writes nothing.

### Dead suspects
- **Entire module** — zero external callers by grep (all 12 `__all__` names + module
  name); relative imports broken since archival, so it cannot even be imported.

### Redundancy suspects
- `build_beta_rows_with_grad` vs `build_beta_rows_with_grad_components` (lines 207 /
  310): ~95-line near-verbatim copies. They differ *physically*: the `_components`
  variant adds an `l*radial/q` term to `base_rad` (line 373) that the base version
  (line 271) lacks, and uses `q_safe = np.where(q>0,q,1.0)` (line 358) vs `1e-12`
  (line 254). Classic diverged copy-paste pair.
- `group_assignments_by_pseudo` + `prepare_pseudo_bundle`/`prepare_pseudo_cache` vs
  `build_vnl_plan` (line 484): two parallel species-grouping/precompute paths within
  the same file (bundle-dataclass style vs plan-dict style).
- Whole module duplicates the active `psp/dft_operators.py` vnl machinery
  (`build_vnl_kdata`/`extract_vnl_channel_data`/`build_Z_and_dZ`) and
  `psp/vnl_ops.py::vnl_velocity_matrix` (used by `psp/get_dipole_mtxels.py:92`).
- `compute_V_NL_velocity_k` (analytic) vs `compute_V_NL_velocity_k_numeric` (FD):
  old cross-check pair kept side by side.

### Weird code
- Lines 302-306: smuggles gradient components as attributes on a NumPy array
  (`Z_cat._dZ_rad = dZ_rad` inside `try/except`). `np.ndarray` forbids attribute
  assignment, so this silently does nothing — the "optional consumers" never get the
  data. Hypothesis: vestigial hack from before `_components` variant existed.
- `q_safe` regularizer inconsistency: `1e-12` (line 254) vs `1.0` (line 358) for the
  q→0 guard; with `1e-12` the `radial/q_safe` angular term can blow up at G=0 unless
  radial→0 faster.
- Line 373: extra `+ l*radial/q_safe` term present only in the `_components` copy —
  a sign that one of the two gradient formulas is wrong (or uses a different
  radial-function convention F vs F/q^l).
- Lines 459-463 sign-convention comment: "v^NL_i = (i/ħ)[V_NL, r_i] = −(∇_q+∇_q′)V_NL
  ... apply a minus sign, no extra i" — this is the p−vNL/BGW-convention territory
  flagged in project memory (`project_lorrax_velocity_sign`); the archived code
  carries the convention in a comment rather than in a named helper.
- `wfn_k` and `Gk_crys` parameters of both `build_beta_rows_with_grad*` functions are
  entirely unused (pure signature baggage).
- `__all__` (line 467) is defined mid-file *before* four of the functions it exports.
- `compute_V_NL_velocity_k` mutates its input `plan` dict (line 431:
  `plan[key]['fprime_mode'] = fprime_mode`) — side effect on caller state.
- Line 688: recovers reciprocal lattice B via `np.linalg.lstsq(K_crys, K_cart)`
  instead of taking bvec as an argument — fragile (fails if k-points are coplanar).

---

## 2. src/psp/archive/charge_density.py (521 LOC)

### Purpose
Legacy density/XC builder: ρ_val(r) from IBZ wavefunctions with k-weights and a
G-space star-average symmetrization; NLCC core density ρ_core(r)/ρ_core(G) from UPF
radial data; |∇ρ|² and PBE/LDA V_xc(r) via autodiff (White–Bird divergence form).
Superseded by `psp/scf_potential.py` (reads QE charge-density.hdf5,
`build_rho_val_from_wfn`), `psp/ionic_gspace.py::build_ionic_and_core` (JAX-scan core
density; its docstring line 8 says it "replaces the Python loops ... in
build_core_density"), and `psp/xc.py::compute_V_xc` + `dft_operators.compute_V_H_and_V_xc`.

Category: **archived legacy: DFT density + V_xc builder (mostly dead; one straggler test caller)**.

### Function list
| Function | Role |
|---|---|
| `build_density_from_ibz` | ρ(r) = Σ_ik wk Σ_ns |ψ|² over IBZ, then G-space symmetrization; prints integral check |
| `_symmetrise_density` | ρ(G) → (1/N_sym) Σ_S ρ(S·G) star average — **documented broken** (dev_status.md:176: "increases density error 4.5×. Not used in the current pipeline") |
| `build_core_density` | NLCC ρ_core: radial FT → uniform-q table → interpolate onto FFT G-grid with structure factors; returns (ρ_core_r, ρ_core_G·N) |
| `compute_grad_rho_sq` | |∇ρ|² via iG in G-space, with optional precise `rhog_core` de-aliasing |
| `build_V_xc` | Dispatch LDA/PBE V_xc on the FFT grid |
| `_build_V_xc_lda` | LDA V_xc = d(ρε)/dρ via `jax.grad` |
| `_build_V_xc_gga` | GGA White–Bird V_xc with QE thresholds (1e-10/1e-6/1e-10) |
| `_build_G_cart_grid` | wrapper: bvec·blat → `build_G_cart` |
| `build_G_cart` | Cartesian G grid from FFT freqs (the *original*; moved to dft_operators in 5bba57e, copy left behind here) |

### Entry points / callers
Grep `build_density_from_ibz|compute_grad_rho_sq|build_V_xc|build_core_density|build_G_cart|_symmetrise_density`
across src/tests/tools/scripts:
- `build_core_density, compute_grad_rho_sq, build_V_xc <- src/psp/tests/test_dft_hamiltonian.py:44`
  (only caller; a standalone `python -m psp.tests.test_dft_hamiltonian` script, NOT
  collected by pytest since `testpaths=["tests"]`).
- All other hits on these names resolve to `psp.dft_operators.build_G_cart` (the
  moved copy) or docstring mentions (`scf_potential.py:116`, `orbit_syms.py:19`,
  `ionic_gspace.py:8`, `dft_operators.py:341`).
- `build_density_from_ibz`, `_symmetrise_density`: zero code callers.

### Import breakage
Line 30: `from psp.radial_jax import (radial_weights, make_core_charge_table,
make_uniform_q_grid)` — **stale path**; module lives at `psp/radial/radial_jax.py`
(re-exported by `psp/radial/__init__.py`), and `psp` is a namespace package with no
`__init__.py` aliasing. So the module raises `ModuleNotFoundError` at import —
meaning its one caller, `src/psp/tests/test_dft_hamiltonian.py`, is **also broken**
as it stands.

Also: `from jax_xc_local.pbe import pbe_xc` (lines 412, 436, function-local) —
`jax_xc_local` is not in the repo src tree nor in `.venv` site-packages; same deferred
import appears in active `psp/xc.py:44`, so it presumably comes from external
PYTHONPATH — flagged for the refactor map as an invisible dependency.

### I/O
- Reads: in-memory `WfnLoader` (wavefunctions, kweights, cell, atoms) — no direct
  file reads; UPF pseudo objects for NLCC (`pp_nlcc`, `pp_mesh`).
- Writes: nothing. Prints density-integral diagnostics to stdout.

### Dead suspects
- `build_density_from_ibz`, `_symmetrise_density`: zero callers by grep across
  src/tests/tools/scripts; `_symmetrise_density` additionally documented broken in
  `psp/dev_status.md:176`.
- `build_G_cart` / `_build_G_cart_grid` (archive copies): zero callers — every live
  reference resolves to `psp.dft_operators.build_G_cart`.
- `build_core_density`/`compute_grad_rho_sq`/`build_V_xc`: only caller is the
  non-pytest-collected `src/psp/tests/test_dft_hamiltonian.py`, which cannot import
  anyway due to the stale `psp.radial_jax` path. Effectively dead; the test should be
  migrated to `ionic_gspace.build_ionic_and_core` + `xc.compute_V_xc` or retired.

### Redundancy suspects
- `build_core_density` vs `psp/ionic_gspace.py::build_ionic_and_core` (explicit
  successor per ionic_gspace.py:8).
- `_build_V_xc_lda`/`_build_V_xc_gga`/`build_V_xc` vs `psp/xc.py::compute_V_xc`,
  `_vxc_lda`, `_vxc_gga` (active autodiff XC stack).
- `compute_grad_rho_sq`'s rhog_core de-aliasing block duplicated verbatim inside
  `_build_V_xc_gga` (lines 353-357 vs 483-488) and again in active
  `dft_operators.compute_V_H_and_V_xc` (lines 316-318).
- `build_G_cart` duplicated at `dft_operators.py:165`.

### Weird code
- Lines 77-96: stream-of-consciousness normalization derivation in comments
  ("Wait — let me be careful.") — left-in reasoning scratchpad.
- Lines 334-355 in `compute_grad_rho_sq`: long comment block that argues itself into
  a corner ("The right approach: pass ρ_val_r separately. But we don't have it
  separately here.") then implements the workaround anyway; the same comment admits
  FFT(IFFT(x)) reasoning is circular. The final code is fine but the trail is
  confusing.
- Line 243: NLCC gate compares `str(cc) != 'UpfLogical.T'` — string-typed enum
  comparison against a repr; brittle if the UPF parser enum changes.
- Line 296: returns a tuple `(rho_core_r, rho_core_G_scaled)` from a function whose
  signature/docstring say it returns a single array ("Returns: rho_core_r") — the
  docstring was never updated when the second return value was bolted on.
- `_symmetrise_density`: known-broken star average (misses fractional-translation
  phases e^{-iG·t_S}, which is the plausible cause of the documented 4.5× error;
  `sym.sym_matrices`/`R_grid` used with no translation term).
- Spin: `spin_factor = 2 if nspinor == 1 else 1` — correct for TRS-paired systems but
  spin-blind for magnetic ones (consistent with the known `project_lorrax_vxc_spin_blind`
  limitation, which this archived code shares/originated).

---

## 3. src/psp/archive/build_projectors.py (238 LOC)

### Purpose
Oldest-generation fully-relativistic (spinor) projector builder on complex spherical
harmonics: complex Y_lm tables, spin-angular functions Ω_{ℓjm_j} for j=ℓ±½
(Clebsch–Gordan built by hand), radial Hankel transforms F_ℓ(q) with cubic-spline
interpolation, D-matrix block expansion, and species-level spinor Z-rows. Superseded
by `psp/radial/build_projectors_qe.py` (QE-convention real harmonics; note it
contains its own `spherical_hankel_transform_l_np` at line 351).

Category: **archived legacy: FR spinor projector builder (dead code)**.

### Function list
| Function | Role |
|---|---|
| `_sphY` | scipy `sph_harm` / `sph_harm_y` compatibility shim (complex Y_lm) |
| `compute_Ylm_list` | Complex Y_lm(K̂) for l=0..lmax, shape (2l+1, nG) per l |
| `compute_spinor_omegas` | Ω_{ℓjm_j}(K̂) spin-angular functions for j=ℓ±½, hand-rolled CG coefficients |
| `spherical_hankel_transform_l_np` | F_ℓ(q) = ∫ r² β_ℓ(r) j_ℓ(qr) dr (NumPy/SciPy, rab or trapezoid weights) |
| `compute_projector_form_factors_on_K` | Tabulate F_ℓ(q) on uniform grid, spline (`ext=1` → zero outside), evaluate at |K| |
| `expand_D_block` | Expand species D_ij onto (ℓ, j, m_j)-grouped row space → (R,R) |
| `build_species_Z_rows` | Species Z_base (R, 2, nG) spinor rows + (β, ℓ, j, m_j) index arrays + D_mat |

### Entry points / callers
Grep `compute_Ylm_list|compute_spinor_omegas|spherical_hankel_transform_l_np|compute_projector_form_factors_on_K|expand_D_block|build_species_Z_rows|build_projectors\b`
across src/tests/tools/scripts: **zero external callers**. The only other hit is the
*independent reimplementation* of `spherical_hankel_transform_l_np` in
`src/psp/radial/build_projectors_qe.py:351` (name collision, not an import).

### I/O
None. Pure functions over in-memory pseudo objects (`pp_beta`, `pp_dij`, `pp_mesh`).
Writes nothing.

### Dead suspects
- **Entire module**: zero callers for all 7 public functions by grep; not referenced
  by the other archive files either.

### Redundancy suspects
- `spherical_hankel_transform_l_np` duplicated by name and near-body in active
  `psp/radial/build_projectors_qe.py:351` — the canonical "old copy left in archive,
  live copy diverges" pattern.
- Whole module parallels `psp/radial/build_projectors_qe.py` (real-harmonic path) —
  complex-Y/spinor-Ω formulation vs QE real-harmonic formulation of the same physics.

### Weird code
- Line 11: `from scipy.special import sph_harm_y as _sph_harm_y` is **unguarded**
  while `sph_harm` (line 7) gets a try/except — on scipy < 1.15 the module dies at
  import despite the compat shim; the fallback logic is inverted from its intent
  (`sph_harm` is the *deprecated* one, removed in scipy 1.17).
- Lines 60-81: hand-written Clebsch–Gordan index gymnastics
  (`m = (two_mj - 1) // 2` vs `(two_mj + 1) // 2`, sign flip `-d * getY(m)` for
  j=ℓ−½) with no reference or test — correct-looking but unverifiable without the
  ℓ±½ spinor-spherical-harmonic formulas; classic silent-sign-bug habitat.
- Line 144: spline `ext=1` (zero outside tabulated q range) silently truncates
  F_ℓ(q) if `K_norm` exceeds the tabulated `qmax` (grid built *from* max(K_norm), so
  only an issue if reused with different K) — contrast with active code that plumbs
  an explicit `q_max`.
- `cell_volume` parameter of `compute_projector_form_factors_on_K` and
  `build_species_Z_rows` is accepted but never used (the 4π/√Ω prefactor lives in
  the caller in the newer pipeline) — signature fossil.

---

## Cross-cutting refactor verdict

All three files are import-dead or import-broken:
- `projector_pipeline.py`: broken relative imports (`.build_projectors_qe`,
  `.radial_jax` → actual location `psp/radial/`), zero callers.
- `charge_density.py`: stale `psp.radial_jax` import path, single caller is a
  non-collected legacy test that therefore also cannot run.
- `build_projectors.py`: importable but zero callers.

Recommended action for the refactor: delete the directory outright after migrating
`src/psp/tests/test_dft_hamiltonian.py` to the active stack
(`ionic_gspace.build_ionic_and_core` + `xc.compute_V_xc`/`dft_operators.compute_V_H_and_V_xc`)
or retiring it. The `__init__.py` docstring already declares the directory
historical-only; the broken imports prove nothing depends on it at runtime.
