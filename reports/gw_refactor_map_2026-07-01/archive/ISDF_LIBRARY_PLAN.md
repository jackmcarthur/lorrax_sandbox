# ISDF Mini-Library Extraction — Execution Plan

**Repo:** `sources/lorrax_D` · **Branch:** `agent/memplanner-cleanup`
**Target file:** `src/gw/isdf_fitting.py` (2733 L, verified)
**Goal:** carve the neutral ISDF core (ψ + centroids → ζ) into a reusable
`src/isdf/` package so a future direct-BSE path (ISDF-fit → solve BSE, **no GW
step**) reuses it. `gw/isdf_fitting.py` shrinks to the thin GW orchestrator
`fit_zeta_to_h5`. **No new abstractions** (no ABCs, plugins, registries,
accumulator-callbacks, dataclasses). Procedural numpy/jax, code moved verbatim.

Scope of this document: **planning + verified facts only.** No edits performed.

---

## 1. GO / NO-GO

### VERDICT: **GO.** The split is import-safe. No cycle is created.

Dependency graph traced end-to-end. Evidence (all re-confirmed by reading):

- **The only `gw.` import in the whole 2733-line file** is
  `from gw.gw_config import SlabIOBackend` at **L1909**, and it sits *inside*
  `fit_zeta_to_h5` (def L1824). It is orchestrator-local and unreachable from
  the core. Every other module-level import is `common.*` + `jax` + (func-local)
  `ffi.cusolvermp`.
- The ~106 GW-coupling refs (`SlabIO`, `isdf_header`, `coulomb_sphere`/vcoul,
  `symmetry_maps`/IBZ, gflat writer, h5py) are **all func-local inside
  `fit_zeta_to_h5`** (imports at L1909, 2035, 2114, 2245, 2280–2283, 2422, 2477,
  2492, 2682, 2705). Confirmed fact (b).
- The `common/` modules the core imports (`gamma_matrices`, `cholesky_2d`,
  `fft_helpers`, `wfn_transforms`, `psi_G_store`, `Meta`, `timing`) contain **no**
  `gw`/`isdf` import → no `common → isdf` back-edge → `isdf → common → …` stays
  acyclic. `ffi.cusolvermp` is likewise clean of gw/isdf. **No cycle.**

### The two CORE-reaches-into-GW smells and their mitigations

Neither is a GW coupling; both are soft reaches into neutral `common` metadata.
Do NOT re-plumb them during the move (behavior-preserving); note them and leave
as-is unless a follow-up wants them severed.

| # | Reach | Where | Verdict / mitigation |
|---|-------|-------|----------------------|
| A | `z_q_from_psi_sm` reads `psi_G_store.meta.fft_grid` | **L526** | `psi_G_store` is `common/psi_G_store.py` (neutral); `.meta` is `common.Meta` (neutral crystal/grid bag), **not** `LorraxConfig`. Admissible in core. Move verbatim. Optional later hardening: pass `fft_grid` as an explicit scalar arg (caller `_make_fit_one_rchunk_kernel` already holds `meta.fft_grid`). **Not required for GO.** |
| B | `fit_one_rchunk` / `_make_fit_one_rchunk_kernel` take `meta: Meta` | L1547, 1696 | Only reads `n_rmu`, `nk_tot`, `nspinor`, `fft_grid`, `kgrid`, `n_rmu_padded` (L1583–1596, 1732–33). `Meta` is neutral — keep the `meta` param (honors the "pass bundles not 15 args" memo). Move verbatim. |

**One external private-cache reach**, not a cycle, but must be repointed
(see §5 step 4): `gw/gw_init.py:700–703` does
`from gw import isdf_fitting as _isdf; … _isdf._fit_one_rchunk_cache.clear()`.
After the move that cache lives in `isdf/`. Mechanical repoint (a downward
`gw → isdf` edge). Verified present.

---

## 2. THE CUT LINE

`fit_zeta_to_h5` (L1824→EOF, ~910 L) is a single function; every other
top-level symbol precedes it. The cut falls exactly at the **L1824 def
boundary**. All symbols L26–1793 are array/scalar-in/array-out on neutral
`common` deps — **except** the two mem-debug helpers, which stay with the
orchestrator (their only callers are `fit_zeta_to_h5` and `gw_init`).

### MOVE to `src/isdf/` — CORE (verified line numbers)

| Symbol | L | Public API? | Notes |
|--------|---|-------------|-------|
| `_pair_density_cache` | 160 | — | jit cache for `pair_density` |
| `_pair_pipeline_sm_cache` | 161 | — | shared cache: `c_q_from_psi_sm` (L373) + `z_q_from_psi_sm` (L563) |
| `pair_density` | 164 | **yes** | consumed by `centroid/pivoted_cholesky.py:858` |
| `_isdf_pipeline_cache` | 202 | — | jit cache for `gram_q0_from_pair` |
| `gram_q0_from_pair` | 213 | **yes** | consumed by `centroid/pivoted_cholesky.py:858` |
| `_chol_2d_cache` | 309 | — | jit cache used at L1134 in `factor_c_q` |
| `c_q_from_psi_sm` | 338 | **yes** | clean ψ+kgrid+mesh+γ → C_q |
| `z_q_from_psi_sm` | 460 | **yes** | reads `psi_G_store.meta.fft_grid` (smell A) |
| `_identity_pad_block_diagonal` | 853 | — | pad-block math |
| `_resolve_solver_kind_charge` | 909 | — | str override → solver name |
| `_resolve_solver_kind_transverse` | 945 | — | " |
| `_resolve_solver_kind` | 976 | — | " |
| `factor_c_q` | 992 | **yes** | consumed by `bandstructure/htransform.py:27`, tests |
| `_solve_cache` | 1155 | — | jit cache for `solve_zeta` |
| `_reshard_zeta_mu_X_r_Y_to_mu_XY` | 1158 | — | reshard util |
| `_reshard_zeta_r_XY_to_mu_XY` | 1183 | — | reshard util |
| `solve_zeta` | 1203 | **yes** | consumed by tests |
| `_fit_one_rchunk_cache` | 1542 | — | cleared by `gw_init:703` (see §5 step 4) |
| `_make_fit_one_rchunk_kernel` | 1545 | — | fuses z_q∘solve for one r-chunk |
| `fit_one_rchunk` | 1686 | **yes** | per-chunk workhorse; both consumers loop this |
| `_band_norms_slice` | 1793 | — | pseudoband weights → jnp; np/jnp only |
| gamma re-imports L16–18: `gamma_perm_phase as _gamma_perm_phase_mu`, `gamma_double_contract` | | re-export | keep in core.py; `_gamma_perm_phase_mu` used at L1724 |

**7 public core symbols:** `pair_density`, `gram_q0_from_pair`,
`c_q_from_psi_sm`, `z_q_from_psi_sm`, `factor_c_q`, `solve_zeta`,
`fit_one_rchunk`.

### STAY in `src/gw/isdf_fitting.py` — ORCHESTRATOR

| Symbol | L | Reason |
|--------|---|--------|
| `fit_zeta_to_h5` | 1824 | the entire GW packaging: wfn/sym/meta/centroid_indices/output_file, `SlabIOBackend`, `IsdfHeader`/`write_isdf_header`/`mark_zeta_done`, IBZ q-reduction (`slice_q_full_to_ibz`), `zeta_cutoff_ry` sphere (`coulomb_sphere`), gflat writer, `vcoul`. All GW imports func-local inside body. |
| `mem_probe` | 29 | env-gated HBM probe; called by `fit_zeta_to_h5` body **and** `gw_init:1085`. Pure jax/nvidia-smi, no ISDF math. Leave with orchestrator (diagnostic utility, not library API). |
| `_nvsmi_used_mb_local_gpu` | 87 | nvidia-smi sampler backing `mem_probe`. |
| `_NVSMI_PEAK_MB` / `_NVSMI_LAST_MB` | 26–27 | mem-probe running-max state. |

### SHARED helpers — duplication/relocation calls

- **None require duplication.** Every jit cache is single-consumer within the
  core and moves with its function. No parallel old/new path is created
  (honors no-redundancy memo).
- `_band_norms_slice` (L1793): confirm at edit time whether only
  `fit_zeta_to_h5` calls it or also `fit_one_rchunk`. Grep first; if
  orchestrator-only, it MAY stay in gw; but it is pure np/jnp with no GW dep, so
  **default: move to core** with the rest (single source of truth; the future
  BSE r-chunk loop needs the same pseudoband norms).
- `mem_probe` is the one genuinely-shared-but-non-ISDF symbol: it stays in the
  orchestrator and `gw_init` keeps importing it from `gw.isdf_fitting` (no
  change to that import). Do **not** move it into `isdf/` — it is not ISDF math.

---

## 3. `isdf/` PUBLIC API + I/O CONTRACT

### `src/isdf/__init__.py` (explicit re-export, not `*`)

```python
"""ISDF mini-library: ψ + centroids -> ζ interpolation vectors.

Neutral array-in/array-out core. Consumers: gw.isdf_fitting.fit_zeta_to_h5
(GW), centroid.pivoted_cholesky (centroid selection), bandstructure.htransform,
and a future direct-BSE fit. NO LorraxConfig / GW meta packaging / h5 / V_q
here. Depends only on common/ (Meta, timing, gamma_matrices, psi_G_store,
cholesky_2d, fft_helpers, wfn_transforms) and ffi/ (cusolvermp).
"""
from isdf.core import (
    pair_density,        # centroid-selection Gram building block
    gram_q0_from_pair,   # q=0 Gram (centroid selection)
    c_q_from_psi_sm,     # centroid ψ -> C_q metric
    z_q_from_psi_sm,     # ψ(G) -> Z_q rhs (internal-ish; exported for tests)
    factor_c_q,          # C_q -> L_q (chol factor / indefinite passthrough)
    solve_zeta,          # (L_q, Z_q) -> ζ chunk
    fit_one_rchunk,      # fused per-r-chunk ζ workhorse; consumers loop this
)

__all__ = [
    "pair_density", "gram_q0_from_pair",
    "c_q_from_psi_sm", "z_q_from_psi_sm",
    "factor_c_q", "solve_zeta", "fit_one_rchunk",
]
```

Privates stay reachable as `isdf.core._fit_one_rchunk_cache` (needed by the
`gw_init` cache-clear repoint). `z_q_from_psi_sm` is exported because
`tests/test_zq_from_psi_sm_bit_identity.py` imports it directly (L35).

### Design decision: expose COMPOSABLES, not a monolithic `fit_zeta()`

The ζ-fit is a 3-phase composition where **only the r-chunk accumulation target
differs** between consumers:

1. `C_q = c_q_from_psi_sm(...)` — build CCT metric (identical GW/BSE)
2. `L_q = factor_c_q(C_q, ...)` — factor/prepare (identical)
3. loop r-chunks: `zeta_chunk = fit_one_rchunk(...)` → **accumulate** (DIFFERS)

GW accumulates each `zeta_chunk` into a **G-flat Coulomb sphere**
(`accumulate_rchunk_to_gflat`, isdf_fitting.py ~L2292/2313) and never
materializes full-r ζ — that is the entire memory model. A BSE consumer
accumulates into a BSE-kernel target. Folding the loop into the core would force
an accumulator-callback abstraction — the exact plugin pattern the project keeps
rejecting. **So the loop stays consumer-side; the core exposes phases.** The two
consumers are line-for-line identical through `C_q → factor_c_q →
fit_one_rchunk` and diverge only at the accumulate call.

### I/O contract (arrays + scalars in, `jax.Array` out — no config objects)

`mesh_xy` = `jax.sharding.Mesh(('x','y'))`; `kgrid=(nqx,nqy,nqz)`;
`gamma_L/R` = `(perm, phase)` tuple or `None` (identity = charge channel).
`meta` (on `fit_one_rchunk` only) = neutral `common.Meta` grid bag.

| Fn | Inputs | Output |
|----|--------|--------|
| `pair_density` | `psi_rmuT_X`(nk,n_rmu,nb,ns), `psi_rcol_Y`(nk,nb,ns,n_col), mesh | (nk,ns,ns,n_rmu,n_col) |
| `gram_q0_from_pair` | two pair densities (nk,ns,ns,n_rmu,n_rmu), `k_weights`(nk), `gamma_L/R`, mesh | (n_rmu,n_rmu) Hermitian PSD |
| `c_q_from_psi_sm` | `psi_{l,r}_X`(nk,n_rmu,nb,ns), `psi_{l,r}_Y`(nk,nb,ns,n_col), `gamma_L/R`, `kgrid`, mesh | C_q (nq,n_rmu,n_rmu) |
| `z_q_from_psi_sm` | `psi_G_store`(common duck-type), ψ chunk arrays, `kgrid`, mesh | Z_q (nq,n_rmu,n_zchunk) |
| `factor_c_q` | C_q, mesh, scalars `vertex_mu_L`, `n_rmu_logical`, `solver_kind` | L_q (nq,n_rmu,n_rmu) |
| `solve_zeta` | L_q, Z_q, mesh, scalars `q_chunk_size`,`vertex_mu_L`,`solver_kind`, opt `cct_trace_per_q` | ζ chunk (nq,n_rmu,n_zchunk) |
| `fit_one_rchunk` | `psi_G_store` (closure, NOT jit arg), `psi_{l,r}_rmuT_X_fit`, `L_q`, `norms_{l,r}`, `r_start_dyn`, band ranges, `kvecs_frac`(np), `q_irr_full_idx`(np\|None), `vertex_mu_L`, `solver_kind`, `meta` | ζ chunk (nq_disk,n_rmu,n_rchunk) |

### How `fit_zeta_to_h5` becomes a thin consumer

```python
from isdf import c_q_from_psi_sm, factor_c_q, fit_one_rchunk
# (also: mem_probe stays local; slice_q_full_to_ibz, coulomb_sphere, slab_io,
#  isdf_header, gw_config imports stay func-local — all GW packaging)

def fit_zeta_to_h5(wfn, sym, meta, centroid_indices, mesh_xy, chunk_r, output_file, ...):
    # --- GW packaging (STAYS): IBZ decision, sphere, headers, SlabIO open ---
    write_ibz_only = _finalize_ibz(sym, centroid_indices, wfn, meta, vertex_mu_L)
    q_irr_full_idx = sym.q_irr_full_idx if write_ibz_only else None
    ...  # coulomb sphere pkg, mf/isdf headers, SlabIOBackend open

    # --- CORE (identical to a BSE consumer) ---
    C_q = c_q_from_psi_sm(psi_l_X_fit, psi_l_Y_fit, psi_r_X_fit, psi_r_Y_fit,
                          gamma_L, gamma_R, kgrid=meta.kgrid, mesh_xy=mesh_xy)
    C_q = slice_q_full_to_ibz(C_q, q_irr_full_idx) if write_ibz_only else C_q  # GW IBZ
    L_q = factor_c_q(C_q, mesh_xy, vertex_mu_L=vertex_mu_L, n_rmu_logical=meta.n_rmu)

    # --- CORE loop, GW accumulation target (G-flat Coulomb sphere) ---
    for chunk in range(num_chunks):
        psi_G_store.begin_rchunk(r0, r1)
        zeta_chunk = fit_one_rchunk(psi_G_store=psi_G_store, L_q=L_q, meta=meta,
                                    r_start_dyn=r0, kvecs_frac=q_irr_frac,
                                    q_irr_full_idx=q_irr_full_idx,
                                    vertex_mu_L=vertex_mu_L, ...)
        gflat_acc = accumulate_rchunk_to_gflat(gflat_acc, zeta_chunk, sphere_idx, ...)  # GW
        psi_G_store.end_rchunk()
    zeta_io.write_slab(gflat_acc, ...)   # GW h5 write
```

---

## 4. PACKAGE STRUCTURE

**Recommendation: `src/isdf/` top-level package, single `core.py` module.**

```
src/isdf/
  __init__.py   # the 7-name public surface above
  core.py       # ALL moved symbols (~1500 L) + their 6 jit caches
```

- **Single `core.py`, do NOT split kernels/primitives.** The jit caches are
  module-level dicts closed over by their factory functions; splitting fractures
  the cache/factory coupling for zero benefit. One ~1500-line module is the
  minimal choice, single source of truth.
- **Top-level `isdf/`, not `common/isdf_core.py`.** Two non-GW consumers already
  exist (`bandstructure/htransform`, `centroid/pivoted_cholesky`) and `src/bse/`
  already exists as the future consumer's home — the deliverable *is* a library,
  so name it as one. `from isdf import factor_c_q` reads as "use the ISDF
  library". `__init__.py` earns its keep as the API surface and leaves an obvious
  home for a future `isdf/bse.py` without re-plumbing.
- **Packaging note (verified):** setuptools flat/src auto-discovery; egg-info
  `top_level.txt` is a static list (currently: bandstructure, bse, centroid,
  common, ffi, file_io, gw, mixing, postprocess, psp, runtime, solvers — no
  isdf). Adding `src/isdf/` requires a re-run of the editable install
  (`uv pip install -e .` / `uv sync`) so the finder + egg-info pick up the new
  top-level package. Include this as an explicit step (§5 step 1b).

### Where `fit_zeta_to_h5` lives

**Stays as `gw/isdf_fitting.py` for the extraction commit.** Keep the risky
pure-move (must stay bit-identical) in ONE commit. **Follow-up separate commit**:
rename `gw/isdf_fitting.py` → `gw/zeta_fit.py` (the honest name once it only does
zeta→h5/IBZ/config packaging; matches the MEMORY note). Rename blast radius is
small: `gw_init.py:444` (`fit_zeta_to_h5`), `gw_init.py:1085` (`mem_probe`),
`gw_init.py:700` + doc-only refs. Splitting rename from move keeps a regression
bisectable.

---

## 5. ORDERED EXECUTION STEPS (behavior-preserving)

Every ζ-fit change feeds all 4 physics gates (Σ_X → Σ → QP), so the fitting
math must stay **bit-identical**. Move code verbatim; do not "improve" it.

**Step 0 — safety net.** Confirm branch `agent/memplanner-cleanup`, clean-ish
tree for these files. Baseline: `uv run python -m pytest -q` (record pass set,
esp. `test_padding`, `test_zq_from_psi_sm_bit_identity`, `test_psi_g_store`,
`test_band_chunk_size_floor`). Capture a baseline ζ artifact from one known-good
run for byte/numeric diff (see gates).

**Step 1a — create the package.** `src/isdf/__init__.py` (empty stub first) +
`src/isdf/core.py`. Move the CORE block (symbols in §2 MOVE table, L26 caches
through L1793 `_band_norms_slice`, **excluding** `mem_probe`/`_nvsmi_*`/NVSMI
state) verbatim into `core.py`.
**Step 1b — register package.** Re-run editable install (`uv sync` /
`uv pip install -e .`) so `isdf` is importable and egg-info updates.

**Step 2 — rewrite CORE's imports in `core.py`.** Add the `common.*` + `jax` +
`jax.experimental.io_callback` imports the moved code needs (copy the relevant
subset of L14–127: `Meta`, `timing`, `gamma_matrices` incl. the
`gamma_perm_phase as _gamma_perm_phase_mu` alias + `gamma_double_contract`,
`jax_profile`, `cholesky_2d`, `fft_helpers`, `wfn_transforms`, `psi_G_store`
duck-type usage; func-local `ffi.cusolvermp` imports stay inside their
functions). Do NOT import anything from `gw`. Fill `__init__.py` with the 7-name
re-export.
**Gate:** `python -c "import isdf; isdf.factor_c_q; isdf.fit_one_rchunk"` import
smoke (run under `srun --jobid=$JID` per no-login-node rule).

**Step 3 — thin the orchestrator.** In `gw/isdf_fitting.py`: delete the moved
symbols; keep `mem_probe`/`_nvsmi_*`/NVSMI state + `fit_zeta_to_h5`. Add
`from isdf import c_q_from_psi_sm, factor_c_q, fit_one_rchunk` (and any other core
symbol its body still calls — grep the body for each name before deleting).
Move the `common.load_wfns.load_centroids_band_chunked` import (L125) down to
orchestrator scope if now unused at module top.
**Gate:** `python -c "from gw.isdf_fitting import fit_zeta_to_h5, mem_probe"`.

**Step 4 — repoint every importer** (exact, verified):

| # | File:line | Old | New |
|---|-----------|-----|-----|
| 1 | `bandstructure/htransform.py:27` | `from gw.isdf_fitting import factor_c_q` | `from isdf import factor_c_q` |
| 2 | `centroid/pivoted_cholesky.py:858` | `from gw.isdf_fitting import (pair_density, gram_q0_from_pair)` | `from isdf import pair_density, gram_q0_from_pair` |
| 3 | `common/isdf_zeta_mode_test.py:48` | `from gw.isdf_fitting import factor_c_q, solve_zeta` | `from isdf import factor_c_q, solve_zeta` |
| 4 | `tests/test_padding.py:291,355,389,422` | `from gw.isdf_fitting import factor_c_q` | `from isdf import factor_c_q` |
| 5 | `tests/test_zq_from_psi_sm_bit_identity.py:35` | `from gw.isdf_fitting import z_q_from_psi_sm` | `from isdf import z_q_from_psi_sm` |
| 6 | `tests/test_zq_from_psi_sm_bit_identity.py:411` | `from gw.isdf_fitting import _gamma_perm_phase_mu` | `from common.gamma_matrices import gamma_perm_phase` (true source; drop alias hop) OR `from isdf.core import _gamma_perm_phase_mu` to stay verbatim |
| 7 | `gw/gw_init.py:700–703` | `from gw import isdf_fitting as _isdf` … `_isdf._fit_one_rchunk_cache.clear()` | `from isdf import core as _isdf_core` … `_isdf_core._fit_one_rchunk_cache.clear()` |
| — | `gw/gw_init.py:444` | `from gw.isdf_fitting import fit_zeta_to_h5` | **unchanged** (orchestrator stays) |
| — | `gw/gw_init.py:1085` | `from gw.isdf_fitting import mem_probe` | **unchanged** (mem_probe stays) |

Doc-only refs (comments/docstrings, no code change; fix opportunistically):
`pivoted_cholesky.py:13,813,816`, `gflat_memory_model.py:5`,
`psi_G_store.py:20-21,350`, `symmetry_maps.py:124`, `coulomb_sphere.py:12`,
`slab_io.py:30`, `zeta_reader.py:41,62`, `gw_init.py:68,662`.

**Step 5 — verification (gated, in order).**
1. **Import smoke** (all repointed modules import): `srun … python -c "import
   bandstructure.htransform, centroid.pivoted_cholesky, gw.gw_init, gw.isdf_fitting, isdf"`.
2. **Full unit suite**: `uv run python -m pytest -q` — must match Step 0 baseline
   exactly (esp. `test_padding`, `test_zq_from_psi_sm_bit_identity`,
   `test_psi_g_store`, `test_band_chunk_size_floor`). These are the direct ζ-fit
   guards.
3. **4 physics gates** (ζ-fit feeds every one): run the standard gate set on GPU
   (`srun --jobid=$JID` / `lxrun`) and diff ζ + Σ_X + Σ + QP against the Step-0
   baseline artifact. Byte-identical ζ is the pass bar (pure code relocation, no
   math touched). CrI3 gates use 4×4 mesh / 80 GB HBM per the always-16-GPU rule.

**Step 6 — checkpoint** (per `skills/checkpoint/SKILL.md`): pytest, commit the
extraction as one self-contained unit on `agent/memplanner-cleanup`, update
report + CHANGELOG.

**Step 7 — follow-up commit (separate):** rename `gw/isdf_fitting.py` →
`gw/zeta_fit.py`; fix `gw_init.py:444,1085` + doc refs; re-run smoke + suite.

---

## 6. HOW FAR TO GO (blunt)

**Do exactly this and stop:**
- One `src/isdf/` package, one `core.py`, one `__init__.py` with 7 re-exports.
- Move code **verbatim**. Rewrite only the import block. Repoint the 7 importers.
- `fit_zeta_to_h5` stays in `gw/` as the thin consumer.

**Explicitly do NOT:**
- No `fit_zeta()` monolith in the core (would force an accumulator-callback /
  plugin abstraction — rejected). Consumers own their r-chunk loop.
- No ABCs, no plugin registry, no `IsdfFitter` class, no `BzIbzTable`/`SymAction`
  wrappers, no new dataclasses. Procedural numpy/jax on plain arrays
  (no-new-API-layers memo).
- No splitting `core.py` into kernels/primitives/solve submodules — fractures the
  jit-cache/factory coupling for zero gain.
- No severing the `psi_G_store.meta.fft_grid` (smell A) or `meta: Meta` (smell B)
  reaches during this move — `Meta` is neutral; changing signatures risks the
  bit-identity the gates require. Defer to a later optional hardening pass if ever.
- No moving `mem_probe` into `isdf/` — it is diagnostics, not ISDF math.
- No `fetch_X_dyn`-style parallel paths, no leaving the old symbols in `gw/` as
  shims (no-redundancy memo — delete on move, single source of truth).
- Do NOT combine the rename into the move commit (bisectability).

---

## 7. RISK TABLE + ROLLBACK

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed intra-file reference: orchestrator body still calls a moved symbol not re-imported | Med | ImportError at runtime (gates) | Before deleting each symbol from `gw/`, grep `fit_zeta_to_h5`'s body (L1824–2733) for its name; add to the `from isdf import …` line. Import smoke (Step 5.1) catches it. |
| jit-cache identity break (moved cache dict is a *different* object → recompile / lost warm cache) | Med | Perf only, not correctness; `gw_init:703` clear targets wrong object | Repoint #7 to `isdf.core._fit_one_rchunk_cache`. Caches are correctness-neutral (memoization); a cold cache only recompiles. |
| Numeric drift in ζ (should be impossible for pure relocation) | Low | Wrong Σ_X/Σ/QP — all 4 gates | Byte-diff ζ artifact vs Step-0 baseline; `test_zq_from_psi_sm_bit_identity` is the ULP guard. Any drift ⇒ a copy was not verbatim ⇒ rollback. |
| Package not discovered (editable install stale, egg-info missing `isdf`) | Med | ImportError `No module named isdf` | Step 1b re-run `uv sync`/`-e .`; confirm `isdf` in `top_level.txt`. |
| `_gamma_perm_phase_mu` alias test (repoint #6) breaks if aliased differently | Low | 1 test fails | Prefer `from common.gamma_matrices import gamma_perm_phase` (true source); or keep `from isdf.core import _gamma_perm_phase_mu`. |
| Doc-only refs now point at a nonexistent path after rename (Step 7) | Low | Stale docs only | Sweep the doc-only list; mkdocstrings build (optional) surfaces broken refs. |

**Rollback:** the entire change is on `agent/memplanner-cleanup` as isolated
commits. Extraction is one commit; `git revert <sha>` (or `git checkout` the two
touched files + `rm -r src/isdf`) restores the monolith. Rename is a separate
commit, independently revertible. No `main` involvement; nothing pushed.

---

*Verified facts in this plan (read at plan time):* file = 2733 L; symbol lines
per §2; only `gw.` import in file = L1909 (inside `fit_zeta_to_h5`);
`z_q_from_psi_sm` reaches `psi_G_store.meta.fft_grid` at L526; `gw_init` cache
clear at L700–703; external consumers = htransform:27, pivoted_cholesky:858,
isdf_zeta_mode_test:48, gw_init:444/700/1085, test_padding (×4),
test_zq_from_psi_sm_bit_identity:35/411. No `src/isdf/` exists yet; `src/bse/`
does. setuptools auto-discovery; egg-info re-run needed.
