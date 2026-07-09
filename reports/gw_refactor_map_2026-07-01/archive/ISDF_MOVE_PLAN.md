# ISDF_MOVE_PLAN — `common/isdf_fitting.py` → `gw/isdf_fitting.py`

MAP §4 #2. Repo: `sources/lorrax_D`, branch `agent/memplanner-cleanup`. Planning pass; no
code was edited to produce this. All file:line references verified against the tree on
2026-07-02/03.

Target file: `src/common/isdf_fitting.py` (2744 L, the A2 ζ-fit STAGE; also hosts C3 FFI
Cholesky/cusolvermp solves and C6 memory probes). Destination: `src/gw/isdf_fitting.py`.

The roadmap rule is enforced here: **STEP 1 is a PURE MOVE (rename + fix imports, zero logic
change, verify green). STEP 2 (probe lift + cache naming) is a SEPARATE, LATER, GATED commit.**
ζ-fit feeds all three e2e gates (cohsex / gnppm / ibz_full_bz), so any value-breaking move
fails the suite today — that is our safety net.

---

## 1. GO / NO-GO on the pure move

**GO.** There is **no circular-import blocker.** No shim, no dependency-reordering needed.

Evidence (verified):
- `isdf_fitting` imports from `gw/` exactly once, and it is **function-local**:
  `isdf_fitting.py:1920 from gw.gw_config import SlabIOBackend`. No top-level `gw` edge.
- All of `isdf_fitting`'s top-level dependencies live in `common/` or external packages
  (`common.gamma_matrices`, `common.timing`, `common.cholesky_2d`, `common.fft_helpers`,
  `common.load_wfns`, `common.wfn_transforms`, `common.jax_profile`, `common.meta`,
  jax/numpy). None of those `common/*` modules import `gw` at load time, so the chain
  `gw.isdf_fitting → common.* →` terminates without re-entering `gw`.
- `gw/__init__.py` (`from .gw_init import ...`, `from .gw_config import ...`) and
  `gw/gw_init.py` reference `isdf_fitting` **only function-locally** (`gw_init.py:444,700,1085`),
  never at top level. So `gw` package init completes before any `isdf_fitting` body runs — no
  partially-initialized-module trap.
- No `common/*` module imports `isdf_fitting` at load time (the mentions in
  `common/psi_G_store.py:20-21` and `common/coulomb_sphere.py:12` are **comments**, not imports).
- Destination name is free: `gw/` contains `w_isdf.py` but no `isdf_fitting.py`. No collision.
- Neither `common/__init__.py` nor `gw/__init__.py` re-exports `isdf_fitting` — **no
  `__init__.py` edits required.**

Direction of dependency stays `gw → common`, which is already the established direction. The
move is architecturally clean.

**The ONE thing that makes a naive `git mv` break:** the file uses **`from .`-relative imports
whose targets all live in `common/`**. The instant the file sits in `gw/`, `.` resolves to
`gw` siblings that do not exist, and the module fails at import time — taking down every gate
(ζ-fit feeds all of them). Therefore the relative-import rewrite is **part of STEP 1, in the
same commit** — it is non-deferrable, not an optional cleanup.

---

## 2. STEP 1 — PURE MOVE (one commit, ZERO logic change)

### 2a. The move
```
git mv src/common/isdf_fitting.py src/gw/isdf_fitting.py
```
Do **not** move `src/common/isdf_zeta_mode_test.py` — it physically stays in `common/`; only
its import string changes (2c). Optionally relocate it later for tidiness; not required.

### 2b. Rewrite the 10 in-file relative imports → absolute `common.*` (ZERO logic change)
These are the sites that break under the move. Verified line numbers in the moved file:

| line | OLD (relative) | NEW (absolute) |
|---|---|---|
| 14 | `from . import Meta` | `from common import Meta` |
| 15 | `from . import timing` | `from common import timing` |
| 16 | `from .gamma_matrices import (…)` | `from common.gamma_matrices import (…)` |
| 129 | `from .cholesky_2d import (…)` | `from common.cholesky_2d import (…)` |
| 134 | `from .fft_helpers import (…)` | `from common.fft_helpers import (…)` |
| 139 | `from .load_wfns import load_centroids_band_chunked` | `from common.load_wfns import load_centroids_band_chunked` |
| 140 | `from .wfn_transforms import to_rchunk_inner` | `from common.wfn_transforms import to_rchunk_inner` |
| 535 | `from .psi_G_store import _PSI_G_FLAT_SPEC` (func-local) | `from common.psi_G_store import _PSI_G_FLAT_SPEC` |
| 2032 | `from . import gamma_matrices as _gm` (func-local) | `from common import gamma_matrices as _gm` |
| 2125 | `from .symmetry_maps import slice_q_full_to_ibz` (func-local) | `from common.symmetry_maps import slice_q_full_to_ibz` |

Note line 14: `Meta` is re-exported by `common/__init__.py:1 (from .meta import Meta)`, so
`from common import Meta` is correct (do **not** rewrite to `from gw import Meta` — `gw` does
not export it). The lone `gw.gw_config` import at 1920 and all `ffi.cusolvermp` /
`file_io.*` / `centroid.*` imports are already absolute and stay untouched.

This is a pure mechanical repath. **No function body, no einsum, no cache key, no solver
policy is touched.**

### 2c. Repoint the 6 importer files (12 statements) → `gw.isdf_fitting`

Runtime (production):
| file:line | OLD | NEW |
|---|---|---|
| `src/bandstructure/htransform.py:27` (top-level) | `from common.isdf_fitting import factor_c_q` | `from gw.isdf_fitting import factor_c_q` |
| `src/centroid/pivoted_cholesky.py:858` (func-local, multi-symbol block: `pair_density, gram_q0_from_pair`) | `from common.isdf_fitting import (…)` | `from gw.isdf_fitting import (…)` |
| `src/gw/gw_init.py:444` (func-local) | `from common.isdf_fitting import fit_zeta_to_h5` | `from gw.isdf_fitting import fit_zeta_to_h5` (or `from . import isdf_fitting` — now intra-package) |
| `src/gw/gw_init.py:700` (func-local) | `from common import isdf_fitting as _isdf` | `from gw import isdf_fitting as _isdf` |
| `src/gw/gw_init.py:1085` (func-local) | `from common.isdf_fitting import mem_probe as _mem_probe` | `from gw.isdf_fitting import mem_probe as _mem_probe` |
| `src/common/isdf_zeta_mode_test.py:48` (top-level, test module) | `from common.isdf_fitting import factor_c_q, solve_zeta` | `from gw.isdf_fitting import factor_c_q, solve_zeta` |

Tests:
| file:line | OLD | NEW |
|---|---|---|
| `tests/test_padding.py:291,355,389,422` (func-local ×4) | `from common.isdf_fitting import factor_c_q` | `from gw.isdf_fitting import factor_c_q` |
| `tests/test_zq_from_psi_sm_bit_identity.py:35` (top-level) | `from common.isdf_fitting import z_q_from_psi_sm` | `from gw.isdf_fitting import z_q_from_psi_sm` |
| `tests/test_zq_from_psi_sm_bit_identity.py:411` (func-local) | `from common.isdf_fitting import _gamma_perm_phase_mu` | `from gw.isdf_fitting import _gamma_perm_phase_mu` |

Highest-risk site: `htransform.py:27` and `isdf_zeta_mode_test.py:48` /
`test_zq…:35` are **top-level** imports — if missed, the importing module fails at collection.
The rest are function-local (fail only when exercised, which the gates do).

### 2d. `__init__.py` re-export edits
**None.** Verified neither package `__init__` re-exports any `isdf_fitting` symbol.

### 2e. Doc-comment path references (NO runtime impact — do NOT touch in STEP 1)
~20 docstring/comment mentions of `common/isdf_fitting.py` remain accurate-only (e.g.
`pivoted_cholesky.py:13,773`, `gw_init.py:68`, `gflat_memory_model.py:*`,
`psi_G_store.py:20-21`, `file_io/*`, `symmetry_maps.py:124`). These break nothing. Leave them
for STEP 1; optionally sed them for legibility in STEP 2 or a trailing tidy commit. Do not
inflate the pure-move diff with them.

### 2f. Verification (all must be GREEN before committing — this is the whole safety argument)
1. `python -c "import gw.isdf_fitting"` — proves the relative-import rewrite is complete and
   no cycle exists.
2. `python -m py_compile` on the moved file + all 6 importer files (fast structural check).
3. Full suite: `uv run python -m pytest -q` — must stay **241 passed / 0 failed**.
4. The 3 e2e gates (cohsex / gnppm / ibz_full_bz, ≤1 GPU). Because ζ-fit feeds all three, a
   value-identical result across all gates is the proof that the move changed zero logic. A
   pure move MUST leave these bit-for-bit identical.

If any gate value moves, the "pure move" was not pure — revert (§4) and re-inspect the diff
for an accidental logic edit.

Commit message: `refactor: pure-move isdf_fitting common→gw (no logic change)`.

---

## 3. STEP 2 — EXTRACTIONS (separate, later, gated commit — only after STEP 1 is green)

Per the extraction analysis, do **only** the two low-effort/high-clarity items below. Do NOT
build new registry/API layers (violates the "no new API layers" and "no redundancy" rules).

### 3a. Lift the C6 memory probes (WORTH DOING)
Create `src/common/mem_probe.py` (sits naturally beside `timing.py` / `jax_profile.py`) holding:
- `mem_probe(label, *, only_rank0)` (currently `isdf_fitting.py:43–98`)
- `_nvsmi_used_mb_local_gpu()` (101–127)
- globals `_NVSMI_PEAK_MB`, `_NVSMI_LAST_MB` (28–29)
- a small shared `sample_peak_bytes() -> int` factoring the duplicated
  `memory_stats()['peak_bytes_in_use'] → nvsmi fallback` logic.

Seams / edits:
- **Delete** `_mem_report` (31–40) and `_MEM_PROFILE` (21) — verified **zero callers**
  anywhere in `src/`. Dead code; do not migrate it.
- Repoint `gw_init.py:1085` import to `from common.mem_probe import mem_probe as _mem_probe`.
- In the moved ζ-fit file: import `mem_probe` from `common.mem_probe`; the in-loop call sites
  (`1934, 2553, 2566, 2601, 2629, 2739` and the `_mem_probe = mem_probe` alias at 2456) are
  **named lifecycle markers — leave them at their points in the physics outline**, only their
  binding changes.
- `_track_peak` closure (2472–2486) + `_peak_bytes`: **LEAVE IN PLACE.** `_peak_bytes` is the
  function's return value (contract at 1913/2744, consumed by the planner). Only dedup: replace
  its inline sampler body with `_peak_bytes = max(_peak_bytes, sample_peak_bytes())`. The
  `nonlocal`/return machinery stays. Do NOT lift it out.

Per-item verification: probes are env-gated no-ops under the suite, so they cannot change a
gate *value* — BUT `peak_bytes` flows to the planner and one runtime line (`_track_peak`'s
sampler) changes, so **re-run the 3 gates + full suite** anyway. Net effect: `isdf_fitting`
loses LOC; one small new module.

### 3b. Name / document the 6 hand-keyed jit caches (WORTH DOING — docstrings only)
Add a one-line docstring above each of the six module-level cache dicts stating what it
memoizes + the `id()`-lifetime caveat. Do **NOT** hoist them into a shared cache module.

| dict | def line | key tag / discriminants |
|---|---|---|
| `_pair_density_cache` | 171 | `('pair_density', id(mesh), nk, n_rmu, nb, ns, n_col)` |
| `_pair_pipeline_sm_cache` | 172 | `('c_q_from_psi_sm', …)` (382) **and** `('z_q_from_psi_sm_streaming', …)` (568) — two producers, one dict |
| `_isdf_pipeline_cache` | 213 | `('gram_q0_from_pair', …)` |
| `_chol_2d_cache` | 320 | `('chol_2d', id(mesh), J, block_size)` |
| `_solve_cache` | 1166 | `('solve_from_L', …, use_lu)` |
| `_fit_one_rchunk_cache` | 1553 | fused-kernel key incl. `id(psi_G_store)`, `id(mesh)` |

Two comments worth adding (correctness legibility, not behavior change):
- On each cache: "keyed on `id(mesh)`/`id(psi_G_store)`; relies on those objects outliving the
  entry — true for all current callers (held live for the whole `fit_zeta_to_h5` call)." This
  is a real latent silent-wrong-answer window only if a mesh/store is GC'd and its address
  recycled cross-call; flag it, do not rewrite.
- At `jax.clear_caches()` (line ~2175): note it clears JAX's compile cache, **not** these
  Python dicts (JAX transparently recompiles).

**Coupling to keep in sync:** `gw_init.py:694–703` reaches into `_isdf._fit_one_rchunk_cache`
and calls `.clear()`. If STEP 2 renames that dict, update line 703. (Recommendation: do NOT
rename it — docstring only — so this call site is untouched.)

### 3c. Explicitly LEAVE ALONE
- Do **not** build `common/jit_cache.py` or any cache registry/hub — no shared key space
  justifies it; it would create the exact reach-back-in hub the conventions forbid.
- Do **not** extract `_track_peak` / `_peak_bytes` out of `fit_zeta_to_h5` — risks the return
  contract for ~15 LOC of no value.
- Do **not** move `pair_density` / `gram_q0_from_pair` (175–311) in STEP 1 — they serve only
  `centroid.pivoted_cholesky` (q=0 Gram for centroid scoring), but they ride along fine as part
  of the moved file and stay importable. Relocating them nearer `centroid/` is a *possible*
  later step, out of scope here.
- Do **not** touch the C3 FFI solve path (`factor_c_q`, `solve_zeta`, the three lazy
  `ffi.cusolvermp` imports) — absolute imports, location-independent, no edit needed in either
  step.

---

## 4. RISK TABLE & ROLLBACK

| Risk | Likelihood | Mitigation |
|---|---|---|
| Naive `git mv` leaves `from .` relative imports → module fails at import, all gates die | Certain if skipped | §2b rewrites all 10 sites **in the same commit**; `python -c "import gw.isdf_fitting"` catches it instantly |
| Circular import `gw→common→gw` | None (verified) | No load-time `gw` edge in `isdf_fitting` or its `common` deps; direction stays `gw→common` |
| Missed a top-level importer (`htransform.py:27`, `isdf_zeta_mode_test.py:48`, `test_zq:35`) → collection failure | Low | §2c enumerates all 12; py_compile + full pytest collection catches it |
| Missed a function-local importer (gw_init, test_padding, pivoted_cholesky) → runtime fail when exercised | Low | The 3 gates + suite exercise every one of these paths |
| Accidental logic change smuggled into the "pure" move | Low | Gates must stay **bit-identical**; any value drift ⇒ not pure ⇒ revert & re-diff |
| STEP 2 probe lift changes `_track_peak` sampler → wrong `peak_bytes` to planner | Low | Separate commit; re-run gates + suite; compare `peak_bytes` before/after |
| STEP 2 cache rename desyncs `gw_init.py:703` `_fit_one_rchunk_cache.clear()` | Low | Recommendation is docstring-only (no rename); if renamed, update 703 in the same commit |
| Heavier import (moved file now triggers `gw/__init__` side effects: `JAX_ENABLE_X64=1`, eager `gw_init`/`gw_config`, `h5py`) | Benign | No correctness impact; already the norm for any `gw` import |

**Rollback:** STEP 1 is a rename + import repath — `git revert <commit>` restores the file to
`common/` and all import lines atomically, clean, no data/state migration. STEP 2 is an
independent commit and reverts independently without disturbing STEP 1. Keeping the two as
separate commits is what makes each rollback clean.

---

## 5. EFFORT / VALUE — is this worth doing now?

- **STEP 1 (pure move): worth doing, low effort, low risk.** ~12 import statements + 10 in-file
  repaths, all mechanical and fully enumerated above. No `__init__` edits, no cycle, no logic.
  The suite + 3 gates give a hard bit-identical checkpoint. This is exactly the kind of MAP
  move the roadmap wants: it relocates the A2 ζ-fit STAGE into `gw/` where its consumers live,
  with a provable no-op. Do it as its own commit.
- **STEP 2 (extractions): worth doing, but strictly smaller than it looks, and separate.** Only
  the probe lift (nets negative LOC, deletes dead `_mem_report`) and cache docstrings. Everything
  else the analyses considered (cache registry, `_track_peak` extraction, `pair_density`
  relocation) is over-engineering and is explicitly excluded. Modest value; do it *after* STEP 1
  is green, in its own commit, re-verified against the gates.
- **Separate commits: YES, required.** STEP 1 must land and go green on its own so its
  bit-identical guarantee is auditable; STEP 2 touches `gw_init` and one runtime line and must
  be revertible without unwinding the move. No blocker was found — proceed.

---

## 10-LINE SUMMARY

1. GO. Pure move is safe — no circular import; direction stays `gw→common` (verified).
2. Not a bare `git mv`: 10 in-file `from .` imports (all target `common/*`) must be rewritten
   to absolute `common.*` in the SAME commit or the module fails at import and every gate dies.
3. Importers to repoint: 6 files, 12 statements (2 top-level src, 3 func-local in `gw_init`,
   1 test in `common/`, 6 in `tests/`). Full old→new table in §2c.
4. No `__init__.py` re-export edits. No name collision in `gw/`. No FFI path touched.
5. STEP 1 verification: `import gw.isdf_fitting` + py_compile + full suite (241/0) + 3 e2e
   gates, all bit-identical (ζ-fit feeds every gate). Zero logic change.
6. Top risk: skipping the 10-site relative-import rewrite — certain breakage, caught instantly
   by a one-line import smoke test.
7. STEP 2 worth doing but small: lift C6 probes to `common/mem_probe.py` (delete dead
   `_mem_report`), add docstrings to the 6 jit caches. Leave `_track_peak`, the caches'
   location, and `pair_density` alone. No cache registry.
8. Keep `gw_init.py:703 _fit_one_rchunk_cache.clear()` in sync only if STEP 2 renames it —
   recommendation: don't rename, docstring only.
9. Rollback is a clean `git revert` (rename + repath, no state migration); the two commits
   revert independently.
10. STEP 1 and STEP 2 are SEPARATE commits. Proceed with STEP 1 now.
