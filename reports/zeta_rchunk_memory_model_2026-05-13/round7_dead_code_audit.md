# Round 7 — Dead-code audit (Agent 1, WS-B)

**Branch**: `sources/lorrax_B agent/zeta-bc-scan-shardmap @ f567aa0` (Round 6 commit, pre-back-pad-fix).

**Method**: grepped every `.py` file under `sources/lorrax_B/` for each item's name and adjacent regressions. Distinguished `def`/`class`/property definitions from comment-only references. For load-bearing items, identified production call sites.

**Cleanup posture per the mission**: catalog only, queue cleanup for after G1 re-passes (after Agent 2's back-pad fix lands).

---

## Summary table

| Item | Definition still present? | Production callers? | Status |
|---|---|---|---|
| `gflat_to_rchunk` (standalone helper) | ❌ deleted | 0 | **DEAD** — only stale comments remain |
| `to_rchunk_inner` | ✅ at `wfn_transforms.py:340` | 1 (`isdf_fitting.py:623`) | **LOAD-BEARING** — production-critical |
| `_GFLAT_TO_RCHUNK_CACHE` (module-level dict) | ❌ deleted | 0 | **GONE** — no cleanup needed |
| `PsiGStore.psi_G_device_full` (lazy property) | ❌ deleted | 0 | **DEAD** — only stale comments + a regression test that *guards* the removal |
| Orphan `return` at `isdf_fitting.py:770-772` (Round 6 [CONCERN]) | ✅ unreachable dead code | 0 (never executes) | **DEAD** — 3-line cleanup pending |

**Net for cleanup commit (after G1 re-passes)**: 1 source-line block (the orphan return, 3 lines) + 5 stale comment/docstring lines across 4 files. No production-API impact.

---

## 1. `gflat_to_rchunk` (standalone helper)

### Definition

✅ **Already removed.** `grep -rn "^def gflat_to_rchunk"` in `sources/lorrax_B/` returns 0 hits. The function body that lived in `wfn_transforms.py` (D-ranged Round 3 design at commit `d7eaf1c`) is gone as of `f567aa0`.

### Remaining references (all comment-only — non-functional)

| Site | Type | Content |
|---|---|---|
| `tests/test_wfn_transforms.py:333` | comment in test docstring | Cross-reference for a related test pattern. |
| `src/common/psi_G_store.py:316` | docstring of `g_index` property | "Used by ``gflat_to_rchunk``." — stale; the actual consumer is `z_q_from_psi_sm._local`'s scan body (via `g_index_spec` in_spec, isdf_fitting.py:541). |
| `src/common/wfn_transforms.py:515` | section header comment | References to the helper as a sibling of `accumulate_rchunk_to_gflat`. |
| `src/common/wfn_transforms.py:525` | docstring of `gflat_to_rmu` | Cross-references the now-deleted forward helper. |
| `src/common/wfn_transforms.py:590, 741, 813, 991` | comments in `gflat_to_rmu` body | Cross-references and design lessons. |
| `src/gw/gw_init.py:627` | comment about deleted cohsex knob | Notes that `gflat_to_rchunk_chunk_size` was retired. Accurate; cleanup-worthy because it documents an absent knob. |
| `src/gw/aot_memory_model/kernels/fit_one_rchunk.py:455` | AOT stub comment | **STALE**: describes the AOT model as emitting "one ``gflat_to_rchunk`` shard_map+scan" — actually it emits the new per-bc io_callback scan inside `z_q_from_psi_sm._local`. The AOT stub class (`_AotStubPsiGStore`, line 72) correctly implements the new API, but this comment lags. |

### Recommended action

- **`fit_one_rchunk.py:455`** — rewrite the comment block (lines ~448-459) to describe the new flow: "one per-bc io_callback per scan iter inside `z_q_from_psi_sm._local`, one all_gather across `('x','y')` on the band axis, one pair-density accumulate per bc, one Cholesky solve". The stub class itself is already correctly shaped; only the comment lags.
- **`psi_G_store.py:316`** — replace "Used by ``gflat_to_rchunk``." with "Used by ``z_q_from_psi_sm._local`` via the shard_map's `g_index_spec` in_spec."
- Other comment references (in `wfn_transforms.py`, `tests/test_wfn_transforms.py`, `gw_init.py`) are minor archeology and can stay or be lightly updated in the same cleanup commit.

---

## 2. `to_rchunk_inner`

### Definition

✅ Present at `src/common/wfn_transforms.py:340`. Exported via `__all__` at line 54.

### Production call sites

✅ **1 caller in production code**:
- `src/common/isdf_fitting.py:623` — inside `z_q_from_psi_sm._local`'s scan body. This is the IFFT primitive that maps per-rank ψ(G_local) → ψ(r_chunk) before the per-iter `all_gather`. **Load-bearing.**

Other references (none are production callers):
- Imported at `src/common/isdf_fitting.py:47`.
- Tested at `tests/test_wfn_transforms.py:207-260` (three bit-identity tests vs `to_rchunk`).
- Referenced in docstrings.

### Status

**`to_rchunk_inner` is NOT dead.** It's the per-rank pure-jax IFFT body that `z_q_from_psi_sm._local`'s scan invokes per iter (after `io_callback`, before `all_gather`). Deleting it would break the Round 6 kernel. Tests guard the contract.

**No cleanup action.** Keep as-is.

---

## 3. `_GFLAT_TO_RCHUNK_CACHE`

### Definition

❌ **Completely removed.** `grep -rn "_GFLAT_TO_RCHUNK_CACHE"` returns 0 hits across all `.py` files.

This was a module-level cache dict in the old `gflat_to_rchunk` helper. When the helper was deleted, the cache went with it. No orphan reference remains.

### Status

**Nothing to clean up.** Audit confirms the removal is complete.

---

## 4. `PsiGStore.psi_G_device_full` (lazy property) + `_psi_G_device_full` (cache slot)

### Definition

❌ **Property and cache slot both removed** from `src/common/psi_G_store.py`. The Round 6 commit `f567aa0` rewrote the consumer flow so the kernel no longer needs a full-device ψ(G) tile — instead each bc is pulled per scan iter via `_slice_local_tile_bc`.

### Remaining references (all comment-only or test-guard)

| Site | Type | Content |
|---|---|---|
| `tests/test_psi_g_store.py:159-166` | **regression test** | `test_psi_G_device_full_property_removed()` — `assert not hasattr(store, "psi_G_device_full")` / `assert not hasattr(store, "_psi_G_device_full")`. **Keep this test** — it guards against accidental reintroduction of the buggy lazy-cache pattern (tracer-leak source per Round 5 mission). |
| `src/common/psi_G_store.py:255` | section-header comment in `_slice_local_tile_bc` block | Notes that the helper was originally added (commit `cdd0fba`), removed during the flat-axis `psi_G_device_full` path, and restored in Round 6. Accurate archeology — leave or trim. |
| `src/gw/aot_memory_model/kernels/fit_one_rchunk.py:451` | AOT stub comment | **STALE**: describes the AOT stub as having a `psi_G_device_full` property. The actual `_AotStubPsiGStore` class (line 72) implements `_slice_local_tile_bc` / `g_index` / `kvecs_frac` — NOT `psi_G_device_full`. Comment lags. |

### Recommended action

- **`fit_one_rchunk.py:451`** — same comment block as `gflat_to_rchunk:455` (above); fix both in the same edit. The line range to rewrite is roughly 448-459.
- **Keep the regression test** at `tests/test_psi_g_store.py:159-166`. It's exactly the kind of "guard the removal" check that prevents the lazy-property tracer-leak bug from coming back.

---

## 5. Round 6 [CONCERN] still outstanding — orphan `return` statement

### Location

`src/common/isdf_fitting.py:770-772` (tab-indented, inside `z_q_from_psi_sm` after the real return at line 761-763):

```
^Ireturn _pair_pipeline_sm_cache[cache_key](
^I^Ipsi_l_X, psi_l_Y, psi_r_X, psi_r_Y,
^I^Iperm_L, phase_L, perm_R, phase_R)
```

### Why it parses

Python's tokenizer skips comments/blank lines for INDENT/DEDENT tracking. Between the real return (line 761-763) and this orphan (line 770-772) there's a comment block at column 0 (`# Backward-compat shim removed`) and a blank line — neither emits a DEDENT token. So Python parses these lines as INSIDE `z_q_from_psi_sm` (function body indent preserved), just unreachable after the prior `return`.

### Why it's harmless at runtime

- Never executed (dead code after the first `return`).
- The `NameError` on `psi_l_Y` / `psi_r_Y` only fires if Python evaluates the expression, which it never does.
- Module loads fine; tests pass (G0 didn't catch this because it's not executed).

### Why it should still be cleaned up

- Reader confusion: a future reader skimming `z_q_from_psi_sm` will find a second `return` referencing `psi_l_Y` (which is a real parameter of `c_q_from_psi_sm` at line 258) and may misread the function signature.
- Linters (pyflakes, ruff) flag unreachable code; commit-time pre-commit hooks could fail.
- The accompanying comment block (lines 766-768, "Backward-compat shim removed — old z_q_from_psi_sm signature ... is gone") is *correct* and can stay — only the orphan `return` 3 lines need to go.

### Recommended action

3-line delete of lines 770-772 (the orphan return statement). Keep the comment block at 766-768 (accurate archeology).

---

## Cleanup-commit shape (for after G1 re-passes)

A single commit on `lorrax_B agent/zeta-bc-scan-shardmap` with these changes:

1. **`src/common/isdf_fitting.py`** — delete lines 770-772 (orphan return statement).
2. **`src/common/psi_G_store.py:316`** — replace `gflat_to_rchunk` reference in `g_index` docstring with `z_q_from_psi_sm._local`.
3. **`src/common/psi_G_store.py:255`** — optional trim of the `_slice_local_tile_bc` archeology header; leave if Agent 2 prefers archeology-preserving.
4. **`src/gw/aot_memory_model/kernels/fit_one_rchunk.py:448-459`** — rewrite the AOT stub comment block to describe the new flow (per-bc io_callback + all_gather + pair-density accumulate, NOT `psi_G_device_full` + `gflat_to_rchunk`).
5. **`src/gw/gw_init.py:627`** — optional: simplify the retired-knob comment now that it's been retired for a round.
6. **Various other comment references** in `wfn_transforms.py` to `gflat_to_rchunk` — leave unless cleanup includes a documentation pass.

**Do NOT delete**:
- `to_rchunk_inner` (load-bearing).
- The regression test `tests/test_psi_g_store.py:test_psi_G_device_full_property_removed()` (guards the removal).
- The `_AotStubPsiGStore` class itself (correctly shaped for the new API).

**Estimated cleanup-commit size**: ~3 source-line deletions + ~10 comment-line edits across 4 files. Net negative diff.

---

## Open questions / non-actions

- Is `_AotStubPsiGStore` itself still exercised? Agent 1's lens is SPMD-safety, not AOT validity. **Defer to Agent 3** — the AOT model is part of the planner pipeline, and Agent 3 owns the planner-side audit in Round 7's plan-record amendments work.
- Do any test files import `gflat_to_rchunk` or `psi_G_device_full` directly? Checked — no direct imports (only string references in docstrings/tests-that-verify-absence). No test files need updating.

---

## Status

**Agent 1 round 7 audit done.** Cleanup queued for after G1 re-passes (Agent 2's back-pad fix landing). Standing by to re-review Agent 2's diff when the fix posts.
