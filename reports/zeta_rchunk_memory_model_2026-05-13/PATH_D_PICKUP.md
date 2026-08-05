# Path D pickup notes (2026-05-13 → next session)

You (the future me) walked into this with `consensus.md`, `hlo_findings.md`, and `agent_2_structural_fix.md` already read. This file is the **delta** — what's done, what's not, and the half-hour of context that's *not* in those reports because it lived in implementation while-I-was-typing.

## 0. The principle this is all in service of (READ FIRST)

> **There should be zero replicated intermediates or work done in this entire procedure.**

Per the user, 2026-05-13. Re-read this if you find yourself reasoning about "how to fit the unsharded pool into the budget" or "how conservative can the planner be." Those are wrong-framed questions. **Any** replicated buffer or repeated computation in `fit_one_rchunk` (or the surrounding pipeline) is a defect to fix, not a resource to model around.

What this means for the work in front of you:

- The planner's `band_fft_pool` term landed in commit `ff5873c` on `agent/zeta-r-chunk-fixes-2026-05-13` is a **stopgap** — it makes the existing code safe by refusing infeasible configs, but it codifies a bug as a budgeted line item. Don't extend that pattern. Don't tune `S_fft` or `band_fft_slots`. Don't refine the planner's accuracy against XLA's actual aliasing. Fix the underlying bug instead.
- Path D's success criterion is **slot count → 0 unsharded transient**, not "slot count ≤ 3 and the planner predicts tightly." The HLO validation after the kernel rewrite should show NO `c128[k_chunk, bc, ns, nx, ny, nz]` slots in any allocation listing — full stop. Three aliased slots is still three replicated intermediates and is still a bug.
- Other known instances of the same defect (audit and fix, not budget):
  - **Peak A — centroid-load FFT box** (`gflat_memory_model._peak_A_centroid_load`, line ~139): same `c128[nk, band_chunk, ns, n_rtot]` shape, single slot but unsharded. Single-slot is no excuse — the *factor* differs but the principle violation is the same. Add to Path D scope.
  - **`fetch_psi_rchunk`'s inner k-chunk Python loop** (`psi_G_store.py:368-378`): introduces `n_kchunk` × FFT-box slot pile-up *within* each bc iter. Path D §F (the hybrid) collapses this; Path D §C/D collapse the whole bc-loop and incidentally fix this too.
  - **`solve_zeta`'s q-batch Python loop** (`isdf_fitting.py:1119-1141`): the comment there says Python is "the only approach" — that was the right call given what the author had tried, but the principle says the q-batch should also collapse. Whether that's same-shape (move loop inside a shard_map / use scan-inside) or different-shape (cuSolverMp call across batches without staging through a sharded Z_col) is an open structural question. **Not in Path D scope today, but log it as the next defect after Path D lands.**
  - Anywhere else in the pipeline where the HLO shows `n_*` concurrent slots with the same shape, or where Python-level iteration sits inside a jit body. Spend 30 min grepping `for .* in .*range\|for .* in .*chunk` inside any `@jax.jit`-decorated function to find them.

The current planner term (`band_fft_pool`) and feasibility raise can stay as a safety net while Path D is in flight, but **mark them for removal** in the same commit that lands Path D 4c-e — they describe a defect that no longer exists.

## What's already committed (don't redo)

### `sources/lorrax_A` — branch `agent/zeta-r-chunk-fixes-2026-05-13`, commit `ff5873c`
The planner-accounting fix. Three sub-fixes:
- `_bytes_centroids_LR` helper replaces the broken `2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy)` formula at three sites in `gflat_memory_model.py`. Confirms 4× correction on balanced mesh.
- `band_fft_unsharded` term in `_peak_C_fit_one_rchunk` — accounts for the Python-unrolled bc-loop's `N_BC × S_fft` slot pile-up.
- `plan_gflat_chunks` raises `ValueError` with mitigations when the `band_fft_pool` term alone exceeds budget.
- `gw_init.fit_zeta` threads `cfg.memory.psig_k_chunk_size` through.

### `sources/lorrax_B` — branch `agent/zeta-bc-scan-shardmap`, commit `cdd0fba`
Path D scaffolding only. Two helper functions, three tests, no kernel rewrite yet:
- `common.wfn_transforms.to_rchunk_inner` — pure-jax body of `to_rchunk` (no shard_map wrapper). Callable from inside another shard_map or scan body. Three tests in `tests/test_wfn_transforms.py` verify bit-identity against `to_rchunk` on 1×1 mesh: no-phase, with-phase, traced `r0`.
- `common.psi_G_store.PsiGStore._slice_local_tile_bc` — host-tile slicer that takes a *traced* `bc_idx`, returns padded `(nk, _bpd_max, ns, ngkmax)`. Added `_bpd_max` field to `__init__`. No unit test on this one — exercised only via the eventual scan-body integration; if you change it before that integration, write a stand-alone test.

**Both branches build and pass all relevant unit tests.** Lorrax_B's branch is the one to continue Path D on — it has the scaffolding already.

## What's still to do — Path D §4c-e (the load-bearing piece)

From `agent_2_structural_fix.md` §4c-e. Order matters:

### Step 1: Decide on the band-mask vs band-slice approach for the per-bc L/R split

The scan body has to take per-bc ψ data (shape `(nk, bpd_max, ns, r_chunk)`) and produce L and R slices. The two approaches:

- **Mask approach** (agent_2 §4c sketch): pre-build per-bc L/R slice tables (`l_lo_tbl`, `l_hi_tbl`, etc.) as static arrays. Inside scan, `jnp.where` zeros out bands outside the L (resp. R) window. Pro: uniform shape, easy to scan over. Con: wasted FLOPs on zero-masked bands; relies on `0 * anything == 0` semantics.
- **Slice approach**: use `lax.dynamic_slice_in_dim` with traced indices from the tables. Pro: no wasted FLOPs. Con: requires the L slice and R slice to have a static *length* at trace time — only possible if all bcs have the same `nb_L` and `nb_R` per-bc (often false in the last bc).

Pick mask approach first (simpler). If the wasted FLOPs are >5% measurable in profiling, revisit.

### Step 2: Implement `z_q_from_psi_sm` (and `c_q_from_psi_sm`) with scan inside shard_map

The new signature drops the pre-computed `psi_l_Y` and `psi_r_Y` arguments and instead takes:
- `psi_l_X`, `psi_r_X` (the X-sharded centroid-side tensors — same as today)
- `psi_G_store` (closure, for `_slice_local_tile_bc`)
- `band_chunk_ranges`, `band_range_left`, `band_range_right` (static)
- `fft_grid`, `r_start_dyn`, `r_chunk_size` (`r_start_dyn` traced)

Inside the new `_local`:
1. Init rank-5 `P_l_acc` / `P_r_acc` accumulators at shape `(nk, ns, r_chunk_loc, mu_loc, ns)`.
2. `lax.scan` over `jnp.arange(n_bc)`. Body:
   - `io_callback(_slice_local_tile_bc, ..., x_idx, y_idx, bc_idx)` → padded `psi_G_bc`.
   - `to_rchunk_inner(psi_G_bc, ...)` → `psi_Y_bc` per-rank-local.
   - Apply L/R masks via tables (see Step 1).
   - Two einsums into accumulators.
3. Existing post-pair pipeline (IFFT → γ̃ → FFT → transpose) on `P_l_acc`, `P_r_acc`.

`c_q_from_psi_sm` is structurally identical — `n_col == n_rmu` instead of `r_chunk`. Probably worth factoring the shared body into a helper that takes a `col_dim` parameter.

### Step 3: Simplify `_make_fit_one_rchunk_kernel._kernel`

Delete the Python `for bc_range in band_chunk_ranges` loop and the `jnp.concatenate(psi_Y_parts, axis=1)`. Call the new `z_q_from_psi_sm` directly.

### Step 4: Validation

In order:
1. **CPU bit-identity** on MoS2 3×3 synth WFN — run `tests/test_isdf_fit.py` (if it exists; otherwise write one for `c_q_from_psi_sm` first). Need rtol=1e-10, atol=1e-12.
2. **GPU run** on MoS2 3×3 — full ζ-fit end-to-end against the lorrax_A baseline; eqp0.dat should match to existing tolerance.
3. **3rd HLO dump** at CrI3 6×6 80 Ry with `psig_k_chunk=6` (the OOMing config). Expected: the 58 FFT-box slots collapse to ~3. Total preallocated-temp ≤ 30 GiB instead of 200 GiB. **This is the killer validation — if the slot count doesn't collapse, the structural assumption was wrong.**
4. **CrI3 COHSEX vs BGW** — Σ values match the LORRAX_A baseline within existing tolerance.

## Gotchas you'll hit (do not relearn)

1. **`io_callback` inside `lax.scan` inside `shard_map` is novel territory.** Agent 2's sketch shows it but it's not a pattern with prior art in this codebase. If it doesn't work, fall back to **Path C** (driver-level Python loop with donation chains) — `agent_2_structural_fix.md` §C has the sketch; gives ~80% of the memory benefit with mechanically simpler code.
2. **The `_kernel` lives inside `@jax.jit` already** (line 1229ish of `isdf_fitting.py`). The new scan-inside-shard_map is *another* level of nesting. JAX supports this but compile times can grow non-linearly. Profile compile time on MoS2 3×3 before moving to CrI3.
3. **`fetch_psi_rchunk` (the existing method, lines 268-380 of `psi_G_store.py`) is still in use elsewhere** — don't delete it. Path D adds a new code path, doesn't replace the old.
4. **`band_chunk_ranges` is a tuple of `(b_lo, b_hi)` global band indices**, not local. The `_bc_band_offsets` field gives the local (per-rank) offsets. Don't confuse them — the slicer in `_slice_local_tile_bc` uses the *local* offsets correctly; the L/R band mask tables in the scan body need to work in *local* coords too.
5. **`norms_l` / `norms_r` (per-band normalization factors) are sized to the L / R band-window widths**, not the full band axis. The current code does `psi_Y_full[:, _l_lo:_l_hi, :, :] / norms_l[None, :, None, None]`. In Path D you'll need to either: (a) apply the norm divide *inside* the scan body using a per-bc slice of the norm vector, or (b) pre-scale `psi_l_X` / `psi_r_X` outside the scan (cleaner — divides constants once instead of n_bc times).
6. **The `lorrax_A` branch's planner was made over-conservative by today's work.** When you re-run a CrI3 fit on the structural-fixed `lorrax_B` branch, the planner inherits all of `lorrax_A`'s peak-C accounting (after rebase / cherry-pick). The W_wfn term should *drop substantially* once Path D is in — but the planner doesn't know that yet. **Either reduce `band_fft_slots` constant to a post-Path-D value (~1?), or refactor the planner term to be Path-D-aware (preferred).**

## First thing to do tomorrow

Decide between mask vs slice (Step 1) and write a 50-line CPU bit-identity test that pre-fakes a `psi_G_store` mock and exercises just the new `z_q_from_psi_sm._local` body against the current implementation. **Don't touch the kernel until that test passes on the current code as a baseline** — once you have a working bit-identity gate, the refactor is bounded.

## Live state at handoff

- Allocation `52926821` cancelled (no point holding).
- tmux `zeta_team` killed.
- 2nd HLO dump: fit_zeta loop ran past the previously-OOMing point without memory failure — **confirms the memory model fix works**. The run did *not* finish cleanly though: it crashed downstream at `write_qp_wfn_h5` (`U shape (36, 150, 150) inconsistent with (nk=8, nb_active=150)`) — that's a symlink mismatch on `qp_wfn_rotations.h5` from the parent perf-run dir (parent had `nk=8`, our run wants `nk=36`). **Unrelated to the memory work.** Files at `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13/` (~3290 dump files, ~50 MB). Keep until Path D HLO comparison; delete after.
- 1st HLO dump at `lorrax_A_hlo_dump_2026-05-13/` is the OOMing baseline — keep for the slot-count comparison.
- `agent_2_structural_fix.md` §4 has the full implementation sketch. This pickup doc complements it; both are needed.
