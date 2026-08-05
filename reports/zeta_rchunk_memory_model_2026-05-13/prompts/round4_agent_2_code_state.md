# Round 4 — Agent 2: State of the code

Read `round4_discussion.md` first (status snapshot + file-polling protocol).

## Your task

Audit the **complete state of the code** across `sources/lorrax_A` and `sources/lorrax_B` post-Path-D. Three sub-questions:

### 1. What's on lorrax_B vs main (the Path D delta)

`sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap` has 5 commits ahead of main:
- `cdd0fba` — Path D scaffolding (`to_rchunk_inner`, `_slice_local_tile_bc`)
- `d7eaf1c` — `gflat_to_rchunk` forward helper
- `3d0636c` — `qvec_frac` content-hash in `accumulate_rchunk_to_gflat` cache
- `3606138` — `gflat_to_rmu` + `to_rmu_inner` (Defect 3 mirror)
- `5cadd4b` — integration into `_kernel`, `band_fft_pool` removed

Walk the diff (`git log -p main..agent/zeta-bc-scan-shardmap`) and produce a per-commit summary of:
- Files touched and the substantive change.
- New public API surface (helpers, properties, cohsex.in fields, etc.).
- Removed API surface (`psig_k_chunk_size`, `band_fft_pool`, `_slice_local_tile_bc`, etc.).
- Any cohsex.in field changes (added, removed, renamed, defaulted differently).

### 2. What's still on lorrax_A as stopgap

`sources/lorrax_A` branch `agent/zeta-r-chunk-fixes-2026-05-13` @ `ff5873c` has the morning's planner accommodation:
- `_bytes_centroids_LR` helper for the centroid_persist fix.
- `band_fft_unsharded` term in `_peak_C_fit_one_rchunk`.
- Structural feasibility `raise ValueError` if `band_fft_pool > budget`.
- `psig_k_chunk_size` threading.

Which of these survive in lorrax_B post-`5cadd4b`? Which should be cherry-picked back into lorrax_B (or vice versa)? What should be dropped because Path D eliminates the underlying mechanism?

### 3. The cohsex.in user-facing surface

What's the current set of memory-related cohsex.in fields after `5cadd4b`? Diff against pre-Path-D (`488e870` is the last upstream commit on lorrax_B before our work). Are there any new fields the user needs to know about? Any documentation gaps?

## Deliverable

Write to `reports/zeta_rchunk_memory_model_2026-05-13/round4_code_state.md`. Sections:
1. **lorrax_B commits — per-commit summary table**.
2. **lorrax_A vs lorrax_B reconciliation** — what to cherry-pick, what to drop.
3. **cohsex.in surface diff** — what the user actually sets now.
4. **Recommended branch-management action**: merge lorrax_B back to main? Cherry-pick? Squash? Force-push?

Communicate via `round4_discussion.md`. Print "Agent 2 round 4 done" when finished.

Read-only on `sources/`. No compute. ~15–20 min.
