# Round 3 — Agent 4: Defect 3 — `to_rmu` shard_map mirror

You catalogued Defect 3 in `defect_catalog.md`: `load_centroids_band_chunked` + `to_rmu` together materialize an unsharded FFT box at the centroid-load step (Peak A in the planner). Single-slot but still a principle violation — a few GB/rank at CrI3 6×6 80 Ry, smaller in absolute terms than Defect 1 but structurally identical.

**Now**: fix it. Mirror the `gflat_to_rchunk` pattern (Agent 2's commit `d7eaf1c` on `sources/lorrax_B`) for the `to_rmu` direction — produce a `gflat_to_rmu` helper that does the same shard_map + scan, then thread it through `load_centroids_band_chunked` so the centroid-load step stops materializing an unsharded FFT box.

## Comm protocol (file-polling, no orchestrator routing)

- **Shared discussion**: `reports/zeta_rchunk_memory_model_2026-05-13/round3_discussion.md`. Read it at the start of each work cycle. Respond to anything new in "Agent 2 → Agent 4" before continuing.
- **Your work log**: `reports/zeta_rchunk_memory_model_2026-05-13/to_rmu_mirror_log.md` (create it). Append as you go.
- **Blockers needing the human**: append a line starting with `BLOCKER:` to the "Either → Orchestrator" section of `round3_discussion.md`, then stop.

## Scope

Three sub-tasks:

1. **Build `gflat_to_rmu` in `common/wfn_transforms.py`**. Mirror `gflat_to_rchunk` (already in this file, commit `d7eaf1c`) — same shard_map + scan over chunks of the flat `(nk · nb_local)` axis, but the inner body uses `to_rmu_inner` (which you'll extract from existing `to_rmu`, analogously to how `to_rchunk_inner` was extracted from `to_rchunk` in `cdd0fba`) instead of `to_rchunk_inner`.
   - **Signature** (mirror of `gflat_to_rchunk`):
     ```python
     def gflat_to_rmu(psi_G, g_index, r_mu, *, mesh, fft_grid,
                     norm="backward", chunk_size=None) -> jax.Array
     ```
     Returns `(nk, nb_total, ns, n_rmu)` c128, band-axis-sharded.
   - **`to_rmu_inner`**: pure-jax version of `to_rmu`'s shard_map body, takes per-rank-local ψ + g_index + r_mu, returns rank-local `(..., n_rmu)`. Mirrors `to_rchunk_inner` exactly. New helper in `wfn_transforms.py`.

2. **Thread it through the centroid load path**. Find `load_centroids_band_chunked` (in `src/common/load_wfns.py` per the defect catalog) and replace its Python-unrolled bc-loop with a single `gflat_to_rmu` call. The output shape should match what callers currently expect (`(nk, nb_total, ns, n_rmu)`).

3. **CPU bit-identity test in `tests/test_wfn_transforms.py`**. Mirror Agent 2's pattern: `_gflat_to_rmu_reference` does the bc-loop + concat against `to_rmu`; new helper must match to `rtol=1e-10, atol=1e-12`. Two tests minimum (with random `r_mu`, with chunk_size sweep).

## Reference reading order

1. `parallel_helpers_design.md` §1-3 (Agent 1's symmetry framework — directly applicable to your mirror).
2. `accumulate_rchunk_to_gflat` (in `common/wfn_transforms.py`, ~line 805) — the canonical clean reference pattern.
3. `gflat_to_rchunk` (same file, commit `d7eaf1c`) — Agent 2's freshly-built sibling. Your work mirrors its structure exactly modulo the `to_rmu` inner body.
4. Current `to_rmu` and `load_centroids_band_chunked` for what you're replacing.

## Validation gates (do NOT proceed past a failed gate)

1. **CPU pytest** for `test_wfn_transforms.py`. All must pass (existing 27 + your new 2 ⇒ 29).
2. **MoS2 3×3 centroid-load bit-identity** (if the test infra exists; otherwise just confirm the gflat_to_rmu output matches the bc-loop+concat output for a real loaded WFN).
3. **HLO slot-count check** at synth scale: the new `load_centroids_band_chunked` path should show 1 FFT-box slot (not N_BC). Use Agent 2's HLO test pattern in `gflat_to_rchunk` tests as a template if available.

## Commit & log

- Single commit on `agent/zeta-bc-scan-shardmap` (same branch as Agent 2). You touch DIFFERENT FILES from Agent 2 (`load_wfns.py` + new `gflat_to_rmu` in `wfn_transforms.py`); they touch `isdf_fitting.py`, `psi_G_store.py`, `gflat_memory_model.py`. If you find yourselves about to edit the same file at the same line range, flag it in `round3_discussion.md` and stop.
- Append final status to `to_rmu_mirror_log.md`.
- Print "Agent 4 to_rmu done" when committed AND the log is published.

## Constraints

- **Work on `sources/lorrax_B` only**. Do not touch lorrax_A this round.
- Branch: `agent/zeta-bc-scan-shardmap`. Don't create a new branch.
- Re-read `round3_discussion.md` before each substantive edit cycle.
- The principle: zero replicated intermediates. After your commit, the centroid-load step's HLO should show 1 FFT-box-class slot (not N_BC).

Start by reading `round3_discussion.md` and writing your "Agent 4 starting on to_rmu mirror" announcement so Agent 2 sees you're live.
