# Round 4 — Agent 3: State of the memory model

Read `round4_discussion.md` first (status snapshot + file-polling protocol).

## Your task

Audit the **memory model** as it stands now on `lorrax_B` post-`5cadd4b`. The accommodation term (`band_fft_pool`) is removed. What does the planner actually model now? Where is it tight, where is it loose, and where does it not match XLA's actual allocations?

### Specific questions

1. **Per-peak breakdown** — what terms does `plan_gflat_chunks` use for each of Peak A (centroid load), Peak B (CCT/Chol), Peak C (fit_one_rchunk), Peak D (accumulate)? List each term, its formula, its sharding divisor. Note any that are r_chunk-dependent vs invariant.

2. **The new `gflat_to_rchunk_chunk_size` auto-pick** — `gw_init.py` now auto-picks this from `cfg.memory.per_device_gb`. What's the heuristic? Is it principled (based on per-iter FFT box bytes) or hand-tuned? Compare to the recommendation in `parallel_helpers_design.md` §2 and the user's discussion about io_chunk_size vs fft_chunk_size separation (see the recent orchestrator-user conversation; principle: I/O batch ≠ FFT batch).

3. **Reconciliation with the CrI3 HLO**:
   - Planner predicted HWM = **51.93 GB**.
   - XLA actual = **48.63 GiB**.
   - Within 7%. That's much tighter than the morning model (predicted 52 GB, actual 200 GiB, 4× off).
   - Walk through the planner's peak breakdown and the HLO's allocation list. Which terms in the model correspond to which allocations in the HLO? Are any HLO allocations not modeled by the planner? Any modeled terms with no HLO counterpart?

4. **The remat warnings**:
   - 20 warnings on `c128[36, 10, 2, 73648]` reshards from `[1,16,1,1]` → `[1,1,1,4,4]T(1,0)+replicate`.
   - Does the planner account for the cost of these remats? (Probably not — remat is XLA implementational.)
   - Should it? Or should the remat be eliminated by fixing the sharding annotation at the helper/consumer boundary (which is the principle answer)?

5. **The principle scorecard**:
   - For each remaining defect that the model line-items, is it (a) a real defect to fix structurally, (b) a tractable model refinement, or (c) genuinely an emergent XLA cost the model just has to track?
   - List by category.

### Deliverable

Write to `reports/zeta_rchunk_memory_model_2026-05-13/round4_memory_model_state.md`. Sections:
1. **Current planner terms** (table per peak).
2. **`gflat_to_rchunk_chunk_size` auto-pick analysis** (is it right? compared to what design said).
3. **HLO vs model reconciliation** (where each term lands in actual allocations).
4. **Remat warnings — modeled? fixable?**
5. **Principle scorecard** — defects vs model-trackable items.
6. **Recommended next planner refinements** (not new features, just where the existing model is wrong/incomplete).

Communicate via `round4_discussion.md`. Print "Agent 3 round 4 done" when finished.

Read-only on `sources/`. ~15–25 min.
