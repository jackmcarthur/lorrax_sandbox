# Memory planner — what it does, and how maintainable it is

_Written 2026-07-02 after Phase 1 (dead-package delete) + docs/docstring rewrite._
_This is the "is it good?" companion to PLAN.md's "what to cut"._

## What the model is actually doing (the physics)

The GW/ISDF pipeline is **memory-bound, not compute-bound**, on an A100. The
binding constraint is materializing a few large tensors during ζ-fit and V_q,
and the physics fixes their shapes:

- ISDF fits interpolation vectors ζ_μ to the pair densities ψ_i(r)ψ_j(r) by
  least squares. That forces holding, at various moments: **ψ(r) FFT boxes**
  (n_rtot ≈ 1e4–1e5 grid points), the **CCT metric** C_q = MM† and its Cholesky
  factor (n_rmu²), and the **rank-5 pair-density accumulators** P_l/P_r indexed
  by (μ, r_chunk, …). Those are what blow the budget.

The five peaks A–E are the five moments where one of those spikes: centroid
load (A), CCT/Cholesky (B), the fused inner `fit_one_rchunk` (C, usually the
binding peak), the G-flat accumulate (D), and the per-tile V_q (E). The planner
writes each peak as a **closed-form byte count** in the chunk sizes + system
dims, finds the binding one, and picks the largest chunks that keep it under
`target_utilization · budget`. It is analytic, not measured — with one
exception: XLA's cuFFT plan scratch is opaque, so `runtime/aot_memory` queries
it live. That analytic-vs-measured split is the real conceptual spine and it is
sound.

## Is it maintainable / well-consolidated? Partly. Honest ledger.

**Better than it was.** One planner instead of three competing models is the big
win (before, you could not tell which of the "four choosers" drove sizing — it
was gflat, clobbering the rest). Closed-form + microsecond + unit-tested is the
*right* design for a planner. The peaks are already factored into separate
`_peak_A…_peak_E` functions with a clean `GFlatChunkPlan` output boundary. Decent
bones.

**But it is not yet "well-consolidated and easily editable":**

1. **V_q is still doubly-modeled.** gflat has Peak E *and*
   `v_q_tile._aot_fft_model` estimates the same kernel a second way. A future
   editor could update one and not the other. (Phase 3 fixes this.)

2. **The peak coefficients have no maintainable derivation.** `factor_D = 2.0`,
   `pair_density_slots = 3/4`, the sphere-idx buffer counts — I understand what
   each *represents*, but they are empirical, and the only justification ever
   written lived in agent-report archaeology (now removed from the docstring for
   legibility). So the numbers are more "magic" than before. If someone changes a
   kernel's buffer structure, nothing tells them how to update the peak formula.
   This is the real gap; the cleanup improved legibility but did not close it.

3. **It models an opaque compiler, so it is inherently brittle.** `3 slots on GPU,
   4 on CPU` is an XLA BufferAssignment quirk, not physics. A JAX bump can shift
   scheduling and silently break the prediction. The deleted DOE framework was a
   (dead) attempt to auto-recalibrate; deleting it was right, but the *need* —
   periodic re-validation against real HLO — is now carried only by the manual
   planner-refit tests.

## What "well-consolidated and editable" would take (the end state)

Each peak = a small pure function `peak_C(dims, chunks) -> bytes` whose docstring
**derives** its term count from the array shapes (which the physics fixes); each
coefficient pinned by **one** HLO-anchored test that fails loudly if XLA changes;
and **one** V_q model. Then editing a kernel is: change the kernel, update one
peak function, one test tells you if the bytes are right. Achievable from here —
the `_peak_X` factoring is already the seam. It is Phase 2/3 **plus** a
"derive, don't just assert, each coefficient" pass on the formulas (a larger,
separate effort — flagged here so it is not forgotten).

## Verdict

I understand the intent and structure; the cleanup made it materially more
legible and honest (one planner, docs match reality). It is **not finished** as a
maintainable subsystem — the redundant V_q model and the underived coefficients
are the two things between "one clean planner" and "a future model can edit this
confidently."

## Phase 3 correction — DO NOT do the V_q consolidation as planned (2026-07-02)

PLAN.md §2 #10 / §3 assumed gflat Peak E and `v_q_tile._choose_v_q_chunks` are
"two models of the same kernel." **Verified false.** They model two *different*
live V_q code paths:

- gflat `_peak_E_v_q_per_tile_transient` models the **G-flat** per-q kernel
  (`v_q_g_flat.py:271-465`, charge path via `compute_all_V_q_g_flat`). That path
  does NOT call `_choose_v_q_chunks` (grep-confirmed).
- `_choose_v_q_chunks` sizes the **r-space tile** kernel
  (`v_q_tile.compute_V_q_tile`), which is live via `v_q_bispinor.py:396` (the
  7-tile bispinor V_q builds each tile with `compute_V_q_tile`) and a
  `compute_vcoul.py:1054` dispatch branch.

So there is no redundant model to collapse — Peak E predicts the G-flat path for
the HWM log, `_choose_v_q_chunks` chooses chunks for the tile path. Merging them
would break tile/bispinor V_q sizing (risk: OOM). Phase 3 is **cancelled as
scoped**.

The real V_q-sizing untangle (a separate, larger initiative if wanted): the V_q
subsystem has two intermingled drivers (`v_q_g_flat` G-flat + `v_q_tile` r-space
tile, with `v_q_bispinor` building G-flat-shaped output out of tile kernels). A
genuine consolidation would first decide the single V_q driver, then unify
sizing — not a tail-end refactor. `_aot_fft_model` (the tile path's FFT-scratch
estimator) is only worth retiring in favor of `runtime.aot_memory` once that
driver decision is made.

**Net for the memory-planner cleanup:** Phase 1 (dead package, ~8.7k lines) +
docs + gflat docstring + Phase 2 (value-preserving gflat simplifications) landed
and are gated green. Phase 3 correctly stopped after verification showed its
premise was wrong.
