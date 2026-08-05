# Agent 4 — Restart: Comprehensive defect catalog

Read `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` first — **§0 is the load-bearing principle (zero replicated intermediates), and the §0 audit list has the four known instances**.

## The work

Per the principle, **any Python loop materializing N intermediates inside a `@jax.jit` body, or any unsharded buffer that should be sharded, is a defect**. The agents have identified four known instances; your job is to find every other one in the pipeline.

## What to do

1. **Comprehensive grep audit** of `sources/lorrax_A/src/`. The signatures to look for:
   - `for .* in .*range\(` inside any function decorated `@jax.jit` or wrapped in `@partial(shard_map, ...)`.
   - `for .* in .*chunk` inside the same.
   - `for .* in range(` inside the same.
   - Any function name that contains `_kernel`, `_local`, or `_inner` — check whether it has internal Python loops.
   - Any function that uses `_RCHUNK_KERNEL_CACHE` or similar pattern — check whether the cached function has internal Python loops over data axes.

2. **For each hit, determine**:
   - Is it a real principle violation? (Python loop inside a jit body — yes; Python loop in driver code outside a jit — no.)
   - What's the shape/scale at CrI3 6×6 80 Ry (`n_rtot ≈ 1.1M`, `nk = 36`, `nb_total ≈ 310`, `μ ≈ 1500`, `nq = 36`)? Estimate the per-rank bytes that get replicated, and how many iterations.
   - Is it fixable with **scan-inside-shard_map** (the working pattern, per `[[feedback_path_d_scaffolding_pattern]]`)? Or does it need a different structural fix?
   - Is it currently inside a `shard_map` body already? If so, what's blocking the conversion to `lax.scan`?

3. **Don't fix anything** — your job is the inventory, not the implementation. Path D handles the bc-loop / k-chunk loop in `fit_one_rchunk`; future work will handle the others.

4. **Specifically check** (known suspects, verify each):
   - `_make_fit_one_rchunk_kernel._kernel` — the bc-loop at lines ~1273-1276 of `isdf_fitting.py`.
   - `PsiGStore.fetch_psi_rchunk` — the inner k-chunk Python loop at lines ~369-378 of `psi_G_store.py`.
   - `_peak_A_centroid_load` (call site) and the centroid-load code path — is the FFT box single-slot but unsharded? (Single-slot is still a violation by the principle.)
   - `solve_zeta` q-batch loop at `isdf_fitting.py:~1119-1141` — the comment there records failed scan/fori_loop attempts; document this and propose what would have to change for the q-batch to satisfy the principle (might require a different structural approach since the prior attempts hit the WhileOp/SPMD trap).
   - The V_q kernel in `gw/v_q_tile.py` — does it have Python loops inside jit bodies?
   - The σ_X / σ_H computation in `gw/sigma_*.py` — same check.

## Output

Write to `reports/zeta_rchunk_memory_model_2026-05-13/defect_catalog.md`. Structure:

1. **Summary table**: file:line, function, defect type (Python-loop-in-jit / unsharded-intermediate / replicated-temp), estimated cost at CrI3, fixable-by-scan-inside-shard_map (yes/no/different-pattern).
2. **Per-defect details**: code snippet, explanation of why it's a violation, sketch of structural fix.
3. **Prioritization**: which defects bite hardest at CrI3-scale? (Probably the bc-loop is #1 since we measured 200 GiB. Others are likely smaller.)
4. **Not-actually-violations** — any candidate hits you investigated and ruled out (with reasoning), so future audits don't re-flag them.

Print "Agent 4 catalog done" when finished.

## Constraints

- Read-only on `sources/lorrax_A/`. No edits.
- Don't read other restart agents' output until your catalog is done.
- Be exhaustive — the goal is the master inventory for Path D++ and beyond. Better to over-flag and let future-me filter than to miss something.
- Stay in this tmux pane.
