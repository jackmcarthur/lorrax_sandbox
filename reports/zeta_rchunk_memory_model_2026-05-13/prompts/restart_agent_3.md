# Agent 3 — Restart: Audit the "clean reverse-direction reference"

Read `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` first — **§0 is the load-bearing principle (zero replicated intermediates)**.

## The work

Agents 1 and 2 are using `common.wfn_transforms.accumulate_rchunk_to_gflat` (in `sources/lorrax_A/src/common/wfn_transforms.py:468`) as **the reference for what a clean shard_map+scan looks like**. They're building the symmetric forward helper to mirror it.

**Your job: independently verify that `accumulate_rchunk_to_gflat` actually adheres to the principle.** If the supposed reference itself has replicated intermediates or other defects, the agents will be mirroring a flawed pattern.

## What to check

1. **Read the function carefully** (lines 468-end). It claims:
   - One `shard_map` over `('x','y')`.
   - `lax.scan` over chunks of the flat `(n_q · n_mu_local)` axis inside the body.
   - Per-iter FFT box aliased by XLA's scan-internal allocator (so one slot, not N).
   - "No cross-rank collectives in the body" per the docstring.

2. **Verify each claim**:
   - Is the scan actually inside the shard_map body? Or is it outside?
   - Are there any `with_sharding_constraint` calls inside the body that could trip the WhileOp/SPMD trap?
   - Are there any Python loops (`for ... in ...`) inside the shard_map body that we'd expect to be `lax.scan`?
   - Is the FFT box (`c128[chunk_size, nx, ny, nz]` or similar) actually aliased per iter, or could XLA pipeline-keep them?
   - Are all intermediates (FFT box, phase tables, gather scratch, accumulator) per-rank-local?

3. **HLO check** (if you can run a small CPU dump or just reason about it from code):
   - Predict the slot count for a small `(n_q, n_rmu_padded)` config. Should be O(1), not O(n_chunks).
   - If you can't run a dump (no GPU on the login node), reason from the code structure.

4. **Look for other defects**:
   - Anything that scales like `chunk_size · n_rtot` and might not alias.
   - Any "we materialize X once and reuse" pattern that's actually replicated per-rank when it should be sharded.

## Output

Write to `reports/zeta_rchunk_memory_model_2026-05-13/accumulate_audit.md`. Structure:

1. **Verdict** — is `accumulate_rchunk_to_gflat` actually clean? Yes / no / with-caveats.
2. **Evidence** — line-by-line walk through with citations.
3. **Found defects (if any)** — what's wrong, and what the fix would look like.
4. **Recommendations for the parallel forward helper** — given what you found, what should Agents 1 and 2 emulate vs avoid?

Print "Agent 3 audit done" when finished.

## Constraints

- Read-only on `sources/lorrax_A/`.
- Don't read other restart agents' output files until your audit is done.
- Stay in this tmux pane.
- The principle: zero replicated intermediates. Be ruthless — if you find one, document it even if it's "just" in a function the other agents are building on.
