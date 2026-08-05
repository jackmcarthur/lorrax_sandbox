# Agent 2 — structural fix design (avoid the WhileOp/SPMD trap)

Context you already have: CONTEXT.md, your own agent_2.md (v1) and agent_2_v2.md (v2), the consensus.md, the hlo_findings.md, and the orchestrator's general-case-framing broadcast.

**The problem:** `fit_one_rchunk` has a Python-unrolled bc-loop at `src/common/isdf_fitting.py:1273-1276`:

```python
psi_Y_parts = []
for bc_range in band_chunk_ranges:
    psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(
        bc_range, r_start_dyn, actual_n_rchunk))
psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)
```

Each `fetch_psi_rchunk` call materializes an unsharded FFT box of shape `c128[k_chunk, band_chunk, ns, nx, ny, nz]`. XLA pins all N_BC × ~3 of these as concurrent live slots because the Python unroll prevents lifetime aliasing.

**The naive fix (`fori_loop`) won't work** — there's prior art at lines 1115-1127 of the same file (in the comparable `solve_zeta` q-batch loop) that says:

> "fori_loop has the same WhileOp issue [as scan-without-unroll]. The Python loop is the only approach that gives constant DUS offsets AND sequential memory reuse."

The "WhileOp issue" appears to be that XLA's SPMD pass replicates the sharded accumulator under a WhileOp, causing OOM. In the bc-loop case, the natural accumulator (`psi_Y_full`) is itself sharded — so blind `fori_loop` would hit the same trap.

**Your task: design a structural fix that side-steps this trap.**

## Reading list (in order)

1. `src/common/isdf_fitting.py:1115-1335` — the `solve_zeta` prior-art comments + the kernel build comments + the bc-loop itself.
2. `src/common/isdf_fitting.py:254-475` — `c_q_from_psi_sm` and `z_q_from_psi_sm`. These are the einsum kernels the bc-loop output feeds into. Look at the shard_map structure.
3. `src/common/psi_G_store.py` — `fetch_psi_rchunk` (around line 268-380). Where the FFT box gets created. Note the `io_callback` boundary and the `to_rchunk` call.
4. `src/common/wfn_transforms.py` — `to_rchunk` (line 338+) and `to_rchunk_shard_map` (mentioned in the 2026-05-13 CHANGELOG entry). The shard_map version was added recently to avoid an unrelated 506-MiB all-gather; the technique might apply here.
5. `CHANGELOG.md` — the 2026-05-13 entry about `LORRAX_PSIG_RCHUNK_SHARDMAP=1`. Context for the shard_map-based pattern.

## Candidate paths to consider

These are starting points — feel free to disagree or propose your own.

### Path A — `fori_loop` with explicit donation + sharding constraint
Try `fori_loop` anyway, with `jax.lax.fori_loop` + explicit `donate_argnums` on the accumulator + `jax.lax.with_sharding_constraint` on the loop-carry. Hope is that explicit donation lets XLA recognize lifetime boundaries that the bare WhileOp doesn't. Risk: the prior comment suggests this was tried; verify in git history.

### Path B — `jax.lax.scan` with axis-naming for the accumulator
`scan` with the accumulator declared along a named axis might prevent SPMD replication. Risk: scan was also tried per the prior comment.

### Path C — Streaming pair-density accumulation (eliminate `psi_Y_full`)
Restructure: instead of materializing `psi_Y_full` of shape `(nk, nb_total, ns, r_chunk)`, accumulate directly into `C_q` (a rank-3 `(nq, n_rmu, n_col)` tensor) inside the bc-loop. Each iter does:
```
psi_Y_bc = fetch_psi_rchunk(bc_range, ...)
C_q += pair_density_einsum_pipeline(psi_l_X, psi_Y_bc[:, l_slice, :, :], ...)
```
The accumulator `C_q` is tiny (~MB) — no SPMD replication trap. Cost: the pair density pipeline runs N_BC times per r-chunk instead of once; need to verify einsum compositions are correct over band-window slices. Risk: requires re-deriving the math + may not match c_q_from_psi_sm/z_q_from_psi_sm bit-for-bit.

### Path D — Move the bc-loop *inside* `c_q_from_psi_sm._local`
Push the loop into the shard_map body so the FFT box lives entirely in the rank-local context. XLA might alias it then. Risk: shard_map bodies have constraints; not all ops are legal inside.

### Path E — Explicit `donate_argnums` on `fetch_psi_rchunk` + sequence hint
Without restructuring the loop, add explicit donate_argnums to force XLA to free each iter's FFT box before allocating the next. Minimal code change. Risk: io_callback's interaction with donation isn't obvious.

## Deliverable

Write your analysis to `reports/zeta_rchunk_memory_model_2026-05-13/agent_2_structural_fix.md`. Structure:

1. **Diagnosis of the WhileOp/SPMD trap** — your read of why fori_loop / scan-without-unroll caused the 88 GB OOM in solve_zeta, and whether the same mechanism would apply to the bc-loop. Cite lines.
2. **Path evaluation** — for each of paths A-E above (and any you propose), assess: does it side-step the trap? what's the expected memory profile after the change? what's the implementation cost (lines of code, files touched)? what tests would catch correctness regressions?
3. **Recommendation** — pick one path with reasoning.
4. **Implementation sketch** — for the recommended path, show the actual code change (diff or pseudocode) for the bc-loop and any required changes to `c_q_from_psi_sm` / `z_q_from_psi_sm`. Don't write the final code — sketch enough that the orchestrator can implement quickly.
5. **Validation plan** — what's the smallest possible test that would catch a numerical regression? What HLO signal would confirm the fix worked (slot count, allocation total)?

**Constraints:**
- Read-only on `sources/lorrax_A/`. Do not implement yet — propose only.
- No compute. Desk research only.
- Do NOT read `agent_1_hlo_verify.md` (Agent 1's parallel task) until you've finished.
- Stay in this tmux pane.
- Print "Agent 2 structural fix done" when finished.

Goal: a proposal sharp enough that the orchestrator can implement, test, and validate in one sitting after reading your file.
