# Agent 2 — Restart: Implementer / tester of the symmetric forward helper

Welcome back. Read `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` first — **§0 is the load-bearing principle (zero replicated intermediates)**. Then skim `agent_2_structural_fix.md` for context on the prior Path D design.

## The work

Agent 1 (in pane 0) is producing a design doc at `reports/zeta_rchunk_memory_model_2026-05-13/parallel_helpers_design.md`. The goal: **two exactly parallel `shard_map` helpers** for the FFT-box procedures:

- **Forward**: `psi_nk(G)` → `psi_nk(rchunk)` (bands & k sharded on ('x','y'), chunks of FFT workspaces inside a shard_map+scan).
- **Reverse**: `zeta_{q,μ}(rchunk)` → `zeta_{q,μ}(G)` — **already exists**, clean, at `common.wfn_transforms.accumulate_rchunk_to_gflat` (`sources/lorrax_A/src/common/wfn_transforms.py:468`).

The forward currently violates the principle (Python-unrolled k-chunk + bc-loop, 58 concurrent unsharded FFT-box slots in the HLO). Your job is to implement the unified forward helper as the structural twin of `accumulate_rchunk_to_gflat`.

## Your role: implementer + tester

Once Agent 1 has a draft design (watch for `Agent 1 design v1 done` in the conversation routing) — implement on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap` (the Path D scaffolding already lives there: `to_rchunk_inner`, `_slice_local_tile_bc`).

Specifically:

1. **Read what's already done on lorrax_B** (no implementation yet — these are helpers):
   - `common.wfn_transforms.to_rchunk_inner` — pure-jax body of `to_rchunk` (no shard_map wrapper, callable from inside another shard_map / scan body). Three CPU bit-identity tests already exist in `tests/test_wfn_transforms.py`.
   - `common.psi_G_store.PsiGStore._slice_local_tile_bc` — traced-bc_idx host-tile slicer returning padded `(nk, _bpd_max, ns, ngkmax)`. Untested standalone; will be exercised by the integration.

2. **Read the reference** — `accumulate_rchunk_to_gflat` (`sources/lorrax_A/src/common/wfn_transforms.py:468-end`). Understand the chunk-scan pattern: single shard_map over `('x','y')`, `lax.scan` over chunks of the flat `(n_q · n_mu_local)` axis inside the body, per-iter FFT-box aliased across iters by XLA's scan-internal allocator.

3. **Implement** the new forward helper (filename + signature TBD by Agent 1's design — probably `to_rchunk_chunked` in `wfn_transforms.py` or similar). The structural pattern should be:
   - Single shard_map over `('x','y')`.
   - `lax.scan` over chunks of the flat `(nk · nb_local)` axis inside the body.
   - Each iter: io_callback (host-tile slice for the chunk's (k, band) range) → `to_rchunk_inner` → write to output slab via `dynamic_update_slice`.
   - Chunk size is a free integer; default to one-shot (whole flat axis in one scan iter); same key-based caching as `accumulate_rchunk_to_gflat`.

4. **Tests**:
   - **CPU bit-identity** against current `to_rchunk` + bc-loop concatenate. Use the existing `synth_loader` fixture in `tests/test_wfn_transforms.py`. `rtol=1e-10, atol=1e-12`.
   - **Slot-count test on synth WFN** — small geometry, dump HLO, count `*memory-usage-report*` slots that hold an FFT-box-shaped tensor. Expect ≤3 (one per shape variant), not N.

5. **Integration** into `_make_fit_one_rchunk_kernel._kernel` is the *next* step after the helper is bit-identity-validated. Don't do that integration until Agent 1's design covers it explicitly and the helper passes tests.

## Output

Write progress to `reports/zeta_rchunk_memory_model_2026-05-13/parallel_helpers_impl.md` as you go (commits, test results, gotchas). Use `parallel_helpers_discussion.md` to ask Agent 1 questions or flag design issues.

Print "Agent 2 helper v1 done" when the new helper exists, compiles, and the CPU bit-identity test passes. Print "Agent 2 needs input on X" if you hit a design ambiguity.

## Constraints

- Code on `sources/lorrax_B` only. Don't modify lorrax_A this session.
- Don't integrate into `_kernel` until the standalone helper is validated.
- Tests are gates, not afterthoughts. If bit-identity fails, stop and figure out why before doing anything else.
- The principle: zero replicated intermediates. If after your implementation the HLO still shows N concurrent FFT-box slots, the helper is wrong — fix it before claiming done.
- Stay in this tmux pane.
