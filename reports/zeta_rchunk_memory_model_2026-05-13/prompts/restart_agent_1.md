# Agent 1 — Restart: Planner / architect of the symmetric forward helper

Welcome back. We're picking up the zeta-fit memory-model work from this morning. Read `reports/zeta_rchunk_memory_model_2026-05-13/PATH_D_PICKUP.md` first — **§0 is the load-bearing principle (zero replicated intermediates) and supersedes any earlier framing**. Then skim `agent_2_structural_fix.md` (the prior Path D design — useful but predates the unified-helper framing below).

## The new framing (user, just now)

The pipeline needs **two exactly parallel `shard_map` helpers**, structurally identical except for I/O endpoints:

| Direction | Input → Output | Sharded on | Inner scan over | Per-iter body |
|---|---|---|---|---|
| **Forward** | `ψ_{n,k}(G)` → `ψ_{n,k}(rchunk)` | bands & k on (`'x'`,`'y'`) | chunks of (band, k) — FFT-workspace-sized | gather G → FFT box → IFFT → Bloch phase → r-slice → write |
| **Reverse** | `ζ_{q,μ}(rchunk)` → `ζ_{q,μ}(G)` | μ & q on (`'x'`,`'y'`) | chunks of (μ, q) — FFT-workspace-sized | r-slab → pad to FFT box → Bloch phase → FFT → gather sphere → accumulate |

**The reverse already exists, clean**, in `common.wfn_transforms.accumulate_rchunk_to_gflat` (`sources/lorrax_A/src/common/wfn_transforms.py:468`). Read it carefully — it IS the reference pattern (single shard_map over `('x','y')`, `lax.scan` over chunks of the flat `(n_q · n_mu_local)` axis inside the body, free chunk size, no divisibility constraint, per-iter FFT-box aliased across iters by XLA's scan-internal allocator).

**The forward currently doesn't follow this pattern** — `psi_G_store.fetch_psi_rchunk` + `common.wfn_transforms.to_rchunk` use a Python-unrolled k-chunk loop, and `_make_fit_one_rchunk_kernel._kernel` (`isdf_fitting.py:1273-1276`) uses a Python-unrolled bc-loop. Both are principle violations — N concurrent unsharded FFT-box slots in the HLO.

## Your role: planner / architect

Produce a design doc for the unified forward helper that's a structural twin of `accumulate_rchunk_to_gflat`. Specifically:

1. **Read the existing code carefully** (~30 min):
   - `common.wfn_transforms.accumulate_rchunk_to_gflat` (lines 468-end) — the reference.
   - `common.wfn_transforms.to_rchunk` (lines 338-435) — the current forward shard_map helper, but without the bc-loop wrapping.
   - `common.psi_G_store.fetch_psi_rchunk` (lines ~268-380) — the wrapper that loops over k-chunks and concatenates.
   - `_make_fit_one_rchunk_kernel._kernel` (`isdf_fitting.py:~1240-1320`) — the kernel that Python-unrolls the bc-loop.
   - Look at the Path D scaffolding already on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap`:
     - `common.wfn_transforms.to_rchunk_inner` — pure-jax body factored out of to_rchunk (no shard_map wrapper, callable from inside another).
     - `common.psi_G_store.PsiGStore._slice_local_tile_bc` — traced-bc_idx host-tile slicer that returns padded uniform shape.
   - Look at how `accumulate_rchunk_to_gflat` handles the `(n_q · n_mu_local)` flat axis chunking — that's the model for chunking `(nk · nb_local)` in the forward.

2. **Propose the forward helper's signature** in a way that's literally the structural mirror of `accumulate_rchunk_to_gflat`:
   - Input: ψ(G) at `(nk, nb_total, ns, ngkmax)` sharded on bands across `('x','y')` flat (so `nb_local = nb_total / P`).
   - Output: ψ(rchunk) at `(nk, nb_total, ns, r_chunk)` sharded the same way.
   - Chunk knob: a single `chunk_size` integer (rows per scan iteration), defaulting to one-shot, no divisibility constraints.
   - Inside one shard_map body: scan over chunks of the flat `(nk · nb_local)` axis, per-iter gather G → FFT box → IFFT → Bloch phase → r-slice → write.
   - Same caching pattern as `accumulate_rchunk_to_gflat` (key on shapes + shardings + chunk_size).

3. **Identify the symmetry-breakers** (places where the two helpers can't be identical — note them explicitly):
   - File I/O endpoints (`PsiGStore.fetch_psi_rchunk` calls into host-tile via io_callback; `accumulate_rchunk_to_gflat` does not need io_callback — its input is already on device).
   - Bloch phase sign (`exp(+2πi k·r)` vs `exp(-2πi q·r)`) — see `apply_bloch_phase` in wfn_transforms.py, both paths already share that helper.
   - FFT direction (IFFT vs FFT) — distinct calls, same code path.
   - Sphere gather (reverse takes from `sphere_idx[q]`; forward writes to a flat-r slab without gather).

4. **Sketch the integration path** — once the new forward helper exists, how does `_make_fit_one_rchunk_kernel._kernel` change? Today it Python-unrolls a bc-loop and concatenates `psi_Y_parts`. After: a single call to the new helper, no bc-loop, no concatenate. Show the diff in pseudocode.

5. **Validation plan**:
   - CPU bit-identity test on synth WFN — compare against current `to_rchunk` + concatenate. `rtol=1e-10, atol=1e-12`.
   - HLO dump test — predict slot count after the change (should drop from ~58 to ~3 for the FFT box; aliased across scan iters).
   - End-to-end ζ-fit test on MoS2 3×3 — `eqp0.dat` should be bit-identical to current.

## Output

Write to `reports/zeta_rchunk_memory_model_2026-05-13/parallel_helpers_design.md`. **Communicate with Agent 2** (the implementer) via that file plus an open `parallel_helpers_discussion.md` that you can both edit. When you have a draft design Agent 2 can act on, print "Agent 1 design v1 done" so I can route it to Agent 2.

If you get stuck on a structural question, write it to `parallel_helpers_discussion.md` and print "Agent 1 needs input on X". I'll route it to other agents or the user.

## Constraints

- Read-only on `sources/lorrax_A/`. Implementation lives on `sources/lorrax_B` and is Agent 2's job, not yours.
- Do NOT read the other restart-agent prompt files or output files for now (Agent 2's `parallel_helpers_impl.md`, Agent 3's `accumulate_audit.md`, Agent 4's `defect_catalog.md`). I'll route relevant findings to you.
- Stay in this tmux pane.
- The principle (zero replicated intermediates) is non-negotiable. If your design has any term that scales like `N · per-iter-unsharded-bytes`, you've violated it — go back and fix.
