# Round 5 — Agent 1: SPMD / shard_map / mesh-semantics lens

Read `round5_discussion.md` first for mission + working principle. **Your lens is the primary focus, but you're an expert in ALL levels of JAX parallelization — challenge the other agents' assumptions across boundaries when your lens is implicated.**

## The user's framing

The current design's `jnp.concatenate(parts, axis=1)` of band-sharded arrays triggers a block-cyclic → contiguous reshuffle = expensive cross-rank all-to-all. The proposed alternative is scan-inside-shard_map. The user explicitly wants this design pursued *if it's clean* — they're willing to accept the io_callback complication if the principle is satisfied.

## Your specific deep-dives

### 1. SPMD safety of the scan carry inside shard_map

The `solve_zeta` prior-art comment (`isdf_fitting.py:1119-1141`) says `scan`/`fori_loop` with a sharded carry triggers SPMD WhileOp inflation (88 GB OOM at CrI3). Why doesn't this apply to a scan **inside** a shard_map body?

Be precise: walk through what SPMD sees. The shard_map's `in_specs` / `out_specs` define the SPMD boundary; inside the body, what's the sharding contract for the carry? Are there any subtle ways the WhileOp trap could *still* fire — e.g., if the carry uses jax-array operations that propagate sharding annotations even inside shard_map? If so, identify them.

### 2. Carry design

The carry is two rank-5 pair-density accumulators `P_l_acc`, `P_r_acc`, each `c128[nk, ns, r_chunk_local, mu_loc, ns]` per-rank. At CrI3 4×4 mesh: 36·2·4603·376·2·16 ≈ 14.85 GiB per rank per accumulator. The pair lives across all scan iters.

- Initialize how? `jnp.zeros(...)` inside the shard_map body (per-rank-local) — confirm SPMD doesn't replicate this.
- Carry vs scan output: the natural scan accumulator pattern is `(carry, _ = scan(body, init, xs))`. You want only the final carry. Standard JAX idiom, but verify.
- Donation: should the scan carry be donated to avoid an extra alloc?

### 3. `in_specs` / `out_specs` for the wrapping shard_map

The shard_map's inputs are the X-sharded `psi_l_X`, `psi_r_X` (one mesh axis each), plus closure refs to `psi_G_store`, `band_chunk_ranges`, etc. The output is `C_q` / `Z_q` at `P(None, 'x', 'y')`. Inside the body, the scan operates on per-rank rank-5 tensors. **What are the exact specs?**

Compare against `accumulate_rchunk_to_gflat`'s working pattern — same shape of solution applies here.

### 4. L/R per-bc band slicing

Inside the scan body, each iter needs to extract the L-window and R-window slices of *that bc's* bands. Three candidate approaches (also Agent 2's design §4c sketched these):

- **Mask approach**: pre-build per-bc `(l_mask, r_mask)` boolean tables, use `jnp.where(mask, psi_bc, 0)`. Wasted FLOPs in einsum on masked rows, but uniform shape.
- **Slice approach**: `lax.dynamic_slice_in_dim` with per-bc traced offsets/lengths from pre-baked tables. Requires uniform slice length per bc — which the bc design doesn't guarantee unless we pad.
- **Padded uniform slice**: combine — pad bands to bpd_max, use static slice within the body.

Which is cleanest under SPMD? Watch out: any approach that does `dynamic_slice` on a sharded axis with a traced index can trigger an all-gather. Verify this doesn't happen here (the band axis inside shard_map body is rank-local).

### 5. `psi_l_X` per-bc band slicing

`psi_l_X` is sharded `P(None, 'x', None, None)` — μ on `'x'`, band axis replicated within `'y'` rank but striped within `'x'` ranks. Inside the shard_map body the band axis is rank-local. **But** the band-axis chunk we want per scan iter is the bc-th window of the global band axis. The bc band ranges are static (closure). What's the safe way to slice `psi_l_X` band-axis per bc-iter — does this trigger any cross-rank op?

### 6. Output: pair density `P_l_acc`, `P_r_acc` → existing post-pair pipeline

After the scan, the body still needs to do: reshape, IFFT, γ̃ contract, FFT, transpose. That's the existing tail of `z_q_from_psi_sm._local`. Confirm this tail is compatible with the new scan output (it should be — `P_l_acc` after scan has the same shape as the old `P_l`).

## Deliverable

Write to `reports/zeta_rchunk_memory_model_2026-05-13/round5_unified_plan.md` (Agent 4 owns the file; you write your section UNDER the shared template Agent 4 will set up, or write your individual analysis to `round5_agent_1_spmd.md` first and Agent 4 incorporates).

Either way: cite line numbers, be explicit about the SPMD invariants you're relying on, and **call out anything that's "I believe this works but haven't proven it"** vs "this is documented behavior". The user wants thoroughness.

Communicate via `round5_discussion.md` (poll before each work cycle). When you have a draft contribution, write a short Agent 1 → others note pointing to your section. Iterate with other agents until consensus.

Read-only on `sources/`. No compute. Print "Agent 1 round 5 ready" when your section is in the unified plan or you've signed off on Agent 4's draft.
