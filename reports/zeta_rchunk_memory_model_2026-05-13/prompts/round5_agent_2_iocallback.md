# Round 5 — Agent 2: io_callback / host-device / host-tile lifecycle lens

Read `round5_discussion.md` first. **Your lens is io_callback boundary semantics, but you're an expert in ALL levels of JAX parallelization — challenge the other agents.**

## Your specific deep-dives

### 1. Can `io_callback` fire inside `lax.scan` inside `shard_map`? Prior art / docs

This is the load-bearing technical question. In your Round-3 design (`agent_2_structural_fix.md` §4 "novel territory" comment), you flagged this as the risky bit. **Resolve it now.**

- Read JAX docs for `jax.experimental.io_callback`. What's the documented behavior inside `scan` and inside `shard_map`?
- Search the LORRAX codebase for any existing pattern: is `io_callback` ever called from inside `lax.scan`? Inside `shard_map`? Both?
- The `accumulate_rchunk_to_gflat` reference uses `lax.scan` inside `shard_map` but does NOT use `io_callback` inside the scan body. Is the difference fundamental, or is `io_callback` "just another effectful op" that's transparent to scan + shard_map?
- If unclear: propose a 10-minute synth-scale CPU smoke test that would resolve it (which Round 6 can run before committing).

### 2. Host-tile lifecycle

The current `PsiGStore` populates `_host_tiles` via `_populate_from_loader` once per `begin_rchunk` cycle. Each tile is `(nk, sum(bpd_per_bc), ns, ngkmax)` per-rank — bc-stacked.

In the new design:
- We no longer concat all bcs into one device tensor.
- Each scan iter pulls *just this bc's slice* via io_callback.
- The padded uniform-shape return contract (your original `_slice_local_tile_bc`, committed in `cdd0fba` then removed in `5cadd4b`) comes back. **Restore it.**

Walk through the exact host-tile slicing path. What's the per-rank return shape? How does the bc-traced index work? What about the short final bc (pad to `bpd_max`, mask in scan body via Agent 1's analysis)?

### 3. `ordered=True` and scan iteration order

`io_callback(..., ordered=True)` enforces sequential firing. Inside a scan, does this play nicely with the traced loop counter? Specifically:

- Does `lax.scan` guarantee in-order body execution at runtime? (It does for `reverse=False`, the default.)
- Does `ordered=True` add overhead beyond what `scan`'s sequential semantics already provide?
- Is `ordered=True` even needed if scan is already sequential?
- What happens if the scan is `unroll=k` for k > 1? (Don't unroll — but call out the constraint.)

### 4. Latency vs throughput per io_callback call

At CrI3 6×6 80 Ry: N_BC ≈ 20 bcs, N_R_CHUNKS = 16 → 20 × 16 = 320 io_callback calls per channel. Plus the bispinor channels (charge + 3 transverse) eventually = 1280 calls per run.

- Each io_callback round-trip has fixed Python-host overhead (~1-5 ms typically).
- Per-bc payload: `(nk · bpd_max · ns · ngkmax · 16)` bytes ≈ 36 × ~20 × 2 × 60000 × 16 ≈ 1.4 GB per call, transferred HtoD.
- 1.4 GB × 320 calls = 450 GB total H2D transfer per channel. At ~25 GB/s NVLink + Lustre, that's ~20s of pure I/O per channel. Acceptable? Compare against the current implementation's I/O cost.

Quantify: is io_callback overhead per call (Python-host latency × 320 calls) a meaningful chunk of total wall time, or noise?

### 5. Alternative: pre-pull all bcs to device once per begin_rchunk

The original Agent 2 Round-3 design did this and ran into the tracer leak. **The fix**: pre-pull at the *driver* level (outside any jit), pass as a jit argument, not a closure-captured lazy property. This eliminates the leak.

But — that's just the current `psi_G_device_full` design with the cache moved outside the jit. It still requires the concat reshuffle. So this is *not* the scan-inside-shard_map win; it's a smaller patch.

Compare the two paths explicitly:
- **Path A — pre-pull to device, pass as jit arg, then `psi_G_full` is consumed without re-fetch inside scan**. Fixes leak. Still does concat reshuffle.
- **Path B — io_callback inside scan inside shard_map**. Fixes leak AND eliminates concat AND eliminates remat. But assumes the nesting works.

If Path B is technically infeasible (you discover a hard JAX limitation), Path A is the fallback. If Path B is feasible, Path B is the answer. Make this call decisively.

### 6. `begin_rchunk` / `end_rchunk` interaction

When the r-chunk loop iterates and the last r-chunk has a different `r_len`, the kernel re-jits. With the new design, what state needs to be valid?

- Host tiles populated once at construction (for `HostPsiGStore`) — unchanged.
- No device-side cache (no `psi_G_device_full`, no tracer to leak).
- The scan body's `io_callback` re-invokes the host function on every call — fresh pulls.

Verify nothing else in `PsiGStore` accumulates state that survives across jit boundaries.

## Deliverable

Same as Agent 1: contribute your section to `round5_unified_plan.md` (Agent 4 owns), and write an Agent 2 → others note in `round5_discussion.md`. Iterate.

Print "Agent 2 round 5 ready" when finished. Read-only on `sources/`. No compute.
