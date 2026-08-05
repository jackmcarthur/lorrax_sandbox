# Agent 4 — Defect 3 mirror work log

Round 3 mission: build `gflat_to_rmu` shard_map mirror of
`gflat_to_rchunk`, thread it through `load_centroids_band_chunked`.

Branch: `agent/zeta-bc-scan-shardmap` on `sources/lorrax_B`.
Files touched: `src/common/wfn_transforms.py` (additive only),
`src/common/load_wfns.py` (centroid-load driver replacement), and
`tests/test_wfn_transforms.py` (additive tests).  No edits to
existing helpers (`to_rmu`, `to_rchunk`, `to_rchunk_inner`,
`gflat_to_rchunk`).

## Implementation summary

### `src/common/wfn_transforms.py` (additive)

1. **`to_rmu_inner`** — pure-jax body of `to_rmu`: G-flat → FFT box →
   IFFT → optional Bloch-phase on full box → centroid sample at
   `r_mu` triples.  Callable from inside another shard_map or
   `lax.scan` body.  Mirror of `to_rchunk_inner` exactly modulo the
   r-slab → centroid-gather swap.

2. **`gflat_to_rmu`** — shard_map + scan twin of `gflat_to_rchunk`:
   - signature: `(psi_G, g_index, r_mu, *, mesh, fft_grid,
     kvecs_frac=None, norm="backward", chunk_size=None) → jax.Array`.
   - input  `(nk, nb_total, ns, ngkmax)` band-flat-sharded on
     `('x','y')`.
   - output `(nk, nb_total, ns, n_rmu)` same sharding.
   - body: shard_map over `('x','y')` → flat `(nk · nb_local)` axis
     → `lax.scan` over chunks of `cs` rows.  Each iter: per-row
     g_index gather → `_box_kernel` → `jnp.fft.ifftn` (per-rank
     local) → centroid gather → optional per-row Bloch phase at
     centroid cells (post-gather, algebraically identical to the
     full-box pre-gather phase that `to_rmu`/`apply_bloch_phase`
     uses) → `dynamic_update_slice_in_dim` into `out_flat`.
   - cache `_GFLAT_TO_RMU_CACHE` keyed on shape + g_index/kvecs/r_mu
     content hashes + cs/n_chunks/pad_N (same trap as the
     `gflat_to_rchunk` kvecs lesson — phase tables bake into the
     closure, shape-only key would silently reuse stale tables).

### `src/common/load_wfns.py` — `load_centroids_band_chunked`

Replaced the driver bc-loop + optional inner k-chunk loop with a
single `gflat_to_rmu` call.  New flow:

1. `loader.load(bands=band_range, k='full_bz', sharding=P(None,
   ('x','y'), None, None))` once — single G-flat collective.
2. `gflat_to_rmu(psi_G_flat, g_index_full, centroid_idx_np,
   mesh=mesh_xy, fft_grid=meta.fft_grid, kvecs_frac=kvecs_frac_full,
   norm='ortho', chunk_size=cs)` — single shard_map+scan.
3. Single global reshard `{None, XY, None, None} → {None, None,
   None, Y}` plus conjugate-transpose to `{None, X, None, None}` (the
   `_reshard_all` jit, same two-step staging as the legacy
   per-chunk reshard but called once).
4. Slice off `nb_padded → nb_total` band pad (no-op when already
   divisible by mesh.size).
5. User-band-pad zeroing — unchanged.

The legacy `band_chunk_size` / `k_chunk_size` kwargs are preserved
in the signature for caller compatibility (none of the four callers
pass `k_chunk_size`; some pass `band_chunk_size`).  Both are now
hints for deriving the new `chunk_size` (rows per scan iter); the
final cap is the per-rank HBM budget (`gpu_mem_bytes / (ns · n_rtot
· 16 · peak_copies)`), same conservative `peak_copies` constant as
the old k_chunk_size autodetect (4 on single-rank, 9 on multi-rank).

### `tests/test_wfn_transforms.py` (additive)

Five new tests (all pass CPU pytest):

- `test_to_rmu_inner_matches_to_rmu_no_phase` — body-without-wrapper
  byte-equality on synth WFN, no Bloch phase.
- `test_to_rmu_inner_matches_to_rmu_with_phase` — same, with random
  kvecs_frac (phase branch).
- `test_gflat_to_rmu_no_phase` — bc-loop+concat reference vs
  `gflat_to_rmu(cs=None)`, `rtol=1e-10 atol=1e-12`.
- `test_gflat_to_rmu_with_phase` — same with random kvecs_frac and
  three-window band_chunks.
- `test_gflat_to_rmu_chunked_matches_oneshot` — chunk_size sweep
  `{1, 3, N, N+1}` (covers divisor / non-divisor / no-pad / pad
  cases) all match the one-shot output.

## Validation gates

1. **CPU pytest for `test_wfn_transforms.py`** — 23/23 pass (18 prior
   + 5 new) at CPU only.  Full repo CPU run: 217 passed, 2 failed
   in `tests/test_v_q_per_q_g_chunked.py::test_v_q_per_G_matches_legacy_kernel`
   — pre-existing failure at HEAD (`make_v_munu_chunked_kernel()
   missing 1 required positional argument: 'mesh_xy'`), unrelated to
   this work.

2. **End-to-end bit-identity on synth WFN.**  Inline smoke test
   invokes `load_centroids_band_chunked` against the synth WFN
   fixture used by the test suite; compares output to a direct
   `to_rmu(full band range)` call.  Both `psi_rmu_Y` and `psi_rmuT_X`
   match to `rtol=1e-10, atol=1e-12`.  Confirms the new wiring
   preserves the contract callers depend on.

3. **HLO slot-count check** — deferred.  Validation requires a GPU
   node; the SLURM allocation in flight (per round3_discussion.md
   §"Status snapshot") will run this against MoS2 3×3 once Agent 2's
   integration lands too.  Standalone helper bit-identity gate (1+2)
   covers correctness; HLO will confirm the Peak-A FFT-box slot
   collapses from "single but unsharded on every rank" to
   "single per-rank shard with ns × n_rtot / mesh.size bytes."

## What this commit closes

- **Defect 3** from `defect_catalog.md`.  The unsharded FFT-box
  transient in the centroid-load step (Peak A in
  `gw/gflat_memory_model.py`) is replaced by a per-rank-local FFT
  box inside a shard_map.  The §0 zero-replicated-intermediates
  principle is restored at the centroid-load step.

## Open follow-ups

- Update `gw/gflat_memory_model.py:_peak_A_centroid_load` to drop
  the `fft_box_factor` multiplier on the FFT-box term — the new
  helper's box is sharded, not replicated.  Agent 2 owns
  `gflat_memory_model.py` this round; their cleanup will likely
  delete `band_fft_pool` / `band_fft_unsharded`; the Peak-A
  adjustment is the same family of change.  Leaving the Peak-A
  term intact for now is conservative (over-estimates budget) and
  the planner will still pick safe configs.
- HLO slot-count validation on a GPU node (see gate 3 above).
