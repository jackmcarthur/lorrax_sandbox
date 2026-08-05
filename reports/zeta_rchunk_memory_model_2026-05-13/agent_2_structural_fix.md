# Agent 2 — structural fix design for the band-FFT slot multiplication

## 1. Diagnosis of the WhileOp / SPMD trap

### The prior-art evidence (solve_zeta)

`src/common/isdf_fitting.py:1119-1141` (the q-batch loop) carries
this comment:

```text
# NOTE: scan(unroll=8) was attempted but OOMs — XLA pipelines
# adjacent unrolled iterations, keeping 2× preallocated-temp alive
# (18.9 GB). scan without unroll triggers SPMD replication of the
# sharded accumulator (88 GB OOM). fori_loop has the same WhileOp
# issue. The Python loop is the only approach that gives constant
# DUS offsets AND sequential memory reuse.
```

The "WhileOp / SPMD trap" mechanism: when a `lax.scan` or
`lax.fori_loop` body is presented to XLA, SPMD-partitioner sees
the loop carry as **opaque**. The carry's sharding cannot be
analyzed across iterations because the WhileOp's body is a
black box to SPMD. For a SHARDED carry, SPMD therefore falls
back to:

- all-gather the carry into a full replicated value at the WhileOp
  boundary, or
- replicate the carry's allocation on every rank (the 88 GB OOM
  for `Z_col` at `(nq, n_rmu, n_rmu_padded)/P` × P).

That's why solve_zeta's `Z_col` (sharded `P(None,None,('x','y'))`)
cannot be a fori_loop carry: SPMD inflates it `P×`. The working
pattern is `helpers.solve_batch_and_update` jitted with
`donate_argnums=(2,)` and called from a Python `for q0 in
range(...):` driver loop — each iter is a separate dispatched
jit. The donation chain across jit boundaries gives sequential
memory reuse without an enclosing WhileOp.

### How it would fire in the bc-loop

`_make_fit_one_rchunk_kernel._kernel` at lines 1273-1277:

```python
psi_Y_parts = []
for bc_range in band_chunk_ranges:
    psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(
        bc_range, r_start_dyn, actual_n_rchunk))
psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)
```

This is Python-unrolled *inside* a `@jax.jit` (the `_kernel`
decorator at line 1229). XLA sees the unroll as `n_bc`
straight-line `fetch_psi_rchunk` calls; each of those internally
unrolls a `k_chunk` loop (`psi_G_store.py:369-377`); each `to_rchunk`
call materializes an FFT box of shape `c128[k_chunk, bc, ns, nx,
ny, nz]`. The HLO finding: 58 concurrent live slots × 3.22 GiB
unsharded = ~200 GB.

A naive `lax.fori_loop` rewrite with `psi_Y_full` as the loop
carry (using `dynamic_update_slice` to write the bc'th slab) would
hit the trap: `psi_Y_full` has shape `(nk, nb_F, ns, r_chunk)`
sharded `P(None, ('x','y'), None, None)`. SPMD sees the sharded
carry under the WhileOp → all-gather or per-rank replicate. At
CrI3 charge nb_F≈400, n_rtot≈1.1M → carry is 2 GB × P=16 = 32 GB
extra per rank if replicated. That's the same class of failure as
solve_zeta's 88 GB at `Z_col`.

**Key structural observation**: the trap only fires when the loop
carry is at outer-jit-level sharding (visible to SPMD). Two ways
to side-step:

1. **Move the loop to Python driver level** (outside any jit).
   Each iter is its own jit dispatch. XLA sees no WhileOp at all;
   SPMD operates per-jit. Donation chains give sequential reuse.
   This is the solve_zeta pattern.

2. **Move the loop INSIDE a `shard_map` body** with `lax.scan` /
   `lax.fori_loop`. SPMD sees the outer shard_map's in_specs/
   out_specs and stops there — the scan body sees only rank-local
   data with no sharding annotations. The carry is per-rank;
   there is no global sharded carry for SPMD to inflate.

Both work; they differ in fusion opportunity and refactor scope.

## 2. Path evaluation

For each: trap side-step? expected memory profile? implementation
cost? correctness tests?

### Path A — `fori_loop` at outer level with `donate_argnums` + WSC

- **Trap side-step?** No. The prior-art quote covers exactly this
  case: `fori_loop has the same WhileOp issue`. Donation doesn't
  help inside a WhileOp body — donation requires a jit boundary
  for XLA to recognize the buffer-release point. `with_sharding_constraint`
  on the carry inside the loop just tells SPMD what sharding the
  carry HAS, which is already known; the inflation happens at the
  WhileOp boundary regardless.
- **Memory profile**: would carry the all-gathered/replicated
  `psi_Y_full`. Worse than today.
- **Cost**: low (small code change).
- **Recommendation**: reject. Empirically refuted by the prior-art
  comment.

### Path B — `lax.scan` at outer level with axis-named accumulator

- **Trap side-step?** No. Same prior-art quote covers it: scan
  without unroll triggers SPMD replication (88 GB). `unroll=k`
  pipelines k adjacent iters, keeping `k×` slots live (the 18.9
  GB result was at k=8 on Z_col). Neither is a fix.
- **Reject for the same reasons as A.**

### Path C — Streaming pair-density accumulation (driver-level Python loop)

The bc-loop is hoisted to the Python driver of `fit_one_rchunk`
(outside any enclosing jit). Each iter is a separate jit dispatch
with donation chains on rank-5 accumulators `P_l_acc` and
`P_r_acc`.

Mathematical justification (re-derived from `c_q_from_psi_sm` and
`z_q_from_psi_sm`): the pair-density einsum is
`'kmna,knbr->karmb'`. The `n` axis (bands) is SUMMED out
INDEPENDENTLY on each side (L and R have their own einsum). After
the einsum, P_l and P_r have NO band axis remaining — they are
rank-5 `(k, ns, r_chunk, mu, ns)`. The downstream IFFT(k) →
γ̃-contract → FFT(k) is linear in both P_l and P_r separately
(bilinear together). So:

```
P_l_full = Σ_{bc in L_chunks}  einsum(ψ_l_X[:, :, bc_L_slice, :], ψ_l_Y_bc)
P_r_full = Σ_{bc in R_chunks}  einsum(ψ_r_X[:, :, bc_R_slice, :], ψ_r_Y_bc)
Z_q       = (existing IFFT + γ̃·γ̃ + FFT chain on P_l_full, P_r_full)
```

The L/R band ranges differ (`band_range_left = (b0,b3)`,
`band_range_right = (b1,b4)`). Per-bc, slice the bc's bands into
its L-portion and R-portion; a bc that spans only L bands (or
only R) contributes only to one accumulator. This is the same
bookkeeping the current code does on `psi_Y_full[:, _l_lo:_l_hi]`
slices.

- **Trap side-step?** Yes. Python driver loop = no WhileOp. Each
  iter is its own jit. SPMD sees only the per-iter sharded inputs/
  outputs; the rank-5 P_*_acc is sharded `P(None,None,None,'x','y')`
  but lives across jit boundaries via donation, never inside a
  WhileOp.
- **Expected memory**: per rank
  - persistent across bc-loop: `P_l_acc + P_r_acc` ≈ `2 · 16·nk·ns²·mu·r_chunk/P` (= 2 rank-5 slots; ~1 GB each at CrI3 charge with r_chunk≈20k).
  - per-bc transient: 1 FFT box × cuFFT scratch ≈ `16 · k_chunk · bc · ns · nx · ny · nz · F_fft` (unsharded by design, since it's local to one shard_map invocation per jit; XLA frees it at jit return).
  - Peak ≈ B_persist + 2 rank-5 + 1 FFT box ≈ ~15-20 GB at CrI3 charge with `psig_k_chunk=6, bc=16, r_chunk ≈ 20k`.
- **Implementation cost**: ~150-300 lines. Files touched:
  - `isdf_fitting.py`: split `c_q_from_psi_sm` and `z_q_from_psi_sm` into (a) `pair_density_partial` (per-bc, returns/donates rank-5 accumulator) and (b) `post_pair_to_Cq` / `post_pair_to_Zq` (rest of the pipeline: IFFT + γ̃ + FFT). Rewrite `_make_fit_one_rchunk_kernel` so the bc-loop is in Python driver and `_kernel` is replaced by a Python orchestrator that calls multiple jits.
  - `psi_G_store.py`: no change (`fetch_psi_rchunk` still works per-bc).
  - `wfn_transforms.py`: no change.
- **Tests that catch regressions**:
  - `tests/test_isdf_fit.py` (CCT / ZCT bit-identity) — must show pair-density values bit-identical to current code at the rank-5 level.
  - `tests/test_zeta_fit_e2e.py` / MoS2 3×3 ζ residual — same final ζ within float tolerance vs reference.
  - One-bc edge case: if `nb_F ≤ bc_size`, the loop runs once; should match exactly.

### Path D — Push the bc-loop INSIDE the shard_map body

Restructure `c_q_from_psi_sm._local` (and `_z_q_from_psi_sm._local`)
so the per-rank body contains a `lax.scan` over bcs with
io_callback inside:

```python
@partial(shard_map, mesh=mesh_xy, in_specs=(...), out_specs=...)
def _local(psi_l_X_, psi_r_X_, perm_L_, phase_L_, perm_R_, phase_R_, ...):
    x_idx = jax.lax.axis_index('x')
    y_idx = jax.lax.axis_index('y')

    # init P_l_acc, P_r_acc rank-5 per-rank (mu_local, r_chunk_local, ...)
    P_l_acc = jnp.zeros((nk, ns, r_chunk_loc, mu_loc, ns), c128)
    P_r_acc = jnp.zeros((nk, ns, r_chunk_loc, mu_loc, ns), c128)

    def body(carry, bc_idx):
        P_l_acc, P_r_acc = carry
        # fetch this bc's host tile via io_callback (traced bc_idx)
        psi_G_bc = io_callback(_slice_local_tile, out_sds, x_idx, y_idx, bc_idx,
                                ordered=True)
        # local FFT box → IFFT → r-slice → Bloch phase (the to_rchunk body inline)
        psi_r_bc = _local_box_ifft_slice_phase(psi_G_bc, ...)
        # split into L / R band sub-slices (static masks per bc_idx via lax.cond
        # OR pre-bake band-slice tables and dynamic_index)
        psi_l_Y_bc, psi_r_Y_bc = _split_bc_into_LR(psi_r_bc, bc_idx)
        # einsum contributions
        P_l_acc = P_l_acc + jnp.einsum('kmna,knbr->karmb',
                                       psi_l_X_bc_slice(bc_idx), psi_l_Y_bc)
        P_r_acc = P_r_acc + jnp.einsum('kmna,knbr->karmb',
                                       psi_r_X_bc_slice(bc_idx), psi_r_Y_bc)
        return (P_l_acc, P_r_acc), None

    (P_l, P_r), _ = lax.scan(body, (P_l_acc, P_r_acc), jnp.arange(n_bc))

    # existing post-pair pipeline:
    # P_l reshape → IFFT → conj; P_r reshape → IFFT; γ̃-contract; FFT → Z_q
    ...
```

- **Trap side-step?** Yes — but for a different reason than C. The
  scan body's loop carry is `(P_l_acc, P_r_acc)` rank-5 arrays.
  Inside the shard_map body these are **per-rank-local with no
  global sharding annotation**. SPMD has already done its work at
  the shard_map boundary (in_specs/out_specs). The scan inside the
  body operates on rank-local tensors → no SPMD inflation. This is
  the "lax.fori_loop inside shard_map" pattern the CHANGELOG
  follow-up explicitly recommends.
- **Expected memory**: per rank
  - 2 rank-5 accumulators (P_l_acc, P_r_acc) live across scan iters: same as Path C.
  - 1 FFT box per scan iter, **aliased across iters by XLA's scan-internal allocator** (lifetimes don't overlap → 1 slot, not n_bc).
  - Better fusion than Path C because everything sits in one shard_map → one jit, single XLA module.
  - Peak similar to Path C, slightly lower due to fusion.
- **Implementation cost**: ~250-400 lines. Higher than C because:
  - `psi_G_store` needs a new `_slice_local_tile(x, y, bc_idx)` callable that takes a traced `bc_idx`. Currently the callback is closed over a static `b_lo, b_hi`.
  - `wfn_transforms.to_rchunk` body has to be inlined into the new shard_map (can't call the existing shard_map-wrapped `to_rchunk` from inside another shard_map's body cleanly — nested shard_maps are unusual).
  - The L/R band slicing per bc must be expressible inside scan (band ranges per bc are static at trace time, so we can pre-build per-bc slice tables and `jnp.take` them via traced bc_idx, or use `lax.switch` over a static dispatch).
  - The existing post-pair pipeline (IFFT, γ̃-contract, FFT) stays inside the shard_map; we just hoist its pair-density einsum into the scan loop.
- **Files touched**:
  - `src/common/isdf_fitting.py`: major rewrite of `c_q_from_psi_sm` and `z_q_from_psi_sm` (combine bc-loop into shard_map body, scan).
  - `src/common/psi_G_store.py`: new `_slice_local_tile_bc(x, y, bc_idx)` and a way to call it from inside a shard_map's scan body.
  - `src/common/wfn_transforms.py`: factor out `_box_kernel + IFFT + slice + Bloch phase` as a pure-jax function callable from inside another shard_map body (no shard_map wrapper).
  - `_make_fit_one_rchunk_kernel`: drop the bc-loop and the concat; just call the new `z_q_from_psi_sm`.
- **Tests**: same as Path C, plus:
  - HLO inspection: confirm one shard_map call covers all bcs, one FFT box slot in the buffer-assignment dump.
  - io_callback-inside-scan correctness: verify the host-side `_slice_local_tile` actually fires `n_bc` times per scan run.

### Path E — donate_argnums on `fetch_psi_rchunk` + sequence hint

- **Trap side-step?** No structural change. Donation can free a jit
  arg's buffer at jit return, but the bc-loop is inside ONE jit
  (`_kernel`); `fetch_psi_rchunk` calls don't have jit-boundary
  donation among themselves at this nesting level. The `n_bc`
  FFT boxes all live in the same jit trace.
- **Reject**: doesn't address the slot multiplication.

### Other path I would propose (Path F — hybrid)

`fetch_psi_rchunk` already has a K-CHUNK Python loop *inside* its
body that produces multiple FFT boxes per bc call (line 369-377).
Path F: collapse the K-chunk loop FIRST (the inner one in
`fetch_psi_rchunk`) into a `lax.scan` inside `to_rchunk`'s
shard_map body — same shard_map trick as Path D but limited to the
k-axis. This is a strict subset of Path D and gives a partial
improvement (eliminates `n_kchunk` multiplier on slot count, but
leaves the bc multiplier). Useful as a half-step but not a full
fix.

## 3. Recommendation

**Choose Path D.**

Reasons:

- The CHANGELOG follow-up entry explicitly calls out this
  refactor: "Structural: convert `c_q_from_psi_sm` /
  `z_q_from_psi_sm` bc-loop to `lax.fori_loop`. Would alias the
  `n_bc · S_fft` slots into one and recover most of the band-FFT
  pool's memory cost."
- The prior-art trap doesn't apply inside a shard_map body. The
  scan carry is per-rank-local; SPMD has stopped at the shard_map
  boundary.
- It aligns with the codebase's recent direction (2026-05-13
  `LORRAX_PSIG_RCHUNK_SHARDMAP=1` work: "keep G-flat gather,
  local IFFT, r-slice, and Bloch phase inside one shard_map
  region"). Path D extends the same pattern one level out:
  everything pair-density-related ends up in one shard_map.
- Better fusion than Path C: one jit, one XLA module, contiguous
  IFFT → einsum → IFFT → γ̃ → FFT chain. Path C splits the pipeline
  into multiple jits and loses some inter-stage fusion.
- The FFT-box slot count collapses from `n_bc · n_kchunk · S_fft`
  (currently 58) to ~1-3 (aliased across scan iters). That's the
  3.85× model miss the consensus identified.

**Fallback: Path C** if Path D hits an implementation snag with
`io_callback` inside scan-inside-shard_map. Path C is
mechanically simpler (mirrors solve_zeta's pattern) and gives
~80% of the memory benefit. Worth keeping in your back pocket.

**Reject A, B, E.** Empirically refuted by the prior-art note or
structurally insufficient.

## 4. Implementation sketch (Path D)

### 4a. New host-tile slicer in `psi_G_store.py`

Add a helper that takes a **traced** `bc_idx`:

```python
# In PsiGStore (or subclass):
def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx_traced):
    # x_idx, y_idx, bc_idx_traced are int32 scalars (traced)
    x, y, bc = int(x_idx), int(y_idx), int(bc_idx_traced)
    tile = self._host_tiles[(x, y)]
    b_lo = self._bc_band_offsets[bc]
    b_hi = self._bc_band_offsets[bc + 1]
    # NOTE: bc-iter band-chunk sizes may differ (last bc is short).
    # Pad to uniform bc_size_max so scan body has a static shape.
    bpd_max = max(b_hi_i - b_lo_i for b_lo_i, b_hi_i in zip(self._bc_band_offsets[:-1], self._bc_band_offsets[1:]))
    out = np.zeros((nk, bpd_max, ns, ngkmax), dtype=self._dtype)
    out[:, :b_hi-b_lo, :, :] = tile[:, b_lo:b_hi, :, :]
    return out
```

The shape returned must be static (scan requires it). Pad the
last (short) bc with zeros — zeros contribute zero to the einsum
(no math contamination, just wasted FLOPs on `≤ bc_size_max`
extra bands).

### 4b. Inline `to_rchunk` body in `wfn_transforms.py`

Factor out a pure-function variant (no shard_map wrapper):

```python
def to_rchunk_inner(psi, g_index, fft_grid_t, r0, r_len_i, kvecs_frac, norm="ortho"):
    """Per-rank-local: G-flat → FFT box → IFFT → r-slice → Bloch phase.
    Callable from inside another shard_map's body."""
    nx, ny, nz = fft_grid_t
    n_rtot = nx * ny * nz
    box = _box_kernel(psi, g_index, ngkmax=int(psi.shape[-1]))
    rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
    rb_flat = rb.reshape(*rb.shape[:3], n_rtot)
    slab = jax.lax.dynamic_slice_in_dim(rb_flat, r0, r_len_i, axis=-1)
    if kvecs_frac is not None:
        slab = apply_bloch_phase_on_slice(slab, kvecs_frac, fft_grid_t, r0, r_len_i)
    return slab  # (nk, bc, ns, r_len_i) per-rank-local
```

### 4c. Rewrite `z_q_from_psi_sm` (and its sibling `c_q_from_psi_sm`)

The new shard_map ingests `psi_l_X`, `psi_r_X`, NOT psi_Y. Inside,
a scan over bc fetches ψ_G_bc via io_callback and accumulates.

```python
def z_q_from_psi_sm(
    psi_l_X, psi_r_X,                          # rank-4 X-shard, full band axis
    psi_G_store,                                # closure -- pulls per-bc via io_callback
    band_chunk_ranges,                          # static: tuple of (b_lo, b_hi)
    band_range_left, band_range_right,          # static
    fft_grid_t, r_start_dyn, r_chunk_size,      # bc-loop arg
    gamma_L=None, gamma_R=None,                 # γ̃ tuples
    *, kgrid, mesh_xy,
):
    n_bc = len(band_chunk_ranges)
    bpd_max = max(b1 - b0 for b0, b1 in band_chunk_ranges)
    # Build per-bc band-slice tables (static):
    l_lo_tbl = jnp.asarray([max(0, band_range_left[0] - b0)
                            for b0, b1 in band_chunk_ranges], dtype=jnp.int32)
    l_hi_tbl = jnp.asarray([min(b1 - b0, band_range_left[1] - b0)
                            for b0, b1 in band_chunk_ranges], dtype=jnp.int32)
    # Same for R...
    # Pre-compute per-bc L/R band mask + slice tables so the scan body can
    # use jnp.where to zero out non-contributing bands.

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(P(None,'x',None,None), P(None,'x',None,None), P(), P()),
             out_specs=P(None,'x','y'),
             check_rep=False)
    def _local(psi_l_X_, psi_r_X_, perm_L_, phase_L_, perm_R_, phase_R_):
        nk_loc = psi_l_X_.shape[0]
        mu_loc = psi_l_X_.shape[1]
        # r-chunk extent: passed via closure
        r_loc = r_chunk_size            # this rank's r-slab (full extent — per-rank-local)
        # Init rank-5 accumulators
        P_l_acc = jnp.zeros((nk_loc, ns, r_loc, mu_loc, ns), c128)
        P_r_acc = jnp.zeros((nk_loc, ns, r_loc, mu_loc, ns), c128)

        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        def body(carry, bc_idx):
            P_l_acc, P_r_acc = carry
            # Pull this bc's tile from host (traced bc_idx — see 4a)
            psi_G_bc = io_callback(
                lambda x,y,b: psi_G_store._slice_local_tile_bc(x,y,b),
                jax.ShapeDtypeStruct((nk_loc, bpd_max, ns, ngkmax), c128),
                x_idx, y_idx, bc_idx, ordered=True)
            # Local IFFT + r-slice (see 4b)
            psi_Y_bc = to_rchunk_inner(
                psi_G_bc, g_index_l, fft_grid_t, r_start_dyn, r_loc,
                kvecs_frac_l, norm="ortho")
            # Slice per-bc into L/R band sub-ranges. Build masks indexed by bc_idx.
            l_lo = l_lo_tbl[bc_idx];  l_hi = l_hi_tbl[bc_idx]
            r_lo = r_lo_tbl[bc_idx];  r_hi = r_hi_tbl[bc_idx]
            # Zero-mask non-contributing bands so a uniform slice runs each iter:
            band_idx = jnp.arange(bpd_max)
            l_mask = (band_idx >= l_lo) & (band_idx < l_hi)
            r_mask = (band_idx >= r_lo) & (band_idx < r_hi)
            psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
            psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
            # Pull the bc's band slice of psi_l_X (analogously masked)
            psi_l_X_bc = jax.lax.dynamic_slice_in_dim(
                psi_l_X_, band_chunk_ranges_lo[bc_idx], bpd_max, axis=2)
            psi_r_X_bc = jax.lax.dynamic_slice_in_dim(
                psi_r_X_, band_chunk_ranges_lo[bc_idx], bpd_max, axis=2)
            # Mask the X-side bands consistently:
            psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
            psi_r_X_bc = jnp.where(r_mask[None, None, :, None], psi_r_X_bc, 0)

            # Pair-density einsum contributions
            P_l_acc = P_l_acc + jnp.einsum(
                'kmna,knbr->karmb', psi_l_X_bc, psi_l_Y_bc, optimize=True)
            P_r_acc = P_r_acc + jnp.einsum(
                'kmna,knbr->karmb', psi_r_X_bc, psi_r_Y_bc, optimize=True)
            return (P_l_acc, P_r_acc), None

        (P_l, P_r), _ = lax.scan(body, (P_l_acc, P_r_acc), jnp.arange(n_bc))

        # Existing post-pair pipeline, unchanged: IFFT → γ̃ → FFT
        P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
        P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0,1,2), norm='forward')
        P_l_R_conj = jnp.conj(P_l_R)
        del P_l, P_l_3d, P_l_R
        P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
        P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0,1,2), norm='forward')
        del P_r, P_r_3d
        Z_R = gamma_double_contract(P_l_R_conj, P_r_R,
                                     perm_L=perm_L_, phase_L=phase_L_,
                                     perm_R=perm_R_, phase_R=phase_R_,
                                     spin_axes=(3, 6))
        Z_q_3d = jnp.fft.fftn(Z_R, axes=(0,1,2), norm='forward')
        return jnp.transpose(
            Z_q_3d.reshape(nkx*nky*nkz, r_loc, mu_loc), (0, 2, 1))

    return _local(psi_l_X, psi_r_X, perm_L, phase_L, perm_R, phase_R)
```

### 4d. Simplified `_kernel`

`_make_fit_one_rchunk_kernel._kernel` becomes:

```python
@jax.jit
def _kernel(psi_l_X, psi_r_X, L_q, norms_l, norms_r, r_start_dyn,
            gamma_perm, gamma_phase, cct_trace_per_q):
    # band_norms applied INSIDE the scan body via a small (nb,) jnp.array
    # closed over by the new z_q_from_psi_sm. Or scale psi_l_X up-front.
    Z_q = z_q_from_psi_sm(
        psi_l_X, psi_r_X,
        psi_G_store,
        band_chunk_ranges, band_range_left, band_range_right,
        meta.fft_grid, r_start_dyn, actual_n_rchunk,
        gamma_L=(gamma_perm, gamma_phase) if not is_charge else None,
        gamma_R=(gamma_perm, gamma_phase) if not is_charge else None,
        kgrid=kgrid, mesh_xy=mesh_xy)
    # IBZ gather + solve (unchanged)
    if q_irr_idx_j is not None:
        L_q_for_solve = L_q[q_irr_idx_j]
        Z_q_for_solve = Z_q[q_irr_idx_j]
        cct_trace_for_solve = (cct_trace_per_q[q_irr_idx_j]
                                if cct_trace_per_q is not None else None)
    else:
        L_q_for_solve = L_q; Z_q_for_solve = Z_q
        cct_trace_for_solve = cct_trace_per_q
    return solve_zeta(L_q_for_solve, Z_q_for_solve, mesh_xy, q_chunk_size,
                      solver_kind=solver_kind,
                      cct_trace_per_q=cct_trace_for_solve)
```

### 4e. Subtleties to handle in implementation

- **Uniform bc size.** Pad the last bc to `bpd_max` and zero-mask
  it. The einsum runs at uniform shape every iter (same compiled
  einsum kernel reused across scan iters — that's the whole point
  of scan).
- **`band_range_left/right` overlap with bcs.** Pre-compute the
  per-bc L/R slice tables at trace time, pass as closure arrays.
  Use `jnp.where` with per-bc masks (constructed by gather on the
  tables via traced `bc_idx`).
- **`pseudobands` norms.** Currently `psi_*_Y_sm = psi_Y_full[...] / norms[...]`.
  In the new scheme, fold the norm divide into either (a)
  pre-multiplying `psi_l_X` / `psi_r_X` by `1/norms` once before
  the kernel, or (b) divide inside the scan body via a per-bc
  norm slice. (a) is cleaner.
- **`io_callback` inside scan**: ensure `ordered=True` so the
  host-side fetch is sequenced. Test on CPU first.
- **`psi_G_store` lifecycle.** The store's host tiles must remain
  valid for the duration of the jit's execution (no python-side
  freeing). Current code already handles this (`begin_rchunk` /
  `end_rchunk` wrap the jit call). No change needed.
- **`k_chunk` (psig_k_chunk_size).** Currently fetch_psi_rchunk
  has an inner k-chunk Python loop. Either:
  - Drop k_chunk (no longer needed at the structural fix's memory
    profile), or
  - Push the k-chunk into a NESTED scan inside the bc-scan body.
    Probably unnecessary — the FFT box's per-rank size at one bc
    is `c128[nk, bc, ns, n_rtot] · F_fft / (P or 1)`; per-rank if
    sharded, dense-per-rank if not. At CrI3 charge bc=16 and full
    nk=36 unsharded: 16·36·16·2·1.1M·4 = 80 GB. Still too big.
    With bc=16, k_chunk=6: 16·6·16·2·1.1M·4 = 13 GB. So the
    k_chunk loop is STILL needed at CrI3. Nest it as an inner scan.

## 5. Validation plan

### 5a. Smallest numerical regression test

**MoS2 3×3 ζ-fit, charge channel, single rchunk, no pseudobands.**
- Existing test infrastructure: `tests/test_isdf_fit.py` (or
  whatever runs the CCT/ZCT bit-identity check).
- Reference: a saved ζ from the current code on MoS2 3×3 charge
  (small enough to commit a few KB of golden values).
- New code's ζ must match within `rtol=1e-10, atol=1e-12` over
  the full `(nq, n_rmu, n_rtot)` tensor. The pair-density chain
  is bilinear in ψ and the IFFT/FFT pair restores ψ unchanged
  numerically; the only floating-point source of drift is the
  scan accumulator's order-of-summation (sum-over-bc vs cat-then-sum
  inside einsum). These should differ at most by ULP-class rounding.
- Bispinor transverse: same test on a small system with
  γ̃^{i} ≠ I to exercise the γ̃-fold path.

### 5b. Numerical edge cases

- **Single-bc case** (n_bc = 1): the scan runs once, no padding.
  Output must match the current code byte-for-byte (or within
  ULP).
- **`pseudobands` enabled** (band_norms not None): verify the
  norm scale-pre-multiply gives identical results to the current
  divide-after-concat.
- **Asymmetric L/R band ranges** (L = [b0,b3), R = [b1,b4)): a bc
  that touches only L or only R should contribute to one
  accumulator only — verify via reference comparison.
- **Last bc short**: pad-mask must zero the trailing bands;
  pad-bands contribute zero to einsum. Test with nb_F = bc·k + r
  for r ∈ {1, bc-1}.

### 5c. HLO signal that confirms the fix worked

Two checks:

1. **Slot count**: dump the `module_*.jit__kernel.memory-usage-report.txt`
   from the new compiled kernel at CrI3 6×6 80 Ry with the same
   config as the old HLO (`band_chunk=16, psig_k_chunk=6,
   r_chunk = planner pick`). Count distinct preallocated-temp
   slots holding a `c128[k_chunk, bc, ns, nx, ny, nz]`-class
   buffer (the band-FFT-box shape class).
   - **Pass**: ≤ 3 slots (one box + one cuFFT scratch + maybe one
     accumulator interaction).
   - **Fail**: 58 slots — fix didn't take.

2. **Total allocation**: `XLA_FLAGS='--xla_dump_to=...
   --xla_dump_hlo_pass_re=.*'` then sum the preallocated-temp
   region's reported total. Compare against the old 200 GiB.
   Expect ~10-30 GiB at the new config.

3. **`all-gather` count**: the prior LORRAX_PSIG_RCHUNK_SHARDMAP=1
   work checked `collectives_details.txt` for spurious all-gathers
   on `wfn_transforms.py:109`. After Path D, those should remain
   absent.

### 5d. Memory measurement

`nvidia-smi --query-gpu=memory.used` sampled at the end of the
chunk loop (the existing `_track_peak()` in `fit_zeta_to_h5`).
Compare against the same metric on the old code at the same
config:
- **Old code at planner-picked r_chunk** (the previously-OOMing
  config): 196 GiB (RESOURCE_EXHAUSTED).
- **New code at the same r_chunk**: should be ~10-30 GiB and not
  OOM.
- Run at `r_chunk = 73,328` (the planner's pick), verify success
  and report measured peak.

### 5e. End-to-end correctness

Re-run the CrI3 6×6 80 Ry COHSEX comparison against BGW (the
existing reference run under `runs/CrI3/M_6x6_80Ry_*`). Σ values
must match the pre-fix LORRAX_A baseline within the existing
tolerance — confirms the streamed pair-density accumulation gives
the same ζ as the eager concat.

---

Agent 2 structural fix done
