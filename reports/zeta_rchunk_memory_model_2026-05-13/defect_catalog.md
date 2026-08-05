# Defect catalog — Python loops / replicated intermediates in lorrax_A

Agent-4 inventory, 2026-05-13.  Scope: `sources/lorrax_A/src/**`, read-only.
Principle: *zero replicated intermediates or work done in this entire
procedure* (see `PATH_D_PICKUP.md` §0).  A defect is any place where a
Python-level loop iterates inside a `@jax.jit` trace and pile-ups N
shape-identical intermediates that XLA's BufferAssignment cannot alias,
or any place where a buffer that should be sharded is materialised
unsharded on every rank.

Methodology
-----------

1. AST walk of every non-test `*.py` under `sources/lorrax_A/src/`,
   collecting every `for` statement that lives in the body of a function
   decorated `@jax.jit`, `@partial(jax.jit, ...)`, or
   `@partial(shard_map, ...)`.
2. Second pass: every for-loop in a *factory* function that also defines
   a `@jax.jit`/`@partial(shard_map)` inner function — captures the
   bc-loop-style pattern where the loop is at the kernel-body level of
   the *outer* jit closure (i.e. unrolled at trace time).
3. Targeted reads of the six pre-identified suspects in the prompt.
4. Loops that only iterate over mesh axis names (`for ax in axis_names`)
   or 2-step Gram-Schmidt unrolls were excluded — see §"Not violations".

CrI3 6×6 80 Ry reference scale (4×4 mesh = 16 ranks, p_xy = 16):
`n_rtot ≈ 1.1M` (e.g. 96×96×120), `nk = nq = 36`, `nb_total ≈ 310`,
`μ ≈ 1500`, `ns = 4` (bispinor), `c128 = 16 B`.

## 1.  Summary table

| # | file:line | function | defect type | est. bytes/rank @ CrI3 6×6 80 Ry | fix |
|---|-----------|----------|-------------|----------------------------------|-----|
| 1 | `common/isdf_fitting.py:1274` | `_kernel` (bc-loop in `fit_one_rchunk`) | Python-loop-in-jit, unsharded FFT-box pile-up | **~200 GiB** (n_bc · S_fft · k_chunk · ns · n_rtot · 16; HLO-measured 58 slots) | scan-inside-shard_map (Path D, in-flight on `lorrax_B`) |
| 2 | `common/psi_G_store.py:370` | `PsiGStore.fetch_psi_rchunk` k-chunk loop | Python-loop-in-jit (called from `_kernel`) | within (1) — multiplies the slot count by `n_kchunk` per bc iter | subsumed by Path D §C/D (`to_rchunk_inner` inside the bc-scan); for now the k-chunk knob just trades n_bc bigger for n_kchunk smaller |
| 3 | `common/load_wfns.py:755,789` | `load_centroids_band_chunked` bc-loop / k-chunk loop (Peak A) | driver Python loop + unsharded FFT box transient inside `to_rmu` | per-iter FFT-box ≈ k_chunk · band_chunk · ns · n_rtot · 16 · 4× (fft_box_factor); planner forces small k_chunk_size | (a) immediate: same Path D scan-inside-shard_map pattern for the centroid sample (drop the n_bc/n_kc unroll); (b) structural: keep the band/k loop in driver but make `to_rmu`'s FFT box sharded on `('x','y')` — Single-slot-but-unsharded is still a principle violation per §0 |
| 4 | `common/isdf_fitting.py:1130` | `solve_zeta` q-batch loop | Driver Python loop inside the outer `_kernel` jit (Z_col, zeta accumulator live across iters) | nq_chunks · q_batch · μ · μ · 16 / p_xy per iter, plus 1× zeta accumulator at full nq · μ · n_rchunk | structural: needs single-shot solve via batched LU/Cholesky OR cuSolverMp FFI accepting (q, μ, n_rchunk) cube — prior `lax.scan`/`fori_loop` attempts hit WhileOp/SPMD-replication trap (see code comment lines 1126–1129); not fixable with scan-inside-shard_map per the failure note |
| 5 | `gw/compute_vcoul.py:623` | `_v_q_per_q_g_chunked_jit` G-chunk loop | Python-loop-in-jit, n_chunks intermediates | n_chunks · (μ_L · g_chunk · 16) per rank ≈ 17 · 100 MB ≈ 1.7 GB for L_chunk / R_chunk / L_weighted slot families on full-mesh | `lax.fori_loop` with donated `V` accumulator (1-arg state) — straightforward; only one accumulator, dynamic_slice already in body |
| 6 | `solvers/davidson.py:167` | `_ortho_expand` Gram-Schmidt | Python-loop-in-jit, 2-iter unroll | tiny — `overlap` is (m, n) where m, n are subspace sizes (~hundreds); ~MB scale | not a real defect at current scale; leave (see §"Not violations") |

Total CrI3-scale cost dominated by **(1) + (3)**.  Defect (1) is what's
in Path D scope.  Defect (3) is Peak A and shares the same structural
shape — `_peak_A_centroid_load._kernel` is the same "single-slot but
unsharded FFT box" pattern as Peak C's `band_fft_unsharded` term.

## 2.  Per-defect details

### Defect 1 — bc-loop in `fit_one_rchunk._kernel`

**File**: `common/isdf_fitting.py:1268–1278`

```python
_b0 = int(band_range_full[0])
_l_lo = int(band_range_left[0]) - _b0
_l_hi = int(band_range_left[1]) - _b0
_r_lo = int(band_range_right[0]) - _b0
_r_hi = int(band_range_right[1]) - _b0
psi_Y_parts = []
for bc_range in band_chunk_ranges:
    psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(
        bc_range, r_start_dyn, actual_n_rchunk))
psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)
```

Why it's a violation.  `band_chunk_ranges` is a static Python tuple of
N_bc band-window tuples.  At trace time the loop body is fully unrolled
into the HLO; each iteration's `fetch_psi_rchunk` produces a rank-6
intermediate `c128[k_chunk, bc_size, ns, nx, ny, nz]`.  XLA's
BufferAssignment cannot alias adjacent slots because their lifetimes
overlap (every intermediate is held alive in `psi_Y_parts` until the
post-loop `jnp.concatenate`).  Measured at CrI3 6×6 80 Ry on 4×4 mesh
(2026-05-13 HLO dump `module_0408.jit__kernel`): 58 concurrent live
FFT-box slots ≈ n_bc · S_fft with S_fft ≈ 3 — each *per rank, no mesh
sharding* — total ~200 GiB of unsharded temp pool.

Structural fix.  Path D §4 in `agent_2_structural_fix.md` (Steps 1–3,
copied into `PATH_D_PICKUP.md`).  Replace the Python `for bc_range` with
`lax.scan` over `jnp.arange(n_bc)` inside the `z_q_from_psi_sm`
shard_map body, fetching per-bc ψ tiles via `io_callback` and applying
L/R masks via static tables (the band-mask approach).  After Path D
lands, the FFT box exists once at scan-body scope and aliases across
scan iterations — slot count → ~1 per family.

Currently fixable by: **scan-inside-shard_map (Path D)** — scaffolding
already on `lorrax_B`, branch `agent/zeta-bc-scan-shardmap`
(`to_rchunk_inner` helper + `_slice_local_tile_bc`).

### Defect 2 — k-chunk loop in `PsiGStore.fetch_psi_rchunk`

**File**: `common/psi_G_store.py:369–377`

```python
_k_chunk = max(1, self._k_chunk_size) if self._k_chunk_size > 0 else nk_slice
...
slabs = []
for k0 in range(0, nk_slice, _k_chunk):
    k1 = min(k0 + _k_chunk, nk_slice)
    slabs.append(to_rchunk(
        psi_G_flat[k0:k1], g_index_full[k0:k1], fft_grid_t,
        r_start_dyn, int(r_chunk_size),
        kvecs_frac=kvecs_frac_full[k0:k1], norm="ortho",
        mesh=self.mesh))
return jnp.concatenate(slabs, axis=0)
```

This loop is at the same trace-time level as Defect 1 (called from
inside the bc-loop, which is itself inside the outer `_kernel` jit).
Each `to_rchunk` call produces a `c128[_k_chunk, bpd, ns, nx, ny, nz]`
intermediate that XLA cannot alias across iterations because they all
need to be in scope simultaneously for the post-loop concatenate.

This is the knob that Phase 1 of the memory work uses to cap the
per-call FFT box: setting `psig_k_chunk_size = k_chunk` trades larger
`n_kchunk` (slot pile-up multiplier) for smaller per-slot bytes.  The
planner's `band_fft_unsharded` term in `gflat_memory_model.py:240` is
explicitly sized as `band_fft_slots · k_chunk_eff · nb_padded · ns ·
n_rtot · 16` — i.e. it folds Defect 1 and Defect 2 into one budget
term.

Fix.  Subsumed by Defect 1's Path D plan: `to_rchunk_inner` is called
once inside the bc-scan body, FFT box gets a single sharded slot at
scan-body scope, k-chunking knob becomes irrelevant.  If a residual
need for k-chunking remains after Path D (very large k-grids where even
the per-rank `(nk/p, bpd, ns, n_rtot)` slab overflows), convert to a
nested `lax.scan` over k inside `to_rchunk`'s shard_map body — same
pattern, one level deeper.

Currently fixable by: **scan-inside-shard_map (Path D)** — same fix as
Defect 1; this loop disappears with the bc-loop replacement.

### Defect 3 — `load_centroids_band_chunked` bc-/k-loop + unsharded FFT box

**File**: `common/load_wfns.py:755–815` (driver bc-loop), `to_rmu`
factory at `common/wfn_transforms.py:288–335` (jit'd kernel with
unsharded FFT box transient).

Two coupled issues, both flagged in `PATH_D_PICKUP.md` §0:

(a) The Python loops at `load_wfns.py:755` (`for bc_idx in
range(num_band_chunks)`) and `load_wfns.py:789` (`for kc_idx in
range(num_k_chunks)`) live in *driver* code.  Each iteration calls a
fresh jit (`loader.load` → `to_rmu` → `_reshard_fn`) and synchronizes
via `jax.block_until_ready`.  They are **not** Python-loop-in-jit
violations as defined.  However, they exist as a workaround for…

(b) `to_rmu` materialises the FFT box `c128[nk_chunk, nb_padded, ns,
nx, ny, nz]` UNSHARDED on every rank during the IFFT.  The inline
comment in `load_wfns.py:656–666` records the diagnosis: the
`with_sharding_constraint` on the band axis is lost once XLA's FFT
planner pulls gather + IFFT into one fused HLO module.  The planner
in `gflat_memory_model._peak_A_centroid_load` (line 159–162) accounts
for this with `_bytes_c128(nk, band_chunk, ns, n_rtot, shard=p_xy) *
fft_box_factor` (factor=4 empirical).

The single FFT-box slot is unsharded — even though there's only one,
it's *replicated on every rank*.  Per the §0 principle this is a
defect: a buffer that should be sharded is not.

CrI3 6×6 80 Ry per-rank cost: `1 (slot) · k_chunk · band_chunk · ns ·
n_rtot · 16 · ~4 (fft_box_factor)` ≈ a few GB per chunk depending on
how aggressively the planner shrinks `k_chunk_size` and
`band_chunk_size`.  Bounded — that's why the driver Python loop exists
at all.

Fix.  Two paths, picked per cost / complexity:

- Easy (closes the unsharded gap, keeps the driver loop): re-enter a
  `shard_map` over `('x', 'y')` around the gather+IFFT+sample in
  `to_rmu`, with `in_specs / out_specs` keeping the band axis sharded
  through the FFT.  Mirror of `to_rchunk` (`wfn_transforms.py:338`) —
  which already does this for the r-chunk version.  Use
  `make_sharded_fftn_3d` (the helper that v_q_tile already uses for the
  same problem on the ζ→G FFT path).
- Structural (closes everything): drop the driver bc/k loop entirely
  and turn `load_centroids_band_chunked` into a single scan-inside-
  shard_map of the bc axis, same pattern as Path D.  The centroid
  sample at the end is a single `dynamic_slice`-style gather — cheap
  per iter.

Currently fixable by: **shard_map around to_rmu's FFT** (easy path)
or **scan-inside-shard_map** (structural).  Easy path is enough to
restore the §0 invariant; structural fix is only worth it if the
driver-level Python dispatch overhead becomes measurable (today it's
not — load_centroids is a one-shot per channel).

### Defect 4 — `solve_zeta` q-batch Python loop

**File**: `common/isdf_fitting.py:1118–1141`

```python
# Allocate output buffer
zeta = jnp.zeros_like(Z_col)

# Python loop with async dispatch — each call returns immediately.
# ... NOTE: scan(unroll=8) was attempted but OOMs — XLA pipelines adjacent unrolled
# iterations, keeping 2× preallocated-temp alive (18.9 GB). scan without unroll
# triggers SPMD replication of the sharded accumulator (88 GB OOM). fori_loop
# has the same WhileOp issue. The Python loop is the only approach that gives
# constant DUS offsets AND sequential memory reuse.
for q0 in range(0, nq_padded, q_batch):
    q1 = q0 + q_batch
    zeta = helpers.solve_batch_and_update(L_q[q0:q1], Z_col[q0:q1], zeta, q0)
```

Why it's a violation under §0.  This loop is inside the outer `_kernel`
@jax.jit trace (called from `fit_one_rchunk._kernel`, line 1326 →
`solve_zeta(...)`).  Each iteration appends a `solve_batch_and_update`
jit call that materialises a per-q-chunk `batch_result =
_sharded_cho_solve_batch(L_rep, Z_batch_col)` intermediate.  Donation
on `zeta_acc` collapses the accumulator side, but the per-iter
`batch_result` and the L_rep replicate-allgather form a temporary slot
pile-up exactly analogous to the bc-loop: N_qchunks copies of a
`c128[q_batch, μ, n_rchunk]` shape on every rank.

The code comment explicitly records that the obvious fixes failed:

- `scan(unroll=8)`: XLA pipelines adjacent iters → 2× preallocated-temp
  alive (the agent-B "unrolled-loop slot pile-up" diagnostic, recorded
  in `[[feedback_path_d_scaffolding_pattern]]` and
  `[[feedback_zero_replicated_intermediates_principle]]`).
- `scan` without unroll: SPMD replication of the sharded accumulator,
  88 GB OOM.
- `fori_loop`: WhileOp / SPMD replication trap.

These are the same trap the bc-loop hits.  The fix that worked for the
bc-loop (scan inside shard_map) might also work here — the q-batch
solve is shape-identical across q's, the accumulator is sharded along
`r_chunk` on `('x','y')`, and `solve_batch_and_update` is already
shard_map-decorated.  But the prior attempts predate Path D; they
likely tried scan AT THE OUTER LEVEL, not inside shard_map.

Currently fixable by: **needs investigation**.  Not the same fix as
Defect 1 in mechanics — the q-batch already lives at the post-reshard
boundary, where `Z_col` is sharded along the `r_chunk` axis (not the q
axis).  Scan-INSIDE-shard_map over q would require: (a) re-spec the
sharding so the q-axis is replicated inside the scan body, (b) the
inner cho_solve runs once-per-q with a (μ, n_rchunk) RHS sharded on
`('x','y')`.  Or: drop the q-loop entirely and call the existing
`solve_all_at_once` path with a stricter feasibility gate — current
code falls back to per-q loop when `q_batch < nq`; if cuSolverMp
(LU/Chol) is available, the batched single-shot can take a (nq, μ, μ)
block-diagonal solve as one FFI call.

Status per `PATH_D_PICKUP.md`: "Not in Path D scope today, but log it
as the next defect after Path D lands."

### Defect 5 — `_v_q_per_q_g_chunked_jit` G-chunk loop

**File**: `gw/compute_vcoul.py:589–625`

```python
@partial(jax.jit, donate_argnums=(0,), static_argnums=(4,))
def _v_q_per_q_g_chunked_jit(V_acc, zeta_q_L, zeta_q_R, v_q, g_chunk):
    ngkmax = int(zeta_q_L.shape[-1])
    def body(start, V):
        L_chunk = jax.lax.dynamic_slice_in_dim(zeta_q_L, start, g_chunk, axis=-1)
        R_chunk = jax.lax.dynamic_slice_in_dim(zeta_q_R, start, g_chunk, axis=-1)
        v_chunk = jax.lax.dynamic_slice_in_dim(v_q, start, g_chunk, axis=0)
        L_weighted = jnp.conj(L_chunk) * v_chunk[None, :]
        return V + L_weighted @ R_chunk.T
    n_chunks = ngkmax // g_chunk
    V = V_acc
    for i in range(n_chunks):
        V = body(i * g_chunk, V)
    return V
```

Why it's a violation.  `for i in range(n_chunks)` unrolls inside the
`@partial(jax.jit, donate_argnums=(0,))` body.  Each iteration's
`L_chunk`, `R_chunk`, `v_chunk`, `L_weighted` are shape-identical
across iterations.  XLA *can* alias them when all are temporaries
(no overlapping lifetime — chunk i+1 doesn't use chunk i's
L_weighted), but observed practice (see bc-loop, q-batch loop) is
that the buffer assigner often fails to recognise this and pile-ups
N_chunks copies of each slot family.

CrI3 6×6 80 Ry: ngkmax ≈ 70k, g_chunk = 4096 → n_chunks ≈ 17.
L_chunk per rank: μ_local × 4096 × 16 ≈ (1500/16) × 4096 × 16 = 6 MB.
17 unrolled copies = ~100 MB per slot family × ~3 families ≈ 300 MB —
modest, and the V_acc accumulator is donated so it's the only
persistent buffer.  Lower priority than (1) and (3); still a
principle violation per §0.

Currently fixable by: **`lax.fori_loop` with V as carry**.  Trivial
replacement here because the body is single-output, no donation
hazards, no replicated state.  `lax.scan` over `jnp.arange(n_chunks)`
would also work and is the more idiomatic JAX pattern, but the
3-line `fori_loop` swap is the smaller diff.  No shard_map needed —
this kernel inherits sharding from the caller and there's no
re-trace concern.

### Defect 6 — `_ortho_expand` Gram-Schmidt 2-unroll

**File**: `solvers/davidson.py:151–178`

```python
@jax.jit
def _ortho_expand(V, P):
    for _ in range(2):
        overlap = jnp.einsum('m...,n...->mn', jnp.conj(V), P, optimize=True)
        P = P - jnp.einsum('mn,m...->n...', overlap, V, optimize=True)
    ...
```

Strictly a Python-loop-in-jit, but the loop has fixed length 2 (CGS2
projection) and produces a single rank-2 `overlap` intermediate per
iter that's small (m × n where m, n are subspace sizes in the
hundreds).  XLA aliases the two iters trivially.  Not a measurable
cost; **listed for completeness, not for action**.  See §"Not
violations" below.

## 3.  Prioritization

1. **Defect 1** — Path D in flight on `lorrax_B`.  Biggest hit at
   CrI3-scale (~200 GiB of unsharded pool).  Closes Defect 2 as a
   side effect.
2. **Defect 3** — Peak A unsharded FFT box.  Same shape signature
   ("single-slot but unsharded") as the bc-loop's slots; the §0
   principle treats them identically.  Easy fix path exists
   (shard_map around `to_rmu` mirroring `to_rchunk`).  Estimated cost
   per-rank only a few GB at CrI3 scale — not a current OOM driver,
   but a principle violation that costs nothing to remove once
   Path D's `to_rchunk_inner` is in.
3. **Defect 4** — `solve_zeta` q-batch.  Open structural question.
   Code comment documents three failed approaches.  Worth re-trying
   scan-inside-shard_map (the Path D pattern) once Path D is
   validated, or pursuing a cuSolverMp batched-LU FFI single-shot.
4. **Defect 5** — `_v_q_per_q_g_chunked_jit`.  Cheap `lax.fori_loop`
   fix.  Modest CrI3 cost (~300 MB).  Good first follow-up after
   Path D lands as a confidence-builder for the fori_loop pattern in
   a different physics context.
5. **Defect 6** — leave alone.

## 4.  Not-actually-violations (candidate hits ruled out)

These appeared in the AST/grep scan and were investigated; each is
documented here so future audits don't re-flag.

- **`solvers/davidson.py:_ortho_expand` 2-unroll**.  See Defect 6.
  Listed in the table for principle completeness but operationally
  not actionable.

- **`gw/v_q_tile.py:1272 / 1289 / 1320 / 1423` — `compute_V_q_tile`
  nested q/μ/ν Python loops**.  Driver code that dispatches one
  pre-cached `_make_V_q_tile_kernel` jit per (q-batch, μ-block,
  ν-block) tile, with `block_until_ready` between iterations.  Each
  call's intermediates are freed before the next call (no Python list
  appending; only `V_acc` survives via donation).  Not in the same
  jit trace.  The `_make_V_q_tile_kernel` body itself (`v_q_tile.py:
  704–740`) has zero internal Python loops — `same_zeta` is a
  Python-static branch, not a loop.

- **`gw/v_q_tile.py:503` — `for _ in range(4)` AOT shrink-retry**.
  Driver-side feasibility-check shrink loop; no JAX tracing involved.

- **`gw/v_q_tile.py:1076`, `gw/v_q_g_flat.py:192,209,431,519`,
  `gw/v_q_bispinor.py:567`** — Python loops at module / function
  scope building static numpy arrays (q-coord tables, qpoint
  enumerations, ngkmax shrink loops).  Pure host work, not in any
  jit trace.

- **`gw/ppm_sigma.py:1151` — `_ReduceScatterGpuAccumulator.add_tau`
  ω-batch loop**.  Driver-level Python loop over the ω axis, each
  iteration is a separate jit dispatch of `_project_tau_onto_omega`
  on a `(batch,)` slice.  No outer jit trace.  Per-iter intermediate
  lifetime is bounded by the dispatch sync.

- **`gw/ppm_sigma.py:1303` — `minimax_tau_integrate_sigma` τ loop**.
  Deliberate Python τ loop.  Code comment (line 1268–1272) records
  *"sigma stays a Python τ loop because its per-τ body emits NCCL
  and a monolithic scan regressed MoS2 3×3 by ~80%"*.  Each
  iteration is a separate jit dispatch (`build_sigma_tau(t_j)`); the
  body is single-output and freed between calls.  Not a
  Python-loop-in-jit violation.

- **`bandstructure/htransform.py:756` — `_kpath_batch` over kpath
  batches**, `bandstructure/bse_setup.py:149`, `bandstructure/
  htransform.py:162` — driver-level Python loops; per-iter jit
  dispatches with explicit `block_until_ready`.  Post-loop the
  outputs are concatenated inside a *separate* `_post_kpath` jit.

- **`centroid/kmeans_isdf.py:523,578,638`, `centroid/
  pivoted_cholesky.py:614,621`** — `for ax in axis_names` inside
  shard_map factory bodies.  Iterates over 2 mesh axes ('x', 'y')
  for axis-name string concatenation in `psum`/`reshape` specs.
  No data intermediates pile up — these emit string-level HLO
  config, not buffers.

- **`gw/compute_vcoul.py:324–326` — triple `for qx/qy/qz in
  range(nk*)`**.  Pure numpy host loop building
  `_v_head_avg[qx,qy,qz]` for 3D bulk MC-averaged head correction.
  Not in any JAX trace.

- **`common/load_wfns.py:755 / 789` driver loops considered
  separately from Defect 3(b)**.  The loops themselves are not
  in-jit; only the FFT-box materialisation inside `to_rmu` is the
  defect.  See Defect 3.

- **`psp/run_sternheimer.py:1151,1326,1422`,
  `bse/davidson_absorption.py:206`, `bse/absorption_haydock.py:267`,
  `bse/bse_feast.py:755,777`, `common/cublasmp_w_solve_bisect.py:105`,
  `centroid/current_density.py:147`** — factory-fn-level driver
  loops, each iteration calls a pre-compiled jit kernel.  No outer
  jit trace.  Outside the GW/ζ pipeline scope of this audit anyway.

Agent 4 catalog done
