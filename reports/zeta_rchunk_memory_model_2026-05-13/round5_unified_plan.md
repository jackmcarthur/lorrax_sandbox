# Round 5 — Unified plan: scan-inside-shard_map forward helper

**Owner**: Agent 4 (synthesis lead).
**Contributors**: Agent 1 (SPMD/shard_map/mesh), Agent 2 (io_callback / host-device lifecycle), Agent 3 (XLA/HLO/BufferAssignment), Agent 4 (numerics + validation).
**Status**: ✅ **FINAL** — sign-offs from all four agents collected
in `round5_discussion.md` (A1 line 130, A2 line 295, A3 line 449,
A4 footer below).

This is the **Round-6 implementer's working document**.  Treat as
authoritative; cross-reference standalone analyses for depth:
- `round5_agent_1_spmd.md` — SPMD/shard_map invariants (§1–9).
- `round5_agent_3_hlo.md` — HLO buffer-assignment predictions.
- `round5_discussion.md` `Agent 2 → others` — io_callback /
  host-tile lifecycle (no standalone doc; the discussion message
  IS the deliverable).

**Headline (CrI3 6×6 80 Ry)**: predicted peak preallocated-temp
**~13–15 GiB per rank**, vs today's 48.63 GiB — ~3× memory
reduction, zero remat warnings expected.  Implementation is gated
on a 4-primitive composition smoke test (§3.6) that must pass on
CPU and GPU before kernel rewrite begins.

## 1.  Goal + design statement

Replace the **flat-axis** scan-inside-shard_map design landed in
Round 3 (`gflat_to_rchunk` + `PsiGStore.psi_G_device_full`
concat-reshuffle + slice-then-shard_map at the kernel boundary) with
the **per-bc** scan-inside-shard_map design originally sketched in
`agent_2_structural_fix.md` §4c–d but DEFERRED during Round 3
implementation in favor of the now-buggy flat-axis variant.

The new design:

- Drops `PsiGStore.psi_G_device_full` (and its tracer-leak bug
  caused by the lazy-cached `jax.Array` outliving its source `jit`).
- Drops `gflat_to_rchunk` and its `out_specs = P(None,('x','y'),
  None,None)` boundary — the kernel never materialises
  `psi_Y_full` at any sharding; each rank only ever sees its own bc
  tile's r-slab.
- Replaces them with a `lax.scan` over band-chunks **inside** the
  consumer's shard_map (`z_q_from_psi_sm._local`, with a sibling
  rewrite for `c_q_from_psi_sm._local`).  Per-iter:
  1. `io_callback` pulls this rank's bc tile of ψ(G-flat) from host
     into per-rank-local memory.
  2. `to_rchunk_inner` does the G→r FFT + Bloch phase on the
     per-rank-local box.
  3. Two einsums into the rank-5 `P_l_acc` / `P_r_acc` carries (one
     for L, one for R, with L/R band masks pre-computed at trace
     time and gathered per-bc by traced `bc_idx`).
- After the scan, the existing post-pair pipeline (IFFT → γ̃ → FFT →
  transpose) runs on the *fully accumulated* `P_l`, `P_r` — i.e.
  one IFFT/FFT pair per kernel call, not per bc.

Net effect: the 30 GiB of double-`psi_Y_full` materialisation Agent 1
identified in `round4_improvements.md` §3a–b disappears because
**`psi_Y_full` never exists in this kernel**.  The (k_chunk, bc, ns,
n_rtot) FFT box lives once at scan-body scope, fully per-rank-local
inside the shard_map, and gets scan-aliased across bc iters.

**Converged peak prediction (Agent 1 v3 + Agent 3 v2, §4.1)**:
~13–15 GiB per-rank total preallocated-temp at CrI3 6×6 80 Ry,
**~3× reduction** vs today's 48.63 GiB.  ~4 substantive slots
(2 carry-class `P_l_acc` / `P_r_acc` at 3.71 GiB each, 1 FFT box
~5 GiB scan-aliased, 1 all_gather slab ~340 MB scan-aliased).

### Load-bearing structural requirements (read before §2 / §4)

Three pieces must be in the design for it to compute the correct
math AND fit the predicted peak.  Each is a "naive implementation
will fail silently" hazard.

1. **Per-rank `r_loc = n_zchunk / p_y` carry sizing (Agent 1 §2.3)**.
   See callout below.
2. **Per-iter `all_gather` across `('x','y')` axis=bands, AFTER
   IFFT-and-slice (Agent 1 §2.9, Agent 3 §4.1a)**.  Host tiles are
   band-flat-sharded over the full `('x','y')` mesh, so each rank
   only has `bpd_per_bc = bc_size / P` bands of any given bc (CrI3:
   1 band/rank/bc).  The einsum `'kmna, knbr → karmb'` contracts
   over the band axis — without the gather, each rank's einsum sums
   only 1/P of the bands → **partial result, wrong numerics**.
   Gather AFTER IFFT-and-slice, not before (gather-first would
   force an 80 GB per-rank FFT box; IFFT-first keeps it ~5 GiB).
3. **`out_spec = P(None, 'x', 'y')` boundary, single interleaved
   scan (Agent 1 §2.10)**.  Both accumulators live concurrently
   inside the scan body (no serialization — at 3.71 GiB carry,
   serialization saves <4 GiB during scan but the post-scan γ̃
   contract needs both buffers simultaneously anyway).

### Load-bearing correctness gotcha (Agent 1 §2.3 — repeat here so
nobody misses it)

The scan accumulators' r-dimension is **per-rank**, not full:
`r_loc = n_zchunk / p_y` (where `p_y` is the mesh size along
`'y'`).  The shard_map's `out_spec = P(None, 'x', 'y')` declares
n_zchunk sharded on `'y'`; manual-mode shard_map requires the
per-rank output to be `(nq, mu_loc, n_zchunk / p_y)`, so the carry
that feeds the tail must already be at `r_loc` extent.  Mechanism:
inside the body, `r0_local = r_start_dyn + axis_index('y') * r_loc`,
and `to_rchunk_inner` is called with `(r0_local, r_loc)` per rank.
Pre-flight invariant: `n_zchunk % p_y == 0` (planner enforces).

### Why this is different from the abandoned `solve_zeta` scan/fori
attempts

The `solve_zeta` prior-art comment (`isdf_fitting.py:1126-1130`)
records that `scan` and `fori_loop` at **outer level** (outside any
shard_map) replicated the sharded `zeta` accumulator → 88 GB OOM.
The new scan lives **inside** the shard_map body — `P_l_acc` /
`P_r_acc` are per-rank-local arrays, not globally-sharded.  XLA's
SPMD partitioner does not analyse inside shard_map regions; the
cross-rank communication happens only at the outer shard_map
boundary (`out_specs = P(None, 'x', 'y')`), once per kernel call.
Inside the shard_map XLA's normal scan-carry-aliasing applies.
No WhileOp / SPMD trap.

## 2.  Shard_map + scan structure  ⟨Agent 1 — condensed; full depth at `round5_agent_1_spmd.md`⟩

### 2.1.  WhileOp/SPMD trap does not fire (Agent 1 §1)

The `solve_zeta` prior-art comment (88 GB OOM with scan-without-unroll,
WhileOp issues with fori_loop) was for scan/fori at *outer level* —
the carry was a globally-sharded `jax.Array` and SPMD's WhileOp
analysis couldn't reason about it across iters.  Inside `shard_map`
the partitioner has already stopped at the `in_specs`/`out_specs`
boundary; the scan body is per-rank manual code and the scan lowers
to a WhileOp **within one rank's program**.  Carry is rank-local;
no sharding annotations for SPMD to inflate.  In-repo precedent
(Agent 1 §1.5): `accumulate_rchunk_to_gflat` does exactly this with
a rank-3 carry; Round 4 HLO confirms aliasing (single-slot FFT box).

### 2.2.  shard_map `in_specs` / `out_specs` (Agent 1 §3)

```python
L_spec_X = P(None, 'x', None, None)    # psi_l_X: (nk, n_rmu, nb_l, ns)
R_spec_X = P(None, 'x', None, None)    # psi_r_X: (nk, n_rmu, nb_r, ns)
out_spec = P(None, 'x', 'y')           # Z_q:    (nq, n_rmu, n_zchunk)
gamma_spec = P()                       # perm/phase: replicated
```

Note: `psi_l_Y` / `psi_r_Y` are NOT in the input list — they are
produced inside the body per scan iter, via `to_rchunk_inner` on
ψ(G)-bc pulled by `io_callback`.

### 2.3.  Carry sizing — load-bearing correctness gotcha (Agent 1 §2 + §3)

> **The per-rank scan accumulator's `r_loc` dimension MUST equal
> `n_zchunk / p_y`, NOT the full r-chunk extent.**

This is the most error-prone piece of the design.  Reasoning:
`out_spec = P(None, 'x', 'y')` declares the n_zchunk axis sharded on
`'y'`.  Under `shard_map`'s manual mode, the per-rank output must
have shape `(nq, mu_loc, n_zchunk / p_y)`.  The post-scan tail emits
exactly that, so the scan carry that feeds it must already be in
that per-rank shape:

```python
mu_loc = n_rmu_padded / p_x
r_loc  = n_zchunk / p_y       # ← NOT n_zchunk
P_l_acc = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
P_r_acc = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
```

This in turn means `to_rchunk_inner` inside the body must produce a
*per-rank `r_loc`-sized slab*, not the full r-chunk.  Mechanism:

```python
r0_local = r_start_dyn + jax.lax.axis_index('y') * r_loc
psi_Y_bc = to_rchunk_inner(psi_G_bc, g_index, fft_grid_t,
                            r0_local, r_loc, kvecs_frac, norm="ortho")
```

The IFFT box still spans the full FFT grid (FFT axes are replicated
by `in_spec`); only the post-IFFT *slice* is per-rank.  This is the
same trick `accumulate_rchunk_to_gflat` uses in reverse.

At CrI3 6×6 80 Ry on a 4×4 mesh under **today's**
`L_spec_X = P(None, 'x', None, None)` (μ on `'x'` only): the
actual kernel uses **`n_rmu = 376`** (verified from HLO param
`c128[36, 376, 376]` on `L_q` — see Agent 3's Round 3 retraction).
So `mu_loc = 376/p_x = 376/4 = 94`, `r_loc = n_zchunk/p_y = 73648/4
≈ 18412`. Per-rank carry: `36 · 2 · 18412 · 94 · 2 · 16 = 3.98 GB ≈
3.71 GiB` × 2 accumulators ≈ **7.42 GiB per rank** (both carries
live in interleaved single scan).

Iteration log: this number was retracted twice. Agent 1 v1 had
mu_loc=94 (correct) but with wrong p assumption. Agent 1 v2 + Agent
3 v1 used `n_rmu=1504` from the ISDF basis count → mu_loc=376 →
14.85 GiB carry. Agent 3 v2 then verified from HLO that `n_rmu=376`
inside the kernel, restoring mu_loc=94 → **3.71 GiB per side**.
The final number stands.

**Pre-flight requirement**: verify `n_zchunk % p_y == 0` (planner
must enforce).  Agent 3 to confirm this is a planner invariant; if
not, the design needs an explicit pad-to-mesh-multiple step.

#### 2.3.1.  Round-6 implementation note — IFFT is on FULL r-chunk, r-slice happens AFTER gather (Agent 2 `f567aa0` deviation, see §9)

The original §2.3 sketch said `to_rchunk_inner` inside the body
should produce a *per-rank `r_loc`-sized slab* by passing
`r0_local = r_start + axis_index('y') * r_loc, r_len=r_loc`.

**That order is incorrect.** Agent 2 surfaced this during Round-6
implementation (round6_discussion.md "Bug B"): if each y-rank
computes a different r-slab inside the body, the subsequent
`all_gather` over `('x','y')` on the band axis stacks contributions
that come from **different physical r-positions**.  The post-gather
tensor mixes "band 0's r-slab from y=0" with "band 1's r-slab from
y=1", which the downstream einsum then reads as a single coherent
`(band, r)` tile — silently producing wrong numerics.  The L/R
masks cannot recover (they don't know which gathered-band rows came
from which y-rank's r-slab).

**Corrected sequence**:

```python
# Inside the scan body, per iter:
# (1) Each rank IFFTs its 1/P band-slab over the FULL r-chunk:
psi_Y_bc_local_full_r = to_rchunk_inner(
    psi_G_bc_local, g_index_dev, fft_grid_t,
    r_start_, n_zchunk,                       # ← full n_zchunk, not r_loc
    kvecs_frac=kvecs_frac_dev, norm="ortho")
# Per-rank shape: c128[nk, bpd_max_local, ns, n_zchunk]

# (2) all_gather over ('x','y') on the band axis (§2.9):
psi_Y_bc_full_r = jax.lax.all_gather(
    psi_Y_bc_local_full_r, axis_name=('x','y'), axis=1, tiled=True)
# Per-rank shape: c128[nk, P·bpd_max_local, ns, n_zchunk]

# (3) THEN slice the r-axis to this y-rank's r_loc slab.
#     Coherent: every gathered band now has its full r-vector;
#     all y-ranks select the SAME r-offset interval.
r0_y_offset = y_idx * jnp.int32(r_loc)
psi_Y_bc = jax.lax.dynamic_slice_in_dim(
    psi_Y_bc_full_r, r0_y_offset, r_loc, axis=3)
# Per-rank shape: c128[nk, bpd_max_global, ns, r_loc]
```

**Memory implication for §4**: the per-iter slab inside the body is
larger than the §2.3 sketch implied:
- Per-rank pre-gather slab: `c128[nk, bpd_max_local, ns, n_zchunk]`
  ≈ `36·1·2·73648·16 = 85 MB` per rank at CrI3 6×6 80 Ry (not the
  `r_loc = 18412` extent the sketch implied).
- Per-rank post-gather slab: `c128[nk, P·bpd_max_local, ns, n_zchunk]`
  ≈ `36·16·2·73648·16 = 1.36 GiB` per rank.  This is the maximum
  band-r slab that lives one scan-iter, scan-aliased to a single
  slot.
- Per-rank post-r-slice slab (fed to the einsum):
  `c128[nk, bpd_max_global, ns, r_loc]` ≈ `36·16·2·18412·16 ≈
  340 MB`.  Per-iter, scan-aliased.

**The carry itself is unchanged** at `c128[nk, ns, r_loc, mu_loc,
ns]` = 3.71 GiB per accumulator (§2.3 main).  The reordering only
moves transient cost — total preallocated-temp at peak grows from
the §4.1 prediction by ~1.0 GiB (full-r slab + post-gather slab
land in the per-iter scan-aliased pool).  Updated §4 (§4.1c phase
walk) now matches Agent 2's implementation; see §9.1.

**Why this isn't a fix to the IFFT-FIRST principle of §2.9**: §2.9
requires the IFFT to happen BEFORE the `all_gather` because
gathering bands first would force per-rank IFFTs on FULL bands
(80 GB FFT box, infeasible).  That ordering is preserved.  The
additional constraint is now: the IFFT must produce a result
*spanning the same r-range across all ranks* (so the per-rank
slabs are addressable by a single per-rank `r_loc` slice after the
gather).  Simplest way is full-r; an r-loc-aligned-per-rank
implementation would require an explicit `all_gather` over `'y'`
on the r-axis too, which is strictly more comm.

### 2.4.  Carry init / donation / output (Agent 1 §2)

- **Init**: `jnp.zeros(...)` inside the shard_map body, on each
  rank.  No NamedSharding — the array is rank-local.
- **Donation**: `lax.scan` does not take `donate_argnums`.  XLA's
  buffer assignment aliases the in/out carry automatically — Agent 1
  confirmed against Round 4's HLO for `accumulate_rchunk_to_gflat`'s
  `acc_flat` (single slot, not two).  No explicit step needed.
- **Output**: `(P_l, P_r), _ = lax.scan(body, (P_l_acc, P_r_acc),
  jnp.arange(n_bc))`.  Discard per-iter `None`.

### 2.5.  L/R per-bc band slicing — **mask approach** (Agent 1 §4)

```python
band_idx = jnp.arange(bpd_max)
l_lo, l_hi = l_lo_tbl[bc_idx], l_hi_tbl[bc_idx]
r_lo, r_hi = r_lo_tbl[bc_idx], r_hi_tbl[bc_idx]
l_mask = (band_idx >= l_lo) & (band_idx < l_hi)
r_mask = (band_idx >= r_lo) & (band_idx < r_hi)
psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
```

- ✅ SPMD-safe (pointwise where, no resharding).
- ✅ No cross-rank op (band axis local to rank after io_callback
  returns the padded tile).
- 🟡 ~50% wasted einsum FLOPs per scan iter (acceptable for
  structural fix; revisit only if profiling demands).

Slice + padded-uniform-slice approaches rejected (require static
length at trace time; can't generally be picked since L/R window
lengths vary per bc).

### 2.6.  `psi_l_X` / `psi_r_X` per-bc slicing (Agent 1 §5)

Band axis is **replicated** under `L_spec_X = P(None, 'x', None,
None)`.  Per-bc slice is purely local:

```python
psi_l_X_bc = lax.dynamic_slice_in_dim(
    psi_l_X_, b_lo_tbl[bc_idx], bpd_max, axis=2)
psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
```

**Easy-to-get-wrong detail**: `psi_l_X.shape[2] == band_range_left[1]
- band_range_left[0]`, **NOT** `nb_total`.  `psi_l_X` is already
the L-window slice of the full band axis (built upstream by
`fit_one_rchunk` caller).  Build `b_lo_tbl` from `band_chunk_ranges`
intersected with `band_range_left`, not raw `band_chunk_ranges`.

#### 2.6.1.  XLA `dynamic_slice_in_dim` clamp footgun — REQUIRES symmetric front+back pad on `psi_l_X` / `psi_r_X` (Round 7 BLOCKER fix, commit `c796420`)

**This was the load-bearing miss in the Round-5 plan.**  Agent 4's
Round-6 bit-identity grid (`tests/test_zq_from_psi_sm_bit_identity.py`,
sub-gates G1.1b + G1.1c) caught it as a BLOCKER on top of Agent 2's
`f567aa0` commit; Agent 2's `c796420` ships the fix.  Plan
amendment so future implementers don't relearn it the hard way:

**The hazard**.  `jax.lax.dynamic_slice_in_dim(x, start, size, axis)`
**silently clamps** an out-of-bounds `start` to `max(0, axis_size -
size)`.  This is XLA's documented behavior — it never raises and
never produces NaN; it returns *the wrong physical bands* and lets
downstream code (here: the L/R `jnp.where` mask) carry on as if
nothing happened.  Reproducer:

```
len-5 array, start=4, slice_len=4 → [1.,2.,3.,4.]   (start clamps 4→1)
len-9 array, start=8, slice_len=4 → [5.,6.,7.,8.]   (start clamps 8→5)
```

**Where this fires in the kernel**.  When the per-bc range
`[bc.lo, bc.hi)` *intersects* the L window `[L_lo_g, L_hi_g)` but
the static-length slice `(bc.lo - L_lo_g, bpd_max_global)` extends
past the end of `psi_l_X` (`shape[2] = nb_l = L_hi_g - L_lo_g`):
- The clamp returns the LAST `bpd_max_global` bands of `psi_l_X`.
- The L mask was built from the *logical* index range `[bc.lo,
  bc.lo + bpd_max_global)` and zeros out logical positions that
  fall outside `[L_lo_g, L_hi_g)`.
- Result: the einsum multiplies the *physically wrong*
  (clamp-returned) bands by the mask's logical retain-zero pattern.
  The wrong bands are not zeroed; they contribute spurious values.
- The L mask cannot recover because it indexes by logical (global
  band) position, but the clamp shuffles which physical row sits
  at that position.

**When does it fire**:
- **Charge channel** with `L = R = (0, nb_total)` and `nb_total %
  bc_size == 0`: NEVER.  Slice ends always land at `bc.lo +
  bpd_max_global ≤ L_hi_g = nb_total`.  This is the **only** path
  the Round-6 MoS2 3×3 G1 e2e test exercised (max rel ≈ 1.02e-10).
- **Bispinor transverse**: `L = val window`, `R = cond window`.
  Bcs that span the L-or-R boundary trigger the clamp on at least
  one side.  **Always fires** in production.
- **Short final bc** (`nb_total % bc_size != 0`): the last bc's
  band range goes from `(N - bc_size, N)` where physically only
  the last `nb_total - (N - bc_size)` bands exist; the slice
  `(bc.lo - L_lo_g, bpd_max_global)` extends past the end and
  clamps.  **Always fires** in production for non-divisible
  configs.
- **Asymmetric L/R** (`nb_L != nb_R`, or shifted windows): any bc
  whose end-offset exceeds `nb_L` (for L-side) or `nb_R` (for
  R-side) clamps.  **Always fires** when the bc grid doesn't
  perfectly align to the smaller window.

**The fix (symmetric front+back pad)**.  Extend the existing
front-pad (which handled the negative-offset case: `bc.lo <
L_lo_g`) to a symmetric front+back pad sized so the dynamic_slice
never goes out of bounds.  Front-pad is already in the §2.3 sketch;
the back-pad is new in the Round-7 amendment:

```python
# Pre-shard_map (closure-time, static):
front_pad_l = max(
    (max(0, L_lo_g - lo) for (lo, _hi) in bcr), default=0)
front_pad_r = max(
    (max(0, R_lo_g - lo) for (lo, _hi) in bcr), default=0)
psi_l_X_bc_offset = np.asarray(
    [lo - L_lo_g + front_pad_l for (lo, _hi) in bcr], dtype=np.int32)
psi_r_X_bc_offset = np.asarray(
    [lo - R_lo_g + front_pad_r for (lo, _hi) in bcr], dtype=np.int32)
# Back-pad: largest end-offset across all bcs that exceeds front-padded length.
_max_end_l = max(off + bpd_max_global for off in psi_l_X_bc_offset)
_max_end_r = max(off + bpd_max_global for off in psi_r_X_bc_offset)
back_pad_l = max(0, _max_end_l - (front_pad_l + nb_l))
back_pad_r = max(0, _max_end_r - (front_pad_r + nb_r))

# Inside shard_map body:
psi_l_X_padded = jnp.pad(
    psi_l_X_, ((0, 0), (0, 0), (front_pad_l, back_pad_l), (0, 0)))
psi_r_X_padded = jnp.pad(
    psi_r_X_, ((0, 0), (0, 0), (front_pad_r, back_pad_r), (0, 0)))
# Per-bc slice now safe — every offset+size fits in-bounds.
```

**Why pad bands are math-neutral**.  Both front-pad and back-pad
bands hold zero.  The L/R mask zeroes out any band-position that's
not in `[L_lo_g, L_hi_g)` (resp. R), so:
- Pad positions outside the logical window: `psi_l_X_bc[pad] = 0`
  AND `l_mask[pad] = False`.  The `jnp.where` returns 0; the
  einsum sees a zero × anything = 0 contribution.
- IEEE arithmetic: `0 + x = x` exactly.  No rounding cascade.
**Math-neutral** — only adds `front_pad_l + back_pad_l` wasted
band rows per einsum iter.  In production typically <bpd_max_global
extra rows on each side.

**Same contract on the Y side**.  The per-bc `psi_l_Y_bc` /
`psi_r_Y_bc` come from the post-gather, post-r-slice
`psi_Y_bc` (§2.3.1).  The gathered band axis has exactly
`P · bpd_max_local = bpd_max_global` positions; these align 1:1
with the global band positions `[bc.lo, bc.lo + bpd_max_global)`.
The L/R mask zeroes pad-rows on this side too (`bc_valid =
g_axis < b_hi_global[bc_idx]` handles short final bc).  No
dynamic_slice on the Y side, so no clamp footgun there.

**Plan-record line for future readers**: any `dynamic_slice_in_dim`
on a band-axis-indexed-by-traced-`bc_idx` inside a pair-density
kernel MUST be guarded by symmetric padding on both ends, OR a
proof that the bc grid is strictly inside the windowed band axis.
Treat the absence of either as a BLOCKER bug.

### 2.9.  Band-shard mismatch: per-iter `all_gather` (Agent 1 v2 — load-bearing addition)

The single biggest gap in the original `agent_2_structural_fix.md`
§4c sketch: it implicitly assumed bands replicated everywhere on
host, but **they aren't**. This is the single most-important
structural piece of §2; missing it would make the kernel produce
wrong numerics.

**The mismatch**:
- Host tile (`PsiGStore._host_tiles`): `P(None, ('x','y'), None, None)` — **bands flat-sharded** over the full mesh. Each rank holds `bpd_per_bc = bc_size / P` bands of any given bc. CrI3 80 Ry P=16, bc_size=16 → **1 band per rank per bc**.
- `psi_l_X` in_spec `P(None, 'x', None, None)`: **bands REPLICATED** (every rank has the full L-window band axis).
- Einsum `'kmna, knbr → karmb'` contracts over the band axis `n`. Each rank needs all bands of the bc-windowed slice (or at least, all bands present on the L/R side); with the band-flat-sharded ψ_Y_bc on a rank, the einsum only sums over 1/P of the bands → **partial result, wrong numerics**.

**Resolution**: explicit `jax.lax.all_gather` inside the scan body, after IFFT, before the einsum:

```python
def body_l(carry, bc_idx):
    P_l_acc = carry
    # 1. Pull this rank's 1/P bands of this bc (band-flat-sharded).
    psi_G_bc_local = io_callback(
        psi_G_store._slice_local_tile_bc,
        jax.ShapeDtypeStruct((nk, bpd_max_local, ns, ngkmax), c128),
        x_idx, y_idx, bc_idx, ordered=True)
    # 2. Local IFFT + per-rank r-slab (per §2.3 r_loc trick).
    r0_local = r_start_ + y_idx * r_loc
    psi_Y_bc_local = to_rchunk_inner(
        psi_G_bc_local, g_index_c, fft_grid_t, r0_local, r_loc,
        kvecs_frac=kvecs_frac_c, norm="ortho")            # (nk, bpd_max_local, ns, r_loc)
    # 3. ★ All-gather across both mesh axes along band axis. ★
    psi_Y_bc = jax.lax.all_gather(
        psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)
                                                          # (nk, P·bpd_max_local, ns, r_loc)
    # 4. L mask + psi_l_X bc-slice + einsum into carry (see §2.5, §2.6).
    ...
```

**Comm cost per iter (CrI3 charge)**:
- Outgoing: `36·1·2·18412·16 = 21 MB / rank / iter`.
- Gathered tensor per rank after collective: `36·16·2·18412·16 = 339 MB`.
- Total per kernel (single interleaved scan): 10 bcs × 21 MB out = **210 MB out per rank** per fit_one_rchunk call. The gathered 339 MB lives only one iter at a time (scan-aliased).

**Order matters** (Agent 3 v2 catch + Agent 2 Round-6 correction): the sequence MUST be `io_callback → IFFT (full r-chunk) → all_gather → r-slice → einsum`.  All-gather-first remains rejected (would force per-rank IFFT on FULL bands → 80 GB / rank).  But the original §2.3 sketch's "IFFT to per-rank `r_loc` slab THEN all_gather" is also rejected — see §2.3.1 for the corrected derivation.

**Sequence finalised after Round-6 (corrects original §2.9 v2 ordering)**:

```python
psi_G_bc_local         = io_callback(...)                  # 1/P bands, full G-sphere
psi_Y_bc_local_full_r  = to_rchunk_inner(... r_len=n_zchunk)  # local IFFT, FULL r-chunk per rank
psi_Y_bc_full_r        = all_gather(axis=band, axis_name=('x','y'), tiled=True)  # bands stacked
psi_Y_bc               = dynamic_slice_in_dim(axis=r, r0_y_offset, r_loc)        # NOW slice r
# L/R masks + einsum into carry
```

The IFFT-on-full-r is the only ordering that keeps **(a)** the FFT
box per-rank-small via `bpd_per_bc=1` and **(b)** the gathered band
axis r-coherent.  Per-iter transient cost (per rank) grows from
the §2.9 v2 prediction of `c128[nk, bpd_per_bc, ns, r_loc] ≈ 5 MB`
to **`c128[nk, P·bpd_per_bc, ns, n_zchunk] ≈ 1.36 GiB`** post-gather
(scan-aliased, one slot).  Updated memory accounting in §9.1.

**Why this isn't the SPMD trap returning**: `jax.lax.all_gather(axis_name=...)` inside `shard_map` is the documented manual-mode collective, exact mechanism used throughout the repo (`cholesky_2d.py:167-194`, `load_wfns.py:279`). The carry remains rank-local; the collective operates on a per-iter value, not the carry. There is no global sharded carry for the SPMD partitioner to inflate — that's still true.

**Alternative considered (and rejected)**: `jax.lax.psum(P_l_partial, axis_name=('x','y'))` after the einsum, reducing each rank's partial-band sum to a full sum. Cost: one 15 GiB reduce per scan instead of n_bc × 21 MB gathers. Wall-clock: ~5× worse, plus the reduced carry becomes effectively replicated (every rank holds the same 15 GiB) which then needs to be sharded back to `out_spec=P(None,'x','y')` — extra reshard remat risk. **Reject.**

🟡 **Open question on `tiled=True` axis ordering**: `all_gather(tiled=True, axis_name=('x','y'))` stacks contributions in **mesh-axis-major order** — but the host-tile bc-stacking convention is `_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)` which flattens `('x','y')` into a single P-way axis using NumPy C-order (x-major). Whether the gathered output band axis matches the global band ordering of `band_chunk_ranges[bc_idx]` is a verification step in the reproducer. If they don't match, we add a per-rank permutation (cheap, no comm) inside the body.

### 2.10.  Single interleaved scan with both accumulators (Agent 1 v3 — recommended)

**Iteration log**: my v2 recommended serializing L and R into two separate scans to halve the 14.85 GiB carry to 7.42 GiB live at once. With the corrected 3.71 GiB per-side carry (v3), serialization's savings shrink to <4 GiB during the scan phase, and Agent 3 v2 pointed out that the γ̃ contract at end-of-body needs both `P_l_R_conj` and `P_r_R` simultaneously anyway — peak ≈ 7.42 GiB regardless of serialization. **Drop the serialization recommendation; single interleaved scan is now the cleaner choice.**

```python
def _local(psi_l_X_, psi_r_X_, perm_L, phase_L, perm_R, phase_R, r_start_):
    x_idx = jax.lax.axis_index('x'); y_idx = jax.lax.axis_index('y')

    P_l_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
    P_r_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)

    def body(carry, bc_idx):
        P_l_acc, P_r_acc = carry
        # 1. io_callback: this rank's 1/P bands of bc i (band-flat-sharded).
        psi_G_bc_local = io_callback(...)
        # 2. IFFT + per-rank r-slab.
        r0_local = r_start_ + y_idx * r_loc
        psi_Y_bc_local = to_rchunk_inner(
            psi_G_bc_local, g_index_c, fft_grid_t, r0_local, r_loc,
            kvecs_frac=kvecs_frac_c, norm="ortho")
        # 3. all_gather across ('x','y') on bands.
        psi_Y_bc = jax.lax.all_gather(
            psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)
        # 4. Compute BOTH L and R masks + slices + einsums in one iter.
        global_band_axis = b_lo_global[bc_idx] + jnp.arange(bpd_max_global, dtype=jnp.int32)
        bc_valid = global_band_axis < b_hi_global[bc_idx]
        l_mask = (global_band_axis >= L_lo_g) & (global_band_axis < L_hi_g) & bc_valid
        r_mask = (global_band_axis >= R_lo_g) & (global_band_axis < R_hi_g) & bc_valid
        psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
        psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
        psi_l_X_bc = lax.dynamic_slice_in_dim(psi_l_X_, b_lo_global[bc_idx] - L_lo_g, bpd_max_global, axis=2)
        psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
        psi_r_X_bc = lax.dynamic_slice_in_dim(psi_r_X_, b_lo_global[bc_idx] - R_lo_g, bpd_max_global, axis=2)
        psi_r_X_bc = jnp.where(r_mask[None, None, :, None], psi_r_X_bc, 0)
        delta_P_l = jnp.einsum('kmna,knbr->karmb', psi_l_X_bc, psi_l_Y_bc, optimize=True)
        delta_P_r = jnp.einsum('kmna,knbr->karmb', psi_r_X_bc, psi_r_Y_bc, optimize=True)
        return (P_l_acc + delta_P_l, P_r_acc + delta_P_r), None

    (P_l, P_r), _ = lax.scan(body, (P_l_init, P_r_init), jnp.arange(n_bc, dtype=jnp.int32))

    # Post-pair pipeline (unchanged from today's z_q_from_psi_sm._local).
    P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns); del P_l
    P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0,1,2), norm='forward'); del P_l_3d
    P_l_R_conj = jnp.conj(P_l_R); del P_l_R
    P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns); del P_r
    P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0,1,2), norm='forward'); del P_r_3d
    Z_R = gamma_double_contract(P_l_R_conj, P_r_R, ...); del P_l_R_conj, P_r_R
    Z_q_3d = jnp.fft.fftn(Z_R, axes=(0,1,2), norm='forward')
    return jnp.transpose(Z_q_3d.reshape(nkx*nky*nkz, r_loc, mu_loc), (0,2,1))
```

**Live memory profile**:
- During scan: 2 × 3.71 GiB carries + FFT box transient ~5 GiB + per-iter buffers ~400 MB ≈ **12.4 GiB**.
- At γ̃ contract: P_l_R_conj (3.7 GiB) + P_r_R (3.7 GiB) ≈ **7.4 GiB** (carries already freed).
- Output 1 GiB + params 0.5 GiB.
- **Peak ≈ 13-15 GiB**, vs today's 48.63 GiB → **~3× memory reduction**.

Original v2 serialized design code is retained at git history if we want to revisit at larger n_rmu where the carry size would re-grow.

#### Superseded v2 serialized sketch (kept for archeology)

(Same shape as the interleaved sketch above, but with two separate `lax.scan` calls. Drop in favor of v3 above.)

```python
def _local(psi_l_X_, psi_r_X_, perm_L, phase_L, perm_R, phase_R, r_start_):
    x_idx = jax.lax.axis_index('x'); y_idx = jax.lax.axis_index('y')

    # ─── Scan 1: build P_l_acc, fold into IFFT, free. ───
    P_l_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
    P_l, _ = lax.scan(body_l, P_l_init, jnp.arange(n_bc, dtype=jnp.int32))
    P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns); del P_l
    P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0,1,2), norm='forward'); del P_l_3d
    P_l_R_conj = jnp.conj(P_l_R); del P_l_R

    # ─── Scan 2: build P_r_acc, fold into IFFT, free. ───
    P_r_init = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
    P_r, _ = lax.scan(body_r, P_r_init, jnp.arange(n_bc, dtype=jnp.int32))
    P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns); del P_r
    P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0,1,2), norm='forward'); del P_r_3d

    # ─── Contract and FFT (unchanged from today's _local). ───
    Z_R = gamma_double_contract(P_l_R_conj, P_r_R, ...); del P_l_R_conj, P_r_R
    Z_q_3d = jnp.fft.fftn(Z_R, axes=(0,1,2), norm='forward')
    return jnp.transpose(Z_q_3d.reshape(nkx*nky*nkz, r_loc, mu_loc), (0,2,1))
```

(End of superseded v2 sketch. The v3 interleaved-single-scan design above replaces this — at the corrected 3.71 GiB per-side carry, serialization is no longer worth the 2× host transfers.)

### 2.7.  Post-pair pipeline tail unchanged (Agent 1 §6)

The `reshape → ifftn(k-axes) → conj → ifftn → gamma_double_contract
→ fftn → transpose+reshape` tail is **byte-identical** to today's
`z_q_from_psi_sm._local` (`isdf_fitting.py:429-444`).  Only the
*front* of the body changes (the new scan replaces today's
direct-einsum-on-pre-built-psi_Y).  This preserves charge / bispinor
γ̃-fold logic unchanged.

### 2.8.  SPMD safety checklist (Agent 1 §7)

| Concern | Status | Notes |
|---|---|---|
| Scan carry SPMD inflation | ✅ Safe | shard_map body is per-rank manual |
| accumulate_rchunk_to_gflat precedent | ✅ Working | Round 4 HLO confirms aliasing |
| Collectives inside body | ✅ None proposed | `axis_index` invariant; no `psum` |
| `with_sharding_constraint` leakage | ✅ Audited clean | `apply_bloch_phase_on_slice` pure-local |
| `check_rep=False` consistent | ✅ Yes | matches existing pair pipeline |
| Replicated closure constants | ✅ Yes | band tables / g_index / kvecs |
| io_callback in scan in shard_map | ⚠️ Novel | reproducer-first (§7 below) |
| Per-rank `r_loc` = n_zchunk / p_y | ⚠️ Correctness gotcha | §2.3 |
| Per-bc band slice (mask) | ✅ SPMD-safe | pointwise where, local band axis |
| Per-bc `psi_l_X` slice | ✅ SPMD-safe | band axis replicated |
| Post-pair tail | ✅ Unchanged | byte-identical to today's _local |
| `lax.scan` donation | ✅ Implicit aliasing | XLA buffer-assigns |
| Carry size at CrI3 80 Ry (Agent 1 v3 / Agent 3 v2) | ✅ 7.42 GiB (both live, interleaved single scan) | μ_loc=94 verified from HLO param `c128[36,376,376]` — see §2.3 + §2.10 |
| Band-shard mismatch all_gather | ⚠️ Required per iter, ~21 MB out / 339 MB gathered | see §2.9 — manual-mode collective, not SPMD inflation |
| All_gather order (IFFT-first vs gather-first) | ✅ IFFT-first | gather-first → 80 GB FFT box (Agent 3 v2 catch); see §2.9 |

Net: SPMD-safe modulo two ⚠️ flags — io_callback-novelty (mitigated
by reproducer per §7) and per-rank `r_loc` shape (mechanical;
audit-during-PR).

## 3.  io_callback inside scan  ⟨Agent 2 — full content; cross-references `round5_discussion.md` §"Agent 2 → others"⟩

### 3.1.  Composition feasibility — `io_callback` × `lax.scan` × `shard_map` × `lax.all_gather`

Round-5 escalated the design from three nested primitives (io_callback inside scan inside shard_map) to **four** (the per-iter `lax.all_gather` from §2.9 is required to align the band-flat-sharded host tile with the band-replicated `psi_l_X` / `psi_r_X`).

JAX supports each primitive independently and documents them as composable. `all_gather` inside `shard_map` is a *manual collective* (axis_name comes from the shard_map mesh), which is the intended composition pattern; SPMD partitioner doesn't enter the shard_map body.

**In-tree prior art for each pair:**

| Pair | In-tree example |
|---|---|
| `io_callback` × `lax.scan` × `lax.cond` (`ordered=True`) | `src/common/progress.py:scan_progress` line 149 — library helper; pattern works |
| `shard_map` × `io_callback` | `psi_G_store.py:_pull_full` and per-bc helpers (`5cadd4b`) — production today |
| `shard_map` × `lax.scan` | `wfn_transforms.accumulate_rchunk_to_gflat`, `wfn_transforms.gflat_to_rchunk` — production today |
| `shard_map` × `lax.all_gather` | `solve_zeta` (manual all_gather), `_reshard_zeta_r_XY_to_mu_XY` — production today |

The four-way nesting is novel within LORRAX. Confidence ~85 % that JAX supports it cleanly. **Smoke test (§3.6) gates Round-6 implementation.**

### 3.2.  Restored `_slice_local_tile_bc` signature

Restore the slicer (committed `cdd0fba`, deleted `5cadd4b`):

```python
def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx) -> np.ndarray:
    """Per-rank host-tile slice for one bc, padded to (nk, _bpd_max, ns, ngkmax).

    bc_idx is a TRACED int32 scalar (resolved to Python int inside the host fn).
    The full bc spans all ranks via _PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None);
    rank r holds bands [r·bpd_per_bc, (r+1)·bpd_per_bc) of the bc.  Short final
    bc is zero-filled to bpd_max so the scan body sees a static return shape
    every iter.

    Pad rows hold zeros; consumer's per-iter all_gather + L/R band-mask makes
    them mathematically inert.
    """
    x, y, bc = int(x_idx), int(y_idx), int(bc_idx)
    if not 0 <= bc < len(self.band_chunk_ranges):
        raise ValueError(
            f"_slice_local_tile_bc: bc_idx={bc} not in [0, {len(self.band_chunk_ranges)})")
    tile = self._host_tiles[(x, y)]
    b_lo = self._bc_band_offsets[bc]
    b_hi = self._bc_band_offsets[bc + 1]
    nk, _, ns, ngkmax = tile.shape
    out = np.zeros((nk, self._bpd_max, ns, ngkmax), dtype=tile.dtype)
    out[:, : b_hi - b_lo, :, :] = tile[:, b_lo:b_hi, :, :]
    return out
```

### 3.3.  `bpd_max` closure timing — load-bearing

`io_callback` requires `out_sds` (`jax.ShapeDtypeStruct`) **static at trace time** — JAX uses it to size the device-side buffer; `lax.scan` requires uniform body output shape across iters. Therefore `_bpd_max` MUST be computed at `PsiGStore.__init__` and closed over both:
- `out_sds = jax.ShapeDtypeStruct((nk, _bpd_max, ns, ngkmax), c128)` — io_callback static shape.
- `_slice_local_tile_bc` body — pads return to `(nk, _bpd_max, ns, ngkmax)` regardless of which bc the traced index resolves to.

Pad rows hold zeros. After the per-iter `all_gather` (§2.9), the L/R band-mask zeros out pad rows; they contribute nothing.

### 3.4.  `ordered=True` vs `ordered=False`

| Property | `ordered=True` | `ordered=False` |
|---|---|---|
| Per-rank ordering (within one rank's traced program) | enforced (HLO appearance order) | not enforced |
| Cross-rank ordering | not enforced (each rank has its own host queue) | not enforced |
| XLA can pipeline next-iter dispatch with current-iter compute | NO | YES |
| Required for correctness in this design | NO (each iter's bc_idx is independent; no cross-iter side channel) | NO |

**Recommendation: `ordered=False`** — no correctness need for ordering (each iter consumes its own bc_idx → its own slab); allows XLA scheduler to overlap H2D dispatch with device compute. `lax.scan(unroll=1)` already provides sequential body execution at runtime regardless of `ordered=`.

### 3.5.  `unroll` constraint

**`lax.scan(..., unroll=1)`** — pin explicitly, do not rely on default. Unrolling means k iterations live concurrently → k FFT boxes resident, k io_callback payloads in flight. Defeats the entire structural fix (back to the round-3 pile-up).

Add a comment at the scan call: `"DO NOT unroll — the FFT-box and psi_G_bc aliasing depends on per-iter sequential lifetime"`.

### 3.6.  Smoke test (Round-6 prerequisite)

50 LOC, ~5 min CPU + 5 min 4-rank GPU. Must run BEFORE the kernel rewrite touches `c_q_from_psi_sm` / `z_q_from_psi_sm`.

```python
# tests/test_io_callback_scan_in_shard_map.py
def test_io_callback_scan_shard_map_all_gather_smoke():
    """All four primitives nested.  Verify on 1×1 mesh first (smoke), then
    2×2 GPU mesh (correctness vs global einsum reference)."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), axis_names=('x', 'y'))
    n_bc, ngkmax, bpd_max, ns, nk = 3, 7, 4, 2, 5
    rng = np.random.default_rng(0)
    host_tile = (rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))
                 + 1j*rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax)))

    def host_slice(x, y, bc):
        return host_tile[int(bc)]                   # (nk, bpd_max, ns, ngkmax)
    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    @partial(shard_map, mesh=mesh,
             in_specs=(), out_specs=P(None, None), check_rep=False)
    def _local():
        x = jax.lax.axis_index('x'); y = jax.lax.axis_index('y')
        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x, y, bc_idx, ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'), axis=1, tiled=True)
            return carry + slab_full.sum(), None
        carry, _ = lax.scan(body, jnp.complex128(0), jnp.arange(n_bc))
        return jnp.broadcast_to(carry, (1, 1))

    out = np.asarray(_local())
    np.testing.assert_allclose(out[0, 0], host_tile.sum())
```

Pass criteria:
- (a) callback fires `n_bc` times per rank (verify with a counter side-channel).
- (b) each call gets the right traced `bc_idx` (verify by mapping bc → known sentinel value).
- (c) all_gather post-callback returns full-band-axis tensor on each rank.
- (d) scan output equals direct numpy reference to floating-point precision.
- (e) `ordered=True` and `ordered=False` give the same result (correctness invariant).

If (a)–(e) all pass, Path B is unblocked. If (c) fails on the 4-rank GPU run (i.e., `all_gather` inside scan composition is broken in JAX), fall back to Path A (§3.11).

### 3.7.  Latency / throughput accounting (CrI3 6×6 80 Ry, 4×4 mesh)

**io_callback** per-iter payload (per rank):
- `nk · bpd_max · ns · ngkmax · 16` ≈ 36 · 16 · 2 · 70k · 16 ≈ 1.3 GB / call. (Padded; actual data is `bpd_per_bc · ngkmax · 16` ≈ 80 MB — pad inflates 16×.)
- 320 calls/channel × 1.3 GB ≈ 416 GB H2D per channel per rank. At ~25 GB/s effective bandwidth: ~16 s / channel.
- × 4 channels = ~64 s total H2D.
- Python overhead per call: 1–5 ms × 1280 ≈ 2–6 s.

**all_gather** per-iter (the new collective from §2.9):
- Pre-gather: `(nk, bpd_per_bc, ns, r_loc)` ≈ 36 · 1 · 2 · 18412 · 16 ≈ 21 MB / rank / iter.
- Post-gather: ~340 MB / rank.
- Each rank receives `(P-1) · 21 MB ≈ 315 MB` over the mesh. At NVLink ~100 GB/s effective: ~3 ms / iter / rank.
- 320 iters × 3 ms ≈ 1 s / channel. Trivial.

**Vs Path A**: identical io_callback call count and byte total; only difference is XLA scheduler (Path B) vs Python loop (Path A). No regression.

**Open** for Round-6 measurement: profile whether XLA actually pipelines io_callback against compute. If yes, Path B is faster; if XLA serializes io_callback dispatches, Path B matches Path A.

### 3.8.  Optimization opportunity (Round-6.5 follow-up, not v1)

The io_callback could return only the rank's actual data `bpd_per_bc · ngkmax · 16` ≈ 80 MB instead of the padded `bpd_max · ngkmax · 16` ≈ 1.3 GB. Cuts H2D 16×. Pad happens device-side via `jnp.pad`. Easy to skip on first pass since the CrI3 ratio is already small. **Flag as Round-6.5 follow-up.**

### 3.9.  `PsiGStore` state changes

| Field | Lives | Path-B disposition |
|---|---|---|
| `_host_tiles[(x,y)]` | per-rank numpy, bc-stacked | **keep** — read by io_callback every iter |
| `_bpd_max` | int, set at `__init__` (was in `cdd0fba`, removed in `5cadd4b`) | **restore** |
| `_bc_band_offsets` | tuple, set at `__init__` | unchanged |
| `_g_index_dev` / `_kvecs_frac_dev` | replicated jax.Array, set once via `jax.device_put` | **keep**, captured via closure in `_kernel`. Concrete arrays — no tracer hazard |
| `_psi_G_device_full` | lazy jax.Array cache | **DELETE** — Path B never materializes the full tile on device |
| `psi_G_device_full` property | computes via per-bc concat | **DELETE** |
| `g_index` / `kvecs_frac` properties | thin accessors | **keep** for external callers |

### 3.10.  `begin_rchunk` / `end_rchunk` / lifetime audit

| Field | Set | Cleared | Path-B status |
|---|---|---|---|
| `_host_tiles` | `_populate_from_loader` | `_clear_tiles` (RereadPsiGStore.end_rchunk only) | **must be valid for kernel duration** — io_callback reads per scan iter. Reread mode's `end_rchunk` runs **after** `block_until_ready` (`isdf_fitting.py:2168-2172` `finally:`), so async io_callbacks finish before tiles are freed. ✓ Existing contract. |
| `_g_index_dev` / `_kvecs_frac_dev` | once | NEVER | concrete `jax.device_put` arrays. ✓ |
| `_psi_G_device_full` | property | `_clear_tiles` | **DELETE** in Path B. |

The async-callback / host-tile lifetime contract already exists in production (Reread mode); Path B inherits it. No new lifecycle invariants needed.

### 3.11.  Path A fallback (if §3.6 smoke test fails on (c) — all_gather inside scan)

If JAX rejects the `all_gather` inside scan inside shard_map nesting, the **preferred fallback** is **Agent 4's §7.2 design** (driver-level Python bc-loop with donated jit per bc) — donation chain gives sequential memory reuse à la `solve_zeta`'s pattern, better memory profile than the simpler "pre-pull + single dispatch" alternative.

A second-tier fallback (Path A-2, listed for completeness):
1. At driver level (in `fit_zeta_to_h5`'s chunk loop, BEFORE the kernel jit dispatch), pull `psi_G_full` once via the same per-bc + `jnp.concatenate` pattern as today's `psi_G_device_full` property.
2. Pass `psi_G_full` as a jit argument (not a closure-captured lazy property) — eliminates the tracer leak.
3. Inside `_kernel`, the band-axis is canonical-sharded (`jnp.concatenate` inserted the reshuffle). Slice L/R as today.
4. Keep the `gflat_to_rchunk` integration from `5cadd4b` unchanged.

Both Path A variants fix the leak (mission goal #2) but do NOT fix the concat reshuffle (#1) or the consumer-boundary remat (#3). They are strict subsets of Path B's wins. Acceptable as a stopgap for Round 6 if Path B blows up; reopen Path B in Round 7 with a documented JAX issue.

### 3.12.  Why this design eliminates the `psi_G_device_full` tracer leak

The tracer leak in the current code (`5cadd4b`) is structural: `psi_G_device_full` is a lazy `@property` that computes a `jax.Array` and caches it on `self`. When the property is accessed inside `_kernel`'s tracing, the result is a *traced* `jax.Array`. Cached on `self`. Subsequent accesses (after `_kernel` returns) see the cached tracer, which is invalid outside its tracing scope. → leak warning at compile.

Path B never caches a `jax.Array` on `self`. The io_callback fires *inside* the scan body, producing a fresh per-iter device array that's consumed immediately and freed by XLA's scan-internal allocator. No `self`-state involved → no leak.

## 4.  Expected HLO + memory profile  ⟨Agent 3 — condensed; full at `round5_agent_3_hlo.md`⟩

### 4.1.  Per-rank slot predictions — FINAL (A1↔A3 converged through R2)

**Two corrections from my Round-1 numbers above** (surfaced in the discussion):

1. **`n_rmu` in this kernel is 376, not 1504.** Verified from HLO param shapes (`c128[36, 376, 376]` = L_q). The 1504 in gw.out is an upstream stage.
2. **Host ψ(G) tiles are band-sharded on the FULL `('x','y')` mesh** (`_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)`), so `bpd_per_bc = 16 / p_xy = 1` band per rank per bc. My Round-1 fear of a 20 GB per-iter FFT box was wrong: the host-side sharding does the chunking for us.

Under today's sharding contract: `μ_loc = 376/p_x = 94`, `r_loc = 73648/p_y = 18412`, `bpd_per_bc = 1`.

| Item | Per-rank size | Lifetime | Aliasable? |
|---|---|---|---|
| Carry `P_l_acc` (interleaved single scan) | **3.71 GiB** | full scan body + until γ̃ contract | Carry slot |
| Carry `P_r_acc` (interleaved single scan) | **3.71 GiB** | full scan body + until γ̃ contract | Carry slot |
| `P_l_R_conj` (post-IFFT(k) of P_l) | 3.71 GiB | until γ̃ contract | One slot |
| `P_r_R` (post-IFFT(k) of P_r) | 3.71 GiB | during γ̃ contract | Aliased with γ̃ scratch where possible |
| FFT box inside scan body | **~5 GiB** (incl. cuFFT ×4) | one scan iter | Aliased across iters AND across both scans |
| `all_gather` band-slab (post-IFFT, pre-einsum) | ~340 MB | one scan iter | Aliased across iters |
| `psi_G_bc` (io_callback return) | ~80 MB | one scan iter | Aliased |
| Output Z_q | ~0.99 GiB | live-out | n/a |
| Parameters + constants | ~0.42 GiB | n/a | n/a |

**Substantive preallocated-temp slots: ~4.** At γ̃ contract peak: two 3.71 GiB P-pair-class buffers live simultaneously plus the FFT box.

**Predicted peak per-rank total: ~13–15 GiB.** Compared to today's 48.63 GiB — **~3× memory reduction.** Well below the 28 GiB per-A100; comfortable headroom under cohsex.in's 60 GiB setting.

### 4.1a.  Host-band-sharding complication + all_gather (Agent 1's catch)

`_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)` ⇒ 1 band per rank per bc at CrI3. Einsum needs all 16 bands per rank. **Fix**: per-iter `jax.lax.all_gather(..., axis_name=('x','y'), axis=1, tiled=True)` AFTER per-rank IFFT-and-slice:

```
io_callback → c128[36, 1, 2, 59990]    # 80 MB per rank
IFFT + slice → c128[36, 1, 2, 18412]    # 21 MB per rank (FFT box ~5 GiB transient with cuFFT)
all_gather(tiled, axis=band) → c128[36, 16, 2, 18412]   # 340 MB per rank
mask L/R + einsum into carry
```

**Order critical**: IFFT BEFORE all_gather. Reversed order puts 16 bands per rank pre-IFFT → 20 GB FFT box. Correct order keeps it at ~5 GiB.

The 340 MB all_gather slab is an intentional small replication — justified vs the alternative (3.71 GiB psum on carry every iter).

### 4.1b.  Single interleaved scan with both accumulators (A1 v3 + A3 concur)

Iteration log: my Round-1 estimate above with `n_rmu=1504` predicted
the carry at 14.85 GiB each, which made the serialized two-scan
design (Agent 1 v2) attractive — halving the live carry during
the scan halved live working set.  With the corrected 3.71 GiB
carry, serialization saves <4 GiB during scan, but the post-scan
γ̃ contract needs both `P_l_R_conj` and `P_r_R` simultaneously
either way — peak is at γ̃ contract, not during scan.  So the
serialization buys nothing while costing 2× host transfers from
the io_callback path.  **Single interleaved scan is the final
recommendation** (Agent 1 §2.10 in the unified plan).

Flow:

1. Single scan over bc → both `P_l_acc` and `P_r_acc` updated per
   iter (carry tuple).  FFT box and all_gather slab are
   scan-aliased; per-iter peak ~10–12 GiB.
2. Post-scan: `P_l → ifft(k) → P_l_R_conj`, `P_r → ifft(k) → P_r_R`
   (two 3.71 GiB buffers concurrently).
3. γ̃ double contract → `Z_R` (single 3.71 GiB buffer; P_l_R_conj
   + P_r_R freed).
4. FFT(k) on Z_R → Z_q.

### 4.1c.  Phase-by-phase memory walk

| Phase | Live buffers | Per-rank GiB |
|---|---|---|
| Scan body (per iter) | `P_l_acc` + `P_r_acc` + FFT-box (scan-aliased) + all_gather slab + psi_G_bc + params/output | **~12–14** |
| Post-scan IFFT(k) | `P_l_R_conj` + `P_r_R` + params/output + small Z scratch | **~9–10** |
| **γ̃ contract (peak vs scan: similar)** | `P_l_R_conj` + `P_r_R` + γ̃ scratch + output | ~9–10 |
| FFT(k) of Z_R → Z_q | Z_R + scratch + output | ~5 |

### 4.1d.  Round-1 numbers (RETRACTED — kept for traceability)

My initial Round-1 estimate of 14.85 GiB per accumulator (=29.70 GiB carry, ~38–48 GiB total preallocated) was based on `n_rmu = 1504` and `bpd_per_bc = 16` per rank. Both wrong. The corrected math gives 3.71 GiB per accumulator with 1 band/bc/rank, leading to the 13–15 GiB total above.

### 4.2.  Carry size — RESOLVED through Round 2

(See §4.1 above for the converged numbers.) Both Agent 1's "999 MB per carry" and my "14.85 GiB per carry" were wrong on different terms. The actual per-rank carry is **3.71 GiB each** under today's sharding contract, because:
- `n_rmu = 376` (kernel-scope, verified from HLO L_q param), so `μ_loc = 376/p_x = 94`.
- `r_loc = 73648/p_y = 18412`.
- Carry shape `c128[36, 2, 18412, 94, 2]` = 3.71 GiB per accumulator.

No upstream sharding change needed; today's `psi_l_X : P(None, 'x', None, None)` works as-is. Moving μ to `P(None, ('x','y'), None, None)` would shrink the carry 4× further to ~0.93 GiB each — out of scope this round (§8), but a clean future optimization.

### 4.3.  FFT-box-inside-scan-body — RESOLVED, no chunking needed

My Round-1 framing posed Option A (nested k-scan) vs Option B (flat-axis cs) to control an FFT box I feared would be 20 GB. After Agent 1's host-band-sharding catch (§4.1a), this is moot: with `bpd_per_bc = 1` band per rank per bc at CrI3, the per-iter FFT box is **`c128[36, 1, 2, 1.125M]`** = 1.3 GB raw, ~5 GiB with cuFFT scratch ×4. **Already small enough** without any chunking — the host-side band-sharding does the work.

The converged design: simple **outer-bc-scan** with per-rank 1-band slab per iter. No nested scans, no flat-axis (k·b) reshape. Each scan is one clean WhileOp lowering.

Indexing inside the body is straightforward: `bc_idx` is the scan iterate; closure tables `b_lo_tbl[bc_idx]`, `b_hi_tbl[bc_idx]` (bc-indexed, mask-friendly) provide L/R band masks per Agent 1 §4.

### 4.4.  `psi_Y_full` and remat boundary — confirmed gone

`psi_Y_full` never exists as a global array in the new design — only
per-iter `psi_Y_bc` lives inside the scan body, contracted directly
into `P_l_acc` / `P_r_acc`.  Today's 30 GiB double-materialisation
(Agent 1 R4 §3a + Round 4 status snapshot) **fully eliminated**.

The reshard boundary that caused the 32 `Involuntary full
rematerialization` warnings is gone too — no slice-then-reshard
across an `out_spec` boundary because the post-pair pipeline lives
in the same shard_map body and consumes the carry directly.  G2
gate's "zero remat warnings" criterion (§5.4) is a direct
verification.

### 4.5.  cuFFT batching at the per-iter scale

Per-iter FFT batch under the converged outer-bc-scan design:
`nk · bpd_per_bc · ns = 36 · 1 · 2 = 72` FFTs of size 75·75·200 per
call — comfortably above the cuFFT batched-efficiency floor (~32).
No further chunking needed (per §4.3); the host-side band sharding
already keeps the batch right-sized.

### 4.6.  Compile-time prediction

New kernel adds one `lax.scan` inside the shard_map body, lowering
to a single WhileOp (don't pass `unroll=`).  Compile time should
grow **<2×** vs today.  Profile on MoS2 3×3 (the existing test
scaffold) before CrI3.

### 4.7.  Planner accounting after this lands

Today's `_peak_C_fit_one_rchunk` (Round 4 `5cadd4b`) uses `pair_density_slots = 3`. After the converged single-interleaved-scan design lands, replace the term set with:

```python
# Both carries live across the scan (interleaved):
2 · _bytes_c128(nk, ns, n_zchunk, n_rmu, ns, shard=p_xy)
# γ̃ contract peak — P_l_R_conj + P_r_R simultaneously (same shapes as carry;
# XLA may alias with the carry slots when lifetimes align):
2 · _bytes_c128(nk, ns, n_zchunk, n_rmu, ns, shard=p_xy)
# Per-rank FFT box, one scan-aliased slot — NOT divided by p_xy because
# bpd_per_bc=1 already came from the host-side band-sharding:
_bytes_c128(nk, bpd_per_bc, ns, n_rtot) * fft_box_factor
# Optional small all_gather slab (sub-GiB, may be omitted):
_bytes_c128(nk, bpd_max, ns, n_zchunk, shard=p_y)
```

where `bpd_per_bc = ceil(band_chunk / p_xy)` (= 1 at CrI3 with band_chunk=16, p_xy=16).

Combined with my Round 4 T3 bookkeeping fixes (Peak C centroids `nk → nb_total`, Peak B centroids `ns → nb_total`, Peak D `fft_box_factor=4 → 1`), the planner's predicted Peak C drops from 51.93 GB → **~15 GiB, matching the new HLO within ~10%.**

### 4.8.  HLO acceptance gates (feed G2 in §5.4)

After implementation, HLO dump at CrI3 6×6 80 Ry must show:

1. **No `c128[..., 73648]` unsharded slabs.** psi_Y_full gone.
2. **No `c128[360, 2, 1125000]`-shape slots.** Old `gflat_to_rchunk`'s 12 GiB unsharded FFT box gone.
3. **Zero `Involuntary full rematerialization` warnings in gw.out.**
4. **`while_loop` count in HLO = 1** (one interleaved scan).
5. **`all-gather` op present inside the WhileOp body** (exactly one).
6. **Total preallocated-temp ≤ 20 GiB** (target 13–15 GiB; 5 GiB slack for cuFFT scratch and XLA allocator choices).
7. **Substantive slot count ≤ 5** (P_l_acc, P_r_acc, P_l_R_conj, P_r_R, FFT box — some aliased).

If any fails, design assumption is wrong; do NOT proceed to G3.

### 4.9.  Risk: all_gather inside scan inside shard_map (novel composition)

Three-level nesting that's novel for this repo. Each ingredient has in-repo precedent: `all_gather` inside `shard_map` at `cholesky_2d.py:167-194` and `load_wfns.py:279` (per Agent 1 v3 §2.9); `lax.scan` inside `shard_map` is verified at `accumulate_rchunk_to_gflat` (Round 4 HLO). The combination — `all_gather` inside a `lax.scan` body inside a `shard_map` — is what's new.

**Reproducer required before kernel rewrite.** A small CPU test verifying: (i) compile succeeds, (ii) runtime values correct, (iii) HLO shows one `all-gather` inside the WhileOp body (not lifted out of the scan, not duplicated per iter).

If the reproducer fails, fallback is `psum`-on-carry at scan exit. Agent 1 v3 §2.9 documents why that's strictly worse (~5× wall-clock, reshard-remat risk). Reproducer-first is non-negotiable; tracked in §7 risks.

## 5.  Numerics + edge cases + validation gates  ⟨Agent 4⟩

### 5.1.  Bit-identity contract — NOT bit-equal, ULP-bounded

The new kernel computes mathematically `Z_q = FFT( γ̃·(IFFT(P_l)·
conj·IFFT(P_r)) )`, where `P_l = Σ_n_global x_l y_l*` and the
existing code is bilinear in ψ.  Decomposing the band-axis sum
into per-bc partial sums and accumulating is **algebraically
identical** but **not FP-bit-equal**:

- Current (`isdf_fitting.py:1283-1298`):
  ```
  psi_Y_full = jnp.concatenate(per-bc, axis=1)   # band-axis concat
  P_l = einsum('kmna,knbr→karmb', psi_l_X, psi_l_Y_full)
  P_r = einsum('kmna,knbr→karmb', psi_r_X, psi_r_Y_full)
  ```
  XLA chooses the contraction order for the `n`-axis reduction
  inside one einsum (implementation-defined; typically a balanced
  tree).
- New:
  ```
  for bc in band_chunk_ranges:        # lax.scan-unrolled
      P_l_acc += einsum('kmna,knbr→karmb', psi_l_X_bc, psi_l_Y_bc)
      P_r_acc += einsum('kmna,knbr→karmb', psi_r_X_bc, psi_r_Y_bc)
  ```
  Per-bc einsum reduces over `n ∈ [b_lo_bc, b_hi_bc)`, then the
  outer `+` accumulates over bcs (linear-tree).

The two summation orders agree at the math level but differ by FP
rounding: at most `N_partial · ULP` relative drift, where
`N_partial ≈ n_bc + log2(bpd_max)` (linear-tree across bcs +
balanced-tree within).  At CrI3 6×6 80 Ry charge channel with
`nb_total = 160, n_bc = 10, bpd_max = 16`:
`(10 + 4) · 2^{-52} ≈ 3.1e-15` relative.  Bispinor with
`nb_total ≈ 376, n_bc = 24, bpd_max = 16`: `(24 + 4) · 2^{-52} ≈
6.2e-15` relative.

### 5.2.  Recommended tolerance (cross-check against scaffold)

Existing scaffold tolerances in `tests/test_wfn_transforms.py`:

- **`atol=1e-14, rtol=0`** — used for *inside-vs-outside-wrapper*
  byte-equality tests (`to_rchunk_inner` vs `to_rchunk`,
  `to_rmu_inner` vs `to_rmu`; lines 222, 245, 267, 398, 427).  No
  summation order change.
- **`rtol=1e-10, atol=1e-12`** — used for *summation-order-may-
  differ* tests (`gflat_to_rchunk` chunked vs one-shot, lines 309,
  334, 366; `accumulate_rchunk_to_gflat` lines 477, 507, 540).
  These exercise scan-over-chunks accumulation against a reference
  that does the reduction differently.

The new kernel is in the second category.  **Use `rtol=1e-10,
atol=1e-12`** — mirrors existing scaffold; comfortably above the
~6e-15 ULP-class drift predicted above; protects against pessimistic
einsum-contraction-tree differences if XLA picks something less
balanced under the new code path.

Note: the prompt's first-guess `rtol=1e-12 atol=1e-14` is *tighter*
than scaffold but still well above the predicted drift.  We could
adopt that as a tighter gate, but the win is marginal and the risk
of a flaky test on a less-balanced XLA pick is non-zero.  Stay with
scaffold values.

### 5.3.  Edge cases — explicit behavior for each

#### 5.3.1.  Single-bc case (`n_bc == 1`)

Scan runs one iter, `bpd_max == nb_total`, no pad.  L and R band
ranges are slices into the full single bc.  Code path **must NOT
special-case** — the same `lax.scan` over a length-1 axis runs
trivially.  Verify the L/R mask tables are length-1 and the
gather-by-bc_idx is a single-element lookup.

**Gate**: write a unit test with `band_chunk_size >= nb_total`
exercising charge + transverse on synth WFN.  Output must match the
multi-bc reference to scaffold tolerance.

#### 5.3.2.  Asymmetric L/R band windows (`nb_L != nb_R`)

`band_range_left = [b_l_lo, b_l_hi)`, `band_range_right = [b_r_lo,
b_r_hi)`.  Some bcs may contribute only to L (if `bc ⊆ [b_l_lo,
b_l_hi)` but disjoint from `[b_r_lo, b_r_hi)`), only to R, both, or
neither.

Per-bc L/R masks handle this naturally on the Y side — a bc that
doesn't overlap the L window has `l_mask` all-zeros, and the
einsum contribution to `P_l_acc` is exactly zero (no FP
contamination — `0·x + 0·x = 0` exactly in IEEE).

**X-side per-bc slice REQUIRES symmetric front+back pad** (per
§2.6.1, Round-7 amendment).  Any bc whose `(bc.lo - L_lo_g) +
bpd_max_global > nb_l` (resp. R-side) drives a
`dynamic_slice_in_dim` past the end of `psi_l_X` → XLA silently
clamps the start → wrong physical bands at the einsum.  The
clamp footgun was the round-6 BLOCKER bug; sub-gate G1.1c
("asymmetric L=(0,5) R=(3,8), bcs ((0,4),(4,8))") caught it at
max rel = 5.04.  Symmetric pad zero-fills both ends; pad-rows are
zero in `psi_l_X_bc` so the einsum sees `0·x = 0` and the mask
zeros the corresponding output anyway — **math-neutral, only
FLOP waste**.

**Gate**: unit test on synth WFN with `nb_L = nb_R = nb_total/2`
shifted by `nb_total/4` so the overlap is partial.  Match
reference at scaffold tolerance.  Round 6 G1.1c is exactly this
fixture.  **Test methodology lesson**: the multi-bc charge-only
case alone is INSUFFICIENT to catch the clamp — Agent 4's
methodology audit (`round7_test_methodology_audit.md`) makes
asymmetric L/R + short-final-bc canonical fixtures for any
future pair-pipeline rewrites.

#### 5.3.3.  Short final bc (last bc has `bpd_per_bc < bpd_max`)

**On the Y side (ψ(G) → ψ_Y_bc)**: per `agent_2_structural_fix.md`
§4a / §4e, the short last bc is zero-padded to `bpd_max` on host,
then the L/R mask zero-masks the pad bands inside the body.

FP check: zero-pad band rows of ψ_G_bc → zero rows of ψ_Y_bc after
FFT (FFT is linear; zero in, zero out at band-axis broadcast).
einsum with zero rows in either operand → zero contribution.  No
rounding cascade because `0·x = 0` exactly.  **Math-neutral**;
only FLOP waste.

**On the X side (`psi_l_X` / `psi_r_X`)**: the short final bc
ALSO requires the symmetric front+back pad from §2.6.1 because
the bc's static-length slice `(bc.lo - L_lo_g, bpd_max_global)`
extends past the end of `psi_l_X`.  Without back-pad, XLA's
`dynamic_slice_in_dim` clamps the start → wrong bands.  The
Round-5 plan's §5.3.3 (this section) did *not* surface the
X-side clamp; sub-gate G1.1b ("short final bc
((0,4),(4,8),(8,9)), L=R=(0,9)") caught it at max rel = 11.5 —
catastrophic FP failure, not ULP-class.  Round-7 amendment
restores correctness via the back-pad (§2.6.1 + commit
`c796420`).

**Gate**: unit test with `nb_total = band_chunk_size · k + 1`
forcing a single trailing band in the last bc.  Match reference.
Round 6 G1.1b is exactly this fixture.  **Mandatory test for any
streaming-scan rewrite of a pair-density kernel**: short-final-bc
+ asymmetric L/R together exercise both clamp footgun modes.

#### 5.3.4.  `norms_l` / `norms_r` per-bc divide

Per agent_2 §4e: pre-multiply `psi_l_X` and `psi_r_X` by `1/norms`
ONCE outside the kernel, before the scan starts.  Avoids per-bc
norm slicing inside the scan body.

FP impact: division order moves from "psi_Y / norm" to "psi_X *
(1/norm)".  Mathematically identical via `(x*1/n) * y == x*y/n` up
to multiply order.  Since the einsum is symmetric in X/Y at the FP
level, this is a single extra multiply per X element with the
result fed to the same einsum — ULP-class drift identical to the
ordering already accepted in §5.2.

**Gate**: include a `band_norms ≠ None` (pseudobands) configuration
in the bit-identity test set.  Match scaffold tolerance.

#### 5.3.5.  Bispinor channels (charge + transverse)

`vertex_mu_L = 0` (charge) → `gamma_L = gamma_R = None` →
`gamma_double_contract` short-circuits identity.  `vertex_mu_L ∈ {1,
2, 3}` (transverse) → `gamma_L = gamma_R = (perm, phase)` for that
γ̃ tuple.  The kernel API threads `gamma_perm`/`gamma_phase`
unchanged; the scan body doesn't touch them (γ̃ is applied
post-scan in the IFFT → γ̃ → FFT chain on the accumulated
`P_l`/`P_r`).  **No structural change** for bispinor; the L/R mask
choice and the einsum kernel are agnostic to γ̃.

**Gate**: run bit-identity on a small bispinor configuration
(MoS2 3×3 spin-orbit or synth) with `vertex_mu_L = 1`.

#### 5.3.6.  Per-rank `r_loc` divisibility (Agent 1 §2.3, §9)

This is structural-correctness, not numerics, but it's the gotcha
most likely to produce silent wrong-answer bugs.  The carry's `r_loc
= n_zchunk / p_y` must be a whole integer.  If `n_zchunk % p_y != 0`,
either:

- (a) The planner pads `n_zchunk` to a `p_y`-multiple (mirroring how
  `n_rmu_padded ≡ ∏ p_a` is enforced at centroid load time), and the
  body emits `out_spec`-shaped output that's later sliced to logical
  `n_zchunk` at the kernel exit.  This is the existing pattern
  (`fit_zeta_to_h5` does this with `valid_shape=meta.n_zchunk` at
  the SlabIO seam).
- (b) Raise at planner time if divisibility fails; no fallback.

**Gate**: explicit assertion at trace time inside the new
`z_q_from_psi_sm` factory:
```python
assert n_zchunk % p_y == 0, (
    f"n_zchunk={n_zchunk} must be divisible by mesh.y={p_y}; "
    f"planner must pad or fail upstream.")
```

**Agent 3 to confirm** in §4 whether the planner already enforces
this (it does for `n_rmu_padded`; needs verification for the
r-chunk axis).  Until verified, treat it as a fail-fast check that
prevents silent wrong-answer regressions.

#### 5.3.7.  `solver_kind == 'auto'` (charge vs transverse path
selection)

Irrelevant to the helper.  The new `_kernel` body still passes
`solver_kind` through to `solve_zeta` unchanged (per
`agent_2_structural_fix.md` §4d sketch).  Cholesky for charge, LU
for transverse — same dispatch as today.

**Gate**: end-to-end CrI3 run must produce eqp0.dat values matching
the lorrax_A baseline within COHSEX tolerance (verified by G3
below).

### 5.4.  Validation gates (G1 → G2 → G3)

Each gate is binary go/no-go.  Implementer **must NOT** proceed past
a failed gate.

#### G1 — CPU bit-identity on MoS2 3×3 synth WFN, single channel

**Scope**: charge channel, single rchunk, multi-bc (e.g.
`band_chunk_size = nb_total/3`), no pseudobands.  Run today's
`fit_one_rchunk` (lorrax_B `5cadd4b` body — bug and all) against
the new body on the same inputs.  Run on a 1×1 mesh.

**Pass criterion**: `Z_q` (output of `z_q_from_psi_sm` in both
codepaths) agrees to `rtol=1e-10, atol=1e-12` over the full
`(nq, mu, n_rchunk)` tensor.

**Fail criterion**: any element exceeds tolerance.  Implementer
must diagnose summation-order assumption violation, mask-zero
assumption violation, or a structural rewrite bug.

**Additional sub-gates** (each separately):

- G1a: bispinor `vertex_mu_L = 1` on synth WFN.  Same tolerance.
- G1b: pseudobands `band_norms ≠ None`.  Same tolerance.
- G1c: short-final-bc (`nb_total = bc·k + r` for `r ∈ {1, bc-1}`).
- G1d: asymmetric L/R windows (`nb_L != nb_R` partial overlap).

#### G2 — HLO slot count + zero remat warnings on synth scale

**Scope**: dump HLO for the new `_kernel` at MoS2 3×3 (small enough
for fast iteration).  XLA flags:
```
XLA_FLAGS="--xla_dump_to=/tmp/hlo_g2/xla_dump
           --xla_dump_hlo_pass_re=spmd"
```

**Pass criteria** (all four must hold):

1. **Substantive preallocated-temp slot count ≤ 5**: 2 carry
   accumulators (`P_l_acc`, `P_r_acc`) + 1 FFT box at scan-body
   scope + 1 IFFT/FFT post-pair scratch + 1 small interaction
   buffer.  Count by `grep -c '^\s*[0-9.]*GiB.*allocation' module_*.jit__kernel.*-memory-usage-report.txt`.
2. **No `c128[nk, bpd_max, ns, nx, ny, nz]` slots outside scan
   scope**: the FFT box must be entirely scan-internal.  Inspect
   the report's allocation table — slot must list under the scan's
   buffer-assignment scope, not as a top-level temp.
3. **Zero `[spmd] Involuntary full rematerialization` warnings** in
   stderr during compile.  Grep for the string; must be empty.
   This is the headline fix — the 32 warnings in the current
   `gw.out` must disappear entirely.
4. **No `psi_Y_full` materialisation**: the AFTER HLO must NOT
   contain a `c128[36, 160, 2, 73648]`-class buffer (the 14.85 GiB
   slot Agent 1 §3a flagged).  Even one such slot fails.

**Fail criteria**:
- > 5 substantive slots → scan-internal aliasing didn't take.
- Any remat warning → consumer-boundary reshard didn't get fixed.
- Any `psi_Y_full`-class slot → the design didn't eliminate the
  global ψ materialisation.

#### G3 — End-to-end CrI3 6×6 80 Ry, live allocation

**Scope**: run `gw.gw_jax` on `runs/CrI3/M_6x6_80Ry_2026-05-07/`
with the new kernel, charge channel only (bispinor is a bigger
follow-up that needs all bispinor channels working).  Use the
existing SLURM allocation (4 nodes hbm80g, per round4_discussion.md
status snapshot).  Same cohsex.in as the `lorrax_B_path_d_hlo_2026-05-13`
run.

**Pass criteria** (all four must hold):

1. **Run completes all 16 r-chunks + remainder** without OOM,
   without tracer leak, without remat warnings (carry-over from
   G2).  Output: `eqp0.dat`, `WFN_qp.h5`, `zeta.h5`, `V_q.h5`.
   (The `write_qp_wfn_h5` downstream bug noted in round 4 status
   snapshot is *separate* — ignore if it's the same nk mismatch.)
2. **Total preallocated-temp ≤ 25 GiB** per rank in the new HLO
   (vs 48.63 GiB on lorrax_B `5cadd4b`).  Measure via the
   `memory-usage-report.txt` files.
3. **HBM HWM via `nvidia-smi`** ≤ 28 GB / GPU during the run
   (matches the 0.8 × budget on HBM40 nodes — though HBM80 here is
   uncontested headroom).  Sample at 1 Hz, peak across run.
4. **Σ matches lorrax_A baseline** within existing COHSEX tolerance
   in eqp0.dat (`re_Sigma` columns); compare against the pre-Path-D
   reference run on `lorrax_A` `agent/zeta-r-chunk-fixes-2026-05-13`
   (whichever commit produced a clean eqp0.dat before the band_fft_pool
   stopgap).  This confirms the algebraic equivalence of scan-sum
   vs concat-then-einsum at production scale.

**Fail criteria**:
- OOM at any r-chunk → memory model assumption wrong; revisit §4.
- Σ disagreement > existing tolerance → numerical fix incomplete
  (e.g. masking bug, normalization order wrong, γ̃-fold path bug).

## 6.  Implementation sketch (file-by-file)  ⟨Agent 4 + peer input⟩

_(Stub.  Refined after §2 / §3 / §4 land; for now, the agent_2 §4
sketch + the round 3 actually-shipped delta is the starting
point.)_

### 6.1.  `src/common/psi_G_store.py`  ⟨per Agent 2 §3.9⟩

| Field / property | Action | Notes |
|---|---|---|
| `_host_tiles[(x,y)]` | **Keep** | bc-stacked per-rank numpy; read by io_callback every iter |
| `_bpd_max` | **Restore** (was added in `cdd0fba`, deleted in `5cadd4b`) | int set at `__init__`; closure-static (§3.3) — required for io_callback's static `out_sds` |
| `_bc_band_offsets` | **Keep** | unchanged |
| `_g_index_dev` / `_kvecs_frac_dev` | **Keep** | replicated jax.Array, set once via `jax.device_put`; captured via closure in `_kernel`; concrete arrays — no tracer hazard |
| `_psi_G_device_full` field | **DELETE** | Path B never materialises the full tile on device |
| `psi_G_device_full` property | **DELETE** | the tracer-leak source |
| `g_index` / `kvecs_frac` properties | **Keep** | thin accessors; consumers (`_kernel`) still need them |
| `_slice_local_tile_bc(x_idx, y_idx, bc_idx)` | **Add (restored)** | per §3.2 signature; returns `(nk, _bpd_max, ns, ngkmax)` padded with **`np.zeros`** (not `np.empty` — pad-rows must be exactly zero for mask + einsum to be math-neutral, §5.3.3) |

Lifecycle invariant (§3.10): host tiles must remain valid for the
full duration of `_kernel`'s jit (io_callback fires per scan iter).
`RereadPsiGStore.end_rchunk` already runs **after**
`block_until_ready` (`isdf_fitting.py:2168-2172` `finally:`), so the
async callback always completes before tiles are freed.  No code
change needed; Path B inherits the existing contract.

### 6.2.  `src/common/wfn_transforms.py`

- **Keep** `to_rchunk_inner` (already present, used by the new
  body).
- **Delete** `gflat_to_rchunk` and its `_GFLAT_TO_RCHUNK_CACHE`.
- **Keep** `gflat_to_rmu` (centroid-load path — Defect 3 fix from
  Round 3 — unchanged).  The centroid load is one-shot, not
  per-r-chunk, so its flat-axis design is OK.
- **Keep** `to_rmu_inner` (parallel to `to_rchunk_inner`,
  unchanged).

### 6.3.  `src/common/isdf_fitting.py`

- **Rewrite** `z_q_from_psi_sm` per agent_2 §4c — new signature
  takes `psi_l_X, psi_r_X, psi_G_store, band_chunk_ranges,
  band_range_left, band_range_right, fft_grid_t, r_start_dyn,
  r_chunk_size, gamma_L, gamma_R, kgrid, mesh_xy`.
- **Rewrite** `c_q_from_psi_sm` for the CCT side (same structure;
  scope says "the CCT path mirror is out of scope" — confirm
  whether this lands in Round 6 or is deferred).
- **Rewrite** `_make_fit_one_rchunk_kernel._kernel` body per
  agent_2 §4d — drops the `psi_Y_full = gflat_to_rchunk(...)`
  call, the L/R slice, and the `del psi_Y_full`.  Just calls the
  new `z_q_from_psi_sm` directly.
- **Pre-multiply** `psi_l_X`, `psi_r_X` by `1/norms` before the
  kernel call (§5.3.4 — cleaner than per-bc norm slicing inside
  the body).
- **r_loc sizing**: inside `z_q_from_psi_sm._local`, compute
  `r_loc = n_zchunk // p_y` and `r0_local = r_start_dyn +
  axis_index('y') * r_loc`; pass to `to_rchunk_inner`.  Assert
  `n_zchunk % p_y == 0` at trace time (per §5.3.6).  Carry init:
  `jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)`.  See §2.3 for
  the SPMD reasoning.

### 6.4.  `src/gw/gflat_memory_model.py`

- **Update** `_peak_C_fit_one_rchunk` term for the new design.
  The `c128[N_rows, ns, n_rtot]` FFT box becomes `c128[nk,
  bpd_max, ns, n_rtot]` per-rank (sharded if applicable).
- Coordinate with Agent 3 R4's three bookkeeping fixes (Peak B/C
  centroids typos, Peak D `fft_box_factor=4` removal) — those
  land in the same commit, per round4_next_tasks.md T3.

### 6.5.  Iteration design — Option A bc-aligned scan (Agent 1 v3 + Agent 3 v2 — corrects Agent 4's first draft)

**Iteration log**: Agent 4's first §6.5 went with Option B (flat
(k·b) scan). Agent 1 v3 + Agent 3 v2 push back: **Option B does
NOT work in the forward direction** under band-flat-sharded host
tiles. The accumulate_rchunk_to_gflat flat-axis pattern works in
REVERSE because each (q, μ) row is computationally independent.
In FORWARD the einsum CONTRACTS over the band axis — Option B
would require a `psum` on the 7.42 GiB rank-5 carry per scan end
to aggregate band-shared partials. The `all_gather` on a per-iter
~340 MB slab is dramatically cheaper, AND doesn't risk
SPMD-replication on the carry.

**Recommended (Option A)**: bc-aligned single interleaved scan with
per-iter `all_gather('x','y')` on the bc's bands. Body sketched in
§2.10; restated here with the full closure context:

```python
n_bc = len(band_chunk_ranges)
bpd_per_bc = bc_size // P                    # bands per rank per bc (CrI3: 1)
bpd_max_global = bc_size                     # full bands per bc after gather

b_lo_global = jnp.asarray([bc[0] for bc in band_chunk_ranges], dtype=jnp.int32)
b_hi_global = jnp.asarray([bc[1] for bc in band_chunk_ranges], dtype=jnp.int32)
L_lo_g, L_hi_g = int(band_range_left[0]),  int(band_range_left[1])
R_lo_g, R_hi_g = int(band_range_right[0]), int(band_range_right[1])

def body(carry, bc_idx):
    P_l_acc, P_r_acc = carry
    # 1. Pull this rank's 1/P bands of bc bc_idx (band-flat-sharded host tile).
    psi_G_bc_local = io_callback(
        psi_G_store._slice_local_tile_bc,
        jax.ShapeDtypeStruct((nk, bpd_per_bc, ns, ngkmax), c128),
        x_idx, y_idx, bc_idx, ordered=True)
    # 2. IFFT + per-rank r-slab.
    r0_local = r_start_dyn + y_idx * r_loc
    psi_Y_bc_local = to_rchunk_inner(
        psi_G_bc_local, g_index_c, fft_grid_t, r0_local, r_loc,
        kvecs_frac=kvecs_frac_c, norm="ortho")
    # 3. all_gather across ('x','y') axis=bands (IFFT-FIRST ordering — Agent 3 v2).
    psi_Y_bc = jax.lax.all_gather(
        psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)
    # 4. L/R global-index masks gathered by bc_idx.
    g_axis = b_lo_global[bc_idx] + jnp.arange(bpd_max_global, dtype=jnp.int32)
    bc_valid = g_axis < b_hi_global[bc_idx]
    l_mask = (g_axis >= L_lo_g) & (g_axis < L_hi_g) & bc_valid
    r_mask = (g_axis >= R_lo_g) & (g_axis < R_hi_g) & bc_valid
    psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
    psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
    # 5. psi_l_X / psi_r_X per-bc slice (band axis replicated on rank).
    psi_l_X_bc = lax.dynamic_slice_in_dim(psi_l_X_, b_lo_global[bc_idx] - L_lo_g, bpd_max_global, axis=2)
    psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
    psi_r_X_bc = lax.dynamic_slice_in_dim(psi_r_X_, b_lo_global[bc_idx] - R_lo_g, bpd_max_global, axis=2)
    psi_r_X_bc = jnp.where(r_mask[None, None, :, None], psi_r_X_bc, 0)
    # 6. Two einsums into the carries (both live, interleaved single scan — §2.10).
    delta_P_l = jnp.einsum('kmna,knbr->karmb', psi_l_X_bc, psi_l_Y_bc, optimize=True)
    delta_P_r = jnp.einsum('kmna,knbr->karmb', psi_r_X_bc, psi_r_Y_bc, optimize=True)
    return (P_l_acc + delta_P_l, P_r_acc + delta_P_r), None

(P_l, P_r), _ = lax.scan(body, (P_l_init, P_r_init), jnp.arange(n_bc, dtype=jnp.int32))
# Post-pair pipeline: byte-identical to today's z_q_from_psi_sm._local.
```

**Slicer API impact**: `_slice_local_tile_bc(x_idx, y_idx,
bc_idx_traced)` returns `(nk, bpd_per_bc, ns, ngkmax)` — the
bc-indexed API, **NOT** the flat row-batch API Agent 4 provisionally
asked Agent 2 to spec. This matches the scaffolding originally
added in `cdd0fba` on `lorrax_B` (subsequently deleted in `5cadd4b`).
Agent 2 §3: please confirm this API shape; it's the right one.

**Pre-implementation unit test** (build BEFORE wiring the body):
2-band, 2-k, 2-rank synth case verifying:
1. `_slice_local_tile_bc(x, y, bc_idx)` returns the right bands of
   the right host tile.
2. `all_gather(axis_name=('x','y'), axis=1, tiled=True)` produces a
   tensor where band index `r·bpd_per_bc + j` matches the global
   bc band stored at rank r's local band j. (Verify the axis
   ordering matches the host-tile bc-stacking convention — §2.9
   flagged this; insert a per-rank `jnp.take` permutation if not,
   no comm cost.)
3. The single-bc einsum matches today's `z_q_from_psi_sm` value
   within scaffold tolerance.

(Stale Option B sketch follows — kept for archeology, but DO NOT
implement:)
wiring the rest of the body.**

### 6.6.  Tests

- **Delete** `test_gflat_to_rchunk_*` tests (the helper they exercise
  is gone).
- **Add** tests under `tests/test_isdf_fit.py` (or equivalent) for
  G1's six sub-gates (G1.1 multi-bc, G1a bispinor, G1b
  pseudobands, G1c short-final-bc, G1d asymmetric L/R, G1.1
  single-bc) — see §5.3 for the scope of each.

## 7.  Risks + fallback  ⟨Agent 4, refined with Agent 2 input⟩

### 7.1.  Four-primitive composition novelty (`io_callback` × `lax.scan` × `shard_map` × `lax.all_gather`)

Round-5 escalated the design from three to **four** nested
primitives (the per-iter `all_gather` is required, §2.9).  ~85%
confidence the composition works cleanly per Agent 2 §3.1; the
in-tree pairs are each verified but the four-way nesting is novel.

**Mitigation: §3.6 smoke test is a Round-6 prerequisite** — must
pass on 1×1 CPU mesh AND a 4-rank GPU mesh BEFORE the kernel
rewrite begins.  Agent 2's §3.6 ~50-line test code is the
canonical scaffold; the implementer copies it to
`tests/test_io_callback_scan_shard_map_smoke.py`.  Pass criteria:
output matches a numpy reference at `atol=1e-14, rtol=0`.

Specific composition risks (each addressed):

- (a) **Traced `bc_idx` in `io_callback`**: `out_sds` must be
  static at trace time — `_bpd_max` is closure-static at
  `PsiGStore.__init__` (§3.3).  ✓
- (b) **`ordered=` semantics**: per-rank only, not cross-rank
  (Agent 2 §3.4).  Recommendation `ordered=False` so XLA can
  pipeline host calls against device compute; `lax.scan(unroll=1)`
  already enforces sequential body execution at runtime.  ✓
- (c) **`all_gather` inside `lax.scan` inside `shard_map`**: the
  novel pair.  No documented restriction; manual collective uses
  the shard_map's axis name and lowers to a WhileOp-body local
  op.  Smoke-test on §3.6 verifies; failure mode is "Path A
  fallback" in §7.2.
- (d) **Compile time**: scan-inside-shard_map adds one
  WhileOp lowering; <2× vs today per §4.6.  Profile on MoS2 3×3
  before CrI3; hard fail at 60 min, soft warning at 15 min.

### 7.2.  Fallback Path A — driver-level Python bc-loop with
        donation chains

Per agent_2 §2 Path C ("Streaming pair-density accumulation,
driver-level Python loop"): the driver calls `z_q_from_psi_sm`
once per bc with `P_l_acc` / `P_r_acc` as donated jit args, and
the outer `_kernel` becomes a Python loop calling a smaller jit
per bc.  Pros: mechanical, no novel JAX patterns.  Cons: ~Nx
Python dispatch overhead (~0.3 ms × N_bc per r-chunk; at CrI3
n_bc=10 and 16 r-chunks → 50 ms total — negligible).  Cons (real):
the per-bc compiled jit is *smaller* than the new single-kernel
design, so each bc's intermediates don't get aliased across bcs —
slightly worse memory than the scan-internal alias.  But still
strictly better than the current `psi_Y_full` double-materialise
bug.

**Decision rule**: if the §7.1 risks manifest, fall back to 7.2.
The fallback is bounded: Path D's structural goal (no
`psi_Y_full`, no remat) is still met because each bc's jit is
independent and the L/R window slicing happens inside each
bc-jit's shard_map.

### 7.3.  Risk: `_slice_local_tile_bc` pad-band initialization

The pad-bands-are-zero contract (§5.3.3) relies on the slicer
returning `np.zeros((nk, _bpd_max, ns, ngkmax))` and copying the
valid bands in.  `np.empty` here would leave the pad rows
uninitialized; masked-zero × NaN/Inf = NaN/Inf, contaminating the
einsum even though math-wise the pad bands should contribute zero.

**Mitigation**: §6.1 spec mandates `np.zeros` (Agent 2 §3.9).
Implementer audits the slicer code path during PR review; G1
bit-identity gate catches a regression if the allocator changes
later.

### 7.4.  Lifetime of `psi_G_store._host_tiles` during scan — RESOLVED

Confirmed by Agent 2 §3.10: `RereadPsiGStore.end_rchunk` runs
inside a `finally:` block AFTER `block_until_ready` on the kernel's
output (`isdf_fitting.py:2168-2172`).  All async io_callbacks
complete before tiles are freed; the host-tile lifetime contract
that Reread mode already enforces in production today carries over
unchanged to Path B.  No code change needed; the new scan-internal
io_callbacks inherit the existing guarantee.

## 8.  Out of scope (deferred deliberately)

- **μ-sharding change from `'x'` to `('x','y')` mesh**.  Today
  `psi_l_X` enters as `P(None, 'x', None, None)` — μ on `'x'`
  only.  Under the converged carry math (§4.2: 3.71 GiB per side,
  7.42 GiB total at CrI3 6×6 80 Ry), moving μ to `P(None,
  ('x','y'), None, None)` shrinks the carry 4× further → 1.86 GiB
  total — smaller marginal win than the original (retracted)
  29.7 → 7.4 GiB estimate.  Mechanically the same change but
  requires upstream changes to the centroid loader's `psi_rmuT_X`
  output sharding (Round 3 `gflat_to_rmu` currently emits
  `P(None, 'x', None, None)` matching today's consumer contract).
  **Not urgent**: the converged 13–15 GiB total already fits HBM40
  comfortably; file as Round 7+ follow-up if profiling reveals
  the carry is the wall-clock bottleneck.
- **CCT-side rewrite of `c_q_from_psi_sm`**.  Agent 2 R3
  `round3_integration.md` §"Out of scope" deferred this; Round 5
  same.  Round 6 implementer should rewrite the CCT side with the
  same pattern as Z_q (parallel structure) IF Round 6 budget
  allows — otherwise file as Round 7 follow-up.  Note: CCT runs
  once per channel (pre-rchunk-loop) so its memory profile
  matters less than Z_q's, which runs N_rchunk times.
- **Planner bookkeeping fixes** (Peak B/C centroids typos, Peak D
  `fft_box_factor=4` removal, `_bytes_centroids_LR` cherry-pick).
  These are round4_next_tasks.md T3 — separate commit per Agent 2
  R4's branch-management plan.  Coordinate with §6.4 above so
  both land before merge to main.
- **Defects 4, 5, 6** from `defect_catalog.md`.  Round 4 priority
  N1/N2/N5.  Not blocked by Round 5/6; can land in parallel after
  Path D-fixed validation gate (G3) flips green.
- **io_chunk / fft_chunk nested-scan split** (the user's idea
  from the prompt referenced in round4_next_tasks.md N3).  Gated
  on cuFFT throughput profiling (P1).  Becomes interesting only
  at 8×8 k-grid scales where the FFT box at full nk is still too
  big — at CrI3 6×6 the new design's FFT box is ~1 GiB per rank
  per bc, no further chunking needed.
- **Merging `lorrax_B agent/zeta-bc-scan-shardmap` to main**.
  Agent 2 R4 §4 branch-management plan: do after Round 6 validation
  (G3) passes, as non-fast-forward merge with the bookkeeping
  fixes as a follow-up commit.

---

## Live status — FINAL

- ✅ **Agent 1 round 5 ready** (`round5_discussion.md` lines 111,
  130).  Owns §2 (SPMD/shard_map) and §2.9 / §2.10 (all_gather +
  interleaved scan).
- ✅ **Agent 2 round 5 ready** (line 295).  Owns §3 (io_callback /
  host-tile lifecycle), §6.1 (`psi_G_store.py` state table), and
  §3.6 smoke test (Round-6 prerequisite).
- ✅ **Agent 3 round 5 ready** (lines 424, 449).  Owns §4 (HLO
  predictions); converged peak ~13–15 GiB per rank with seven HLO
  acceptance gates in §4.8.
- ✅ **Agent 4 round 5 done**.  Owns §1, §5 (numerics + edge cases
  + G1/G2/G3 validation gates), §6 (file-by-file), §7 (risks +
  fallback), §8 (out-of-scope).

Plan is **final** as of 2026-05-13.  Round-6 implementer:
1. Run the §3.6 smoke test first (CPU then GPU).  Must pass before
   touching kernel code.
2. Follow §6 file-by-file; §6.5 has the binding body sketch.
3. Run G1 → G2 → G3 in order; each is a hard gate per §5.4.
4. Land the planner bookkeeping fixes (Round 4 T3) in the same
   commit per §6.4.

---

## 9.  Round 6 reality vs Round 5 prediction (Round-7 amendment, 2026-05-14)

This section records what Agent 2 actually shipped in commits
`f567aa0` (Round-6 kernel rewrite) + `c796420` (Round-7 back-pad
BLOCKER fix) vs what §2–§7 of this plan predicted.  Future
implementers reading the plan should expect §2.3 / §2.6 / §2.9 to
be amended in place (per the in-section subsections above); this
§9 collects the deltas in one place for traceability.

### 9.1.  Body sequence — r-slice moves AFTER all_gather

**Plan said (§2.3 v2, §2.9 v2)**:

```
io_callback → IFFT (per-rank r_loc slab) → all_gather (bands) → einsum
```

**Shipped (`f567aa0`)**:

```
io_callback → IFFT (FULL per-rank n_zchunk slab) → all_gather (bands)
            → dynamic_slice_in_dim (r-axis to r_loc)
            → einsum
```

**Root cause**: each y-rank computing a different r-slab pre-gather
produces an r-incoherent post-gather tensor (band `i` from rank
`y=0` would hold its r-slab, but band `j` from rank `y=1` would
hold a different r-slab; the einsum sees them as one coherent
`(band, r)` tile and silently produces wrong numerics).  Surfaced
by Agent 2 during Round-6 debug; documented in §2.3.1 above.

**Memory impact**: per-iter slab inside the body is bigger than
§4.1's prediction by ~1.0 GiB (post-gather full-r tile is `c128[nk,
P·bpd_per_bc, ns, n_zchunk]` ≈ 1.36 GiB at CrI3 6×6 80 Ry).  Both
the pre-gather full-r-local slab (~85 MB) and the post-gather slab
(~1.36 GiB) are scan-aliased to single slots.  Carry and γ̃ peak
are unchanged.  **Net peak prediction adjusts from §4.1's 13–15 GiB
to ~14–17 GiB.**  Still ~3× better than `5cadd4b`'s 48.63 GiB.  G2
HLO dump (Round-7 follow-up) confirms.

### 9.2.  X-side dynamic_slice clamp footgun — symmetric front+back pad

**Plan said (§2.6 v1)**: per-bc `psi_l_X` / `psi_r_X` slice is
"purely local" — `dynamic_slice_in_dim` with traced offset and
static length.  No discussion of OOB behavior.

**Plan said (§5.3.2 / §5.3.3 v1)**: L/R mask zero-masks pad bands
on the Y side; same contract assumed for the X side.

**Shipped (`f567aa0` + Round-7 BLOCKER fix `c796420`)**: requires
symmetric front+back pad on `psi_l_X` / `psi_r_X` so the per-bc
`dynamic_slice_in_dim` never goes out of bounds.  Without it, XLA
silently clamps the start to `max(0, axis_size - slice_size)` →
wrong physical bands → einsum with L/R mask produces catastrophic
FP errors (max rel = 11.5 on short-final-bc, 5.04 on asymmetric
L/R).

**Where it fires**:
- charge with L=R=full + divisible bc: never (passed Round-6 e2e at
  max rel ≈ 1e-10)
- bispinor transverse: always
- short final bc: always
- asymmetric L/R: always when any bc crosses a window boundary

§2.6.1 above gives the full diagnosis and the fix; this §9.2 is the
audit-trail entry.  Test methodology lesson is in Agent 4's
`round7_test_methodology_audit.md`.

### 9.3.  Round-5 vs Round-6 / Round-7 numerics

| Sub-gate | R5 prediction (§5.3) | R6 result (`f567aa0`) | R7 result (`c796420`) |
|---|---|---|---|
| G1.1 charge multi-bc, L=R=full | ULP-class drift | ✅ max rel ~5e-13 | ✅ same |
| G1.1a single-bc | ULP-class | ✅ ~5e-13 | ✅ same |
| G1.1b short final bc | math-neutral | ❌ **max rel 11.5** | ✅ ~5e-13 (post-back-pad) |
| G1.1c asymmetric L=(0,5) R=(3,8) | math-neutral | ❌ **max rel 5.04** | ✅ ~5e-13 (post-back-pad) |
| G1.1d bispinor γ̃^1 | identical to charge | ✅ ~5e-13 | ✅ same |
| G1.1e pseudobands | identical | ✅ ~5e-13 | ✅ same |

After `c796420` (Round-7 back-pad fix), all six sub-gates pass at
the scaffold tolerance (`rtol=1e-10, atol=1e-12`).  Agent 4's
independent re-verification is the formal G1 sign-off.

### 9.4.  SPMD invariants — all unaffected

Agent 1's 6 SPMD invariants from the Round-6 review checklist
remain satisfied across both commits.  The `f567aa0` → `c796420`
diff is a pad-tensor shape change only (`((0,0),(0,0),(front,
0),(0,0))` → `((0,0),(0,0),(front, back),(0,0))`); no shard_map
spec, no carry shape, no collective placement changed.  Agent 1
re-bless on `c796420` posted in `round7_discussion.md`.

### 9.5.  Plan-record discipline notes for future rewrites

Three discipline lessons surfaced by this round:

1. **The "r-slice trick" of §2.3 needs to be paired with a
   coherence argument across whatever mesh axis the pre-gather
   value is sharded on.**  If different ranks compute different
   sub-tiles BEFORE a gather, the gather output is incoherent
   along the un-coordinated axis.  Sanity check: after every
   per-iter gather inside a scan body, the gathered tensor should
   be coherent on ALL non-gathered axes, otherwise the design is
   wrong.

2. **`dynamic_slice_in_dim` with a traced start and static length
   silently clamps OOB starts.**  Any traced-start slice that
   *might* go OOB requires either pre-padding or explicit
   `clip_indices=False` (not available pre-2024 JAX, hence the
   pad approach).  The plan's §2.5 mask approach assumed the
   `dynamic_slice` was always in bounds; that assumption was
   load-bearing and undocumented.

3. **Bit-identity test grids must include adversarial corner
   cases ALONGSIDE production-shaped runs.**  Agent 2's own G1 ran
   on MoS2 3×3 charge with divisible bc count → passed.  Agent 4's
   adversarial 6-sub-gate grid caught two production-relevant
   bugs the e2e run missed.  Future pair-pipeline rewrites should
   include short_final_bc + asymmetric_L_R + bispinor_transverse
   as canonical fixtures from day 1.

### 9.6.  Round-7 follow-ups still open at amendment time

- G1 re-verification by Agent 4 (G1 rerun on `c796420`):
  pending — Agent 4 starting "round 7 G1 rerun" per
  `round7_discussion.md:83`.  Once posted, Agent 3 fires G2 HLO
  dump.
- G2 HLO dump on `c796420`: pending — Agent 3 (this section's
  amender) holds for G1 rerun pass, then fires
  `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round6_hlo_2026-05-13/run_round6_hlo.sh`
  per `round6_discussion.md` Phase-2 plan.
- G3 e2e on `c796420`: pending — Agent 4 supervises after G2.
- Dead-code cleanup commit: queued by Agent 1 (`round7_dead_code_audit.md`),
  lands after G1+G2+G3 all pass.
- Planner accounting fixes (Round 4 T3 + the §9.1 transient-slab
  term): land as a separate small commit alongside the dead-code
  cleanup.
