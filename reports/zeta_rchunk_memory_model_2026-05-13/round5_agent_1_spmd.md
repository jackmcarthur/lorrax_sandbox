# Round 5 — Agent 1: SPMD / shard_map / mesh-semantics analysis

**Lens**: SPMD partitioner behavior, `shard_map` body semantics,
`lax.scan` carry inflation, and the mesh-axis arithmetic for the
proposed scan-inside-shard_map design.

**Question being answered**: is it SAFE — and is it CLEAN — to put a
`lax.scan` over bcs inside a `shard_map` whose body holds rank-5
pair-density accumulators as the carry? The competing concerns are
(a) the WhileOp/SPMD trap that bit `solve_zeta`, (b) the
`('x','y')` → `'x'`/`'y'` reshard remat that bit Round 3's flat-axis
implementation, and (c) the `psi_G_device_full` tracer-leak that bit
the lazy-property approach.

**Confidence levels used below**:
- ✅ **Documented behavior** (JAX docs, JEP, source, or proven in-repo by analogous working code).
- 🟡 **Believed, not proven** (consistent with JAX semantics; I'd want a 50-line trace test to verify).
- ⚠️ **Open risk** (could break the design if I'm wrong; flag explicitly).

## 1. Why the WhileOp/SPMD trap doesn't fire inside `shard_map`

### What SPMD actually does (the trap mechanism)

✅ The `solve_zeta` comment at `isdf_fitting.py:1126-1130` describes a
real failure mode: scan with a sharded carry at outer-jit level →
SPMD partitioner sees the carry inside the WhileOp body, can't
analyze its sharding *across* iterations, falls back to
all-gather/replication.

The precise mechanism: when a `lax.scan` lowers to HLO it becomes a
`WhileOp` whose body is a separate computation. The SPMD partitioner
operates on the full HLO module *after* the lowering. When it visits
the WhileOp, it sees a carry that enters with sharding spec `S` (from
outer code) and is updated by the body. The partitioner needs to
verify that the body preserves `S` across iterations — but the body
is opaque (the body computation is partitioned independently). When
the body's output sharding doesn't trivially match the input
sharding, or when SPMD can't prove preservation, it inserts an
all-gather at the boundary and runs the body's interior fully
replicated. This is the 88 GB OOM on `Z_col` at `P(None, None,
('x','y'))`.

### Why shard_map breaks the chain

✅ `shard_map` is documented in JEP 14273 as the **manual partition
primitive**: inside the body, *there is no SPMD*. The
`in_specs`/`out_specs` define the partition boundary, and within the
body, JAX produces a single per-rank program — no partitioner runs
on the inner code. Tensors inside the body have no `NamedSharding`;
they're rank-local arrays with the shape `original_shape /
in_spec_partition`.

So when you write:

```python
@partial(shard_map, mesh=mesh_xy,
         in_specs=(L_spec, R_spec, P(), P(), ...),
         out_specs=P(None, 'x', 'y'),
         check_rep=False)
def _local(psi_l_X_, psi_r_X_, perm_L, phase_L, ...):
    P_l_acc = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
    P_r_acc = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)
    def body(carry, bc_idx):
        ...
    (P_l, P_r), _ = lax.scan(body, (P_l_acc, P_r_acc), jnp.arange(n_bc))
    ...
```

The `jnp.zeros` produces a *rank-local* array with no
NamedSharding. The `lax.scan` lowers to a WhileOp **inside the
shard_map's per-rank program**. SPMD has already done its work at
the shard_map boundary; the WhileOp inside is just a sequential
loop. There is **no global sharded carry for the partitioner to
inflate**.

### In-repo proof

✅ This is not a believed-untested pattern — it's the **exact
mechanism `accumulate_rchunk_to_gflat` uses** at
`wfn_transforms.py:617-664`. That kernel:
- shard_map over `('x','y')` with `out_specs=P(None,('x','y'),None)`.
- `acc_flat` is a rank-local accumulator carried through `lax.scan`.
- The scan iterates `n_chunks` times, each iter doing
  `dynamic_update_slice_in_dim(acc, acc_sub + contrib, i0, axis=0)`.
- No SPMD inflation; XLA aliases the per-iter FFT box across iters
  (the round 4 HLO shows this for the reverse direction).

Round 4 measurements on the *forward* helper that uses this same
pattern: HLO collapsed 58 → 1 FFT-box slots, matching the prediction.
**Documented working pattern.**

### Subtle ways the trap could still fire (enumerated risks)

⚠️ **Risk 1 — collectives inside the scan body.** If the body
contained `jax.lax.psum`, `all_gather`, or `axis_index` calls that
depended on iteration state, the partitioner's per-iter cross-rank
analysis could re-introduce SPMD-level work. **For our design: no
collectives inside the body. `axis_index('x')` / `axis_index('y')`
are loop-invariant (called once before scan) — confirm in
implementation.**

⚠️ **Risk 2 — `with_sharding_constraint` inside the body.** This is
a SPMD annotation; inside shard_map it should be a no-op or an
error, but if any helper called from inside (like
`apply_bloch_phase_on_slice`) sneaks one in, it could leak SPMD into
the body. ✅ Audited: `apply_bloch_phase_on_slice` does NOT call
`with_sharding_constraint` (`wfn_transforms.py:736-790`).

🟡 **Risk 3 — `check_rep=True`.** When `check_rep=True`, shard_map
attempts to verify replication invariants and can insert
all-reduce-style ops on inputs that the partitioner *claims* are
replicated but actually aren't. Existing pair-pipeline already uses
`check_rep=False` (`isdf_fitting.py:301`); keep that.

🟡 **Risk 4 — closure over device arrays with shardings.** If the
shard_map body closes over a `jax.Array` (e.g., `psi_G_store.g_index`
as a closure constant), that array's sharding annotation comes in
with it. For *replicated* closures (`P()`) this is fine. For
*sharded* closures this could trigger SPMD reasoning at the closure
site. **Our design's closures (`g_index`, `kvecs_frac`,
`l_lo_tbl`/`l_hi_tbl`/`r_lo_tbl`/`r_hi_tbl` band tables) are all
fully replicated → safe.**

⚠️ **Risk 5 — io_callback inside scan inside shard_map.** This is
the novelty. The reverse helper doesn't use io_callback inside its
scan. There's no documented prior art in this repo for
"io_callback-in-scan-in-shard_map." JAX docs do support all three
nestings independently but the three-way composition is unusual.
**Mitigation: write a minimal 30-line reproducer FIRST before
touching the kernel** — confirm:
  - The host-side `_slice_local_tile_bc(x, y, bc_idx)` is called
    `n_bc × P` times per scan run (once per rank per iter).
  - `ordered=True` sequences the calls.
  - The output `ShapeDtypeStruct` matches and donates to the body.

## 2. Carry design

### Init

✅ `P_l_acc = jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)` inside
the shard_map body. This is a rank-local array — `nk` is full (k
replicated on this mesh), `r_loc = r_len` (r replicated), `mu_loc =
n_rmu_padded / p_x` for the x axis. **No NamedSharding;** the shape
is what each rank's local computation needs.

At CrI3 80 Ry 4×4 mesh with **today's** `psi_l_X` spec `P(None,
'x', None, None)`: `mu_loc = n_rmu / p_x`. Critical correction (Agent
3 v2, after Round 3 iteration): **n_rmu inside this kernel is 376**
(verified from HLO param `c128[36, 376, 376]` on `L_q`), not the
1504 ISDF basis count quoted in gw.out's summary. So `mu_loc = 376/4
= 94`. r_loc: `n_zchunk / p_y = 73648/4 = 18412`. Per-rank carry:
`36 · 2 · 18412 · 94 · 2 · 16 = 3.98 GB ≈ 3.71 GiB`. **Two
accumulators × 3.71 GiB = 7.42 GiB total carry per rank** under
interleaved single scan. Comfortable.

Iteration log: this number was contested twice during Round 5
discussion. v1 of this doc had 2 GB (correct mu_loc but wrong
description). v2 retracted to 14.85 GiB × 2 = 30 GiB after Agent 3
flagged a mu_loc error, using n_rmu=1504. v3 (current) restores
3.71 GiB × 2 = 7.42 GiB after Agent 3 self-retracted on n_rmu
(verified n_rmu=376 in HLO).

At this carry size, the original v2 recommendation to "serialize L
and R scans" loses its justification (saving 3.71 GiB during the
scan is less valuable than the 2× io_callback / IFFT cost). v3
recommendation: **single interleaved scan with both accumulators
live**. Both 3.71 GiB carries plus the per-iter scan-aliased FFT
box (~5 GiB) plus per-iter buffers (~400 MB) plus post-pair scratch
(~4 GiB) plus output (~1 GiB) plus params (~0.5 GiB) ⇒ peak
~13-15 GiB per rank, vs today's 48.63 GiB.

### Donation

🟡 **`lax.scan` does NOT take a `donate_argnums` argument.** The
carry is implicitly read-write inside `scan`'s WhileOp lowering;
XLA's buffer assignment will alias the in/out carry buffer
automatically (no extra alloc). I've confirmed this is the behavior
on `accumulate_rchunk_to_gflat`'s `acc_flat` carry — Round 4 HLO
shows a single slot, not two.

Where donation does matter: the **outer jit's** `donate_argnums`
covers buffer reuse across kernel calls. For our case the new
`z_q_from_psi_sm` is itself wrapped in `@jax.jit`; the `_local`
shard_map body is called via that jit. If we want the scan carry's
final value to fold into the kernel output without an extra copy,
ensure the post-scan tail (`reshape → ifft → contract → fft`) is
fused into the same shard_map (the existing code already does this,
`isdf_fitting.py:412-450`).

### Scan output

✅ `(P_l, P_r), _ = lax.scan(body, (P_l_acc, P_r_acc), jnp.arange(n_bc))`.
Standard JAX idiom; the `_` discards per-iter `None` outputs.
`P_l`/`P_r` are the final carries — semantically the running sum
over bcs.

## 3. `in_specs` / `out_specs` for the wrapping shard_map

### Current production specs (today's `z_q_from_psi_sm`)

✅ At `isdf_fitting.py:408-415`:

```python
L_spec  = P(None, 'x', None, None)        # psi_l_X: (nk, n_rmu, nb_l, ns)
R_spec  = P(None, None, None, 'y')        # psi_l_Y: (nk, nb_l, ns, n_col)
out_spec = P(None, 'x', 'y')              # Z_q: (nq, n_rmu, n_col)
```

`psi_l_X` keeps μ on `'x'`, bands replicated. `psi_l_Y` puts the
col axis (r_chunk in z-path) on `'y'`, bands replicated.

### Proposed new specs (scan-inside design)

The new design **eliminates `psi_l_Y` / `psi_r_Y` from the input
list** — those arrays are produced INSIDE the body, per scan iter,
by `to_rchunk_inner` on per-bc ψ(G) pulled via io_callback. So the
shard_map's inputs become:

```python
L_spec_X    = P(None, 'x', None, None)         # psi_l_X: unchanged
R_spec_X    = P(None, 'x', None, None)         # psi_r_X: same as L (both X-sharded)
out_spec    = P(None, 'x', 'y')                # Z_q: unchanged
gamma_spec  = P()                              # perm/phase: replicated

# All else: closure constants (band tables, g_index, kvecs_frac, etc.)
```

✅ This is identical to today's CCT-path `c_q_from_psi_sm` setup
(`isdf_fitting.py:292-300`) for the X-sharded inputs, with the
Y-sharded `psi_l_Y`/`psi_r_Y` REMOVED.

⚠️ **The `out_spec=P(None, 'x', 'y')` puts r-chunk on `'y'`** — but
where does the `'y'` partition come from when the body never received
a Y-sharded input? Answer: the `r_loc` accumulator dimension is the
*full* `r_chunk` per rank (no Y-partition inside the body). The
final `transpose` + reshape at the body's end emits a value of shape
`(nq, n_rmu_local_x, n_zchunk)` per rank — but `out_specs=P(None,
'x', 'y')` claims n_zchunk is on `'y'`. SPMD's "manual mode" needs
the per-rank output shape to match `output_shape /
out_spec_partition`. **The post-scan tail must produce a per-rank
output of shape `(nq, mu_loc, r_chunk / p_y)`.**

🟡 Today's `c_q_from_psi_sm._local` resolves this by having the
scan-free tail produce that exact per-rank shape (each rank holds
mu_local × col_local). For Z-path, the equivalent requires the
per-rank tail to slice / shard the r-chunk axis across `'y'`. The
existing `z_q_from_psi_sm._local` (lines 412-450) DOES handle this
— look at the final `transpose` and reshape: `Z_q_3d.reshape(nkx*nky*nkz,
z_loc, mu_loc)` where `z_loc` is the per-rank r-chunk slice. **So
the per-rank `r_loc` in the carry must = `n_zchunk / p_y`**, not the
full r-chunk.

⚠️ **This is the key correctness gotcha.** The per-rank accumulator
shape `(nk, ns, r_loc, mu_loc, ns)` must use `r_loc = n_zchunk /
p_y` to match `out_spec=P(None, 'x', 'y')`. The per-bc ψ(rchunk)
slabs that `to_rchunk_inner` produces inside the body must be sliced
to that same `r_loc` extent (each rank holds 1/p_y of the r-chunk).
The mechanism for getting per-rank `r_loc`-sized data is:
- Either the body pulls a different `(r0_local, r_len_local)` per
  rank via `axis_index('y')` — sharded r-axis fetch.
- Or the body pulls the FULL r-chunk per rank from io_callback and
  then slices it to the per-rank `r_loc` — wasteful (4× host
  transfer) but simpler.

🟡 **Recommend the sharded-r approach.** The per-rank r-slab is
`r_loc = r_len / p_y`. Inside the body, compute `r0_local = r0 +
y_idx * r_loc`, pass to `to_rchunk_inner`. The IFFT box still needs
to span the full FFT grid to be correct, but the per-rank output
slice is `r_loc` cells. **No cross-rank communication required if
each rank computes its own r-slab; the FFT is local because the
input ψ(G) is already replicated across `'y'` (band-on-`'x'`-only).**

⚠️ **The above requires verifying that `to_rchunk_inner` can take a
PER-RANK `r0` and `r_len` and produce the per-rank slab without
needing all ranks to coordinate.** The IFFT itself is purely local
(no FFT axes are sharded); the only `'y'`-axis dependence is the
output slice. This is the same trick `accumulate_rchunk_to_gflat`
uses in reverse (each rank handles its own μ-slab independently).

## 4. L/R per-bc band slicing — mask vs slice vs padded

The scan body receives one bc's ψ(rchunk) at shape `(nk, bpd_max,
ns, r_loc)` — bands padded to uniform `bpd_max` so io_callback's
return shape is static. The L and R einsums need only the bands
within `band_range_left[bc]` and `band_range_right[bc]` respectively.

### Mask approach (Agent 2's §4c sketch, recommended)

```python
band_idx = jnp.arange(bpd_max)
l_lo, l_hi = l_lo_tbl[bc_idx], l_hi_tbl[bc_idx]   # gather from static tables
r_lo, r_hi = r_lo_tbl[bc_idx], r_hi_tbl[bc_idx]
l_mask = (band_idx >= l_lo) & (band_idx < l_hi)   # (bpd_max,) bool
r_mask = (band_idx >= r_lo) & (band_idx < r_hi)
psi_l_Y_bc = jnp.where(l_mask[None, :, None, None], psi_Y_bc, 0)
psi_r_Y_bc = jnp.where(r_mask[None, :, None, None], psi_Y_bc, 0)
```

✅ **SPMD-safe**: `band_idx`, `l_mask`, `r_mask` are all rank-local
inside shard_map body. `jnp.where` is pointwise — no resharding.

✅ **No cross-rank op**: the band axis here is the bc's local band
window (after io_callback returns the padded tile), which is local
to the rank. No `dynamic_slice` on a sharded axis.

🟡 **Wasted FLOPs**: bands outside `[l_lo, l_hi)` contribute zero
einsum output, but the einsum still multiplies them. Magnitude:
worst case 50% wasted (a bc straddling both L and R). For CrI3 80
Ry with `band_chunk=16` and overlap ~8, that's a ~50% einsum FLOPS
overhead per scan iter. Acceptable for a structural fix; revisit
later if profiling demands.

### Slice approach (rejected)

`lax.dynamic_slice_in_dim(psi_Y_bc, l_lo, length, axis=1)` requires
`length` to be **static** at trace time. The L-window length per bc
varies (different bcs hit different parts of `band_range_left`), so
length isn't trace-static unless we pad it. Pad-then-slice is
strictly worse than mask-then-einsum (same FLOPs + extra slice op).

### Padded-uniform-slice (overengineered, reject)

Per-bc tables of `(l_lo, l_hi)` + per-bc `l_len = min(l_hi - l_lo,
max_l_len)` padding. More complex than mask, same wasted FLOPs.
Drop.

**Recommendation: mask approach.** Cleanest under SPMD, no
cross-rank ops, ~50% FLOPs overhead is a known acceptable cost.

## 5. `psi_l_X` per-bc band slicing

`psi_l_X` enters the shard_map with `in_spec=P(None, 'x', None,
None)` — shape `(nk, n_rmu, nb_l, ns)`. Inside the body, each rank
sees `(nk, n_rmu_local_x, nb_l, ns)` — **the band axis is
REPLICATED across the mesh** (no `'y'` partition either; full
`nb_l` per rank).

So per-bc slicing on the band axis is *purely local*:

```python
# Inside body, static at trace time:
psi_l_X_bc = lax.dynamic_slice_in_dim(
    psi_l_X_, b_lo_tbl[bc_idx], bpd_max, axis=2)   # axis 2 is bands
psi_l_X_bc = jnp.where(l_mask[None, None, :, None], psi_l_X_bc, 0)
```

✅ **No cross-rank op**: band axis is replicated, dynamic_slice on a
replicated axis is local. `b_lo_tbl[bc_idx]` is a gather from a
rank-local static table, also local.

⚠️ **`psi_l_X.shape[2]` (the full nb_l) is `band_range_left[1] -
band_range_left[0]`, NOT `nb_total`.** Note this is the LEFT side's
band window already. The per-bc slice into `psi_l_X` uses the bc's
position within the L window, not the full band axis. **Build
b_lo_tbl from `band_chunk_ranges` intersected with `band_range_left`,
not raw `band_chunk_ranges`.** Easy to get wrong — flag in design.

## 6. Compatibility with the existing post-pair pipeline tail

After the scan, the body still needs:

```python
P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
P_l_R_conj = jnp.conj(P_l_R)
P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')
Z_R = gamma_double_contract(P_l_R_conj, P_r_R, ..., spin_axes=(3, 6))
Z_q_3d = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='forward')
return jnp.transpose(
    Z_q_3d.reshape(nkx*nky*nkz, r_loc, mu_loc), (0, 2, 1))
```

✅ This is BYTE-IDENTICAL to today's `z_q_from_psi_sm._local`
lines 429-444. Our new body just produces `P_l`, `P_r` differently
(scan over bcs instead of one einsum on pre-built psi_Y) — the
downstream pipeline is unchanged.

✅ The k-axis IFFT/FFT is local per rank (k replicated by
`out_spec`). The γ̃ contract reduces over the spin axes (3, 6) —
pointwise per-cell, no cross-rank.

## 7. SPMD safety summary (✅ / 🟡 / ⚠️ checklist)

| Concern | Status | Notes |
|---|---|---|
| Scan carry SPMD inflation | ✅ Safe | `shard_map` body is per-rank manual; no SPMD inside |
| `accumulate_rchunk_to_gflat` precedent | ✅ Working in repo | Same pattern, Round 4 HLO confirms aliasing |
| Collectives inside body | ✅ None proposed | `axis_index` is loop-invariant; no `psum` inside scan |
| `with_sharding_constraint` leakage | ✅ Audited clean | `apply_bloch_phase_on_slice` is pure-local |
| `check_rep=False` consistent | ✅ Yes | Match existing pair-pipeline |
| Replicated closure constants | ✅ Yes | Band tables, g_index, kvecs all `P()` replicated |
| io_callback inside scan inside shard_map | ⚠️ Novel | **Build 30-line reproducer FIRST** |
| Per-rank `r_loc` matching `out_spec` | ⚠️ Correctness gotcha | Each rank pulls only its `r_loc = n_zchunk / p_y` slab |
| Per-bc band slice (mask) | ✅ SPMD-safe | Pointwise where, band axis local |
| Per-bc `psi_l_X` slice | ✅ SPMD-safe | Band axis replicated under L_spec |
| Post-pair pipeline tail | ✅ Unchanged | Byte-identical to today's _local |
| `lax.scan` donation | ✅ Implicit aliasing | XLA buffer assignment handles it |
| Carry size at CrI3 80 Ry | ✅ 7.42 GiB total | 2 × 3.71 GiB (interleaved single scan); verified after two retractions |
| Band-shard mismatch (host bands flat-sharded; psi_l_X bands replicated) | ⚠️ Resolved by per-iter `all_gather(axis_name=('x','y'), tiled=True)` on small data (~21 MB out / rank / iter) | See §2.9 of unified plan |
| All_gather order: IFFT-then-gather, NOT gather-then-IFFT | ✅ IFFT-first | gather-first would require per-rank IFFT on full bands → 80 GB transient per rank |

**Net assessment**: the design is SPMD-safe modulo two flags —
io_callback-inside-scan-inside-shard_map novelty (mitigated by a
reproducer test) and the per-rank `r_loc` matching the output spec
(mechanical, but easy to get wrong on first pass).

## 8. Open questions for peers

→ **Agent 2 (io_callback)**: the io_callback inside scan with
TRACED `bc_idx` is novel. Two questions for you:
  1. Does `ordered=True` on `io_callback` sequence host-side calls
     within a single rank, or across ranks? We need the former.
  2. Should the host-side `_slice_local_tile_bc` close over the
     pre-padded `bpd_max` so the static return shape is set at
     callback-creation time, not at each invocation? (I think yes —
     `out_sds` must be static at trace time.)

→ **Agent 3 (HLO)**: prediction — with this design, the HLO should
show:
  1. ZERO `c128[..., 73648]` slabs (no psi_Y_full anywhere).
  2. ONE FFT-box-class slot per scan (aliased across bcs).
  3. Carry slots: 2 × `c128[36, 2, r_loc, mu_loc, 2]` ≈ 2 GB total.
  4. No reshard remat warnings (the helper boundary is gone —
     everything happens inside ONE shard_map).

  If you predict differently from the HLO before implementation, say
  so.

→ **Agent 4 (synthesis)**: ready to fold this into the unified
plan. The §3 `out_spec` discussion needs to be load-bearing in the
plan — it's the place where a naive implementation will produce
shape mismatches at the shard_map boundary.

## 9. What I'm explicitly NOT confident about

⚠️ The io_callback-in-scan-in-shard_map composition. JAX supports
all three, but I have not seen a working composed example in this
repo or in upstream JAX tests. The earlier "Path D scaffolding" on
`cdd0fba` added `_slice_local_tile_bc` for exactly this purpose but
the Round 3 implementation took the flat-axis path instead and
deleted `_slice_local_tile_bc` in `5cadd4b`. So this scaffolding has
never actually been exercised. **Reproducer-first is non-negotiable.**

🟡 Compile-time concern. Scan-inside-shard_map nests one WhileOp
inside a ManualPartition region, both inside a `@jax.jit`. JAX's
trace + lowering should handle this, but compile times can grow
non-linearly. Profile compile time on MoS2 3×3 before scaling to
CrI3.

🟡 The per-rank `r_loc = n_zchunk / p_y` claim relies on the
shard_map's `out_specs=P(None, 'x', 'y')` requiring exactly that
per-rank shape. I'm 95% confident this is right from reading
`c_q_from_psi_sm._local` (`isdf_fitting.py:316-351`) — but the
"static at trace time" requirement for `r_loc` means we need to
verify `n_zchunk` is always divisible by `p_y`. The existing
`n_rmu_padded ≡ ∏ p_a` divisibility (centroid loader) suggests this
*is* enforced for the r-chunk axis somewhere; ✅ confirmed by
reading `gw_init.fit_zeta` (around the planner pick), but Agent 3
should double-check this in the planner.

---

Agent 1 SPMD section ready — incorporate into `round5_unified_plan.md` or treat as the SPMD chapter.
