# Round 5 — Agent 3: XLA / HLO / BufferAssignment prediction lens

**Author:** Agent 3
**Date:** 2026-05-13
**Scope:** predicted HLO for the scan-inside-shard_map forward helper at
CrI3 6×6 80 Ry under the Path D design (`agent_2_structural_fix.md` §4c–d).

This document is the HLO/BufferAssignment chapter. It complements Agent 1's
SPMD analysis (`round5_agent_1_spmd.md`) and feeds Agent 4's synthesis at
`round5_unified_plan.md`. Where Agent 1 and I disagree on numbers, see §8.

## 1. The HLO that exists today (for grounding)

From `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_path_d_hlo_2026-05-13/xla_dump/module_0293.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`:

```
Total bytes used: 52216482140 (48.63GiB)
allocation 27: size 44.56GiB, preallocated-temp  ← the bulk
  14.85 GiB @ offset 31900781696 — 3 values: c128[2,6922912,2,36], c128[10,36,147296], s8[4194304]
  14.85 GiB @ offset         3200 — 11 values: c128[360,2,1125000], c128[36,160,2,73648], c128[18412,376,6,6,1], c128[2,18412,376,2,6,6,1], c128[36,36824,752], ...
  14.85 GiB @ offset 15950392448 — 4 values: c128[2,6922912,2,36], c128[2,18412,376,2,6,6,1], c128[36,160,2,73648], c128[36,36824,752]
  12.07 GiB @ offset 12960003200 — 2 values: c128[360,2,75,75,200], c128[36,16,2,59990]
allocation 0: 3.71GiB maybe-live-out (output ζ tile)
allocations 1–4: 0.27GiB params (L_q, R_L, R_R)
```

Decoding the four large slots:

| Slot | Size | Dominant value(s) | What it actually holds |
|---|---|---|---|
| A | 14.85 GiB | `c128[2, 6922912, 2, 36]` = `c128[ns, μ·r_chunk_local, ns, nk]` | P_l einsum output (rank-5 factored), `μ·r_chunk_local = 1504·4603 = 6 922 912` |
| B | 14.85 GiB | `c128[36, 160, 2, 73648]` = `c128[nk, nb_R, ns, r_chunk_unsharded]` | **psi_r_Y full materialization (REMAT)** — would have been sharded on `'y'`, but XLA rematerializes it |
| C | 14.85 GiB | `c128[2, 6922912, 2, 36]` | P_r (post-einsum) or P_l_R_conj — shares the same shape as A |
| D | 12.07 GiB | `c128[360, 2, 1125000]` ≡ `c128[nk·b_local, ns, n_rtot]` | **FFT box from `gflat_to_rchunk`'s inner kernel** — unsharded |

The 30 GiB Agent 1 R4 called out as "psi_Y_full materialized TWICE" is
slots B + D. The third 14.85 GiB slot is genuinely the pair density.

## 2. Predicted HLO for the new design (under TODAY's sharding contract)

I'll walk through each section, using the **conservative interpretation**:
the new shard_map keeps today's `in_specs/out_specs`:

```
psi_l_X, psi_r_X : P(None, 'x', None, None)    # μ on 'x' only, p_x=4
out (Z_q)        : P(None, 'x', 'y')           # μ on 'x', r on 'y'
```

Under this contract, **per-rank shapes**: `μ_loc = 1504/4 = 376`,
`r_loc = 73648/4 = 18412`. (Agent 1 §2 uses `μ_loc = 94`, which only
works under a `('x','y')`-combined μ sharding — see §8.)

### 2a. Persistent allocations

| Item | Per-rank bytes | HLO class | Comment |
|---|---|---|---|
| psi_l_X parameter | 36 · 376 · 150 · 2 · 16 / p_x=4 → wait, replicated | parameter | Sharding makes μ local; per-rank holds full nb on its μ slab. Today: 61.96 MB unsharded. |
| psi_r_X parameter | 66.09 MB unsharded | parameter | Same. |
| L_q parameter | 77.66 MB unsharded | parameter | Same. |
| Output Z_q (`c128[36, 376, 18412]` per rank) | 3.71 GiB | maybe-live-out | Same as today |
| Constants (phase tables, g_index baked) | ~150 MB | constant | Closure |
| **Persistent total** | **~4.1 GiB** | | Roughly matches today's non-temp footprint |

### 2b. Carry — the load-bearing question

Two `c128` accumulators, both rank-5, live across all scan iters.
Shape inside the body:

```
P_l_acc : c128[nk_loc, ns, r_loc, μ_loc, ns]
P_r_acc : c128[nk_loc, ns, r_loc, μ_loc, ns]
```

Under today's sharding (`μ_loc=376, r_loc=18412, nk_loc=36, ns=2`):

```
P_l_acc per rank = 36 · 2 · 18412 · 376 · 2 · 16 = 15.94 GB = 14.85 GiB
```

**Two carries × 14.85 GiB = 29.70 GiB per rank.**

This is the dominant cost in the new design. It's already a 15 GiB win
vs today's 44.56 GiB (which holds three slots of the same shape, all
live simultaneously — two of which are actually `psi_Y` rematerializations
in disguise).

### 2c. Per-iter transient (FFT box) — the second load-bearing question

Inside each scan iter, `to_rchunk_inner` runs an IFFT on a box of shape
that depends on what we put into the scan. Three structural choices:

#### Option α — bc-only scan with full `band_chunk` per iter

```python
def body(carry, bc_idx):
    psi_G_bc = io_callback(..., shape=(nk, bpd_max, ns, ngkmax))
    psi_Y_bc = to_rchunk_inner(psi_G_bc, ...)   # FFT box (nk, bpd_max, ns, nx, ny, nz)
    ...
```

FFT box per rank: `c128[nk, bpd_max, ns, nx·ny·nz]` × cuFFT scratch:
`36 · 16 · 2 · 1.125M · 16 = 20.74 GB` bare; **× 4 cuFFT factor = ~80 GB.
NOT FEASIBLE.**

This is the option agent_2_structural_fix.md §4c sketches but §4e bullet
acknowledges needs an inner k-loop at CrI3 scale.

#### Option β — nested scan over k_chunk inside bc

```python
def bc_body(carry, bc_idx):
    psi_G_bc = io_callback(...)
    def k_body(carry_k, k_idx):
        psi_G_bck = psi_G_bc[k_idx*k_chunk:(k_idx+1)*k_chunk]
        psi_Y_bck = to_rchunk_inner(psi_G_bck, ...)
        # einsum into carry_k for this (bc, k_chunk) sub-block
        ...
    lax.scan(k_body, ..., jnp.arange(nk // k_chunk))
    ...
```

Per inner iter: `c128[k_chunk, bpd_max, ns, n_rtot] × 4`. At k_chunk=6,
bpd_max=16: `6·16·2·1.125M·16 = 3.46 GB`, ×4 = 13.8 GB. **One slot, aliased
across both scans.**

Two nested scans — agent 2 §4e proposes exactly this for CrI3 scale.

#### Option γ — flat-axis scan over (k·b) with cs (mirrors `accumulate_rchunk_to_gflat`)

```python
def body(carry, i):
    sub = jax.lax.dynamic_slice_in_dim(psi_G_flat, i*cs, cs, axis=0)
    # sub: (cs, ns, ngkmax)
    box = _box_kernel(sub.reshape(cs, 1, ns, ngkmax), ...)  # (cs, 1, ns, nx, ny, nz)
    psi_Y_sub = ifft + slab + phase   # (cs, ns, r_loc)
    # einsum into carry; bookkeeping for which (k, b) each row belongs to
    ...
```

This is the **exact pattern of `accumulate_rchunk_to_gflat` and
`gflat_to_rchunk`**: flat-axis scan with chunk size `cs`, FFT box
`c128[cs, ns, n_rtot]` per iter aliased to one slot. At cs=8: `8·2·1.125M·16
= 288 MB × 4 = 1.15 GB`. At cs=16: 2.3 GB. At cs=64: 9.2 GB.

The complication: with a flat (k·b) axis, the per-row L/R band-window
membership is a function of `(k, b) = (row // n_b, row % n_b)`. Doable
with `q_row`/`k_row` decoding tables (same as `accumulate_rchunk_to_gflat`'s
`q_row`). NOT structurally harder than today's reverse helper.

### 2d. Predicted total preallocated-temp under each option

| Option | Carry | FFT-box slot | Other transients | **Total per-rank** |
|---|---|---|---|---|
| α (bc-only) | 29.70 GiB | 80 GiB ✗ | – | **infeasible** |
| β (nested k-scan, k_chunk=6) | 29.70 GiB | 13.80 GiB | ~3 GiB | **~46 GiB** |
| γ (flat-axis, cs=16) | 29.70 GiB | 2.30 GiB | ~2 GiB | **~34 GiB** |
| γ (flat-axis, cs=8) | 29.70 GiB | 1.15 GiB | ~2 GiB | **~33 GiB** |

Versus today's 48.63 GiB total. Option γ at small cs gives the cleanest win.

### 2e. Where psi_Y_full goes

**Gone.** In all three options above, `psi_Y_full` (the band-full r-chunk
tensor) is never materialized as a single array. Each scan iter produces
a small `psi_Y_sub` that's contracted into the accumulator on the same
iter and freed (aliased) on the next. Slot B (14.85 GiB) and slot D
(12.07 GiB) in today's HLO both disappear — **27 GiB of unsharded
materialization eliminated.**

### 2f. Where the remat warning goes

**Gone.** The remat is caused by a slice + reshard at the boundary between
the band-flat-sharded `gflat_to_rchunk` output and the r-sharded
consumer. In the new design, no such boundary exists — `psi_Y_bc` is
produced and consumed within the same shard_map body, both per-rank-local.

## 3. BufferAssignment behavior on scan carries (prompt §2)

XLA's BufferAssignment treats `lax.scan` as a `WhileOp` with a state
tuple. For each element of the state, BA chooses between:

- **Reuse** the input slot for the output (in-place update). Requires the
  lifetimes to align — which they do in `lax.scan`'s standard lowering.
- **Allocate** a separate output slot, copy. Wasteful; only happens when
  BA can't prove safety.

For our carry `(P_l_acc, P_r_acc)`:
- Each accumulator is read at the start of each iter, written at the end.
- XLA recognizes this pattern and aliases input→output of the WhileOp.
- **One slot per accumulator, lives across all iters. No double-buffering.**

This is confirmed by `accumulate_rchunk_to_gflat`'s `acc_flat` carry —
Round 4 HLO shows a single 14.85 GiB-class slot for the carry, not two.
Agent 1 §2 makes the same call. We agree.

**Could they be aliased together (1 slot instead of 2)?** No, because both
accumulators are read+written in every iter — their lifetimes fully
overlap inside an iter. Two slots, both alive.

## 4. FFT box aliasing inside scan (prompt §3)

The per-iter FFT box has a clear "born and die within one iter" lifetime:
allocate inside body, consume in einsum, no carry. XLA's scan-internal
allocator aliases this to a single transient slot across iters. This is
the **same mechanism** that gives `accumulate_rchunk_to_gflat` its
single FFT-box slot (Round 4 confirmed: 1 slot, not n_chunks).

### Does io_callback disrupt the aliasing?

**Probably not.** `io_callback` returns a fresh array each call; the
result is used inside the body and falls out of scope at iter end. The
allocator should treat it like any other per-iter intermediate.

**But it's novel composition.** Agent 1 §9 flags this as the highest-risk
unknown. I concur. The validation gate (§7) must include a CPU reproducer:
a tiny scan-inside-shard_map with io_callback that confirms the FFT box
gets one slot, not n_iter slots.

### Risk: `ordered=True` may force serial dispatch but not allocation

`io_callback(..., ordered=True)` ensures host calls happen in scan order.
That's about *execution order*, not *allocation*. Aliasing should still
work — but a sloppy implementation that captures the io_callback return
in a Python list (e.g., for debug) would defeat the aliasing. Agent 2:
verify the design doesn't have any Python-side accumulation.

## 5. The "extra remat" question (prompt §4)

In the new design:

1. Inputs `psi_l_X, psi_r_X` are sharded `P(None, 'x', None, None)`. They
   feed the einsum directly inside the body — same per-rank layout as
   today. ✅ No reshard.
2. Closure tables (g_index, phase, band masks) are baked as constants.
   Replicated on every rank by closure semantics; no resharding involved.
3. The carry `(P_l_acc, P_r_acc)` lives entirely inside the body — no
   resharding.
4. The post-scan tail (IFFT(k) → conj → γ̃ → FFT(k)) runs in the same
   shard_map body. It already lives there in today's `_local`
   (`isdf_fitting.py:325-352`); the new design adds the scan UPSTREAM
   of it, doesn't touch the tail. ✅ No new reshard.
5. The body returns Z_q with shape `(nq, μ_loc, r_loc)` per rank — exactly
   what `out_specs=P(None, 'x', 'y')` expects.

**No remat triggers I can identify.** If I had to name a potential
hazard: if the planner ever picks an r_chunk that doesn't divide p_y
(or μ_total that doesn't divide p_x), the boundary will reshard at
return. But `gflat_memory_model.plan_gflat_chunks` already rounds
r_chunk to a multiple of `p_xy` (line 340), so this is enforced.

## 6. cuFFT batching (prompt §5)

Per-iter FFT batch size determines cuFFT's efficiency. cuFFT amortizes
plan setup over the batch; a batch of 1 is much slower per-FFT than a
batch of 256.

| Option | Per-iter FFT batch | FFT count per scan run |
|---|---|---|
| α (bc-only, infeasible) | 36·16·2 = 1152 | 20 iters × 1152 = 23 040 |
| β (nested k-scan) | 6·16·2 = 192 | (36/6) × 20 × 192 = 23 040 |
| γ (flat-axis, cs=8) | 8·2 = 16 | (360/8) × 16 = 720 |
| γ (flat-axis, cs=16) | 16·2 = 32 | (360/16) × 32 = 720 |
| γ (flat-axis, cs=64) | 64·2 = 128 | (360/64) × 128 = 720 |

Same total FFT work for γ across cs values. Per-call batch size 16 vs 32
vs 128 — all comfortably above cuFFT's setup-overhead break-even (~10–20).

**Compared to today's `gflat_to_rchunk` one-shot**: today's call runs
one batch of `360·2 = 720` FFTs. New design runs (360/cs) batches of `cs·2`
FFTs each. Total FFT work identical. Per-FFT cost slightly higher (more
plan setups) but ≤2× even at the smallest cs.

**Per-FFT wall-clock penalty**: for n_rtot=1.125M FFTs, plan-setup ~10–50 μs,
FFT compute ~5–15 ms. Setup overhead is ~0.3% of compute — negligible
unless cs is set to 1 (don't).

**Verdict**: γ at cs=8–16 is fine for performance. β at k_chunk=6 is also
fine. Neither hits a cuFFT efficiency wall.

## 7. Compile-time prediction (prompt §6)

Today's `_kernel` jit compiles in ~3 min at CrI3 80 Ry. New design adds:

- A `lax.scan` (option γ) or two nested `lax.scan`s (option β) inside the
  shard_map body.
- An `io_callback` per iter.
- More HLO ops in the body (per-row decode, mask construction).

XLA compile time scales roughly with **HLO op count** and **module
complexity**. A single nested scan adds a few WhileOp lowering passes;
not non-linear in the inputs.

**Predicted**: <2× today's compile time. Concrete prediction: 4–6 minutes
at CrI3 80 Ry, vs today's 3 minutes.

**Hazards to avoid**:
- Don't pass `unroll=` to `lax.scan`. Unrolling at n_bc≈20 creates ~20×
  the HLO ops in the body. Even `unroll=2` doubles HLO size.
- Don't put `jax.jit` around the scan body. The scan must lower as a
  WhileOp in the same module as the shard_map; nesting jits creates
  separate modules and breaks the buffer aliasing across the boundary.

## 8. Where I disagree with Agent 1

Agent 1 §2 §3 predicts "2 GB total carry per rank" with `μ_loc=94,
r_loc=4603`. Those numbers require:

- μ sharded on the **combined `('x','y')`** mesh axis (μ_loc = 1504/16 = 94).
  Today's `psi_l_X` is `P(None, 'x', None, None)` — μ on `'x'` only,
  μ_loc = 376.
- r per-rank = 4603, which requires r sharded on the **combined `('x','y')`**
  mesh (r_chunk/p_xy = 73648/16 = 4603). Today's `out_spec=P(None, 'x', 'y')`
  shards r on `'y'` only, r_loc = 18412.

**Both** are sharding changes beyond what `agent_2_structural_fix.md` §4c
proposed. They're valid choices — and if adopted, the carry shrinks
~16× (29.70 GiB → 1.86 GiB) — but they require:

1. **Upstream change**: the centroid loader must produce `psi_l_X` with
   `P(None, ('x','y'), None, None)` sharding. Currently it produces
   `P(None, 'x', None, None)` (μ on `'x'`).
2. **Output spec change**: `out_specs` becomes either
   `P(None, ('x','y'), None)` (μ on full mesh, r replicated) or
   `P(None, ('x','y'), ('x','y'))` (μ AND r on full mesh, but axis names
   conflict — not allowed). The clean form is to make μ sharded on full
   mesh and r replicated, or μ on `'x'` and r on `'y'` (today's form).
3. **Einsum sharding analysis**: `'kmna,knbr->karmb'` with μ on full mesh
   shards the einsum on m — that's fine, m only appears in input1 and
   output, so no cross-rank reduction. But the inputs `psi_l_X` (μ-sharded)
   and `psi_l_Y_bc` (band-replicated, r-replicated?) must coordinate.

**My recommendation**: surface this as a primary design decision in the
unified plan. Two paths forward:

- **Path R5-A (conservative)**: keep today's sharding `P(None, 'x',
  None, None)` for `psi_l_X`. Carry per-rank = 29.70 GiB. Total preallocated
  ≈ 33–46 GiB depending on FFT-box option. **Modest 5–15 GiB win, low
  risk, mechanical refactor.**
- **Path R5-B (aggressive)**: change μ sharding to `('x','y')`-combined,
  reduce carry per-rank to 1.86 GiB. Total preallocated ≈ 6–10 GiB.
  **Massive ~40 GiB win, but requires upstream centroid-loader rework
  + einsum sharding verification.**

Either is consistent with the zero-replicated-intermediates principle.
Path R5-B is structurally cleaner (more sharding = less replication).
Path R5-A is the smaller refactor.

**This is the highest-value design decision Round 5 should make.** I lean
toward Path R5-B because the principle says "more sharding," but I want
Agent 1 to confirm the einsum sharding remains sound under combined-axis
μ sharding before committing.

## 9. Predicted slot count summary

For Option γ + Path R5-A (conservative):

| Slot | Size | Lifetime | Notes |
|---|---|---|---|
| P_l_acc carry | 14.85 GiB | all scan iters + tail | Aliased input→output |
| P_r_acc carry | 14.85 GiB | all scan iters + tail | Aliased input→output |
| FFT box (per-iter) | ~2.3 GiB at cs=16 | one scan iter | Aliased across iters by scan allocator |
| psi_G_bc (per-iter, from io_callback) | ~1.1 GiB | one scan iter | Aliased across iters |
| Post-pair scratch (γ̃ contract, FFT) | ~3.7 GiB | tail only | Reused from old `_local` |
| Output Z_q | 3.71 GiB | live-out | Same as today |
| **Predicted total** | **~37 GiB** | | vs today's 48.63 GiB |

For Path R5-B (aggressive):

| Slot | Size | Lifetime |
|---|---|---|
| P_l_acc carry | 0.93 GiB | all iters |
| P_r_acc carry | 0.93 GiB | all iters |
| FFT box (per-iter) | ~1–2 GiB | one iter |
| psi_G_bc | ~70 MB | one iter (smaller because k may be sharded too) |
| Post-pair scratch | ~0.5 GiB | tail |
| Output | 0.23 GiB | live-out |
| **Predicted total** | **~5 GiB** | |

## 10. Validation gates from the HLO lens

For the implementation in Round 6, the HLO acceptance gates are:

1. **Slot count for FFT box inside scan body = 1.** Dump
   `module_NNNN.jit__kernel.*-memory-usage-report.txt` and count
   `c128[*, *, n_rtot]` or `c128[*, *, nx, ny, nz]` slots; should be 1
   (or aliased within one slot via offset listing).
2. **No `c128[..., 73648]` unsharded slots.** psi_Y_full must be gone.
3. **Zero `Involuntary full rematerialization` warnings in gw.out.** The
   remat boundary must be eliminated.
4. **Scan WhileOp present in HLO.** Confirm via
   `grep -c "while " module_NNNN.jit__kernel.*ir-after-optimization*`
   — should be exactly 1 (for option γ) or 2 (for option β nested).
5. **Carry size matches prediction.** Per Path R5-A: two slots of
   14.85 GiB each; per Path R5-B: two slots of ~0.93 GiB each.

If any of these fails, the design assumption is wrong and we should not
ship.

## 11. Bottom line for the unified plan

The new design is structurally sound at the HLO/BufferAssignment level.
Three load-bearing design decisions remain:

1. **Carry sharding** — Path R5-A (conservative) vs R5-B (aggressive).
   Affects total memory by ~30 GiB. Requires Agent 1 sign-off on einsum
   sharding under R5-B.
2. **FFT-box-inside-body sizing** — Options α/β/γ. Option α is
   infeasible at CrI3 scale; β (nested k-scan) and γ (flat-axis cs)
   both work. γ is structurally simpler (mirrors `accumulate_rchunk_to_gflat`).
3. **io_callback-in-scan-in-shard_map composition** — novel; reproducer-first
   non-negotiable. Agent 2 owns the host-side details.

If we pick R5-B + γ (cs=16), the predicted total is ~6 GiB per rank —
**a 8× memory reduction from today**, with no remats and no replicated
intermediates. That's the version of the design that fully honors the
zero-replicated-intermediates principle.

If we settle for R5-A + γ (cs=16), it's ~37 GiB per rank — a 1.3× reduction,
mechanically simple, and still removes the remat. Acceptable as a Round 6
target with R5-B as a Round 7 follow-up.

## 12. Round-2 correction (post Agent 1 retraction-and-expansion)

**Retraction.** §2b above used `n_rmu = 1504` (from gw.out's "ISDF basis: 1504 centroids"). The actual `n_rmu` inside this kernel is **376**, verified from the HLO parameter shapes:
- `c128[36, 376, 376]` = L_q at `(nq, n_rmu, n_rmu)` ⇒ n_rmu = 376
- `c128[36, 376, 160, 2]` = psi_r_X at `(nk, n_rmu, nb_R, ns)` ⇒ same

The 1504 in gw.out is the centroid count before some upstream reduction or padding — possibly the `n_rmu_padded` field for a different stage. Inside `z_q_from_psi_sm`'s kernel, `n_rmu = 376`.

**Updated carry math under today's `P(None, 'x', None, None)` for psi_l_X** (p_x=4, p_y=4):
- `μ_loc = 376 / p_x = 94`
- `r_loc = 73648 / p_y = 18412`
- Per-rank `P_l_acc` shape: `c128[36, 2, 18412, 94, 2]` = 36·2·18412·94·2·16 = **3.98 GB ≈ 3.71 GiB per accumulator**
- Two carries simultaneously live ≈ **7.42 GiB**

So Agent 1's 999 MB-per-carry math was right (the only mistake was using `r_loc = 4603` instead of 18412 — a 4× error). Combined: per-rank carry ≈ 3.71 GiB each, 7.42 GiB total, **not 1 GB nor 14.85 GiB.**

### 12a. The host-side band-sharding complication (Agent 1's catch)

`_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)` — host tiles shard bands across the FULL mesh. Per rank holds `bpd_per_bc = bc_size / p_xy = 16/16 = 1` band per bc at CrI3. But the einsum needs all 16 bands per rank.

Fix: per-iter `all_gather` over `('x','y')` on the band axis, applied AFTER per-rank IFFT-and-slice. Sequence:

```
io_callback → c128[36, 1, 2, 59990]    # 80 MB per rank, fetched from host
IFFT + slice → c128[36, 1, 2, 18412]    # 21 MB per rank
all_gather(tiled, axis=band) → c128[36, 16, 2, 18412]   # 340 MB per rank
einsum into P_l_acc carry              # contracts band axis to 0
```

The all_gather payload (340 MB per rank) is small relative to the FFT box already in flight (~1.3 GiB raw, ~5 GiB with cuFFT scratch). It's a minor "intentional replication" — a 340 MB per-iter replicated buffer — but contractually unavoidable given the upstream sharding choice, and dwarfed by the FFT box it accompanies.

**Order matters**: do the IFFT BEFORE the all_gather, not after. Doing it after would mean each rank IFFTs the full 16 bands → 20 GB FFT box per rank. Doing it before keeps the FFT box at 5 GiB per rank (per-rank 1 band batch).

### 12b. Updated total prediction (Agent 1's serialized scan design)

Two scans run back-to-back: first computes `P_l`, IFFTs and conjugates it, then second computes `P_r` and IFFTs. The γ̃ contract needs both `P_l_R_conj` and `P_r_R` simultaneously.

| Item | Per-rank | Lifetime |
|---|---|---|
| One live carry during scan (`P_l_acc` then `P_r_acc`) | 3.71 GiB | Scan iters; serialized so only 1 live at a time |
| Per-iter FFT box (aliased across iters and across both scans) | ~5 GiB | Scan transient |
| Per-iter all_gather result (aliased) | ~340 MB | Scan transient |
| Per-iter `psi_G_bc` (aliased) | 80 MB | Scan transient |
| Peak — γ̃ contract holds `P_l_R_conj` AND `P_r_R` | 7.42 GiB | Tail |
| Output Z_q | ~0.99 GiB | live-out |
| Parameters (L_q, psi_l_X, psi_r_X) | ~0.27 GiB | parameter |
| Constants (phase, g_index, band tables) | ~0.15 GiB | constant |
| **Predicted total preallocated** | **~13–15 GiB** | vs today's 48.63 GiB |

**~3× memory reduction**, well below the 28 GiB nominal per-A100 budget. Under cohsex.in's 60 GiB setting, plenty of headroom for downstream solver buffers.

### 12c. New slot-count prediction

For the kernel `_local`:

| Slot | Size | Lifetime | Aliasable? |
|---|---|---|---|
| Carry (live during scan) | 3.71 GiB | one full scan | One slot, in-place updated |
| `P_l_R_conj` (post-scan-1) | 3.71 GiB | until γ̃ contract done | One slot |
| `P_r_R` (post-scan-2) | 3.71 GiB | only during γ̃ contract | One slot, possibly aliased with γ̃ scratch |
| FFT box (per-iter) | ~5 GiB | one scan iter | One slot, aliased across iters AND across both scans |
| Output Z_q | 0.99 GiB | live-out | n/a |

**Substantive slots: 4–5.** Far below today's 4 large slots totaling 44.56 GiB. The dominant cost moves from "3 P_pair concurrent slots" to "FFT box at ~5 GiB + 2 carry-class buffers at 3.71 GiB each at peak."

### 12d. Implications for the planner

Once this design lands, `_peak_C_fit_one_rchunk` in `gflat_memory_model.py` needs:
- Drop the `pair_density_slots = 3` term (no longer 3 concurrent; serialized scans + γ̃ contract gives 2 concurrent at peak).
- Add an `fft_box_in_scan` term: `nk · bpd_per_bc · ns · n_rtot · cuFFT_factor / 1` (unsharded per rank because bpd_per_bc=1 already came from the mesh shard).
- Add an `all_gather_band_slab` term: small, ~340 MB; could omit.

The bookkeeping fixes from my Round 4 analysis (Peak C centroids `nk → nb_total`, Peak B centroids `ns → nb_total`, Peak D `fft_box_factor=4 → 1`) are still required. They'd combine with the Path-D structural changes in the same commit.

### 12e. Risk assessment: all_gather inside scan inside shard_map

Three-level nesting that's novel for this repo:
- shard_map manual-mode body
- `lax.scan` (WhileOp lowering)
- `jax.lax.all_gather` inside the scan body

JAX supports `all_gather` inside shard_map's manual mode (it's the canonical primitive). Inside a scan body, `all_gather` lowers to an HLO `all-gather` op in the WhileOp body. **Should work** but I don't have direct repo precedent — Agent 1 says they haven't seen it either.

**Reproducer required.** A 50-line CPU test with a small scan body containing `all_gather` on a small array, verifying:
1. The compile succeeds.
2. The runtime produces correct values.
3. The HLO shows one `all-gather` inside the WhileOp body (not lifted out, not duplicated per iter).

This is a hard requirement before kernel rewrite. If it fails, the fallback is psum-on-carry (with a larger reduction at scan exit — same final memory but more wall-clock).

Agent 3 round 5 ready
