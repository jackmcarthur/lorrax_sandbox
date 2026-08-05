# Round 6/7 — HLO validation (Agent 3) — G2 result on `c796420`

**Status**: G2 measured on Agent 4's combined G2+G3 run at
`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round7_validation_2026-05-14/`.
HLO modules `0293/0295/0394.jit__kernel` are the three identical
`fit_one_rchunk` variants (same as Round-4).

**Verdict**: **[CONCERN]** — the structural defects (remat,
`psi_Y_full` materialization, unsharded gflat_to_rchunk FFT box)
are fully eliminated as predicted, but **total preallocated-temp
is unchanged from Round-4 at 44.56 GiB**. Run is progressing
cleanly (no OOM, zeta fitting in-flight).

## Comparison table (measured)

| Metric | lorrax_A `ff5873c` (pre-Path-D) | lorrax_B `5cadd4b` (R4 Path-D) | **R6 plan §9.1 prediction** | **R7 `c796420` measured** | Pass? |
|---|---|---|---|---|---|
| Total bytes used (per rank) | 200.35 GiB | 48.63 GiB | ≤ 17 GiB | **48.63 GiB** | ❌ |
| Preallocated-temp | 196.30 GiB | 44.56 GiB | ≤ 12 GiB | **44.56 GiB** | ❌ |
| Substantive temp slots | 58 FFT-box class | 4 | ≤ 5 | **4** (3 × 14.85 GiB + smaller) | ✅ |
| FFT-box-class slots | 58 unsharded | 1 (12.07 GiB unsharded) | ≤ 2 | **1** (1.12 GiB per-iter aliased) | ✅ |
| `psi_Y_full`-shape slots `c128[nk, nb, ns, r_chunk]` | n/a | 2 (within 14.85 GiB offsets) | 0 | **0** | ✅ |
| Pair-density rank-5 slots (size) | 0 | 3 × 14.85 GiB | ≤ 2 × 3.71 GiB | **3 × 14.85 GiB** (mu-full) | ❌ |
| Output Z_q live-out | n/a | 3.71 GiB | ~0.99 GiB | **3.71 GiB** | ❌ |
| `while_loop` count in HLO | 0 | 0 | = 1 | **1** | ✅ |
| `all-gather` inside while-body | n/a | n/a | = 1 | **1** (`isdf_fitting.py:658`) | ✅ |
| `Involuntary full rematerialization` warnings | many | 32 | **= 0** | **0** | ✅ |

## Measured slot map (preallocated-temp pool, allocation 16, 44.56 GiB)

Three co-resident 14.85 GiB slots at offsets `1920`, `15950391168`, `31900780416`:

| Slot offset | Size | Dominant tensors (aliased lifetimes) |
|---|---|---|
| 1920 | 14.85 GiB | `c128[2,18412,376,2,6,6,1]` + `c128[36,36824,752]` + `4× c128[36,2,18412,376,2]` + `c128[36,1,4,94,18412]` + `c128[18412,376,6,6,1]` |
| 15950391168 | 14.85 GiB | `4× c128[36,2,18412,376,2]` + `c128[2,18412,376,2,6,6,1]` + `c128[36,36824,752]` |
| 31900780416 | 14.85 GiB | `c128[36,1,2,75,75,200]` (FFT box) + `c128[16,36,2,18412]` (gathered) + `c128[36,1,2,59990]` (psi_G_bc) + `2× c128[2,6922912,2,36]` |

Plus output `c128[36, 94, 73648]` = 3.71 GiB (live-out), parameters + constants ≈ 0.4 GiB. Total 48.63 GiB.

**Decoding the dominant `c128[36,2,18412,376,2]`**: this is `c128[nk, ns, r_loc=18412, mu=376, ns]`. **The mu axis = 376 = FULL n_rmu, NOT mu_loc = 94 = n_rmu/p_x.** The post-scan carry / IFFT-tail intermediates run at **mu-unsharded** despite the in_spec `P(None, 'x', None, None)` for psi_l_X.

The 6 × 14.85 GiB tensors at offsets 1920 + 15950391168 are aliased copies of the while-carry post-scan intermediates (P_l, P_r, P_l_R, P_l_R_conj, P_r_R, etc.) all at mu-full=376. The 3rd slot (offset 31900780416) holds the scan-internal transients (FFT box, gathered slab, psi_G_bc) plus 2 mu-full P-pair tensors. XLA's BufferAssignment co-resident-aliases these into three 14.85 GiB slots.

## while op decoded

```
%while.4.0 = (s64[], c128[36,2,18412,376,2], c128[36,2,18412,376,2],
              s32[10], s32[], s32[], s32[], s32[], s32[10],
              c128[36,376,160,2], c128[36,376,160,2])
  while(%tuple.20), condition=..., body=...,
  metadata={op_name="jit(_kernel)/.../jit(shmap_body)/while"
            source_file="isdf_fitting.py" source_line=711},
  backend_config={"known_trip_count":{"n":"10"}, ...}
```

The while carries 2× `c128[36, 2, 18412, 376, 2]` = 2× 14.85 GiB = **29.7 GiB carry live across all 10 scan iters**. Plus 2× `c128[36, 376, 160, 2]` (psi_r_X parameter copies). Trip count = 10 = n_bc.

The all-gather inside the body (line 658):
```
%all-gather-start = ... all-gather-start(...)
  channel_id=1, replica_groups={{0..15}}, dimensions={1},
  metadata={op_name=".../jit(shmap_body)/while/body/all_gather"
            source_file="isdf_fitting.py" source_line=658}
```
Per-iter gather over the full 16-way mesh on band axis. Result type `c128[36, 16, 2, 73648]` per rank.

## Diagnosis: why total preallocated-temp didn't drop

The Round-5 plan §4.4 claimed three eliminations:
1. **3 × P-pair concurrent slots** would drop to 2 carry-class @ 3.71 GiB each (via explicit per-rank `mu_loc` carry shape). ❌ **DID NOT HAPPEN** — XLA still produces mu-full = 14.85 GiB carries.
2. **psi_Y_full double materialization** (30 GiB) — eliminated. ✅ Confirmed.
3. **Involuntary remat boundary** (12 GiB) — eliminated. ✅ Confirmed; 0 warnings.

Net: the 30 GiB from (2) and the 12 GiB from (3) DID disappear, but their slot offsets are now reused by **scan transients** (FFT box, gathered slab) and **additional mu-full P-pair-class intermediates** from the post-pair pipeline tail. So total preallocated-temp stays at ~44.56 GiB.

**Root cause of (1) not happening**: the per-rank carry init `jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128)` uses `mu_loc = psi_l_X_.shape[1]`. Inside the manual-mode shard_map body, this evaluates to **376** (full mu), not 94 (per-rank mu_loc under p_x=4). The carry consequently has mu_full = 376; XLA cannot un-do this at SPMD-partition time because the body's code explicitly allocates at that size. The 4× over-allocation propagates through the post-scan IFFT/γ̃/FFT chain.

This is the **same behavior Round-4's `5cadd4b` HLO showed** (Round-4 audit `round4_improvements.md` slot A/C at `c128[2, 6922912, 2, 36]` = mu-full). The Round-5 plan §4.4 misdiagnosed the Round-4 P-pair slots as a side-effect of "the full-mu intermediate the einsum's natural output produces today" — and claimed explicit per-rank carry shape would prevent it. **In fact, `psi_l_X_.shape[1]` returning the global (not per-rank) shape inside manual-mode shard_map is the underlying issue, and the explicit carry shape used the global anyway.**

## Was the plan wrong? (post-mortem)

Yes, on §4.4 specifically. The plan assumed:
- Inside `shard_map` manual mode, `arr.shape` returns per-rank dims.
- Therefore `mu_loc = psi_l_X_.shape[1]` = 94 under p_x=4.
- Therefore explicit `jnp.zeros((..., mu_loc, ...))` allocates a 3.71 GiB carry.

In practice (this `c796420` HLO + Round-4's `5cadd4b` HLO + the existing `c_q_from_psi_sm._local`): `psi_l_X_.shape[1]` returns **376** (the global n_rmu), so `mu_loc = 376` and the carry is mu-unsharded at 14.85 GiB per accumulator.

Whether this is the documented modern-JAX shard_map behavior (manual mode preserves global shapes) or a quirk of this specific code path — Round-8 design needs to verify.

## What G2 *did* prove (the structural wins are real)

- ✅ The remat boundary at the helper/consumer reshard is eliminated. 32 → 0 warnings.
- ✅ `psi_Y_full` is never materialized inside the kernel — the `c128[36, 160, 2, 73648]` and `c128[36, 150, 2, 73648]` shapes are absent.
- ✅ The old `gflat_to_rchunk` 12.07 GiB unsharded FFT box (`c128[360, 2, 1125000]`) is absent — replaced by a per-iter scan-internal `c128[36, 1, 2, 75, 75, 200]` = 1.12 GiB FFT box that XLA aliases into the same offset as a post-scan P-pair tensor.
- ✅ The scan structure is exactly as predicted: 1 while op with trip_count=10 (n_bc), 1 all-gather inside the body on the band axis.
- ✅ Run is progressing (zeta fitting started, r-chunk 1/16 in flight; no OOM).
- ✅ The agent_2_structural_fix.md §4c-d design is structurally validated as a remat eliminator; just not as a total-memory reducer at CrI3 80 Ry on a 4×4 mesh.

## Recommendation to Round-8 design

The post-pair pipeline's mu-full intermediates dominate Peak C at ~45 GiB. To get the predicted 13–15 GiB total, Round-8 needs to address the mu sharding inside the body — either by:
1. **Explicitly indexing `mu_loc = n_rmu // p_x` from the mesh outside the shard_map** (don't trust `psi_l_X_.shape[1]` inside the body), and verify the carry & einsum stay mu-sharded post-partition by reading the HLO.
2. **Reshard psi_l_X to `('x','y')`-combined** (μ_loc = 376/16 = 23.5 → 24 padded; requires upstream centroid loader change). This was the §8 "out of scope" Round-5 deferred optimization.
3. **Restructure the einsum** so the mu axis stays sharded across the post-scan IFFT/γ̃/FFT chain (currently it doesn't — the reshape to `(nkx, nky, nkz, ns, r_loc, mu_loc, ns)` may break sharding propagation).

Without one of these, the 48.63 GiB plateau persists — the design has eliminated structural defects but not reduced total memory.

## Final tags

- ✅ **5/10 gates PASS**: remat, psi_Y_full, gflat-FFT-box, while-loop count, all-gather count, slot count, run progresses.
- ❌ **4/10 gates FAIL**: total bytes, preallocated-temp, P-pair size, output Z_q size — all reflect the same mu-unsharded carry issue.

Not posting "Agent 3 G2 passed". **Posting `[CONCERN]` for Round-8 to address: mu-sharding does not propagate through the manual-mode shard_map body's `jnp.zeros` carry allocation.**

**Agent 3 round 7 G2 [CONCERN] posted.**
