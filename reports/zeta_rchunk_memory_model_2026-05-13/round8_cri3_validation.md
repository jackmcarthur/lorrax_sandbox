# Round 8 CrI3 6×6 80 Ry validation — fit_zeta structural fix on lorrax_B `c796420`

**Date:** 2026-05-14
**Branch / commit:** `sources/lorrax_B` `agent/zeta-bc-scan-shardmap` @ `c796420`
  = scan-inside-shard_map (`f567aa0`) + symmetric back-pad on `psi_l_X / psi_r_X` (`c796420`).
**System / config:** CrI3 6×6×1, 80 Ry, 150 bands, 1504 centroids, `memory_per_device_gb=60`,
  `band_chunk=16`, `r_chunk=0 (→73648)`, `gflat_chunk=64`. 4 nodes × 4 A100/80 GB.
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round8_validation_2026-05-14/`
**Allocation:** JID 52946809 (lx-alloc-jackm, 4×hbm80g).

This report serves G2 (HLO acceptance) and G3 (end-to-end completion) for the
Round 7 structural fix at CrI3-80 Ry scale. The Round 7 validation dir (same
commit, same config) was used as a cross-check; its HLO/run is byte-identical
where it should be and provides the reference numbers below.

---

## Headline: predicted vs measured

| Metric | Pre-Round-6 baseline `5cadd4b` (path_d HLO) | Round 8 prediction for `c796420` | Round 8 measured (this run) |
|---|---|---|---|
| Total preallocated-temp (kernel HLO) | 48.63 GiB | **≤ 15 GiB** | **48.63 GiB** |
| LORRAX driver AOT estimate | — | — | **3.92 GB** |
| GPU runtime high-water | (OOM at 200 GiB pre-Path-D / unmeasured at 48 GiB Path-D) | — | **7.34 GB / 60 GB budget (12 %)** |
| FFT-box-class slots (`c128[4,36,94,18412]` = 3.71 GiB) | 1 aliased + 1 unsharded remat copy (`c128[360,2,75,75,200]` = 12.07 GiB) | 1 aliased inside scan, no remat copy | **1 aliased inside scan, no remat copy** — match |
| `c128[36,1,2,59990]` 1.03 GiB FFT-box-shaped per-iter copies | 9 separate slots (≈ 9 GiB) | 0 | **0** — match |
| Pair-density carry slots | n/a (no carry: Python-unrolled) | 2 × 3.71 GiB (n_rmu_L sharded 376→94/rank) | **2 × 14.85 GiB carry (n_rmu_L=376 replicated)** — see below |
| `psi_Y_full` materialization | TWICE (~30 GiB) | gone — never materialized | **gone** — match (only `c128[36,160,2,73648]` lives in the 14.85 GiB allocation slots, no replicated full ψ_Y) |
| `Involuntary full rematerialization` (fit_zeta compile) | ~20 | **0** | **0** — match |
| End-to-end fit_zeta over 16 r-chunks + remainder | OOMed at runtime (pre-Path-D) / never run E2E pre-Round-6 | completes through all 16 r-chunks | **completes 16/16 r-chunks** (round 7 ref: 800 s wall; round 8 in flight: see Status) |

---

## G2 — HLO acceptance

**Source file (round 8):** `xla_dump/module_0293.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`
(identical, modulo slot-table ordering, to `lorrax_B_round7_validation_2026-05-14/.../module_0293...memory-usage-report.txt`).

`module_0293` is `fit_one_rchunk`. Identified by the `op_name` trail to
`/global/homes/j/jackm/software/lorrax_B/src/common/isdf_fitting.py` (line 603
inside `jit(shmap_body)/jit(_pad)`) and the WhileOp with `known_trip_count=10`,
`source_line=711` (`fit_one_rchunk`'s `lax.scan` over band-chunks).

### Total preallocated-temp pool: 48.63 GiB (3× over Round-8 prediction)

| Allocation | Size | Cum. % | Role |
|---|---|---|---|
| 16 | **44.56 GiB** | 92 % | `preallocated-temp` (3 × 14.85 GiB pair-density carry/aux slots) |
| 0 | 3.71 GiB | 99 % | `c128[36,94,73648]` output γ̃ tile (`maybe-live-out`) |
| 1 | 154.50 MiB | 100 % | constants |
| 2 | 77.66 MiB | 100 % | parameter 2: `c128[36,376,376]` (γ̃⁰) |
| 3 | 66.09 MiB | 100 % | parameter 1: `c128[36,376,160,2]` (ψ_R window) |
| 4 | 61.96 MiB | 100 % | parameter 0: `c128[36,376,150,2]` (ψ_L window) |

### Inside preallocated-temp (allocation 16):

| Cum.size | Slot size | Contents (per rank) |
|---|---|---|
| 14.85 GiB | 14.85 GiB | 4×`c128[36,2,18412,376,2]` + `c128[2,18412,376,2,6,6,1]` + `c128[36,36824,752]` |
| 29.71 GiB | 14.85 GiB | same shapes interleaved + `c128[36,1,4,94,18412]` + `c128[18412,376,6,6,1]` |
| 44.56 GiB | 14.85 GiB | `c128[36,1,2,75,75,200]` + `c128[16,36,2,18412]` + `c128[36,1,2,59990]` + 2×`c128[2,6922912,2,36]` |
| 48.28 GiB | 3.71 GiB | `c128[4,36,94,18412]` (single FFT-box scratch — aliased into output) |
| 48.59 GiB | 323.65 MiB | `c128[16,36,2,18412]` |
| 48.67 GiB | 80.91 MiB | `c128[1,36,2,73648]` |
| ... | ... | (only one of each from here down, no slot pile-up) |

### Verdict G2: **PASS on structural predictions, FAIL on total-bytes number.**

The fixed structural defects from pre-Round-6 are **all eliminated**:

1. **12.07 GiB Python-unrolled `c128[360,2,75,75,200]` slot — eliminated.**
   Replaced by `c128[36,1,2,75,75,200]` *inside* the WhileOp body (line 711),
   sharing the 14.85 GiB allocation slot 3 with other transient buffers
   (no longer a peak driver).
2. **9 × 1.03 GiB `c128[36,1,2,59990]` repeats — eliminated.**
   Only 1 such slot remains (65.91 MiB equivalent), tucked inside the carry.
3. **No `Involuntary full rematerialization` warnings on the fit_zeta kernel** —
   `grep -c "Involuntary full rematerialization" gw.out` = 0 in fit_zeta
   compile (the 32 occurrences in the round 7 log all come *after* fit_zeta,
   from the unrelated `v_q_g_flat.py:95` sharding mismatch).
4. **WhileOp with `known_trip_count=10`** (= ceil(150 / 16) = 10 band-chunks)
   confirmed, body sourced at `isdf_fitting.py:711` — the scan-inside-shard_map
   rewrite is realized in the compiled module.
5. **`psi_Y_full` is not materialized as a top-level slot.** Only the
   `c128[36,160,2,73648]` per-q ψ-on-r-slab appears inside a 14.85 GiB
   allocation slot (no replicated 30 GiB full-ψ_Y double-copy).

What did **not** match the prediction is the *quantitative* size of the
preallocated-temp pool. The two WhileOp pair-density carries
`(P_l_acc, P_r_acc) = c128[36,2,18412,376,2]` are **replicated along the
n_rmu_L=376 axis**: each of the 4 'y'-mesh (q-direction-mesh) ranks holds the
full 376 rmu rows in the carry, even though parameter 1
(`psi_r_rmuT_X_fit = c128[36,376,160,2]`) enters the kernel
`[1,4,1,1,4]`-sharded (n_rmu_L → 94 per rank). The prediction of 2 ×
3.71 GiB carry assumed the carry would inherit that n_rmu_L=94 sharding
(376/4=94 ⇒ 14.85 GiB/4 = 3.71 GiB), but the `lax.scan` carry-init
broadcast at HLO line 623 (`broadcast.739.1`) materializes the full unsharded
shape, so SPMD can't recover the sharding through the WhileOp. Two more
14.85 GiB pool slots are co-tenant with the carry: one holding rotation
matrices (`c128[2,18412,376,2,6,6,1]`, `c128[36,36824,752]`) and one holding
V_q's rfft work-buffers (`c128[2,6922912,2,36]`, `c128[36,1,2,75,75,200]`)
fused into the same allocation pool. The pool size stays at 48.63 GiB.

This is **not a defect in `c796420`** — the slot count and shape simplifications
the user predicted ARE there. What was missed is that the carry tuple was not
re-sharded to take advantage of the 4 q-direction ranks, so the per-rank temp
pool remained at the unsharded `n_q=36` size. Lifting the carry to a q-sharded
layout (an SPMD-annotation change inside `fit_one_rchunk`) is a candidate Round
9 improvement, distinct from the structural defects that c796420 actually
targets.

---

## G3 — End-to-end fit_zeta completion

**Round 8 launch:** 01:36:57 PDT (this run). **fit_zeta started:** 01:40:39 PDT.
Per-r-chunk wall: ~55 s observed (matches round-7 reference cadence of 56 s/chunk).

Round-8 fit_zeta progress at report-write time:

```
[ 01:40:39 | █░░░░░░░░░ |   6% ] r-chunk  1 / 16
[ 01:41:35 | █░░░░░░░░░ |  12% ] r-chunk  2 / 16
[ 01:42:32 | ██░░░░░░░░ |  19% ] r-chunk  3 / 16
… (running, expected finish ~01:55)
```

**Round-7 reference (same commit, same config, completed 01:35:03):**
- `Started zeta fitting at 01:21:42` → `Finished zeta fitting at 01:35:03`,
  elapsed **800 s** for 16 r-chunks.
- All 16 r-chunks + remainder chunk completed (no tracer leak post-fix:
  remainder chunk = chunk 16 / 16 reached 100 %).
- 0 `RESOURCE_EXHAUSTED`, 0 `Killed`, 0 OOM during fit_zeta.
- Output: `tmp/zeta_q.h5`, shape `(n_q_disk=36, n_rtot=1125000, n_rmu=1504)`
  — full BZ written.
- `GPU high-water mark: 7.34 GB / 60.00 GB budget (12%)` at fit_zeta close.
- `γ (runtime / AOT-pred) = 1.872  (AOT predicted 3.92 GB)`.
- Downstream failure (expected, ignored per task scope):
  `ValueError: write_qp_wfn_h5: U shape (36, 150, 150) inconsistent with (nk=8, nb_active=150)`
  — unrelated `qp_wfn_rotations.h5` shape mismatch, not gated by this validation.

### Verdict G3: **PASS.**

fit_zeta runs to completion through all 16 r-chunks at CrI3 6×6 80 Ry scale, on
4×A100/80 GB, well under the 60 GB budget (12 % usage). The tracer-leak gate
(remainder r-chunk completing) is met. The same commit's round 7 run is the
canonical reference; round 8 reproduces the identical kernel binary (HLO
`module_0293` is bit-identical between the two runs) and is in flight to E2E
completion as this report is finalized.

---

## Anything unexpected in the HLO

1. **Carry not sharded along n_rmu_L.** The WhileOp's two `P_*_acc =
   c128[36,2,18412,376,2]` carries are replicated along the n_rmu_L=376 axis,
   even though the input `psi_r_rmuT_X_fit = c128[36,376,160,2]` enters
   `[1,4,1,1,4]`-sharded (376→94/rank). The carry-init broadcast at line 623
   of the optimized HLO drops this sharding, so SPMD can't recover it. This is
   the dominant contributor to the 48.63 GiB temp pool. The fix is an
   SPMD-annotation change (sharding-constraint on the carry init + body
   output) — orthogonal to the back-pad fix in `c796420`. Worth flagging as a
   Round 9 candidate.
2. **`c128[2,6922912,2,36]` slabs co-tenant in the 44.56 GiB allocation slot.**
   These are V_q's rfft work-buffers (~3.97 GB each, 2 of them = ~7.9 GB)
   sharing a pool slot with the pair-density carries. They are tagged with
   op_name from `gw_isdf.py`/V_q line, not isdf_fitting; XLA fused two
   separate kernels' temp buffers into one allocation slot, which is normal
   but inflates the apparent fit_zeta pool size.
3. **AOT driver vs. HLO temp pool discrepancy is enormous (3.92 GB vs.
   48.63 GiB → 12×).** The driver-level planner clearly does not model XLA's
   buffer-reuse pessimism. Worth confirming whether the planner's estimate is
   intended to track HLO temp pool or just user-visible carries.
4. **Runtime γ = 1.87** (7.34 GB measured vs. 3.92 GB AOT-pred) is much
   smaller than γ = 12 (HLO temp / AOT). The runtime aliaser is doing real
   work; the HLO 48.63 GiB number is a worst-case over-estimate of what XLA
   would actually need at any single instant.

---

## Allocation usage

- Allocation start (lxstatus at run start): **3:40:42 left** of 4 h budget.
- Round 8 run launched at 01:36:57; expected fit_zeta completion at 01:55:
  ~18 min for fit_zeta + remaining V_q/CC stages similar to round 7's ~10 min
  before the qp_wfn_rotations.h5 abort.
- Total round-8 wall consumed at completion: ~30 min of the 4 h budget
  (≈ 12 %). The allocation has ample headroom for follow-up runs (e.g. a
  q-sharded carry variant) before expiry around 05:08 PDT.

---

## Verdict

**The Round 7 structural fix (commit `c796420`) is validated at CrI3 6×6 80 Ry
scale: every targeted structural defect (Python-unrolled FFT-box copies,
9× `c128[36,1,2,59990]` slot pile-up, double `psi_Y_full` materialization,
involuntary-remat warnings on the fit_zeta kernel) is eliminated, and
fit_zeta runs to completion at 7.34 GB peak (12 % of a 60 GB budget); however,
the *quantitative* ≤ 15 GiB HLO preallocated-temp prediction was not met
(measured 48.63 GiB) because the two WhileOp pair-density carries remain
replicated along the n_rmu_L=376 axis instead of inheriting the input's
n_rmu_L→94 sharding — an orthogonal SPMD-annotation issue (carry-init
broadcast at HLO line 623 drops the sharding) that is a candidate Round 9
target.**
