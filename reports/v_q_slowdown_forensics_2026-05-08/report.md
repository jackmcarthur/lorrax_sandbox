# V_q tile 10× slowdown forensics — 2026-05-08

Read-only post-mortem of one outlier V_q tile wall (2134 s) vs prior runs (152–218 s) on the same 4-node hbm80g allocation, lorrax_A branch agent/phdf5_padded.

## Hypothesis

The V_q tile slowdown is **read-side, batch-localized I/O contention** (transient Lustre/OST throughput drop), not a code regression in the padded SlabIO commits. The padded `read_slab` adds only one extra small int64 control buffer per call; the JIT/shard_map structure (cached `sm`, single `block_until_ready`) is unchanged from `a5e404f` to `a16f8eb`. Every other stage in the same run (ζ-fit, ζ writes, close-time drains, MPI init) is at-or-faster than the reference fast runs, which would not be the case if a per-call host-blocking overhead had been introduced.

## Evidence summary

ζ-fit and V_q wall, slow vs reference fast runs (all q_chunk=12 unless noted):

| Run | ζ-fit elapsed | V_q elapsed | Notes |
|---|---|---|---|
| baseline_fftcoef2 (15:18) | 251 s | 218 s | `LORRAX_V_Q_FFT_COEF=2.0` |
| stage_breakdown (16:51) | — | 226 s | host-blocked: read 346.8 s, kernel 14.1 s |
| baseline_fftcoef5 (16:29) | 252 s | 274 s | |
| aot_v2 (17:59) | — | 259 s | q_chunk=6 |
| aot_v3 (18:14) | 251 s | **152 s** | chooser picked q_chunk=12 organically |
| **flatq_v2 (19:42, slow)** | **221 s** | **2134 s** | first run after liblorrax_ffi.so rebuild at 19:19:21 |

Slow ζ-fit is *faster* than every fast run. Per-r-chunk ζ-fit timings in the slow run (236.2–248.4 s) bracket the fast aot_v3 (261–272 s). Whatever made V_q slow did not affect ζ-fit at all — both go through the same SlabIO/PHDF5/FFI path.

V_q tile q-point progress (q_chunk=12 → 3 batches of 12):

```
slow flatq_v2:  19:42:47  q1–12   (instant, JIT trace)
                20:00:34  q13–23  (+1067 s)
                20:18:21  q24–36  (+1067 s)
fast aot_v3:    18:14:08  q1–12   (instant)
                18:15:24  q13–23  (+76 s)
                18:16:40  q24–36  (+76 s)
```

Per-batch wall: 711 s slow vs 51 s fast, ~14×. Per-batch read volume is unchanged: shape (12, 1.125 M, 1504) complex128 ≈ 324 GB → ~972 GB total. Effective aggregate read bandwidth: fast 6.4 GB/s, slow 0.46 GB/s. Both runs use the same 16×4 MiB stripe (verified via `lfs getstripe /…/tmp/zeta_q.h5`).

Other diffs ruled out:
- Same job (JID 52664413), same 4 nodes (008508/24/25/28), same Mesh 4×4, same q_chunk=12, same μ_chunk=1504, same `predicted peak/rank=40.70 GB`.
- Same read call site. No errors/warnings unique to the slow run.
- γ (memory ratio): slow 0.660, fast 0.428 — both well under budget; not a thrash signal.
- liblorrax_ffi.so rebuilt at 19:19:21 between aot_v3 and flatq_v2; the immediately preceding run crashed with the 3-arg ABI break. flatq_v2 was thus the first V_q tile run with a fresh JIT cache, so the kernel and read shard_map had to retrace once at q1 — but that would show up only in **batch 1**, not batches 2 and 3. All three batches are uniformly slow, ruling out compile/AOT-trace overhead.

Code path inspection (`_slab_io_ffi.py` L559–654 + diff `a5e404f..a16f8eb`):
- The padded change adds one `_replicated_i64_vector(vshape, mesh)` per call (~tens of µs), threads `valid_shape_local` into the `_per_rank` body, and updates `in_specs=(P(), P())`.
- `sm = jax.jit(sm_bare)` cache and final `result.block_until_ready()` are unchanged.
- The C++ FFI handler now reads only the prefix described by `valid_shape` and zero-fills the rest — there is no plausible way this added 660 s/batch.

## Most-likely cause

Transient Lustre/OST read-side contention on the shared pscratch pool during the 19:42–20:18 window. The V_q tile issues 3 large collective reads of `tmp/zeta_q.h5` (974 GB total); the host thread blocks on each via `result.block_until_ready()` at `_slab_io_ffi.py:653`. If aggregate OST throughput drops (other tenant traffic, OST rebalance/recovery, OSS metadata pressure), every batch stalls uniformly. The padded SlabIO commits did not change this serialization or the underlying collective read.

A code-side contributor cannot be **fully** ruled out without stage-breakdown printing on the slow run (`LORRAX_V_Q_TIME_STAGES` was unset for flatq_v2). But the uniform 14× per-batch ratio and unaffected ζ-fit are far more consistent with system I/O than with any per-call padded-read overhead, which scales as N_reads × tens of µs.

## Specific next steps to test on a fresh allocation

1. **Reproduce with stage timing on.** Set `LORRAX_V_Q_TIME_STAGES=1` and rerun. Compare `read=`, `kernel=`, `v+phase=` totals against the 16:51 stage_breakdown reference. If on a slow rerun `read=` ≫ 350 s while `kernel=` is unchanged, the slowdown is purely read-bound — confirming Lustre/OST.
2. **Back-to-back A/B in one allocation.** Run two flat-q V_q tiles consecutively without changing inputs. If run-to-run wall varies by 10× across one job, the cause is system fluctuation. If both are slow, the cause is repeatable.
3. **Bisect SlabIO.** `git checkout a5e404f` (last pre-padded), rebuild .so, run the same flat-q V_q tile. Expected ~150–220 s. If it is also slow when system is loaded, regression is conclusively excluded; if it is consistently fast while padded HEAD is consistently slow, focus on the C++ FFI handler diff.
4. **Hoist ζ to a single read.** As diagnostic, replace the per-batch reads with a single `read_slab` of the full (36, 1.125 M, 1504) ζ tensor before the q loop. Removes per-call dispatch overhead; if speedup is large, points to per-call latency, not throughput.
5. **Lustre health probe at allocation start.** Cheap pre/post `dd if=…zeta_q.h5 of=/dev/null bs=4M count=1024` per node. Log throughput. Gives a quantitative system baseline for any future runs and allows distinguishing "fluke" from "regression" in seconds.

## Key files

- `…/run_logs/flatq_v2_20260507_191934.log` (slow)
- `…/run_logs/aot_v3_20260507_180502.log` (fast 152 s)
- `…/run_logs/stage_breakdown_20260507_164127.log` (host-blocked breakdown reference)
- `…/run_logs/flatq_check_20260507_191207.log` (immediately preceding crash, 3-arg ABI break)
- `…/run_logs/ffi_rebuild_v2_20260507_191908.log` (rebuild at 19:19:21)
- `lorrax_A/src/file_io/_slab_io_ffi.py` (read_slab L559–654)
- `lorrax_A/src/gw/v_q_tile.py` (read calls L969, L1006)
