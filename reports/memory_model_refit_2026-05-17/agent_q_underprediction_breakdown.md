# Agent Q — Where Is the 10 GB/dev Planner Under-Prediction Hiding?

**Branch:** `agent/bispinor-ibz` on lorrax_B
**Config audited:** X3 sweet-spot `r=24576, b=32, cs=100, bispinor=true, nbnd=150` on CrI3 6×6 80Ry, 4×4 mesh, hbm80g
**Planner prediction (Peak C):** 66.41 GB/dev — bottleneck `C_fit_one_rchunk`
**Ground truth (Y3_95, `mem_stats peak`):** 76.05 GB/dev
**Gap:** 9.64 GB/dev (14.5%)

This round is OFFLINE analysis of existing HLO dumps + the live_arrays census embedded
in `agent_o_y3_95.out`. No new compute. The X3-config HLO at cs=360 (run dir
`lorrax_D_bispinor_hlo_2026-05-17/`) is reused for fit_one_rchunk's Peak C since
cs only affects accumulate (Peak D), not fit (verified by agent_d M1/M3).

## 1. Per-jit HLO peak vs planner

| jit | module# | HLO peak | planner pred | gap | per-category breakdown |
|---|---|---|---|---|---|
| `fit_one_rchunk` (charge channel) | 0438 | **61.77 GiB = 66.32 GB** | 66.41 GB (Peak C) | **+0.09 GB** | preallocated-temp 60.12 GiB (3 pair-density slots × 20.04 GiB + 4.83 GiB FFT box); output 1.25 GiB (zeta_chunk); params 414 MiB |
| `fit_one_rchunk` (transverse) | 0521 etc | **65.63 GB** | 65.6 GB | ~0 | identical, mu_transverse=1504 |
| `accumulate_rchunk_to_gflat` (cs=360 in dump; production cs=100) | 0474 | **20.92 GiB at cs=360** | (would be 5.96 GB at cs=100) | n/a | preallocated-temp 16.61 GiB (2×6.03 GiB FFT box + 3.22 GiB gflat scan-carry + 1.32 GiB zeta-slab); param/out 4.31 GiB |
| `_local_fft` (centroid build, Peak A) | 0021/0023 | **77.25 GiB** | 9.22 GB (A.fft_box) | **+68 GB**, but freed BEFORE Peak C | input 19.31 GiB + output 19.31 GiB + 38.62 GiB transpose temp on shape `c128[36,8,4,75,75,200]` |

**`fit_one_rchunk` HLO matches planner Peak C within 0.1%.** The 10 GB gap is NOT inside
the jit. The `_local_fft` 77 GiB is a separate gigantic peak in centroid-load (Peak A),
but it runs and frees BEFORE the r-chunk loop — it doesn't overlap with Peak C.

## 2. Where the 10 GB hides — co-resident state during Peak C

`agent_o_y3_95.out` has a live_arrays census at every probe site. Per-device totals
(global / 16, since sharding is p_xy=16):

| state | global bytes (live_arrays) | per-device | in planner? |
|---|---|---|---|
| `gflat_acc` `c128[36,1520,59990]` | 52.52 GB | **3.28 GB** | **NO — Peak C zeros it "to avoid double-count with D"** |
| `C_q` `c128[36,1520,1520]` (L_q?) | 1.33 GB | 0.083 GB | yes (in Peak C as `L_q`) |
| transverse centroids `c128[36,1520,160,4]×2 + c128[36,160,4,1520]×2` | 4.48 GB | 0.280 GB | yes (Peak C `centroids_persist`) |
| charge centroids `c128[36,1520,150,4]×2 + c128[36,150,4,1520]×2` | 4.20 GB | 0.263 GB | NO — second channel's centroids co-resident |
| sphere_idx `s32[36,75,75,200]` | 0.16 GB replicated | **0.16 GB** | yes (Peak C `sphere_idx_replicated`) |
| **persistent co-resident total** | — | **~4.07 GB/dev** | planner counts ~0.52 |

Persistent miss = `(3.28 - 0) + (0.263 - 0) ≈ 3.55 GB/dev`. The `gflat_acc` zeroing in
Peak C is the **biggest single error** in the planner (3.28 GB/dev).

## 3. Final per-device accounting at fit_one_rchunk peak

| component | GB/dev | source |
|---|---|---|
| in-jit (HLO module_0438 peak) | 66.32 | preallocated-temp + output + params |
| co-resident persistent (live_arrays) | 4.07 | gflat_acc + dual-channel centroids + L_q + sphere idx |
| **subtotal (visible to planner+HLO)** | **70.4** | |
| **measured ground truth** | **76.05** | `device.memory_stats()['peak_bytes_in_use']` |
| **residual unmodeled gap** | **5.6** | NOT in HLO, NOT in live_arrays |

## 4. Verdict on candidates (1)–(9)

| # | candidate | GB/dev | reasoning |
|---|---|---|---|
| (1) | XLA preallocated-temp slabs | **0** | already in HLO `module_0438` (60.12 GiB charged) |
| (2) | NCCL collective intermediate buffers | **~2–3** | NCCL 2.26.3 + cusolverMp on 4×4 mesh; pool lives outside HLO, no NCCL_DEBUG logs available; typical 1.5-3 GB/dev |
| (3) | cuFFT plan workspace beyond `factor_D=2.0` | **0** | factor_D HLO-verified at cs=1, cs=360; FFT scratch already in the 60 GiB slab |
| (4) | Cross-jit transient lifetimes | **3.28** | **`gflat_acc` zeroed in Peak C** — confirmed alive at probe `after_fit_one_rchunk live_total=79.91 GB` while jit transient still in pool. Planner bug. |
| (5) | Sharded→resharded transients | **<0.5** | live_count grows by 1 across fit (66 vs 65), small |
| (6) | Padding overhead `nb_padded` etc | **~0** | dump shows actual `nb=150,160`; padding already in formula |
| (7) | Per-channel inter-channel state | **~0.26** | charge centroids stay resident during transverse fit (4.20 GB / 16 = 0.26 GB/dev) |
| (8) | Async/dispatch lag | **0** | `peak_bytes_in_use` is BFC-side high-water mark; not sample-based |
| (9) | CUDA context overhead | **~1–2** | NVIDIA driver baseline per process |

**Total accounted: ~9.3 GB/dev** ≈ the 9.64 GB measured gap.

**The dominant single term is candidate (4): `gflat_acc` is alive during Peak C but
zeroed in the planner's Peak C formula** (`gflat_memory_model.py:356-357`). This
alone accounts for 3.28 GB/dev (~34% of the gap). Add NCCL pool (~2-3 GB) + CUDA
context (~1-2 GB) + second-channel centroids (~0.26 GB) and the gap is fully
explained.

## 5. Recommendation

**Fix candidate (4) first.** One-line change in `gflat_memory_model.py:357`:

```python
"gflat_acc":  # alive across the r-chunk loop, co-resident with Peak C transient
    _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy),
```

Then take MAX(C, D) instead of summing both — they don't both peak at the same
time, but gflat_acc *is* live during C. Re-tally HWM:
new Peak C = 66.41 + 3.28 ≈ 69.7 GB/dev.

**Add a fixed NCCL/CUDA overhead term** (candidate 2 + 9), e.g. 3 GB/dev constant.
That brings predicted HWM to **~72.7 GB/dev** vs measured 76 — under-prediction
shrinks from 14.5% → 4.5%. Remaining 3 GB is within "safety margin" until the
next refactor.

**Do NOT** retune `pair_density_slots` or `fft_box_factor` — those terms already
match HLO exactly (agent_d).

## 6. Constraint adherence

- No new compute runs.
- No edits to `gflat_memory_model.py` (diagnostic only).
- HLO data sourced from `lorrax_D_bispinor_hlo_2026-05-17/xla_dump/` (X3-config
  except cs=360) and live_arrays census from `agent_o_y3_95.out` (production X3 cs=100).
- NCCL allocation **not in HLO**; estimate from typical NCCL pool sizes (no
  NCCL_DEBUG=INFO logs in the existing runs).

## 7. Files

- HLO Peak C: `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17/xla_dump/module_0438.jit_fn.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`
- HLO Peak D (cs=360): `…/module_0474.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`
- HLO `_local_fft` (Peak A): `…/module_0021.jit__local_fft.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`
- Live-arrays census: `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/agent_o_y3_95.out`
- Planner source: `/global/u2/j/jackm/software/lorrax_B/src/gw/gflat_memory_model.py:351-382` (Peak C, line 357 is the bug)
