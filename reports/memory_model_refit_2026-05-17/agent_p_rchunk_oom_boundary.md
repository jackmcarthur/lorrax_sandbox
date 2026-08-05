# Agent P (Round 9a) — empirical r_chunk OOM boundary post-Round-4/6 fixes

**Branch:** `agent/bispinor-ibz` on lorrax_B
**System:** CrI3 6x6x1 80Ry SOC bispinor, 16 GPUs (4x4 mesh, hbm80g)
**JID:** 53087968
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**Goal:** locate the post-Round-4/6 OOM cliff on r_chunk_size; answer "can r go 4x larger?"; cross-check planner against ground truth at the new failure point.

## Headline verdict

**Empirical OOM boundary is unchanged from pre-fix: the cliff sits between r=24576 (OK) and r=28672 (OOM).** Round-4 cache-leak (1.13 GB/rank) and Round-6 sphere consolidation (0.32 GB/dev) freed steady-state HBM but did NOT widen the in-jit transient ceiling — that ceiling is dominated by Peak C, the single (n_q_pp, n_rmu, n_gflat) materialization which is what scales linearly with r_chunk.

The user's "can r go 4x?" hypothesis is **NO**: even 1.17x (24576 -> 28672) is already over the cliff, on both `platform` and BFC@95% allocators. r=98304 (4x) would need a single ~256 GB block.

## 1. Empirical sweep table

Probe: `LORRAX_MEM_DEBUG=1 LORRAX_RCHUNK_DEBUG=1 LORRAX_MAX_RCHUNKS=3 LORRAX_EXIT_AFTER_ZETA=1 LORRAX_FORCE_FULL_BZ=1`,
cohsex.in overrides: `r_chunk_size=N b=32 gflat=100`, 4 nodes x 4 A100-80GB.

| config | r_chunk | planner Peak-C (GB/dev) | nvsmi peak | mem_stats peak | result | failure mode |
|---|---|---|---|---|---|---|
| **Z0** | **24576** (X3 sweet) | **66.41** | **8.67** GB | 76.05 GB (from Agent-O Y3_95) | **OK** — 3 r-chunks completed, clean exit | n/a |
| **Z1** | 28672 | 77.39 | n/a (died) | n/a (died) | **OOM** | cuFFT scratch alloc failed: 4.83 GB on top of in-jit working set (platform allocator) |
| **Z1_y3_95** | 28672 (BFC pre 0.95) | 77.39 | n/a | n/a (died) | **OOM** | XLA `Out of memory while trying to allocate 75.31 GB` single block; rematerializer reports "can't reduce memory use below 69.75 GB; only reduced to 71.60 GB" |
| **Z2** | 32768 | 88.38 | n/a | n/a (died) | **OOM** | XLA `Failed to allocate request for 80.16 GB` single block (platform) |
| Z3 (r=49152) | — | (would be ~120 GB/dev) | — | — | **not run** (stop after 2 consecutive OOMs) | — |
| Z4 (r=98304) | — | (would be ~240 GB/dev) | — | — | **not run** | — |

Source files: `agent_p_z0_r24576.out`, `agent_p_z1_r28672.out`, `agent_p_z1_r28672_y3_95.out`, `agent_p_z2_r32768.out`.

## 2. Did the Round-4/6 fixes move the boundary?

**No — boundary at r=28672 is identical to the pre-fix v10 sweep** (`/tmp/bispinor_80ry_sweep_v10.tsv` C1 OOM at r=28672 with `gflat=360`). The Round-4/6 fixes freed steady-state working set (live_total dropped by ~1.5 GB/dev), but the in-jit transient — the (n_q_pp, n_rmu, n_gflat) Peak-C buffer — was NOT touched by either fix. That buffer is the OOM-trigger, and it scales linearly with r.

Quantitative cross-check: at r=24576 (Z0 succeeds), `mem_probe rchunk_start` shows live_total=58.92 GB (post-fix); pre-fix v10 sweep at r=24576 showed live_total ≈ 60.4 GB. Difference ≈ 1.5 GB freed by Round 4+6. The transient single-allocation (the 60 GB Peak-C block seen in Agent O Y2/Y3_50) is unchanged. Pool budget = 80 GB - working_set; widening working set by 1.5 GB reduces the headroom for Peak-C by 1.5 GB, which is less than the 11 GB Peak-C step from r=24576 (66.41 GB) to r=28672 (77.39 GB).

So: **+0 r vs pre-fix.** The fixes are real and worthwhile (they free 1.5 GB of replicated buffers and prevent slow growth across the 46 r-chunks) but do not move the single-rchunk cliff.

## 3. The 4x hypothetical — answer with data

| factor on r | r_chunk | planner pred | will it OOM? | reason |
|---|---|---|---|---|
| 1.0x | 24576 | 66.41 | OK | Peak-C 60 GB single block fits in cudaMallocAsync (transient) and BFC@95% (76 GB total < 80 GB pool) |
| 1.17x | 28672 | 77.39 | **OOM** | single block ≈ 70 GB requested; remat irreducible at 71.60 GB; total > 80 GB pool |
| 1.33x | 32768 | 88.38 | OOM | single block ≈ 80.16 GB requested; > pool |
| 2.0x  | 49152 | ~125 (est)  | OOM | single block ≈ 120 GB; nowhere close |
| 4.0x  | 98304 | ~245 (est) | OOM, far past | single block ≈ 240 GB; would need >3x A100-80GB per device |

**4x is not feasible.** The hardware ceiling on a single A100-80GB device is roughly r ≈ 26000 under cudaMallocAsync (the actual cliff sits in the narrow window [24576, 28672]); pushing past that would require either (a) shrinking the Peak-C buffer footprint (band-chunk smaller, gflat-chunk smaller — already at cap 100), or (b) rematerializing the n_q_pp axis, which the XLA pass already failed to do at r=28672.

## 4. Planner calibration at the new threshold

The Y3_95 BFC run at Z1 (r=28672) is the cleanest cross-check of the planner against an XLA-arena single allocation. XLA tried to allocate **75.31 GB** in one block; planner predicted Peak C = **77.39 GB/dev**. **Planner is +2.7% over-predicting the single block** — almost exactly on. The rematerializer's reported "can't reduce below 71.60 GB" is the same Peak-C transient seen from the optimizer's frame, again within ~7% of the planner.

Compared with Agent O's Z0 (r=24576) calibration — planner 66.41 vs measured 76.05 (-13.6% under-pred) — the under-prediction GAP narrows at the cliff itself. The planner is more accurate where it matters (near the OOM boundary) than at the sweet spot (where there's 10 GB of slack the planner doesn't model).

**Practical implication:** the planner's *override-r_chunk* warning is well-calibrated. r=24576 (66.41 pred) succeeds; the natural cap of r=20688 (which the planner picks unprompted) is conservative by ~19% in r-units, ~21% in GB. Lifting the natural cap by ~15% (to r=23800 or so) would be safe; lifting further is *not*.

## 5. Recommendations

1. **Keep the user-side r=24576 override**: it's right at the safe edge of the cliff, +19% past the planner's natural cap. No headroom for more.
2. **Do NOT recalibrate the planner cap down** (the conservative natural choice at r=20688 is fine for hands-off operation).
3. **Future work** to break the ceiling requires the cited single-block remat (split the (n_q_pp, n_rmu, n_gflat) buffer along q or rmu) — not a memory-model tweak.
4. **Round-4/6 fixes remain net-positive** for steady-state HBM but should not be claimed as cliff-movers in the changelog.

## 6. Artifacts

| file | purpose |
|---|---|
| `agent_p_z0_r24576.out` | baseline confirm (succeeded, nvsmi_peak=8.67 GB) |
| `agent_p_z1_r28672.out` | first OOM past sweet spot (cuFFT scratch failure, platform) |
| `agent_p_z1_r28672_y3_95.out` | BFC@95% reproducer; XLA single-block OOM = 75.31 GB; remat irreducible at 71.60 GB |
| `agent_p_z2_r32768.out` | confirms cliff (single-block OOM = 80.16 GB, platform) |
| `cohsex_p_r{24576,28672,32768}.in` | per-r cohsex.in (auto-generated by launcher) |
| `_launch_agent_p.sh` | per-variant launcher (r_chunk + alloc variant) |
