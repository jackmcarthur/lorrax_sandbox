# Memory-model refit — what the 80Ry sweep showed (2026-05-17)

## System under test

CrI3 6×6×1 80 Ry SOC bispinor on 16 GPUs (hbm80g, 4×4 mesh).
- `nk = 36`, `nspinor = 2`, `nval = 70`, `ncond = 80`, `nband = 150`
- `n_rmu_charge = 1508`, `n_rmu_transverse = 1504` (n_rmu_padded ≈ 1520 on 16-rank mesh)
- `ngkmax = 59990`, `n_rtot = 1,125,000`, `fft_grid ≈ (60, 60, 200)`
- `memory_per_device_gb = 70.0` (user-set; physical HBM = 80 GB)
- ζ-fit runs **4 separate channels** (charge + 3 transverse); each does its own r-chunk loop.

## Sweep results (b=32 = best per-rank-FFT/iter-tradeoff at large r)

| config | r_chunk | n_chunks | proj ζ-fit (4ch) | planner HWM | bottleneck | status |
|---|---|---|---|---|---|---|
| A1 (r=4096, b=16) | 4096 | 275 | 129.9 min | — | — | ok |
| A4 (r=8192, b=16) | 8192 | 138 | 92.3 min | — | — | ok |
| A5 (r=8192, b=32) | 8192 | 138 | 91.6 min | 55.77 GB | D | ok |
| B1 (r=16384, b=16) | 16384 | 69 | 68.8 min | — | — | ok |
| **B5 (r=24576, b=32)** | **24576** | **46** | **59.6 min** | **65.99 GB (94% of 70 GB)** | **C** | **ok (best)** |
| C1 (r=28672, b=32) | 28672 | 40 | — | 76.98 GB (110%) | C | **OOM** |
| C2 (r=32768, b=32) | 32768 | 35 | — | — | — | OOM (request 80 GB single alloc) |

## Planner natural pick (no overrides)

`gflat_memory_model.py:plan_gflat_chunks` with all chunk knobs = 0:
- `band_chunk = 32`, `r_chunk = 21232` (53 chunks), `gflat_chunk_size = 707`
- HWM estimate = 57.03 GB (81% of 70 GB budget), bottleneck = C_fit_one_rchunk
- Projected ζ-fit ≈ 62 min (within 5% of empirical optimum)

## Peak D anatomy (the user's "GB and GB of overhead")

At planner natural pick:
- `gflat_acc` persistent (sharded, per-rank): `36 × 1508 × 59990 × 16 / 16 = 2.71 GB`
- `zeta_chunk` transient: `36 × 1508 × r_chunk × 16 / 16` → 7.1 GB at r=8192, 21.4 GB at r=24576
- **`accumulate_fft_box` = `gflat_chunk_size × n_rtot × c128 × fft_factor` ≈ 707 × 1.125M × 16 × 4 = 51 GB** ← dominates D

The FFT-box term is **r_chunk-independent** because gflat_chunk_size is set by Peak D's own budget logic, not by Peak C's r_chunk. So D stays at ~55 GB regardless of whether r is 4096 or 24576.

## Per-r-chunk timing breakdown (from new `LORRAX_RCHUNK_DEBUG=1` print)

Charge channel (Cholesky): z_q_build ~2s, solve ~3s, write ~0.7s, total ~6s/r-chunk
Transverse channel (LU): z_q_build ~2s, solve ~5s, write ~0.7s, total ~8s/r-chunk
**Solve dominates per-chunk wall** (not pair-density / FFT).
**Bispinor has 4 channels (3 LU + 1 Chol)**, so total wall scales with 4× per-chunk × n_chunks_per_channel.

## User's diagnosis

The planner picks (band_chunk, r_chunk, gflat_chunk_size) **sequentially**:
1. `band_chunk` first (capped at 0.5×target / FFT-box)
2. `r_chunk` from `headroom_C / α_C` after band fixed
3. `gflat_chunk_size` last (one-shot if it fits; else divide down)

This sequential order means **gflat_chunk_size is locked too high** (the one-shot is too aggressive), eating ~50 GB of phantom Peak D, which limits how big r_chunk can grow. **Joint optimization** would trade gflat_chunk_size DOWN to free Peak D headroom for r_chunk UP.

## User's proposal

- Treat band_chunk (ψ-side) and gflat_chunk_size (ζ-side) as **symmetric parameters** governed by the same memory-vs-iter-count tradeoff.
- Each has a **minimum floor**: 4× one FFTbox (≈ `n_rtot × c128 × 4 ≈ 72 MB`) plus the chunk's ψ/ζ slab being processed (per-rank).
- Maximize `r_chunk` first (the expensive axis — its iters are slow), then **shrink band_chunk and gflat_chunk_size** to whatever fits the remaining HBM budget, since their iters are FFT-cost-cheap.
- ψ-side and ζ-side chunk sizes are essentially the same FFT-batch concept, with:
  - (a) ψ gets unfolded from k_irr → k_full (extra k-axis factor)
  - (b) typically more ζ rows (`n_rmu > n_band`), so even at the same chunk size, more iters for the ζ side

## Goal of the refit

A `plan_gflat_chunks` that:
1. Picks `r_chunk` first to max Peak C subject to budget
2. Then picks `gflat_chunk_size` and `band_chunk` to fit Peak D and Peak A/C-FFT-terms in the remaining budget, with floor = ~4× one FFTbox
3. Reports each peak's components individually so the user can see exactly where the memory goes
