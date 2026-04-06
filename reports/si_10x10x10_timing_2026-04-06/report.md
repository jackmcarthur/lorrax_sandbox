# Si 10×10×10 Timing Benchmark (Incomplete) + OOM Analysis

**Date**: 2026-04-06
**Run**: `runs/Si/03_si_10x10x10_nosym_timing/`
**Purpose**: Time LORRAX vs BGW on a production-scale 3D k-grid.

## Setup

Si diamond, ecutwfc=25 Ry, 60 bands, 480 centroids, GN-PPM (freq_dep 3),
`nosym=.true.`, 1000 k-points. Perlmutter A100-40GB GPUs.

## Results (partial — LORRAX OOMs, BGW sigma killed for time)

| Step | BGW (16 GPUs, 4 nodes) | LORRAX (4 GPUs, 1 node) |
|------|------------------------|-------------------------|
| QE NSCF | 35 s | (shared) |
| QE pw2bgw+wfn2hdf | 84 s | (shared) |
| Epsilon / ISDF | **9463 s (2.6 hr)** | OOM |
| Sigma | ~50 s/k-pt × 1000 ≈ **~14 hr** (killed) | OOM |
| **Total GW** | **~16.6 hr** (estimated) | — |

### For reference: Si 4×4×4 timings (same parameters, 64 k-points)

| Step | BGW (8 GPUs) | LORRAX (4 GPUs) |
|------|-------------|-----------------|
| Epsilon + Sigma | 115 s | — |
| LORRAX total | — | **50 s** |
| MAE vs BGW | — | **11.7 meV** |

## OOM Analysis

### What fails

LORRAX OOMs during `get_sharded_wfns()` in `common/load_wfns.py`. This
function FFTs all k-point wavefunctions from G-space to real space, applies
phases, and extracts centroid values — all in a single JIT-compiled kernel
operating on the full k-point array simultaneously.

### The memory budget

XLA's log at the point of failure:

```
Can't reduce memory use below 14.12 GiB by rematerialization;
only reduced to 22.66 GiB, down from 22.66 GiB originally
```

```
GPU_3_bfc ran out of memory trying to allocate 5.66 GiB
```

This means: the JIT kernel needs **22.66 GB of execution buffer** per device,
but only **14.12 GB is free** (the rest is occupied by `psi_Gtot_local`
shards and JAX/XLA runtime overhead).

### Array inventory at the OOM point

| Array | Total | Per device (4 GPUs) | Sharding |
|-------|-------|--------------------|---------| 
| `psi_Gtot_local` | 26.5 GB | 6.6 GB | k-sharded |
| XLA runtime + staging | ~6 GB | ~6 GB | per-device |
| **Free** | — | **14.1 GB** | — |
| **Kernel needs** | — | **22.7 GB** | — |

The 22.7 GB kernel buffer includes: FFT temporaries for all local k-points
(each k-point: 60 bands × 2 spinors × 13824 r-points × 16 bytes = 26 MB;
times 250 local k-points = 6.5 GB just for inputs), the FFT output, phase
multiplication buffers, centroid gather indices, and the output `psi_y` /
`psi_x` arrays being assembled.

### Why 4×4×4 works

With 64 k-points, `psi_Gtot_local` is 1.7 GB total (0.4 GB per device),
and the XLA kernel buffer is ~8 GB. Peak measured: **9.49 GB** per device
on 4 A100s — well within the 40 GB limit.

### The fix

**K-point chunking in `get_sharded_wfns`**: instead of processing all 1000
k-points in one JIT call, process them in batches of ~64-128, accumulating
`psi_y` and `psi_x` (the centroid-space representations) incrementally.
Each batch's FFT intermediates are freed before the next batch starts. The
output arrays (`psi_y`: 0.92 GB, `psi_x`: 0.92 GB) are small enough to
accumulate across all batches.

This does NOT affect the ISDF fitting or sigma pipeline — only the
wavefunction extraction stage.

### Broader implications

For production 3D calculations, the k-grid typically has 100-1000+ k-points.
The current code is limited to ~200 k-points on 4 A100-40GB GPUs (the 6×6×6
= 216 k-point case would be borderline at ~5.7 GB for `psi_Gtot`). With
k-point chunking, the limit would be set by the centroid arrays (0.92 GB
for 1000 k-points) rather than the FFT-box arrays (26.5 GB), enabling
10×10×10 and beyond.

## BGW Performance Notes

BGW epsilon on 1000 q-points took 2.6 hours on 16 GPUs. The dominant cost
is the polarizability matrix build (~15s per q-batch × 1000 q-points).
BGW sigma at 1000 k-points was killed after ~35 minutes, having completed
only a small fraction — estimated total ~14 hours on 16 GPUs.

The 4×4×4 comparison (64 k-points) showed LORRAX at 50s on 4 GPUs vs BGW
at 115s on 8 GPUs. Extrapolating, a working 10×10×10 LORRAX would likely
complete in minutes (ISDF + sigma scale favorably), while BGW takes ~16 hours.
