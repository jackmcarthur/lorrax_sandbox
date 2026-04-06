# Memory Assay Report: Si 4×4×4, 35 bands, 240 centroids, 4 GPUs

**Date**: 2026-04-06
**System**: Si diamond, nk=64, nb=35 (pad to 36), nspinor=2, FFT 24³, n_rmu=240
**Hardware**: 4× A100-40GB, 2×2 mesh, single process

## Key finding: multi-GPU peak is ~9× shard, not 4×

On a single GPU, the FFT peak is 4× the input array (measured and verified).
On 4 GPUs with band-sharding, the peak is **~9× the per-device shard**.
The extra 5× comes from collective communication buffers and resharding
intermediates that don't exist in the single-GPU case.

| bc | bc_pad | shard/dev (GB) | 4× pred (GB) | measured peak (GB) | ratio |
|----|--------|----------------|--------------|-------------------|-------|
| 9 | 12 | 0.085 | 0.340 | 0.74 | 8.7× |
| 18 | 20 | 0.142 | 0.566 | 1.33 | 9.4× |
| 35 | 36 | 0.255 | 1.019 | 2.29 | 9.0× |

The 4× model accounts for:
1. psi_G shard (FFT input): 1× shard
2. psi_r shard (FFT output): 1× shard
3. FFT staging buffer 1: 1× shard
4. FFT staging buffer 2: 1× shard

The additional ~5× likely includes:
5. All-gather receive buffer during {-,XY,-,-} → {-,Y,-,-}: ~2× shard
   (each Y-device receives bands from X-neighbors)
6. All-to-all buffer during {-,Y,-,-} → {-,-,-,Y}: ~2× shard
   (swapping bands for centroids/r-points)
7. JIT output materialization (psi_rmu + psi_rmuT before return): ~1× shard

**Recommendation**: Use **9× shard** as the multi-GPU memory model, not 4×.
The 4× model is correct for single-GPU-only profiling.

## R-chunk has negligible impact on peak memory

The r-chunk extraction does NOT increase the peak above what the centroid
extraction already established. This is because:
- The r-chunk is extracted from the G-space cache (already on device)
- The r-chunk reshard intermediate is small compared to the FFT peak
- The peak_bytes_in_use from the centroid extraction dominates

| bc | rc | r-chunk peak (GB) | centroid peak (GB) | r > centroid? |
|----|------|------------------|-------------------|---------------|
| 35 | 1152 | 2.29 | 2.29 | NO |
| 35 | 3456 | 2.43 | 2.43 | NO |
| 35 | 6912 | 2.43 | 2.43 | NO |
| 35 | 13824 | 2.43 | 2.43 | NO |

The slight increase from 2.29 to 2.43 for rc>1152 is from the G-space
cache being larger (0.14 GB more), not from the r-chunk reshard.

## Timing: larger chunks are faster

| bc | rc | n_bc | n_rc | centroid (s) | r-chunk (s) |
|----|------|------|------|-------------|------------|
| 9 | 1152 | 4 | 12 | 3.67 | 0.64 |
| 9 | 13824 | 4 | 1 | 0.28 | 0.77 |
| 35 | 1152 | 1 | 12 | 0.75 | 0.35 |
| 35 | 13824 | 1 | 1 | 0.23 | 0.33 |

- Centroid extraction is fastest with bc=35 (1 band chunk = no loop overhead)
- R-chunk extraction time is relatively flat (~0.3-0.7s) regardless of chunking
- The first centroid run at each bc size includes JIT compilation (~3s for bc=9)
- Total time difference between most/least chunked: ~4s (dominated by JIT)

## G-space cache memory

| bc | bc_pad | G-cache used/dev (GB) | pred shard (GB) | ratio |
|----|--------|----------------------|-----------------|-------|
| 9 | 12 | 0.37 | 0.085 | 4.4× |
| 18 | 20 | 0.98 | 0.142 | 6.9× |
| 35 | 36 | 1.21 | 0.255 | 4.7× |

The G-cache holds the band-sharded FFT box arrays for reuse across r-chunks.
The 4-7× ratio above the shard size includes all band chunks in the cache
(e.g., bc=9 has 4 band chunks cached, so 4×0.085 = 0.34 GB matches the
measured 0.37 GB).

## Implications for 10×10×10

With nk=1000, nb_pad=64, mesh 4×4 (16 devices):
- Shard/dev = 1000 × 4 × 2 × 13824 × 16 / 1e9 = 1.77 GB
- Predicted multi-GPU peak = 9 × 1.77 = **15.9 GB** — fits in 40 GB
- Reshard intermediate = 1000 × 16 × 2 × r_chunk × 16 / 1e9
  - At r_chunk=13824: 7.1 GB (plus the 15.9 GB peak = 23 GB — fits)
  - At r_chunk=6912: 3.5 GB
  - At r_chunk=3456: 1.8 GB

The 10×10×10 OOM (6.5 GB allocation) is the reshard intermediate
appearing ON TOP of the ~26 GB already used by other arrays and
multi-process overhead. The fix: account for 9× shard (not 4×) in
the memory model, plus the reshard intermediate.
