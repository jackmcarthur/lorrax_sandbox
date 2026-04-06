# Si 10×10×10 OOM — Handoff Notes for Next Session

**Date**: 2026-04-06
**Branch**: `main` (all changes pushed)

## Current state

Si 10×10×10 (1000 k-points, 60 bands, 480 centroids, 16 GPUs) OOMs during
the ISDF fitting r-chunk pipeline. The wavefunction centroid loading itself
works (k-chunking fixes that path), but the r-chunk extraction path in
`get_sharded_wfns_rchunk_slice` has a sharding annotation problem.

## The specific OOM

XLA log:
```
Can't reduce memory use below 14.12 GiB by rematerialization;
only reduced to 22.66 GiB
```

This means XLA compiled a kernel that needs 22.66 GB, but only 14.12 GB is
free. The 22.66 GB is from **involuntary full rematerialization** caused by
a gather operation at `load_wfns.py:548` that transitions from
`{devices=[1,16,1,1]}` to `{devices=[1,1,1,4,4]}` — XLA can't reshard
the gather output without recomputing the entire input tensor.

## Root cause

`get_sharded_wfns_rchunk_slice` (line ~290 in load_wfns.py) does:
1. FFT the band-sharded psi_G → psi_r (sharded on bands across 16 devices)
2. Flatten to (nk, nb, ns, n_rtot) — still band-sharded
3. Slice to r-chunk: `psi_r[:, :, :, r_start:r_end]` — still band-sharded
4. `jnp.take` with linear indices for the gather
5. Reshard to output sharding `{-, -, -, Y}` (centroids on Y axis)

The transition from `{-, 16-way-band, -, -}` to `{-, -, -, 4×4}` at step 5
forces XLA to rematerialize the entire tensor. With 1000 k-points, that's
22.66 GB.

## What was already fixed

`get_sharded_wfns_centroids` (the centroid extraction path) was fixed to use
direct 3D spatial indexing instead of flatten+linear-gather, avoiding the
rematerialization. But `get_sharded_wfns_rchunk_slice` (the r-chunk path)
still uses the old pattern.

## What to do next

1. Fix `get_sharded_wfns_rchunk_slice` with the same approach: either
   - Use 3D spatial indexing (convert r_start:r_end to 3D grid coordinates)
   - Or add intermediate resharding constraints that XLA can handle

2. More fundamentally: the r-chunk path extracts a contiguous slice of
   flattened r-space. This is inherently 1D, so 3D indexing doesn't apply
   directly. The fix may need to keep bands replicated (not sharded) during
   the r-chunk extraction, or shard on k-points instead of bands.

3. The memory model (4 copies of the shard during FFT) is now calibrated
   and correct for the centroid path. The r-chunk path may have different
   memory characteristics because the output is a different shape.

## Measured memory model

From 1-GPU profiling of Si 4×4×4:

| Step | Peak | Arrays alive |
|------|------|-------------|
| FFT | 4× input | input + output + 2 FFT staging buffers |
| Phase multiply | 3× input (staging freed) | input + output + phased |
| Centroid gather | 3× + small | above + psi_rmu (tiny) |

Peak factor: **4 countable copies** of the per-device shard during FFT.
Per-device shard = nk × ceil(nb / n_devices) × nspinor × n_rtot × 16 bytes.

## Files changed in this session

- `src/common/load_wfns.py`: k-chunking, make_array_from_callback,
  3D centroid gather, memory model
- `src/common/isdf_fitting.py`: extracted from load_wfns
- `src/common/fft_helpers.py`: extracted from load_wfns
- `src/common/bispinor_init.py`: extracted from load_wfns
- `src/gw/compute_vcoul.py`: fused V_q pipeline, GPU-side accumulation
- `src/common/symmetry_maps.py`: improper spinor fix + multihost fix
