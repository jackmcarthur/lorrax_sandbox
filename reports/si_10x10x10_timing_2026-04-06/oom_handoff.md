# Si 10×10×10 OOM — Updated Handoff (2026-04-06 evening)

## Status

Memory model updated to 9× shard for multi-GPU (measured from assay).
16 GPUs should now fit 1000 k-points in a single centroid-extraction chunk
(predicted 16 GB peak < 28 GB budget). NOT YET TESTED — allocation expired.

## What changed this session

1. **load_wfns.py refactor**: 2000 → 700 lines. ISDF fitting extracted to
   `isdf_fitting.py`, FFT helpers to `fft_helpers.py`, bispinor to
   `bispinor_init.py`. All tests pass, numerically identical.

2. **Centroid extraction OOM fix**: keep padded bands through reshard, trim
   outside JIT. Split JIT prevents FFT rematerialization. VERIFIED working
   (10×10×10 passed centroid stage on 16 GPUs, reached r-chunk stage).

3. **R-chunk reshard**: split JIT same as centroids. Two-step {-,XY,-,-} →
   {-,Y,-,-} → {-,-,-,Y}. The {-,Y,-,-} intermediate is the binding
   constraint (nb_pad/p_y bands × full r_chunk per device).

4. **Memory model**: FFT peak is 4× on 1 GPU, 9× on multi-GPU (measured).
   The extra 5× is from collective communication buffers during the
   band-axis reshard. Validated on Si 4×4×4 with 4 GPUs across 12
   configurations (3 band chunks × 4 r chunks).

5. **V_q pipeline**: fused kernels, GPU-side accumulation via .at[].set().

6. **Improper spinor fix**: mirrors/S6 get correct SU(2) spinor.

## Next steps

1. **Test 10×10×10 on 16 GPUs** with the 9× model. Prediction: 16 GB peak,
   fits in 28 GB budget, no k-chunking needed.

2. If it passes centroid+r-chunk: run full sigma and time it vs BGW.

3. The r-chunk reshard intermediate may still OOM if the chunk solver picks
   too large an r_chunk. The chunk solver in `compute_optimal_chunks`
   needs the reshard buffer term added (documented in MEMORY_MODEL.md
   but not yet in the solver code).

## Files on LORRAX main

All pushed. Key commits: `1808c15` (9× model), `6df6ac4` (split JIT reshard),
`1911929` (centroid padded bands fix), `892ea9c` (load_wfns refactor).
