# Memory Assay: Status and Handoff

**Date**: 2026-04-06
**System**: Si 4×4×4, nk=64, nb=35 (pad=36), ns=2, FFT 24³, n_rmu=240
**Location**: `runs/Si/04_si_4x4x4_memory_assay/`

## CRITICAL: all measurements were single-process (INVALID for production)

All memory profiling in this directory used `srun -n 1` (1 process, 4 GPUs).
This is NOT the production configuration. Production uses 1 process per GPU
(`srun -n 4` on 1 node, `-n 16` on 4 nodes). **Never run single-process
multi-GPU** — it hides NCCL inter-process overhead and gives misleading
memory numbers.

The `run_4proc.out` ran 4 processes correctly (`Processes: 4, Devices: 4`)
but has NO per-stage memory data because the instrumentation wasn't in the
code path.

## What exists and is valid

### `assay_results.json` + `assay.out`
Systematic sweep of 3 band_chunk × 4 r_chunk configurations. Measured
centroid extraction peak, G-cache load, and r-chunk extraction timing.
**Single-process only.** Key finding: peak scales linearly with band_chunk,
r_chunk has negligible impact on peak.

### `detailed_trace.out`
Step-by-step memory trace through centroid + r-chunk pipeline. **Single-
process only.** Shows per-step used/peak for 13 stages. This is the most
detailed data available.

### `collective_buffers.out`
Attempted to isolate all-gather and all-to-all costs. Two valid data points
(centroids and full-n_rtot all-gather). The 10×10×10 tests OOMed.
**Single-process only.**

## What the next agent should do

1. **Add `_mem_report()` calls to `isdf_fitting.py`** at each STEP boundary
   (after load_wfns, after CCT, after cholesky, before/after each r-chunk,
   after ZCT, after solve, after H5 write). The function `_mem_report` is
   already defined in the file — just add calls gated on `_MEM_PROFILE`.

2. **Run the full LORRAX calculation (`python3 -u -m gw.gw_jax -i cohsex.in`)
   with `LORRAX_MEM_PROFILE=1` and `srun -N 1 -n 4`** (4 processes, 1 GPU
   each). This gives real multi-process memory data.

3. **Re-derive the memory model from multi-process data.** The current model
   in `load_wfns.py` uses `peak_copies = 9` for multi-GPU, derived from
   single-process measurements. This needs validation on multi-process.

4. **Run the same on 4 nodes (`-N 4 -n 16`)** if 10×10×10 is the target.
   The NCCL inter-node overhead may add 1-5 GB per device that doesn't
   appear in single-node measurements.

5. **Update `docs/MEMORY_MODEL.md`** with the validated multi-process model.
   The current doc has the single-GPU 4× model (correct) but the multi-GPU
   section needs rewriting with actual multi-process data.

## Buffer model (single-process, needs multi-process validation)

### Centroid extraction
```
peak_per_device = 4 × shard + 0.3 GB
shard = nk × ceil(nb_pad / P) × nspinor × n_rtot × 16
```
4 countable FFT buffers (input + output + 2 staging) + XLA fixed overhead.

### R-chunk reshard
```
peak_per_device = 2×shard + 1.55×intermediate + output
shard = nk × ceil(nb_pad / P) × nspinor × n_rtot × 16  (G-cache + rchunk)
intermediate = nk × ceil(nb_pad / p_y) × nspinor × B_r × 16  (all-gather)
output = nk × nb × nspinor × ceil(B_r / p_y) × 16  (final Y-sharded)
```
The 1.55× = 1.0 array + 0.55 NCCL all-gather buffer. All-to-all reuses
the all-gather buffer (0 extra).

### IMPORTANT: nb_pad must be divisible by P

The band count is padded to `ceil(nb / P) × P` for clean sharding. Band
trimming must happen OUTSIDE the JIT to prevent XLA from rematerializing
the FFT to satisfy output sharding on the non-divisible actual band count.

## Files

| File | What | Valid? |
|------|------|--------|
| `run_assay.py` | Systematic chunk sweep script | Yes (single-proc) |
| `assay_results.json` | 12-config sweep results | Yes (single-proc) |
| `assay.out` | Sweep stdout | Yes (single-proc) |
| `detailed_trace.out` | 13-step memory trace | Yes (single-proc) |
| `detailed_trace_4proc.out` | 4-proc attempt (no distributed init) | INVALID |
| `collective_buffers.out` | Isolated collective costs | Partial (2 of 6 configs) |
| `run_4proc.out` | 4-proc full LORRAX run | Valid but NO memory data |
| `assay_report.md` | Initial report (9× claim) | Superseded by this file |
| `detailed_buffer_inventory.md` | Per-step buffer sizes | Valid (single-proc) |
