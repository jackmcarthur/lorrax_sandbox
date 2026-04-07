# Detailed Buffer Inventory at Memory Bottlenecks

**System**: Si 4×4×4, nk=64, nb=35 (pad=36), ns=2, FFT 24³, n_rmu=240
**Hardware**: 4× A100-40GB, 2×2 mesh (p_x=2, p_y=2), single process
**Shard**: nk × (nb_pad/P) × ns × n_rtot = 64 × 9 × 2 × 13824 = 0.255 GB/dev

## Centroid extraction pipeline

| Step | GPU0 used | GPU0 peak | Delta from prev | What happened |
|------|-----------|-----------|-----------------|---------------|
| Clean start | 0.000 | 0.000 | — | Nothing on device |
| psi_G on device | 0.268 | 0.268 | +0.268 | 1 shard (0.255) + 0.013 sharding metadata |
| **After FFT** | 0.523 | **1.329** | +0.255 used, **+1.061 peak** | psi_G + psi_r survive; peak from staging |
| After phase multiply | 0.819 | 1.329 | +0.296 | psi_G + psi_r + psi_phased (staging freed) |
| After freeing psi_G, psi_r | 0.296 | 1.329 | -0.523 | Only psi_phased + phase remain |
| After centroid gather | 0.555 | 1.329 | +0.259 | psi_phased + psi_flat(=reshape) + psi_rmu |
| After freeing FFT arrays | 0.019 | 1.329 | -0.536 | Only psi_rmu (tiny: 0.009 GB/dev) |
| After reshard step 1 (XY→Y) | 0.036 | 1.329 | +0.017 | psi_rmu + psi_rmu_Y |
| After reshard step 2 (Y→outY) | 0.045 | 1.329 | +0.009 | + psi_rmu_final |
| After cleanup | 0.023 | 1.329 | -0.022 | Only psi_rmu_final |
| After transpose + X-shard | 0.032 | 1.329 | +0.009 | psi_rmu_final + psi_rmuT |

**FFT peak decomposition (1.329 GB):**
- psi_G shard: 0.255 GB (input, kept alive)
- psi_r shard: 0.255 GB (output)
- FFT staging ×2: 0.510 GB (freed after FFT completes)
- Subtotal 4 copies: 1.019 GB
- Remaining: 1.329 - 1.019 = **0.310 GB unexplained**
  - Phase array (replicated): 0.014 GB
  - shard_map metadata/buffers: ~0.03 GB
  - XLA JIT compilation cache: ~0.27 GB (first run only)

**After first JIT run, the 0.27 GB XLA cache persists in peak but not in `used`.**

## R-chunk extraction pipeline (with G-cache)

| Step | GPU0 used | GPU0 peak | Delta | What's on device |
|------|-----------|-----------|-------|-----------------|
| G-cache loaded | 0.268 | 1.329 | +0.268 | G-cache (1 band chunk = 1 shard) |
| After r-chunk slice | 0.792 | 1.329 | +0.524 | G-cache + psi_flat + psi_rchunk (2 shards) |
| **Reshard step 1 (XY→Y)** | 1.329 | **2.147** | **+0.537/+0.818** | + reshard intermediate |
| **Reshard step 2 (Y→outY)** | 1.838 | **2.348** | **+0.509/+0.201** | + final output |
| After cleanup | 1.046 | 2.348 | -0.792 | G-cache + psi_rchunk_Y2 |

**R-chunk reshard step 1 peak decomposition (2.147 GB):**
- G-cache: 0.268 GB (1 shard)
- psi_rchunk band-sharded: 0.255 GB (1 shard)
- psi_flat (reshape of G-cache): 0.268 GB (alias, but may be separate in XLA)
- Reshard intermediate {-,Y,-,-}: nk × (nb_pad/p_y) × ns × n_rtot / 1e9
  = 64 × 18 × 2 × 13824 × 16 / 1e9 = **0.510 GB**
- Subtotal: 0.268 + 0.255 + 0.268 + 0.510 = 1.301 GB
- Measured `used`: 1.329 GB → 0.028 GB overhead
- Measured `peak`: 2.147 GB → **0.818 GB above `used`** = XLA execution buffers
  during the all-gather + all-to-all collectives

**R-chunk reshard step 2 peak decomposition (2.348 GB):**
- Everything from step 1 + the final output {-,-,-,Y}
- Final output: nk × nb × ns × (n_rtot / p_y) = 64 × 35 × 2 × 6912
  = 0.510 GB per device (but nb=35 not nb_pad=36, so slightly less)
- Peak increase: 2.348 - 2.147 = 0.201 GB (just the output being allocated)

## Scaling predictions for 10×10×10

For nk=1000, nb_pad=64, mesh 4×4 (p_x=4, p_y=4), n_rmu=480:

| Buffer | Per device formula | 4×4 mesh value |
|--------|-------------------|----------------|
| psi_G shard | nk × (nb_pad/P) × ns × n_rtot × 16 | 1.77 GB |
| psi_r shard (=psi_G) | same | 1.77 GB |
| FFT staging ×2 | 2 × above | 3.54 GB |
| **FFT peak (4 copies)** | **4 × shard** | **7.08 GB** |
| phase (replicated) | nk × n_rtot × 16 | 0.22 GB |
| XLA JIT cache | ~0.3 GB | 0.30 GB |
| **Total centroid peak** | | **~7.6 GB** |
| | | |
| G-cache (all band chunks) | nk × (nb_pad/P) × ns × n_rtot × 16 | 1.77 GB |
| Reshard intermediate {-,Y,-,-} | nk × (nb_pad/p_y) × ns × n_rtot × 16 | 7.08 GB |
| Reshard collective buffers | ~1.5× intermediate | ~10.6 GB |
| **Total r-chunk reshard peak** | | **~19.5 GB** |

The r-chunk reshard peak (19.5 GB) is the binding constraint. The centroid
extraction peak (7.6 GB) fits comfortably.

**Note**: The "reshard collective buffers" (~1.5× intermediate) is measured,
not derived from countable buffers. It likely includes NCCL send/receive
buffers for the all-gather and all-to-all operations. On 16 processes
(vs single process), additional per-process NCCL state may increase this.

## Isolated collective buffer measurements

Tested individual `with_sharding_constraint` calls in standalone JITs
(no FFT context) to measure pure collective costs.

### All-gather along X: {-,XY,-,-} → {-,Y,-,-}

| Config | Input/dev | Output/dev | Peak delta | NCCL overhead | Ratio |
|--------|-----------|------------|------------|---------------|-------|
| centroids (0.004→0.009) | 0.004 GB | 0.009 GB | 0.034 GB | 0.025 GB | 2.8× output |
| full n_rtot (0.255→0.510) | 0.255 GB | 0.510 GB | 1.301 GB | 0.536 GB | 1.1× output |

The NCCL overhead for an all-gather is approximately **1× the output size**.
This is the send/receive buffer that NCCL allocates for the ring all-gather.

**Model**: `peak_allgather = input + 2 × output`
- input: nk × (nb_pad/P) × ns × last_dim × 16 / dev
- output: nk × (nb_pad/p_y) × ns × last_dim × 16 / dev
- NCCL buffer: ≈ output

### All-to-all along Y: {-,Y,-,-} → {-,-,-,Y}

Not measured (the 10×10×10 full-r case OOMed — XLA tried to allocate
19.78 GB for the all-to-all, triggering rematerialization of the full
46 GB unsharded tensor). This confirms that even in a standalone JIT,
XLA's SPMD partitioner may rematerialize when the array doesn't divide
cleanly or when the collective buffer exceeds device memory.

### Remaining unknowns

1. **XLA JIT cache**: ~0.27 GB on first compilation, persists in peak.
   Scales with HLO graph complexity, not array size. Not yet measured
   as a function of graph size.

2. **shard_map overhead**: ~0.013 GB per shard_map call. Small, fixed.

3. **Multi-process overhead**: Not measured in this assay (single process
   with 4 GPUs). On 16 processes, NCCL creates per-process communicator
   state that may add 1-5 GB per device. This is the main unknown for
   the 10×10×10 scaling.

4. **All-to-all buffer scaling**: Unknown. The centroids all-to-all is
   too small to measure (0.009 GB). The full-r all-to-all OOMed before
   we could measure it. Need intermediate sizes to characterize.
