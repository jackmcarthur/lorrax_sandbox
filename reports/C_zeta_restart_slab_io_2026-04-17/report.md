# Zeta + V_q + restart-tensor SlabIO migration (lorrax_C)

**Date:** 2026-04-17 (afternoon, alloc 51700616)
**Branch:** `agent/C-unified-slab-io`
**Commits:** `21bf8ec` (zeta + V_q), `bb77390` (zeta batched write), `c6a68e5` (restart)

## What's migrated

Every big sharded-array HDF5 write in a gw_jax run now routes through
[`file_io.slab_io.SlabIO`](../../sources/lorrax_C/src/file_io/slab_io.py):

| Site | File(s) | Route via |
|---|---|---|
| Static COHSEX + dynamic Σ_c(ω) | `sigma_mnk.h5` | `write_sigma_omega_h5` (session 1) |
| zeta_q ISDF output | `tmp/zeta_q.h5` | `isdf_fitting.fit_zeta_chunked_to_h5` |
| g0_mu (ζ at G=0) | `tmp/zeta_q.h5` | still rank-0 h5py (small, one-shot) |
| V_q read path (zeta read) | `tmp/zeta_q.h5` | `compute_vcoul.compute_all_V_q_from_zeta_h5` |
| Restart tensors | `tmp/isdf_tensors_N.h5` | `tagged_arrays.write_restart_state_to_h5` |

`cfg.use_ffi_io` (from LorraxConfig / cohsex.in) is threaded end-to-end.

## Verification

Si 4x4x4 25 Ry GN-PPM full gw_jax run, 4 GPU single-node, both
backends:

| File | Datasets | Max diff |
|---|---|---|
| sigma_mnk.h5 | 6 × (hartree, omega, sigma_c, sigma_sx, sigma_total, sigma_xc_qsgw) | 0.000e+00 |
| tmp/zeta_q.h5 | zeta_q (4,4,4,480,13824) + g0_mu (4,4,4,4,480) | 0.000e+00 |
| tmp/isdf_tensors_480.h5 | V_qmunu, W0_qmunu, G0_mu_nu, enk_full, psi_full_y, restart_format_version | 0.000e+00 |

`eqp0.dat`: byte-identical.

## Timing

| Path | Total | zeta write | V_q compute | ppm_sigma |
|---|---|---|---|---|
| allgather | 53.2 s | 3.3 s | 2.8 s | 9.7 s |
| FFI       | 56.1 s | 5.3 s | 11.1 s | 10.0 s |

FFI is slower here (and so is zeta on FFI).  Two causes:

1. **V_q read path.** 64 q-points × 1 collective MPI-IO call each = 64 barriers.  Allgather backend reads every rank independently via cached h5py handles on a warm page cache (~50 ms per read).  FFI has per-call collective overhead that dominates for many small reads.
2. **zeta write on FFI.** Even after batching to one 5-D hyperslab per r-chunk (was 64 per-q calls — session mid-point optimisation), the reshape across a replicated leading axis adds some XLA reshaping cost.  Still 60 % faster than the per-q version.

Expected: multi-node runs and cold-cache big files are where the FFI wins — the [prior 16-GPU bench](../C_unified_slab_io_2026-04-17/report.md) showed 8× over gather for 4 GB collective writes.

## What's still allgather-only

- `_accumulate_kij_stream` in `ppm_sigma.py` (stream mode, single-process-only fallback anyway).
- Small one-off metadata: `g0_mu` write-back in `gw_init.py`, `W0_ready` dataset-attribute (bse_io reads it as an h5py attr).  These are rank-0 h5py calls outside the main SlabIO context.
- WFN reads (ragged G-vectors, separate design).
- BSE eigenvectors, kin_ion, DFT matrix elements — still on old pattern; low priority.

## Internal design notes that came out of this session

- **Per-rank cached h5py handle** in `_slab_io_allgather.read_slab` — every rank opens its own 'r' handle once per SlabIO lifetime, matching the pre-migration pattern.  Broadcasting from rank 0 via `multihost_utils.broadcast_one_to_all` turned out to be 4× slower than per-rank reads on the V_q loop (many small reads + per-call coordinator overhead).
- **`as_numpy=True`** flag on `read_slab` — skip the H2D+D2H round-trip when the caller is going to stage into a host numpy buffer anyway (the V_q pattern).  Brought compute_all_V_q from 10.3s → 2.8s on the allgather path.
- **Multi-mesh-axis sharding** in the FFI.  zeta_chunk uses `P(None, ('x','y'), ...)` — dim 1 sharded over BOTH mesh axes.  The C++ previously rejected that; now `axis_count_per_dim[]` + `axis_flat[]` encode arbitrary multi-axis layouts, and the C++ un-ravels coordinates through the product of participating mesh axes.
- **Deferred attr writes on FFI backend** — rank-0 h5py open + attr-set happens AFTER `close_ctx` flushes the MPI-IO handle; prevents metadata races.
- **`W0_ready` compatibility shim** — bse_io reads it as a dataset-level HDF5 attr.  `write_restart_state_to_h5` sets it via a rank-0 h5py 'a' open after the main SlabIO context closes, with a `sync_global_devices` barrier afterwards.

## Next steps

- Follow-up session: migrate wfn reads (ragged G-vectors → needs hyperslab-per-k helper) and the BSE eigenvector / kin_ion / DFT matrix-element writers.
- Multi-node bench on a bigger system to quantify the FFI win on restart-state writes (V_qmunu is ~2 GB at Si 4x4x4; ~100 GB at production sizes).
