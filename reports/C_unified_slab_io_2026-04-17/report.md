# Unified SlabIO — first GPU verification (lorrax_C)

**Date:** 2026-04-17  **Agent:** lorrax_C  **Branch:** `agent/C-unified-slab-io`

## TL;DR

Unified [`file_io.slab_io`](../../sources/lorrax_C/src/file_io/slab_io.py) ships a single helper (`SlabIO` + `write_slab` / `read_slab` / `accumulate_slab`) behind one `use_ffi_io: bool = False` flag.  Default (`False`) is `process_allgather` → rank-0 h5py (byte-identical to the old hand-rolled pattern); `True` routes through the collective-MPI-IO `ffi.phdf5`.

**Verified end-to-end on 4-GPU Perlmutter** (2-node cluster alloc, Si 4x4x4 25 Ry GN-PPM + MoS2 4x4 30 Ry GN-PPM):

- `phdf5_write_test` round-trip after N-D FFI refactor: **PASS**, `max|diff| = 0.000e+00` (serial h5py + parallel FFI read-back).
- `gw_jax` full GN-PPM run, both backends: **bit-for-bit identical output**.
  - All six sigma datasets: `max|diff| = 0.000e+00`.
  - `eqp0.dat` byte-identical.
- Wall-clock: essentially tied (see table).

## Numbers

| System | Backend | Wall | `ppm_sigma` | `zeta_fit_chunked` | Note |
|---|---|---|---|---|---|
| Si 4x4x4 25 Ry, 60 band, 480 cent | allgather | 54.1 s | 10.0 s | 18.4 s | sigma_mnk.h5 = 313 MB |
| Si 4x4x4 25 Ry, 60 band, 480 cent | FFI | 53.2 s | 10.7 s | 18.0 s | 6/6 datasets identical |
| MoS2 4x4 30 Ry, 80 band, 640 cent | allgather | 49.3 s | (inside Total 33.5) | 19.0 s | sigma_mnk.h5 = 275 MB |
| MoS2 4x4 30 Ry, 80 band, 640 cent | FFI | 51.3 s | (inside Total 34.0) | 18.5 s | 6/6 datasets identical |

No speed-up on single-node 4-GPU (as expected).  The `sigma_mnk.h5` write is only ~300 MB and fits in the Linux page cache — wall-clock is dominated by GW compute (`zeta_fit_chunked`, `ppm_sigma`), and neither backend spends measurable time in actual I/O here.

## Where we expect the FFI to win (not tested here)

- Multi-node: the sigma write bandwidth scales with number of aggregators; the `phdf5_vs_gather_bench` result at 4 nodes / 16 GPUs showed 8× over gather for 4 GB writes.
- Cold cache: writes that actually hit disk (large files written by prior jobs, file sizes > per-node RAM) will show the big difference.
- Bigger sigma: tensors that exceed rank-0 RAM budget in the allgather path.

To test these, a multi-node allocation would be needed.  This run was single-node by design (per user: `sources/lorrax_C` gets one alloc, sibling agents own the other two slots).

## Bugs caught during this test

Two real bugs not catchable from login-node alone — committed as `3f37918`:

1. **FFI array attrs must be numpy.ndarray, not tuple.**  JAX's `jax.ffi.ffi_call` passes tuples through as hashable scalars; XLA then fails with `Unsupported attribute type`.  Fix: `offset_base`, `mesh_shape`, `axis_for_dim` all built as `np.asarray(..., dtype=np.int64)`.
2. **`_sharding_to_axis_for_dim` has to walk `A.ndim`, not `spec` length.**  JAX canonicalises `PartitionSpec(None, None)` to `PartitionSpec()`, so iterating over the spec returned an empty tuple for fully replicated arrays, tripping the FFI's `axis_for_dim.size != A.ndim` assertion.  Fix: iterate `range(ndim)` with the spec, treating missing trailing entries as `None`.

Both bugs only show up on a real dispatch through `jax.ffi.ffi_call` on a 4D sharded array — can't reproduce from static import / ast parsing / the 2-D `phdf5_write_test` round-trip (which uses an explicit-length tuple spec, masking the second bug).

## Recommendation

v1 of SlabIO is ready to merge.  The default (allgather) path is risk-free — it's the current code's behaviour consolidated behind one API.  The FFI path is proven correct; timing benefit is a multi-node / I/O-dominated story for the next bench round.

## Repro

```bash
# 4-GPU Si verification
cd runs/Si/C_unified_slab_io/00_gnppm_allgather
LORRAX_NGPU=4 $LORRAX_SRC/ffi/common/cpp/run_shifter.sh env \
    XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async HDF5_USE_FILE_LOCKING=FALSE \
    python3 -u -m gw.gw_jax -i cohsex.in

cd ../01_gnppm_ffi   # adds "use_ffi_io = true" to cohsex.in
LORRAX_NGPU=4 ... same command

# diff check
python3 -c "
import h5py, numpy as np
f1 = h5py.File('00_gnppm_allgather/sigma_mnk.h5', 'r')
f2 = h5py.File('01_gnppm_ffi/sigma_mnk.h5', 'r')
for k in f1.keys():
    print(k, np.max(np.abs(f1[k][...] - f2[k][...])))
"
```
