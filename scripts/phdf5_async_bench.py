"""Dispatch-vs-complete benchmark for the phdf5 FFI write handler.

Measures:
  - dispatch_ms:  time from Python calling write_slab to write_slab
                  returning (main-thread block)
  - ready_ms:     time from write_slab returning to token.block_until_ready()
                  completing (actual H5Dwrite duration)

Async handler: dispatch_ms ~= 1-5 ms; ready_ms absorbs H5Dwrite.
Sync handler:  dispatch_ms ~= ready_ms summed into the write_slab call;
               block_until_ready is a no-op.

Usage:
    # in a shell with lorrax_C module loaded + SLURM_JOBID exported
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \\
        HDF5_USE_FILE_LOCKING=FALSE \\
        lxrun python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/phdf5_async_bench.py \\
          --n-chunks 4 --chunk-mb 270
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import tempfile
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Bootstrap jax.distributed (SLURM-aware)
_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                         os.environ.get("JAX_NUM_PROCESSES",
                         os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"

_maybe_init_jax_distributed()

from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from file_io.slab_io import SlabIO


def _log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--chunk-mb", type=float, default=270.0,
                    help="target MB per rank per chunk (F64)")
    args = ap.parse_args()

    world = jax.process_count() * jax.local_device_count()
    if world != 4:
        _log(f"expected 4 devices, got {world}; results may be skewed")

    # 2x2 mesh
    devices = jax.devices()[:4]
    mesh = Mesh(np.asarray(devices).reshape(2, 2), ("x", "y"))

    # Pick per-rank shape to hit chunk-mb bytes (F64 = 8 bytes/elt).
    # Per-rank elts = chunk_mb * 1024**2 / 8.  Using square shards.
    elts_per_rank = int(args.chunk_mb * 1024**2 / 8)
    n_local = int(np.sqrt(elts_per_rank))
    n_rows = n_local * 2   # mesh dim x=2
    n_cols = n_local * 2
    _log(f"world={world}, mesh=(2,2), n={n_rows}x{n_cols}, "
         f"per-rank={n_local}x{n_local} = "
         f"{n_local*n_local*8/1024/1024:.1f} MB F64")

    # Build a sharded dummy array on each rank.
    @jax.jit
    def make_A(key):
        return jax.random.normal(key, (n_rows, n_cols), dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    A = jax.device_put(make_A(key), NamedSharding(mesh, P('x', 'y')))
    A.block_until_ready()

    # MPI-IO H5Fcreate is collective; every rank must see the SAME path.
    # PID differs per rank, so use a fixed filename.
    tmpdir = os.environ.get("LORRAX_BENCH_DIR", tempfile.gettempdir())
    path = os.path.join(tmpdir, "phdf5_async_bench.h5")
    if jax.process_index() == 0 and os.path.exists(path):
        os.remove(path)
    _log(f"file: {path}")

    with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=True) as w:
        backend = w._backend
        # Pre-create all datasets (so first-call ensure_dataset doesn't
        # skew the dispatch measurement of chunk 0).
        ds_name_list = [f"ds{i}" for i in range(args.n_chunks)]
        for name in ds_name_list:
            w.create_dataset(name, shape=(int(n_rows), int(n_cols)),
                             dtype=jnp.float64)

        results = []
        for i, name in enumerate(ds_name_list):
            t0 = time.perf_counter()
            w.write_slab(name, A, global_shape=A.shape)
            t_dispatch = time.perf_counter() - t0
            t1 = time.perf_counter()
            # Drain the Python worker — wait for this task to complete.
            backend._drain_pending()
            t_ready = time.perf_counter() - t1
            results.append((i, t_dispatch, t_ready))
            _log(f"  chunk {i}: "
                 f"dispatch={t_dispatch*1000:9.2f} ms   "
                 f"ready={t_ready*1000:9.2f} ms   "
                 f"total={(t_dispatch+t_ready)*1000:9.2f} ms")

        _log("")
        if len(results) > 1:
            r = results[1:]   # drop first call (compile + dset-alloc overhead)
            mean_d = np.mean([x[1] for x in r]) * 1000
            mean_r = np.mean([x[2] for x in r]) * 1000
            _log(f"mean (chunks 1..N):  dispatch={mean_d:.2f} ms   "
                 f"ready={mean_r:.2f} ms")

    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
