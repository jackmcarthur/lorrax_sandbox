"""Minimal probe: open + (optional) create_dataset + close."""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
import sys, time, tempfile
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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

from jax.sharding import Mesh, PartitionSpec as P
from file_io.slab_io import SlabIO

def log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)

def main():
    n_ds = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    do_writes = len(sys.argv) > 2 and sys.argv[2] == "write"

    devs = jax.devices()[:4]
    mesh = Mesh(np.asarray(devs).reshape(2, 2), ("x", "y"))

    # Path must be identical on all ranks (MPI-IO collective H5Fcreate).
    # Use a non-PID-dependent name to avoid rank-local divergence.
    path = os.path.join(tempfile.gettempdir(), "phdf5_probe_bench.h5")
    if jax.process_index() == 0 and os.path.exists(path):
        os.remove(path)
    log(f"path={path}  n_ds={n_ds}  writes={do_writes}")

    log("about to SlabIO open")
    t0 = time.perf_counter()
    w = SlabIO(path, mode="w", mesh=mesh, use_ffi_io=True)
    log(f"SlabIO open done in {(time.perf_counter()-t0)*1000:.1f}ms")

    for i in range(n_ds):
        log(f"about to create_dataset ds{i}")
        t0 = time.perf_counter()
        w.create_dataset(f"ds{i}", shape=(256, 256), dtype=jnp.float64)
        log(f"  create_dataset ds{i} done in {(time.perf_counter()-t0)*1000:.1f}ms")

    if do_writes:
        from jax.sharding import NamedSharding
        A = jax.device_put(jnp.ones((256, 256), dtype=jnp.float64),
                           NamedSharding(mesh, P('x', 'y')))
        A.block_until_ready()
        for i in range(n_ds):
            log(f"about to write ds{i}")
            t0 = time.perf_counter()
            w.write_slab(f"ds{i}", A, global_shape=(256, 256))
            t1 = time.perf_counter()
            w._backend._drain_pending()
            log(f"  ds{i}: dispatch={(t1-t0)*1000:.2f}ms  ready={(time.perf_counter()-t1)*1000:.2f}ms")

    log("about to close")
    t0 = time.perf_counter()
    w.close()
    log(f"close done in {(time.perf_counter()-t0)*1000:.1f}ms")
    try:
        os.remove(path)
    except OSError:
        pass

if __name__ == "__main__":
    main()
