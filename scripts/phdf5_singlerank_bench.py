"""Single-rank async FFI probe: directly call the FFI custom_call without
shard_map, to isolate the per-call Python-thread block.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
import sys, time, tempfile
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ffi.common import ffi_loader
from ffi.common.ffi_loader import get_lib

def main():
    get_lib()

    # World size 1, mesh 1x1.  Open via the ctypes open call directly.
    path = "/tmp/phdf5_singlerank_bench.h5"
    if os.path.exists(path):
        os.remove(path)

    fh = ffi_loader.phdf5_open(
        path=path, p=1, q=1, rank=0, world_size=1, mode_flag=0)
    print(f"opened fh={fh}")

    n = 2048
    A = jax.numpy.ones((n, n), dtype=jnp.float64)
    A.block_until_ready()

    ds_id = ffi_loader.phdf5_ensure_dataset(
        fh, "ds0", (n, n), "float64")
    print(f"ensure_dataset ds0 → ds_id={ds_id}")

    from ffi.phdf5.write import ffi_write_call

    def call_one(tag):
        t0 = time.perf_counter()
        tok = ffi_write_call(A,
            ctx_handle=int(fh), ds_id=int(ds_id),
            offset_base=(0, 0),
            mesh_shape=(1, 1),
            axis_count_per_dim=(0, 0),
            axis_flat=(),
        )
        t1 = time.perf_counter()
        # Separate block_until_ready
        if hasattr(tok, 'block_until_ready'):
            tok.block_until_ready()
        t2 = time.perf_counter()
        print(f"  {tag}: ffi_call return={(t1-t0)*1000:8.2f}ms   "
              f"tok.ready={(t2-t1)*1000:8.2f}ms")

    # Run via jit: XLA async FFI Future should surface as a non-blocking
    # returned array.
    for i in range(4):
        ds_name = f"ds{i}"
        ds_id_i = ffi_loader.phdf5_ensure_dataset(fh, ds_name, (n, n), "float64")
        def _inner(A):
            return ffi_write_call(A,
                ctx_handle=int(fh), ds_id=int(ds_id_i),
                offset_base=(0, 0),
                mesh_shape=(1, 1),
                axis_count_per_dim=(0, 0),
                axis_flat=(),
            )
        jitted = jax.jit(_inner)
        t0 = time.perf_counter()
        ts0 = int(time.time() * 1000)
        tok = jitted(A)
        t1 = time.perf_counter()
        ts1 = int(time.time() * 1000)
        tok.block_until_ready()
        t2 = time.perf_counter()
        print(f"  jit ds{i}: t0={ts0%100000}ms  t1={ts1%100000}ms  "
              f"jit_call={(t1-t0)*1000:6.1f}ms  ready={(t2-t1)*1000:6.2f}ms")

    ffi_loader.phdf5_close(fh)
    if os.path.exists(path):
        os.remove(path)

if __name__ == "__main__":
    main()
